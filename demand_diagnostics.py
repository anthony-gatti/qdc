"""Optional application-demand diagnostics for SeQUeNCe QPQ reservations.

The collector instruments resource-manager instances at runtime.  It does not
modify SeQUeNCe and deliberately keeps the detailed records out of schema-v2
CSV output.
"""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from types import MethodType

from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.entanglement_management.generation.single_heralded import SingleHeraldedA
from sequence.components.bsm import SingleHeraldedBSM
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.resource_management.memory_manager import MemoryInfo


def reservation_key(reservation) -> tuple:
    return (
        reservation.identity,
        reservation.initiator,
        reservation.responder,
        int(reservation.start_time),
        int(reservation.end_time),
    )


def edge_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


_PROTOCOL_CLASS_INSTRUMENTED = False


def _install_protocol_class_instrumentation() -> None:
    """Install allocation-free wrappers once for official application EG."""
    global _PROTOCOL_CLASS_INSTRUMENTED
    if _PROTOCOL_CLASS_INSTRUMENTED:
        return
    _PROTOCOL_CLASS_INSTRUMENTED = True
    original_start = SingleHeraldedA.start
    original_received = SingleHeraldedA.received_message
    original_emit = SingleHeraldedA.emit_event

    def diagnostic(protocol):
        return getattr(getattr(protocol, "owner", None), "qdc_demand_diagnostics", None)

    def start(protocol):
        collector = diagnostic(protocol)
        edge = collector._edge_for_protocol(protocol) if collector else None
        before = len(protocol.scheduled_events)
        if edge is not None:
            edge["protocol_start_calls"] += 1
        result = original_start(protocol)
        if edge is not None and len(protocol.scheduled_events) > before:
            edge["starts_that_schedule_an_emission"] += 1
        return result

    def received_message(protocol, src, msg):
        collector = diagnostic(protocol)
        edge = collector._edge_for_protocol(protocol) if collector else None
        before = len(protocol.scheduled_events)
        result = original_received(protocol, src, msg)
        if edge is not None:
            scheduled = protocol.scheduled_events[before:]
            if any(event.process.activation == "emit_event" for event in scheduled):
                edge["protocol_callbacks_that_schedule_emission"] += 1
        return result

    def emit_event(protocol):
        collector = diagnostic(protocol)
        edge = collector._edge_for_protocol(protocol) if collector else None
        memory = protocol.memory
        previous = memory.excited_photon
        result = original_emit(protocol)
        photon = memory.excited_photon
        if edge is not None and photon is not None and photon is not previous:
            reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
            photon.qdc_generation_marker = {
                "source": "application",
                "reservation_key": list(reservation_key(reservation)),
                "edge": list(edge_key(protocol.owner.name, protocol.remote_node_name)),
                "node": protocol.owner.name,
                "protocol": protocol.name,
                "round": protocol.ent_round,
            }
            edge["endpoint_emissions"] += 1
        return result

    SingleHeraldedA.start = start
    SingleHeraldedA.received_message = received_message
    SingleHeraldedA.emit_event = emit_event


class ApplicationDemandDiagnostics:
    """Collect comparable application demand and generation metrics."""

    def __init__(
        self,
        network_topo,
        query_specs: list[dict],
        backend: str,
        memory_budget: int,
        topo_json_path: str | None = None,
        include_events: bool = True,
    ):
        self.topology = network_topo
        self.timeline = network_topo.get_timeline()
        self.query_specs = {spec["query_id"]: spec for spec in query_specs}
        self.backend = backend
        self.memory_budget = memory_budget
        self.reservations = {}
        self.events = []
        self.memory_snapshots = []
        self.physical_snapshots = []
        self._physical_config = self._physical_config_fingerprint(topo_json_path)
        self.include_events = include_events
        self._installed = False

    def install(self, routers: list) -> None:
        if self._installed:
            return
        self._installed = True
        _install_protocol_class_instrumentation()
        for router in routers:
            router.qdc_demand_diagnostics = self
            self._wrap_resource_manager(router)
        for bsm_node in self.topology.get_nodes_by_type("BSMNode"):
            for component in bsm_node.get_components_by_type(SingleHeraldedBSM):
                self._wrap_bsm(component)

    def _wrap_resource_manager(self, router) -> None:
        manager = router.resource_manager
        original_generate = manager.generate_load_rules
        original_load = manager.load
        original_send = manager.send_request
        original_update = manager.update

        def generate_load_rules(_manager, path, reservation, timecards, memory_array_name):
            self.register_reservation(path, reservation)
            return original_generate(path, reservation, timecards, memory_array_name)

        def load(_manager, rule):
            self.record_rule_installed(router, rule)
            return original_load(rule)

        def send_request(_manager, protocol, req_dst, req_condition_func, req_args):
            self._instrument_protocol(protocol, router)
            self.record_protocol_requested(router, protocol, req_dst)
            return original_send(protocol, req_dst, req_condition_func, req_args)

        def update(_manager, protocol, memory, state):
            self.record_protocol_update(router, protocol, state)
            return original_update(protocol, memory, state)

        manager.generate_load_rules = MethodType(generate_load_rules, manager)
        manager.load = MethodType(load, manager)
        manager.send_request = MethodType(send_request, manager)
        manager.update = MethodType(update, manager)

    @staticmethod
    def _stable_hash(value) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _physical_config_fingerprint(self, topo_json_path: str | None) -> dict:
        if topo_json_path is None:
            return {}
        with open(topo_json_path) as source:
            topology = json.load(source)
        physical = json.loads(json.dumps(topology))
        for template in physical.get("templates", {}).values():
            template.pop("adaptive_max_memory", None)
        return {"sha256": self._stable_hash(physical), "config": physical}

    def _instrument_protocol(self, protocol, router) -> None:
        if not isinstance(protocol, EntanglementGenerationA) or getattr(protocol, "_qdc_instrumented", False):
            return
        protocol._qdc_instrumented = True
        reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
        edge_record = self._edge_record(
            reservation, (router.name, protocol.remote_node_name)
        )
        if edge_record is None:
            return
        edge_record["unique_protocol_instances"] += 1

    def _edge_for_protocol(self, protocol) -> dict | None:
        reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
        return self._edge_record(
            reservation, (protocol.owner.name, protocol.remote_node_name)
        )

    def _wrap_bsm(self, bsm) -> None:
        original_get = bsm.get

        def get(_bsm, photon, **kwargs):
            now = _bsm.timeline.now()
            prior = None
            if _bsm.photon_arrival_time == now:
                prior = next(
                    (candidate for candidate in _bsm.photons if candidate.location != photon.location),
                    None,
                )
            result = original_get(photon, **kwargs)
            if prior is None:
                return result
            markers = [
                getattr(prior, "qdc_generation_marker", None),
                getattr(photon, "qdc_generation_marker", None),
            ]
            if not all(marker and marker.get("source") == "application" for marker in markers):
                return result
            first = markers[0]
            record = self.reservations.get(tuple(first["reservation_key"]))
            if record is None:
                return result
            edge_record = record["edges"].get("|".join(first["edge"]))
            if edge_record is not None:
                edge_record["bsm_attempts"] += 1
            return result

        bsm.get = MethodType(get, bsm)

    def _record_for(self, reservation) -> dict | None:
        if reservation is None or reservation.__class__.__name__ == "ReservationAdaptive":
            return None
        return self.reservations.get(reservation_key(reservation))

    def register_reservation(self, path: list[str], reservation) -> None:
        if reservation.__class__.__name__ == "ReservationAdaptive":
            return
        key = reservation_key(reservation)
        if key in self.reservations:
            return
        spec = self.query_specs.get(reservation.identity, {})
        round_num = 1 if int(reservation.start_time) == int(spec.get("start_time", -1)) else 2
        edges = [edge_key(a, b) for a, b in zip(path, path[1:])]
        self.reservations[key] = {
            "key": list(key),
            "backend": self.backend,
            "acp_memory_budget": self.memory_budget,
            "query_id": reservation.identity,
            "round": round_num,
            "path": list(path),
            "start_time_ps": int(reservation.start_time),
            "deadline_ps": int(reservation.end_time),
            "entanglement_number": reservation.entanglement_number,
            "edges": {
                "|".join(edge): {
                    "edge": list(edge),
                    "elementary_pair_demand": reservation.entanglement_number,
                    "compatible_cached_pair_available": 0,
                    "cache_adoption_attempted": 0,
                    "cache_adoption_succeeded": 0,
                    "fresh_generation_required_after_adoption": reservation.entanglement_number,
                    "fresh_generation_rules_requested": 0,
                    "fresh_generation_rules_installed": 0,
                    "fresh_generation_protocol_started": 0,
                    "fresh_generation_protocol_started_by_deadline": 0,
                    "protocol_request_objects": 0,
                    "unique_protocol_instances": 0,
                    "protocol_start_calls": 0,
                    "starts_that_schedule_an_emission": 0,
                    "protocol_callbacks_that_schedule_emission": 0,
                    "endpoint_emissions": 0,
                    "bsm_attempts": 0,
                    "completed_physical_attempts": 0,
                    "endpoint_success_updates": 0,
                    "successful_physical_pairs": 0,
                    "fresh_generation_succeeded": 0,
                    "fresh_generation_succeeded_by_deadline": 0,
                    "fresh_generation_cancelled_or_expired": 0,
                    "demand_unsatisfied_at_deadline": 0,
                    "blocking_reasons": {},
                }
                for edge in edges
            },
        }
        self.timeline.schedule(Event(reservation.start_time, Process(self, "snapshot_start", [key]), -2))
        self.timeline.schedule(Event(reservation.end_time, Process(self, "snapshot_deadline", [key]), -2))

    def _edge_record(self, reservation, edge) -> dict | None:
        record = self._record_for(reservation)
        if record is None:
            return None
        return record["edges"].get("|".join(edge_key(*edge)))

    def record_cache_attempt(self, reservation, edge, compatible: bool, succeeded: bool, reason: str = "") -> None:
        edge_record = self._edge_record(reservation, edge)
        if edge_record is None:
            return
        edge_record["cache_adoption_attempted"] += 1
        edge_record["compatible_cached_pair_available"] += int(compatible)
        edge_record["cache_adoption_succeeded"] += int(succeeded)
        edge_record["fresh_generation_required_after_adoption"] = max(
            0,
            edge_record["elementary_pair_demand"] - edge_record["cache_adoption_succeeded"],
        )
        if reason:
            reasons = edge_record["blocking_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1

    def record_rule_installed(self, router, rule) -> None:
        reservation = getattr(rule, "reservation", None)
        record = self._record_for(reservation)
        if record is None:
            return
        action_name = getattr(getattr(rule, "action", None), "__name__", "")
        if not action_name.startswith("eg_rule_action"):
            return
        args = getattr(rule, "action_args", {})
        path = args.get("path", record["path"])
        index = args.get("index", path.index(router.name))
        neighbor = path[index + 1] if "request" in action_name and index < len(path) - 1 else (
            path[index - 1] if index > 0 else None
        )
        if neighbor is None:
            return
        edge_record = self._edge_record(reservation, (router.name, neighbor))
        if edge_record is not None:
            edge_record["fresh_generation_rules_requested"] += 1
            edge_record["fresh_generation_rules_installed"] += 1

    def record_protocol_requested(self, router, protocol, req_dst) -> None:
        if req_dst is None or not isinstance(protocol, EntanglementGenerationA):
            return
        reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
        edge_record = self._edge_record(reservation, (router.name, protocol.remote_node_name))
        if edge_record is None:
            return
        edge_record["protocol_request_objects"] += 1
        # Deprecated schema-v1 compatibility alias.  This never counted start().
        edge_record["fresh_generation_protocol_started"] += 1
        if self.include_events:
            self.events.append({
                "event": "fresh_generation_protocol_requested",
                "time_ps": self.timeline.now(),
                "backend": self.backend,
                "query_id": reservation.identity,
                "round": self._record_for(reservation)["round"],
                "edge": list(edge_key(router.name, protocol.remote_node_name)),
                "node": router.name,
            })

    def record_protocol_update(self, router, protocol, state) -> None:
        if protocol is None or not isinstance(protocol, EntanglementGenerationA):
            return
        reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
        remote = getattr(protocol, "remote_node_name", None)
        if remote is None:
            return
        edge_record = self._edge_record(reservation, (router.name, remote))
        if edge_record is None:
            return
        if state == MemoryInfo.ENTANGLED:
            edge_record["endpoint_success_updates"] += 1
            if router.name < remote:
                edge_record["completed_physical_attempts"] += 1
                edge_record["successful_physical_pairs"] += 1
                edge_record["fresh_generation_succeeded"] += 1
        elif router.name < remote:
            edge_record["completed_physical_attempts"] += 1

    def snapshot_start(self, key: tuple) -> None:
        record = self.reservations.get(key)
        if record is None:
            return
        path_edges = {edge_key(*edge["edge"]) for edge in record["edges"].values()}
        for node_name in record["path"]:
            node = self.timeline.get_entity_by_name(node_name)
            counts = defaultdict(int)
            for info in node.resource_manager.memory_manager:
                memory = info.memory
                active = [
                    reservation
                    for reservation in node.network_manager.timecards[info.index].reservations
                    if reservation.start_time <= self.timeline.now() <= reservation.end_time
                ]
                app_owned = any(r.__class__.__name__ != "ReservationAdaptive" for r in active)
                adaptive_owned = any(r.__class__.__name__ == "ReservationAdaptive" for r in active)
                background = getattr(memory, "qdc_generation_source", "") == "background"
                counts["total_memories"] += 1
                counts["free_memories"] += int(info.state == MemoryInfo.RAW and not active)
                counts["application_owned_memories"] += int(app_owned)
                counts["adaptive_background_owned_memories"] += int(adaptive_owned)
                counts["entangled_background_memories"] += int(background and info.state in (MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED))
                link = tuple(sorted(getattr(memory, "qdc_link_endpoints", ())))
                counts["useful_background_memories_for_path"] += int(background and link in path_edges)
                counts["memories_transferred_to_application"] += int(bool(getattr(memory, "qdc_application_reservation", "")))
                counts["memories_blocked_from_application"] += int(adaptive_owned)
            path_neighbors = {
                neighbor
                for edge in path_edges
                if node_name in edge
                for neighbor in edge
                if neighbor != node_name
            }
            future_bins = []
            for neighbor in path_neighbors:
                middle = node.map_to_middle_node.get(neighbor)
                channel = node.qchannels.get(middle)
                if channel is None:
                    continue
                future_bins.extend(
                    channel.timebin_to_time(time_bin, channel.frequency)
                    for time_bin in channel.send_bins
                    if channel.timebin_to_time(time_bin, channel.frequency) >= self.timeline.now()
                )
            counts["reserved_path_quantum_channel_bins"] = len(future_bins)
            counts["furthest_reserved_path_channel_bin_ps"] = max(future_bins, default=self.timeline.now())
            counts["path_channel_backlog_ps"] = max(0, counts["furthest_reserved_path_channel_bin_ps"] - self.timeline.now())
            self.memory_snapshots.append({
                "backend": self.backend,
                "acp_memory_budget": self.memory_budget,
                "query_id": record["query_id"],
                "round": record["round"],
                "time_ps": self.timeline.now(),
                "node": node_name,
                **dict(counts),
            })
        self._snapshot_physical_state(record)

    def _snapshot_physical_state(self, record: dict) -> None:
        pending = [event for event in self.timeline.events if not event.is_invalid()]
        for edge in record["edges"].values():
            left, right = edge["edge"]
            left_node = self.timeline.get_entity_by_name(left)
            middle_name = left_node.map_to_middle_node[right]
            middle = self.timeline.get_entity_by_name(middle_name)
            bsm = middle.components[middle.first_component_name]
            path_nodes = {left, right, middle_name}
            relevant_events = [
                event for event in pending
                if getattr(getattr(event.process, "owner", None), "name", None) in path_nodes
                or getattr(getattr(getattr(event.process, "owner", None), "owner", None), "name", None) in path_nodes
            ]
            self.physical_snapshots.append({
                "backend": self.backend,
                "acp_memory_budget": self.memory_budget,
                "query_id": record["query_id"],
                "round": record["round"],
                "time_ps": self.timeline.now(),
                "edge": edge["edge"],
                "middle": middle_name,
                "bsm_photons_buffered": len(bsm.photons),
                "bsm_photon_arrival_time_ps": bsm.photon_arrival_time,
                "detector_next_detection_time_ps": [detector.next_detection_time for detector in bsm.detectors],
                "detector_photon_counters": [detector.photon_counter for detector in bsm.detectors],
                "pending_path_events": len(relevant_events),
                "pending_emit_events": sum(
                    event.process.activation == "emit_event" for event in relevant_events
                ),
                "pending_protocol_start_events": sum(
                    event.process.activation == "start" for event in relevant_events
                ),
                "endpoint_next_excite_time_ps": {
                    node_name: [
                        info.memory.next_excite_time
                        for info in self.timeline.get_entity_by_name(node_name).resource_manager.memory_manager
                    ]
                    for node_name in (left, right)
                },
            })

    def snapshot_deadline(self, key: tuple) -> None:
        record = self.reservations.get(key)
        if record is None:
            return
        for edge_record in record["edges"].values():
            edge_record["fresh_generation_protocol_started_by_deadline"] = edge_record[
                "fresh_generation_protocol_started"
            ]
            edge_record["fresh_generation_succeeded_by_deadline"] = edge_record[
                "fresh_generation_succeeded"
            ]
            satisfied = (
                edge_record["cache_adoption_succeeded"]
                + edge_record["fresh_generation_succeeded_by_deadline"]
            )
            unsatisfied = max(0, edge_record["elementary_pair_demand"] - satisfied)
            edge_record["demand_unsatisfied_at_deadline"] = unsatisfied
            if unsatisfied:
                reasons = edge_record["blocking_reasons"]
                pending = self._pending_generation_count(record, tuple(edge_record["edge"]))
                if pending:
                    reasons["generation_protocol_waiting"] = pending
                if edge_record["fresh_generation_rules_installed"] == 0:
                    reasons["application_rule_not_installed"] = unsatisfied
                elif edge_record["fresh_generation_protocol_started"] > edge_record["fresh_generation_succeeded"]:
                    reasons["generation_attempts_failed_or_contended"] = unsatisfied
                elif not reasons:
                    reasons["other_unknown"] = unsatisfied
            edge_record["fresh_generation_cancelled_or_expired"] = max(
                0,
                edge_record["fresh_generation_protocol_started"]
                - edge_record["fresh_generation_succeeded"],
            )

    def _pending_generation_count(self, record: dict, edge: tuple[str, str]) -> int:
        total = 0
        for node_name in edge:
            node = self.timeline.get_entity_by_name(node_name)
            for protocol in node.resource_manager.pending_protocols + node.resource_manager.waiting_protocols:
                reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
                if reservation is not None and reservation_key(reservation) == tuple(record["key"]):
                    total += 1
        return total

    def aggregate(self) -> dict:
        totals = defaultdict(int)
        reason_totals = defaultdict(int)
        for reservation in self.reservations.values():
            for edge in reservation["edges"].values():
                for name in (
                    "elementary_pair_demand",
                    "compatible_cached_pair_available",
                    "cache_adoption_attempted",
                    "cache_adoption_succeeded",
                    "fresh_generation_required_after_adoption",
                    "fresh_generation_rules_requested",
                    "fresh_generation_rules_installed",
                    "fresh_generation_protocol_started",
                    "fresh_generation_protocol_started_by_deadline",
                    "protocol_request_objects",
                    "unique_protocol_instances",
                    "protocol_start_calls",
                    "starts_that_schedule_an_emission",
                    "protocol_callbacks_that_schedule_emission",
                    "endpoint_emissions",
                    "bsm_attempts",
                    "completed_physical_attempts",
                    "endpoint_success_updates",
                    "successful_physical_pairs",
                    "fresh_generation_succeeded",
                    "fresh_generation_succeeded_by_deadline",
                    "fresh_generation_cancelled_or_expired",
                    "demand_unsatisfied_at_deadline",
                ):
                    totals[name] += edge[name]
                for reason, count in edge["blocking_reasons"].items():
                    reason_totals[reason] += count
        return {
            "schema_version": "2",
            "backend": self.backend,
            "acp_memory_budget": self.memory_budget,
            "totals": dict(totals),
            "blocking_reasons": dict(reason_totals),
            "reservations": list(self.reservations.values()),
            "memory_snapshots": self.memory_snapshots,
            "physical_snapshots": self.physical_snapshots,
            "events": self.events,
            "metric_definitions": {
                "fresh_generation_protocol_started": "deprecated alias for protocol_request_objects",
                "protocol_request_objects": "request-side protocol objects passed to ResourceManager.send_request",
                "unique_protocol_instances": "endpoint protocol objects (normally two per physical attempt)",
                "protocol_start_calls": "actual protocol start() callbacks, including both endpoints and both rounds",
                "starts_that_schedule_an_emission": "emission events scheduled synchronously inside start()",
                "protocol_callbacks_that_schedule_emission": "message callbacks that schedule an emit_event",
                "endpoint_emissions": "photons emitted by application protocol endpoints",
                "bsm_attempts": "coincident two-photon application windows presented to a BSM",
                "completed_physical_attempts": "request-side two-round protocols ending in RAW",
                "endpoint_success_updates": "endpoint memory ENTANGLED updates (two per physical pair)",
                "successful_physical_pairs": "elementary pairs, deduplicated across endpoint updates",
                "fresh_generation_succeeded": "compatibility alias for successful_physical_pairs",
            },
            "input_fingerprints": self.input_fingerprints(),
        }

    def input_fingerprints(self) -> dict:
        workload = sorted(self.query_specs.values(), key=lambda item: item["query_id"])
        actual_paths = sorted(
            (
                reservation["query_id"],
                reservation["round"],
                reservation["start_time_ps"],
                reservation["path"],
            )
            for reservation in self.reservations.values()
        )
        round1_paths = [path for path in actual_paths if path[1] == 1]
        return {
            "workload_sha256": self._stable_hash(workload),
            "initial_application_paths_sha256": self._stable_hash(actual_paths),
            "round1_application_paths_sha256": self._stable_hash(round1_paths),
            "physical_config_sha256": self._physical_config.get("sha256", ""),
            "workload": workload,
            "initial_application_paths": actual_paths,
            "round1_application_paths": round1_paths,
        }

    def write(self, path: str) -> None:
        with open(path, "w") as output:
            json.dump(self.aggregate(), output, indent=2, sort_keys=True)
