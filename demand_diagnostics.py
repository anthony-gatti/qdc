"""Optional application-demand diagnostics for SeQUeNCe QPQ reservations.

The collector instruments resource-manager instances at runtime.  It does not
modify SeQUeNCe and deliberately keeps the detailed records out of schema-v2
CSV output.
"""

from __future__ import annotations

import json
from collections import defaultdict
from types import MethodType

from sequence.entanglement_management.generation import EntanglementGenerationA
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


class ApplicationDemandDiagnostics:
    """Collect comparable application demand and generation metrics."""

    def __init__(self, network_topo, query_specs: list[dict], backend: str, memory_budget: int):
        self.topology = network_topo
        self.timeline = network_topo.get_timeline()
        self.query_specs = {spec["query_id"]: spec for spec in query_specs}
        self.backend = backend
        self.memory_budget = memory_budget
        self.reservations = {}
        self.events = []
        self.memory_snapshots = []
        self._installed = False

    def install(self, routers: list) -> None:
        if self._installed:
            return
        self._installed = True
        for router in routers:
            router.qdc_demand_diagnostics = self
            self._wrap_resource_manager(router)

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
            self.record_protocol_requested(router, protocol, req_dst)
            return original_send(protocol, req_dst, req_condition_func, req_args)

        def update(_manager, protocol, memory, state):
            self.record_protocol_update(router, protocol, state)
            return original_update(protocol, memory, state)

        manager.generate_load_rules = MethodType(generate_load_rules, manager)
        manager.load = MethodType(load, manager)
        manager.send_request = MethodType(send_request, manager)
        manager.update = MethodType(update, manager)

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
        edge_record["fresh_generation_protocol_started"] += 1
        self.events.append({
            "event": "fresh_generation_protocol_started",
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
        if remote is None or router.name > remote:
            return
        edge_record = self._edge_record(reservation, (router.name, remote))
        if edge_record is not None and state == MemoryInfo.ENTANGLED:
            edge_record["fresh_generation_succeeded"] += 1

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
                    "fresh_generation_succeeded",
                    "fresh_generation_succeeded_by_deadline",
                    "fresh_generation_cancelled_or_expired",
                    "demand_unsatisfied_at_deadline",
                ):
                    totals[name] += edge[name]
                for reason, count in edge["blocking_reasons"].items():
                    reason_totals[reason] += count
        return {
            "schema_version": "1",
            "backend": self.backend,
            "acp_memory_budget": self.memory_budget,
            "totals": dict(totals),
            "blocking_reasons": dict(reason_totals),
            "reservations": list(self.reservations.values()),
            "memory_snapshots": self.memory_snapshots,
            "events": self.events,
        }

    def write(self, path: str) -> None:
        with open(path, "w") as output:
            json.dump(self.aggregate(), output, indent=2, sort_keys=True)
