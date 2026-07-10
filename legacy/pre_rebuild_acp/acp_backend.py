"""Historical pre-rebuild ACP backend; retained for reference only.

This module depends on the removed ``external/acp`` package. It is intentionally
outside the supported backend registry and must not be used for current runs.
"""

import sys
import os
import json
from collections import defaultdict
from typing import Optional

ACP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "acp"))

if os.path.isdir(ACP_DIR) and ACP_DIR not in sys.path:
    sys.path.insert(0, ACP_DIR)

from sequence.topology.router_net_topo import RouterNetTopo
from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
from sequence.constants import MILLISECOND
from sequence.kernel.quantum_manager import QuantumManager
from sequence.entanglement_management.generation import (
    EntanglementGenerationA,
    EntanglementGenerationB,
)
from sequence.entanglement_management.purification.bbpssw_protocol import BBPSSWProtocol
from sequence.entanglement_management.swapping import (
    EntanglementSwappingA,
    EntanglementSwappingB,
)
from sequence.kernel.event import Event
from sequence.kernel.process import Process

from router_net_topo_adaptive import RouterNetTopoAdaptive
from request_app import RequestAppTimeToServe

from backends.base import BackendBase
from backends.collectors import collect_qpq_results
from results import BackendResult, RequestResult
from qpq_app import QPQApp, QPQResult
from common import MILLISECOND
from demand_diagnostics import ApplicationDemandDiagnostics


class CacheLifecycleDiagnostics:
    """Structured ACP cache lifecycle snapshots for focused diagnostics."""

    def __init__(self, network_topo, query_specs: list, topo_json_path: str):
        self.network_topo = network_topo
        self.query_specs = query_specs
        self.topo_json_path = topo_json_path
        self.snapshots = []
        self.router_graph = self._build_router_graph(topo_json_path)

    def _build_router_graph(self, topo_json_path: str) -> dict[str, set[str]]:
        with open(topo_json_path) as f:
            topo = json.load(f)
        bsm_to_routers = defaultdict(list)
        for qc in topo.get("qchannels", []):
            src = qc["source"]
            dst = qc["destination"]
            if dst.startswith("BSM"):
                bsm_to_routers[dst].append(src)
        graph = defaultdict(set)
        for routers in bsm_to_routers.values():
            if len(routers) == 2:
                a, b = routers
                graph[a].add(b)
                graph[b].add(a)
        return graph

    def _shortest_path(self, src: str, dst: str) -> list[str]:
        if src == dst:
            return [src]
        queue = [(src, [src])]
        seen = {src}
        while queue:
            node, path = queue.pop(0)
            for neighbor in sorted(self.router_graph.get(node, [])):
                if neighbor in seen:
                    continue
                next_path = path + [neighbor]
                if neighbor == dst:
                    return next_path
                seen.add(neighbor)
                queue.append((neighbor, next_path))
        return []

    def _needed_links(self) -> set[tuple[str, str]]:
        links = set()
        for spec in self.query_specs:
            path = self._shortest_path(spec["src"], spec["dst"])
            for a, b in zip(path, path[1:]):
                links.add(tuple(sorted((a, b))))
        return links

    def schedule(self):
        tl = self.network_topo.get_timeline()
        for start_time in sorted({spec["start_time"] for spec in self.query_specs}):
            snapshot_time = max(0, start_time - 1)
            process = Process(self, "snapshot", [snapshot_time])
            tl.schedule(Event(snapshot_time, process, priority=0))

    def snapshot(self, label_time_ps: int):
        needed_links = self._needed_links()
        pairs = []
        local_records = 0
        useful_records = 0
        ages_ms = []
        adaptive_memory_by_node = {}
        for router in self.network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            acp = getattr(router, "adaptive_continuous", None)
            if acp is None:
                continue
            adaptive_memory_by_node[router.name] = getattr(acp, "adaptive_memory_used", 0)
            for pair in sorted(getattr(acp, "generated_entanglement_pairs", [])):
                local_records += 1
                link = tuple(sorted((pair[0][0], pair[1][0])))
                metadata = getattr(acp, "generated_pair_metadata", {}).get(pair, {})
                generation_time = metadata.get("generation_time_ps")
                age_ms = None
                if generation_time is not None:
                    age_ms = (router.timeline.now() - generation_time) / MILLISECOND
                    ages_ms.append(age_ms)
                if link in needed_links:
                    useful_records += 1
                pairs.append({
                    "node": router.name,
                    "pair": pair,
                    "link": link,
                    "useful_for_workload": link in needed_links,
                    "age_ms": age_ms,
                    "metadata": metadata,
                })
        unique_pairs = {
            tuple(sorted(((pair["pair"][0][0], pair["pair"][0][1]), (pair["pair"][1][0], pair["pair"][1][1]))))
            for pair in pairs
        }
        useful_unique_pairs = {
            tuple(sorted(((pair["pair"][0][0], pair["pair"][0][1]), (pair["pair"][1][0], pair["pair"][1][1]))))
            for pair in pairs
            if pair["useful_for_workload"]
        }
        self.snapshots.append({
            "time_ps": label_time_ps,
            "time_ms": label_time_ps / MILLISECOND,
            "needed_links": sorted(needed_links),
            "local_pair_records": local_records,
            "unique_pair_records": len(unique_pairs),
            "useful_local_pair_records": useful_records,
            "useful_unique_pair_records": len(useful_unique_pairs),
            "pair_age_ms_min": min(ages_ms) if ages_ms else None,
            "pair_age_ms_max": max(ages_ms) if ages_ms else None,
            "adaptive_memory_by_node": adaptive_memory_by_node,
            "pairs": pairs,
        })

    def aggregate(self) -> dict:
        counters = defaultdict(int)
        lifecycle_events = []
        max_adaptive_memory_by_node = defaultdict(int)
        for router in self.network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = getattr(router, "app", None)
            for name, value in getattr(app, "diagnostic_counters", {}).items():
                if name.startswith("unmapped_") or name.startswith("pre_start_") or name.startswith("duplicate_"):
                    counters[f"qpq_{name}"] += value
            lifecycle_events.extend(getattr(app, "diagnostic_events", []))
            acp = getattr(router, "adaptive_continuous", None)
            if acp is None:
                continue
            for name in (
                "generation_attempts",
                "generation_successes",
                "background_generation_successes",
                "background_generation_failures",
                "cache_checks",
                "cache_hits",
                "cache_misses",
                "background_pairs_expired",
                "background_pairs_consumed_by_app",
                "background_pairs_removed_other",
                "fresh_app_pairs_generated",
                "probability_updates",
                "application_pause_entries",
                "application_pause_rejections",
                "application_pause_wakeups",
                "background_rules_quiesced_for_application",
                "background_emissions_during_application",
                "invariant_stale_adaptive_cleanup_on_app_memory",
                "invariant_fresh_generation_on_adopted_memory",
            ):
                counters[name] += getattr(acp, name, 0)
            max_adaptive_memory_by_node[router.name] = max(
                max_adaptive_memory_by_node[router.name],
                getattr(acp, "adaptive_memory_used", 0),
            )
            lifecycle_events.extend(getattr(acp, "lifecycle_events", []))
            for reason, count in getattr(acp, "cache_rejections", {}).items():
                counters[f"cache_rejected_{reason}"] += count
            for info in router.resource_manager.memory_manager:
                memory = info.memory
                if getattr(memory, "qdc_application_reservation", ""):
                    counters["cleanup_app_owned_memories"] += 1
                if getattr(memory, "qdc_claimed_by_reservation", ""):
                    counters["cleanup_claimed_memories"] += 1
        generated_pairs = {
            (
                event.get("time_ps"),
                tuple(sorted(tuple(endpoint) for endpoint in event.get("pair", ()))),
            )
            for event in lifecycle_events
            if event.get("event") == "background_pair_available"
        }
        adopted_pairs = {
            (
                event.get("time_ps"),
                tuple(sorted(tuple(endpoint) for endpoint in event.get("app_pair", ()))),
                event.get("app_reservation", ""),
            )
            for event in lifecycle_events
            if event.get("event") == "background_pair_adopted_by_application"
        }
        counters["background_physical_pairs_generated"] = len(generated_pairs)
        counters["background_physical_pairs_adopted"] = len(adopted_pairs)
        counters["background_generation_endpoint_updates"] = counters.get(
            "background_generation_successes", 0
        )
        counters["background_inventory_endpoint_records_created"] = counters.get(
            "generation_successes", 0
        )
        counters["fresh_application_endpoint_updates"] = counters.get(
            "fresh_app_pairs_generated", 0
        )
        return {
            "counters": dict(counters),
            "snapshots": self.snapshots,
            "lifecycle_events": lifecycle_events,
            "max_adaptive_memory_by_node": dict(max_adaptive_memory_by_node),
            "metric_definitions": {
                "background_physical_pairs_generated": "deduplicated elementary pairs; canonical background generation count",
                "background_generation_endpoint_updates": "endpoint success callbacks; normally two per physical pair",
                "background_inventory_endpoint_records_created": "local cache records; normally two per physical pair",
                "background_physical_pairs_adopted": "deduplicated elementary pairs atomically transferred to applications",
                "background_pairs_consumed_by_app": "initiator-side adoption records; retained compatibility field",
                "fresh_application_endpoint_updates": "endpoint ENTANGLED updates; normally two per fresh elementary pair",
                "fresh_app_pairs_generated": "deprecated ambiguous alias for fresh_application_endpoint_updates",
                "useful_local_pair_records": "endpoint cache records on workload path edges at a snapshot",
                "unique_pair_records": "deduplicated physical cache pairs at a snapshot",
            },
        }


class ACPBackend(BackendBase):
    """Backend using ACP infrastructure.

    With adaptive_max_memory=0, this is the ODO (on-demand only) baseline.
    With adaptive_max_memory>0, this is ACP with continuous pre-generation.
    """

    def __init__(
        self,
        adaptive_max_memory: int = 8,
        update_prob: bool = True,
        background_enabled: bool = True,
        application_priority: bool = True,
        cache_strategy: str = "freshest",
        name_override: Optional[str] = None,
    ):
        """
        Args:
            adaptive_max_memory: memories per node for ACP background generation.
            update_prob: whether ACP updates neighbor selection probabilities.
                Only relevant when adaptive_max_memory > 0.
        """
        if adaptive_max_memory <= 0:
            raise ValueError(
                "ACPBackend requires adaptive_max_memory > 0. "
                "Use ODOBackend for the on-demand baseline."
            )
        self._adaptive_max_memory = adaptive_max_memory
        self._update_prob = update_prob
        self._background_enabled = background_enabled
        self._application_priority = application_priority
        if cache_strategy not in {"freshest", "random"}:
            raise ValueError(f"Unsupported ACP cache strategy: {cache_strategy}")
        self._cache_strategy = cache_strategy
        self._name_override = name_override

    @property
    def adaptive_max_memory(self) -> int:
        return self._adaptive_max_memory

    @property
    def name(self) -> str:
        if self._name_override is not None:
            return self._name_override
        return f"acp_m{self._adaptive_max_memory}"

    def run(
        self,
        topo_json_path: str,
        request_queue: list,
        config: dict,
    ) -> BackendResult:
        """Run simulation with ACP/ODO protocol.

        Supports two modes based on config:
        - "pair" mode (default): request_queue is list of pair request tuples
        - "qpq" mode: request_queue is list of QPQ query spec dicts
        """
        mode = config.get("workload", {}).get("mode", "pair")

        if mode == "qpq":
            return self._run_qpq(topo_json_path, request_queue, config)
        else:
            return self._run_pair(topo_json_path, request_queue, config)

    def _run_pair(
        self,
        topo_json_path: str,
        request_queue: list,
        config: dict,
    ) -> BackendResult:
        """Run with individual pair requests (Phase 1 mode)."""
        self._configure_current_sequence_stack()
        network_topo = RouterNetTopoAdaptive(topo_json_path)
        tl = network_topo.get_timeline()

        name_to_app = {}
        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = RequestAppTimeToServe(router)
            name_to_app[router.name] = app

            router.adaptive_continuous.has_empty_neighbor = True
            router.adaptive_continuous.update_prob = self._update_prob
            router.adaptive_continuous.print_prob_table = False
            router.adaptive_continuous.background_enabled = self._background_enabled
            router.adaptive_continuous.application_priority = self._application_priority
            router.adaptive_continuous.strategy = self._cache_strategy
            update_period_s = config.get("acp", {}).get("update_period_s")
            if update_period_s is not None:
                router.adaptive_continuous.update_period(round(update_period_s * 10**12))
            forced_tables = config.get("diagnostics", {}).get("force_probability_table", {})
            if router.name in forced_tables:
                router.adaptive_continuous.forced_probability_table = forced_tables[router.name]
            if not self._background_enabled:
                router.active = False

        for request in request_queue:
            req_id, src_name, dst_name, start_time, end_time, \
                memo_size, fidelity, entanglement_number = request

            if src_name not in name_to_app:
                continue
            app = name_to_app[src_name]
            app.start(dst_name, start_time, end_time,
                      memo_size, fidelity, entanglement_number, req_id)

        tl.init()
        tl.run()

        from pair_app import collect_pair_results
        seed = config.get("topology", {}).get("random_seed", 0)
        return collect_pair_results(name_to_app, request_queue, self.name, seed)

    def _run_qpq(
        self,
        topo_json_path: str,
        query_specs: list,
        config: dict,
    ) -> BackendResult:
        """Run with QPQ multi-round queries (Phase 3 mode)."""
        self._configure_current_sequence_stack()
        network_topo = RouterNetTopoAdaptive(topo_json_path)
        tl = network_topo.get_timeline()

        name_to_app = {}
        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = QPQApp(router)
            name_to_app[router.name] = app

            router.adaptive_continuous.has_empty_neighbor = True
            router.adaptive_continuous.update_prob = self._update_prob
            router.adaptive_continuous.print_prob_table = False
            router.adaptive_continuous.background_enabled = self._background_enabled
            router.adaptive_continuous.application_priority = self._application_priority
            forced_tables = config.get("diagnostics", {}).get("force_probability_table", {})
            if router.name in forced_tables:
                router.adaptive_continuous.forced_probability_table = forced_tables[router.name]
            if not self._background_enabled:
                router.active = False

        demand_diagnostics = None
        diag_cfg = config.get("diagnostics", {})
        if diag_cfg.get("application_demand", False):
            demand_diagnostics = ApplicationDemandDiagnostics(
                network_topo,
                query_specs,
                self.name,
                self.adaptive_max_memory,
                topo_json_path,
                diag_cfg.get("application_demand_events", True),
            )
            demand_diagnostics.install(
                network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
            )

        for spec in query_specs:
            src_name = spec["src"]
            if src_name not in name_to_app:
                continue

            app = name_to_app[src_name]
            app.submit_query(
                query_id=spec["query_id"],
                responder=spec["dst"],
                start_time=spec["start_time"],
                end_time=spec["end_time"],
                database_size_log=spec["database_size_log"],
                fidelity=spec["fidelity"],
                round_deadline_ps=spec["round_deadline_ps"],
            )

        diagnostics = None
        if diag_cfg.get("cache_lifecycle", False):
            diagnostics = CacheLifecycleDiagnostics(network_topo, query_specs, topo_json_path)
            diagnostics.schedule()

        tl.init()
        tl.run()

        for app in name_to_app.values():
            app.finalize_unfinished_queries(tl.now())

        self._print_acp_counters(network_topo)
        self._print_qpq_diagnostic_counters(name_to_app)
        if diagnostics is not None:
            self._write_cache_lifecycle_diagnostics(diagnostics, config)
        if demand_diagnostics is not None:
            demand_diagnostics.write(diag_cfg["application_demand_output"])

        return collect_qpq_results(name_to_app, config, self.name)

    def _configure_current_sequence_stack(self) -> None:
        QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementGenerationA.set_global_type("single_heralded")
        EntanglementGenerationB.set_global_type("single_heralded")

    def _print_acp_counters(self, network_topo: RouterNetTopoAdaptive) -> None:
        counters = defaultdict(int)
        counter_names = (
            "start_invocations",
            "start_events_scheduled",
            "start_events_no_memory",
            "start_events_select_none",
            "start_events_after_response",
            "ac_request_sent",
            "ac_request_received",
            "ac_respond_sent",
            "ac_respond_received",
            "blocked_on_memory_entries",
            "blocked_on_memory_wakeups",
            "blocked_on_memory_duplicate_wakeups_avoided",
            "application_pause_entries",
            "application_pause_rejections",
            "application_pause_wakeups",
            "background_rules_quiesced_for_application",
            "background_emissions_during_application",
            "reservation_schedule_attempts",
            "reservation_schedule_successes",
            "reservation_schedule_failures",
            "adaptive_rule_load_batches",
            "adaptive_rules_scheduled",
            "request_rule_load_batches",
            "request_rules_scheduled",
            "rule_load_invocations",
            "generation_protocol_starts",
            "app_generation_protocol_starts",
            "background_generation_protocol_starts",
            "app_generation_successes",
            "background_generation_successes",
            "app_generation_failures",
            "background_generation_failures",
            "swap_starts",
            "swap_successes",
            "swap_failures",
            "generation_attempts",
            "generation_successes",
            "cache_checks",
            "cache_hits",
            "cache_misses",
            "background_pairs_expired",
            "background_pairs_consumed_by_app",
            "background_pairs_removed_other",
            "fresh_app_pairs_generated",
            "probability_updates",
            "invariant_stale_adaptive_cleanup_on_app_memory",
            "invariant_fresh_generation_on_adopted_memory",
        )
        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            acp = getattr(router, "adaptive_continuous", None)
            if acp is None:
                continue
            for name in counter_names:
                counters[name] += getattr(acp, name, 0)

        tl = network_topo.get_timeline()
        timeline_summary = (
            f"timeline_scheduled={getattr(tl, 'schedule_counter', 0)}, "
            f"timeline_run={getattr(tl, 'run_counter', 0)}, "
            f"timeline_pending={len(getattr(tl, 'events', []))}, "
            f"timeline_now_ps={tl.now()}, "
            f"timeline_stop_ps={getattr(tl, 'stop_time', 0)}"
        )
        counter_summary = ", ".join(f"{name}={counters[name]}" for name in counter_names)
        print(f"    ACP counters: {counter_summary}, {timeline_summary}")

    def _print_qpq_diagnostic_counters(self, name_to_app: dict) -> None:
        counters = defaultdict(int)
        for app in name_to_app.values():
            for name, value in getattr(app, "diagnostic_counters", {}).items():
                counters[name] += value
        if counters:
            counter_summary = ", ".join(
                f"{name}={counters[name]}" for name in sorted(counters)
            )
            print(f"    QPQ diagnostic counters: {counter_summary}")

    def _write_cache_lifecycle_diagnostics(
        self,
        diagnostics: CacheLifecycleDiagnostics,
        config: dict,
    ) -> None:
        diag_cfg = config.get("diagnostics", {})
        output_path = diag_cfg.get("cache_lifecycle_output")
        if not output_path:
            return
        data = diagnostics.aggregate()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"    ACP cache lifecycle diagnostics wrote {output_path}")
    
    def _collect_pair_results(
        self,
        name_to_app: dict,
        config: dict,
    ) -> BackendResult:
        """Extract per-request metrics from apps after simulation."""
        time_to_serve = {}
        fidelities = {}
        for app_name, app in name_to_app.items():
            time_to_serve.update(app.time_to_serve)
            fidelities.update(app.entanglement_fidelities)

        request_results = []
        for reservation, tts in sorted(time_to_serve.items()):
            fid_list = fidelities.get(reservation, [])
            fid = fid_list[0] if fid_list else None

            src = getattr(reservation, 'initiator', str(reservation))
            dst = getattr(reservation, 'responder', '')
            start = getattr(reservation, 'start_time', 0)

            request_results.append(RequestResult(
                request_id=hash(reservation) % (2**31),
                src=src,
                dst=dst,
                start_time_ps=int(start),
                time_to_serve_ms=tts / MILLISECOND,
                fidelity=fid,
                success=True,
            ))

        num_nodes = len(name_to_app)
        seed = config.get("topology", {}).get("random_seed", 0)

        return BackendResult(
            backend_name=self.name,
            seed=seed,
            num_nodes=num_nodes,
            request_results=request_results,
        )

    def _collect_qpq_results(
        self,
        name_to_app: dict,
        config: dict,
    ) -> BackendResult:
        """Extract QPQ query results from QPQ apps.

        Populates failure_reason and per-pair arrival timestamps.
        Pair timestamps are in ms, relative to the query's round-1 start.
        """
        request_results = []

        for app_name, app in name_to_app.items():
            if not isinstance(app, QPQApp):
                continue

            # Build per-query pair arrival timestamps from the app's state
            # We need to join: query -> rounds -> reservation -> timestamps
            for qpq_result in app.get_results():
                query = app.queries.get(qpq_result.query_id)

                # Compute pair arrivals in ms relative to round-1 start
                pair_arrivals_ms = []
                if query is not None:
                    round1_start_ps = 0
                    if 1 in query.rounds:
                        round1_start_ps = query.rounds[1].start_time_ps

                    for round_num in (1, 2):
                        if round_num not in query.rounds:
                            continue
                        rnd = query.rounds[round_num]
                        if rnd.reservation is None:
                            continue
                        # app.entanglement_timestamps is keyed by reservation
                        ts_list = app.entanglement_timestamps.get(rnd.reservation, [])
                        for ts_ps in ts_list:
                            rel_ms = (ts_ps - round1_start_ps) / MILLISECOND
                            pair_arrivals_ms.append(rel_ms)

                # Average fidelity across rounds, if successful
                avg_fid = None
                if qpq_result.success:
                    avg_fid = (qpq_result.round1_avg_fidelity +
                               qpq_result.round2_avg_fidelity) / 2

                request_results.append(RequestResult(
                    request_id=qpq_result.query_id,
                    src=qpq_result.src,
                    dst=qpq_result.dst,
                    start_time_ps=0,
                    time_to_serve_ms=qpq_result.total_time_ms,
                    fidelity=avg_fid,
                    success=qpq_result.success,
                    failure_reason=qpq_result.failure_reason or "",
                    pair_arrival_ms=pair_arrivals_ms,
                ))

        num_nodes = len(name_to_app)
        seed = config.get("topology", {}).get("random_seed", 0)

        return BackendResult(
            backend_name=self.name,
            seed=seed,
            num_nodes=num_nodes,
            request_results=request_results,
        )
