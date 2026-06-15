"""
Backend that uses RouterNetTopoAdaptive from the ACP codebase.

Covers both ODO (adaptive_max_memory=0) and ACP (adaptive_max_memory>0).
The routing algorithm is shortest-path in both cases; the difference is
whether pre-generated entanglement pairs are available.

Requires: the acp/ fork to be on sys.path (or installed).
"""

import sys
import os
from collections import defaultdict
from typing import Optional

ACP_DIR = os.environ.get(
    "QDC_ACP_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "acp")),
)

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

# ACP imports — these come from the acp/ fork
try:
    from router_net_topo_adaptive import RouterNetTopoAdaptive
    from request_app import RequestAppTimeToServe
except ImportError as e:
    raise ImportError(
        "Could not import ACP modules. Set QDC_ACP_DIR to the patched ACP repo, "
        "or install ACP so router_net_topo_adaptive.py is importable."
    ) from e

from backends.base import BackendBase
from backends.collectors import collect_qpq_results
from results import BackendResult, RequestResult
from qpq_app import QPQApp, QPQResult


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
        purify = config.get("hardware", {}).get("purify", True)

        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = RequestAppTimeToServe(router)
            name_to_app[router.name] = app

            router.adaptive_continuous.has_empty_neighbor = True
            router.adaptive_continuous.update_prob = self._update_prob
            router.adaptive_continuous.print_prob_table = False
            router.resource_manager.purify = purify
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

        return self._collect_pair_results(name_to_app, config)

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
        purify = config.get("hardware", {}).get("purify", True)

        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = QPQApp(router)
            name_to_app[router.name] = app

            router.adaptive_continuous.has_empty_neighbor = True
            router.adaptive_continuous.update_prob = self._update_prob
            router.adaptive_continuous.print_prob_table = False
            router.resource_manager.purify = purify
            if not self._background_enabled:
                router.active = False

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

        tl.init()
        tl.run()

        for app in name_to_app.values():
            app.finalize_unfinished_queries(tl.now())

        self._print_acp_counters(network_topo)
        self._print_qpq_diagnostic_counters(name_to_app)

        return collect_qpq_results(name_to_app, config, self.name)

    def _configure_current_sequence_stack(self) -> None:
        QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
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
            "probability_updates",
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
