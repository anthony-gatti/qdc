"""
Vanilla upstream SeQUeNCe ODO backend.

This backend uses RouterNetTopo directly. It expects the topology JSON to set:
    formalism = "bell_diagonal"
    encoding_type = "single_heralded"

Do not manually overwrite tl.quantum_manager here. RouterNetTopo / Timeline should
create the correct QuantumManager from the topology config.
"""

from sequence.topology.router_net_topo import RouterNetTopo
from sequence.constants import MILLISECOND
from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
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

from backends.base import BackendBase
from backends.collectors import collect_qpq_results
from results import BackendResult, RequestResult
from qpq_app import QPQApp
from demand_diagnostics import ApplicationDemandDiagnostics


class ODOBackend(BackendBase):
    @property
    def name(self) -> str:
        return "odo"

    @property
    def adaptive_max_memory(self) -> int:
        return 0

    def run(self, topo_json_path: str, request_queue: list, config: dict) -> BackendResult:
        mode = config.get("workload", {}).get("mode", "qpq")
        if mode != "qpq":
            raise NotImplementedError("ODOBackend spike currently supports only QPQ mode.")
        return self._run_qpq(topo_json_path, request_queue, config)

    def _run_qpq(self, topo_json_path: str, query_specs: list, config: dict) -> BackendResult:
        # Use SeQUeNCe's Bell-diagonal + single-heralded stack.
        # This must be set before RouterNetTopo is constructed, because BSM nodes
        # create their EntanglementGenerationB protocol during topology loading.
        QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)

        EntanglementGenerationA.set_global_type("single_heralded")
        EntanglementGenerationB.set_global_type("single_heralded")

        network_topo = RouterNetTopo(topo_json_path)
        tl = network_topo.get_timeline()

        name_to_app = {}

        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = QPQApp(router)
            name_to_app[router.name] = app

        demand_diagnostics = None
        diagnostics_config = config.get("diagnostics", {})
        if diagnostics_config.get("application_demand", False):
            demand_diagnostics = ApplicationDemandDiagnostics(
                network_topo, query_specs, self.name, self.adaptive_max_memory
            )
            demand_diagnostics.install(
                network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
            )

        for spec in query_specs:
            src_name = spec["src"]
            if src_name not in name_to_app:
                continue

            name_to_app[src_name].submit_query(
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

        result = collect_qpq_results(name_to_app, config, self.name)
        if demand_diagnostics is not None:
            demand_diagnostics.write(diagnostics_config["application_demand_output"])
        self._print_diagnostic_counters(tl, result)
        return result

    def _print_diagnostic_counters(self, tl, result: BackendResult) -> None:
        max_tts_ms = max(
            (rr.time_to_serve_ms or 0.0 for rr in result.request_results),
            default=0.0,
        )
        max_pair_arrival_ms = max(
            (arrival for rr in result.request_results for arrival in rr.pair_arrival_ms),
            default=0.0,
        )
        n_success = sum(1 for rr in result.request_results if rr.success)
        print(
            "    ODO counters: "
            f"timeline_scheduled={getattr(tl, 'schedule_counter', 0)}, "
            f"timeline_run={getattr(tl, 'run_counter', 0)}, "
            f"timeline_pending={len(getattr(tl, 'events', []))}, "
            f"timeline_now_ps={tl.now()}, "
            f"timeline_stop_ps={getattr(tl, 'stop_time', 0)}, "
            f"result_successes={n_success}, "
            f"result_count={len(result.request_results)}, "
            f"max_tts_ms={max_tts_ms:.3f}, "
            f"max_pair_arrival_ms={max_pair_arrival_ms:.3f}"
        )
