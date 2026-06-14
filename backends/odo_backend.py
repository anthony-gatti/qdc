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

from backends.base import BackendBase
from backends.collectors import collect_qpq_results
from results import BackendResult, RequestResult
from qpq_app import QPQApp


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

        EntanglementGenerationA.set_global_type("single_heralded")
        EntanglementGenerationB.set_global_type("single_heralded")

        network_topo = RouterNetTopo(topo_json_path)
        tl = network_topo.get_timeline()

        # Temporary debug print for this spike.
        print("    sequence formalism:", tl.quantum_manager.get_active_formalism())
        print("    quantum manager:", type(tl.quantum_manager).__name__)
        print("    EGA type:", EntanglementGenerationA.get_global_type())
        print("    EGB type:", EntanglementGenerationB.get_global_type())
        print("    BBPSSW formalism:", BBPSSWProtocol.get_formalism())

        name_to_app = {}
        purify = config.get("hardware", {}).get("purify", True)

        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = QPQApp(router)
            name_to_app[router.name] = app
            router.resource_manager.purify = purify

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

        return collect_qpq_results(name_to_app, config, self.name)
