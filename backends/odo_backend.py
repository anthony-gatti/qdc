from sequence.topology.router_net_topo import RouterNetTopo

from backends.base import BackendBase
from backends.collectors import collect_qpq_results
from qpq_app import QPQApp


class ODOBackend(BackendBase):
    """Vanilla SeQUeNCe on-demand shortest-path backend.

    This backend has no ACP dependency.
    """

    @property
    def name(self) -> str:
        return "odo"

    @property
    def adaptive_max_memory(self) -> int:
        return 0

    def run(
        self,
        topo_json_path: str,
        request_queue: list,
        config: dict,
    ):
        mode = config.get("workload", {}).get("mode", "qpq")
        if mode != "qpq":
            raise NotImplementedError(
                "ODOBackend currently supports only QPQ workload mode."
            )

        return self._run_qpq(topo_json_path, request_queue, config)

    def _run_qpq(
        self,
        topo_json_path: str,
        query_specs: list,
        config: dict,
    ):
        network_topo = RouterNetTopo(topo_json_path)
        tl = network_topo.get_timeline()

        name_to_app = {}
        purify = config.get("hardware", {}).get("purify", True)

        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            app = QPQApp(router)
            name_to_app[router.name] = app

            if hasattr(router.resource_manager, "purify"):
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

        return collect_qpq_results(name_to_app, config, self.name)