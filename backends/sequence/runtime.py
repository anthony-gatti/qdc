"""Common SeQUeNCe runtime used by clean workload/algorithm plugins."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.entanglement_management.purification.bbpssw_protocol import BBPSSWProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.quantum_manager import QuantumManager
from sequence.topology.router_net_topo import RouterNetTopo

from algorithms.acp import AdaptiveContinuous
from algorithms.odo import ShortestPathOnDemand
from backends.sequence.acp_topology import ACPRouterNetTopo
from pair_app import PairRequestApp, collect_pair_results


class SequenceRuntime:
    """Owns SeQUeNCe construction, app attachment, and instrumentation."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_diagnostics: dict = {}

    def run_single_pair(self, workload, algorithm) -> object:
        self._configure_sequence()
        adaptive_memory = getattr(algorithm, "adaptive_max_memory", 0)
        topology_config = workload.topology(adaptive_memory=adaptive_memory)
        topology_path = self.output_dir / "topologies" / f"{algorithm.name}_topology.json"
        topology_path.parent.mkdir(parents=True, exist_ok=True)
        topology_path.write_text(json.dumps(topology_config, indent=2) + "\n")

        if isinstance(algorithm, AdaptiveContinuous):
            network_topo = ACPRouterNetTopo(topology_config, {
                "adaptive_max_memory": algorithm.adaptive_max_memory,
                "acp_strategy": algorithm.cache_strategy,
                "acp_update_prob": algorithm.update_prob,
                "acp_period_ps": algorithm.period_ps,
                "acp_delta": algorithm.delta,
                "acp_background_enabled": algorithm.background_enabled,
            })
        elif isinstance(algorithm, ShortestPathOnDemand):
            network_topo = RouterNetTopo(topology_config)
        else:
            raise TypeError(f"Unsupported algorithm: {algorithm!r}")

        apps = {
            router.name: PairRequestApp(router)
            for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        }
        requests = workload.requests()
        for request in requests:
            identity, src, dst, start, end, memory, fidelity, pairs = request
            apps[src].start(dst, start, end, memory, fidelity, pairs, identity)

        tl = network_topo.get_timeline()
        tl.init()
        tl.run()
        result = collect_pair_results(apps, requests, algorithm.name, workload.seed)
        self.last_diagnostics = self._collect_diagnostics(network_topo, algorithm.name)
        return result

    def _configure_sequence(self) -> None:
        QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementGenerationA.set_global_type("single_heralded")
        EntanglementGenerationB.set_global_type("single_heralded")

    def _collect_diagnostics(self, network_topo, algorithm_name: str) -> dict:
        counters = Counter()
        max_memory = {}
        memory_high_watermark = {}
        probability_tables = {}
        lifecycle_events = []
        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            acp = getattr(router, "adaptive_continuous", None)
            if acp is None:
                continue
            counters.update(acp.counters)
            max_memory[router.name] = acp.adaptive_memory_used
            memory_high_watermark[router.name] = acp.counters.get("adaptive_memory_high_watermark", 0)
            probability_tables[router.name] = {
                ("None" if key is None else key): value
                for key, value in acp.probability_table.items()
            }
            lifecycle_events.extend(acp.lifecycle_events)
        return {
            "algorithm": algorithm_name,
            "counters": dict(counters),
            "adaptive_memory_at_end": max_memory,
            "adaptive_memory_high_watermark_by_node": memory_high_watermark,
            "probability_tables": probability_tables,
            "lifecycle_events": lifecycle_events,
        }
