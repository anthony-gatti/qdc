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
from algorithms.qcast import QCAST
from backends.sequence.acp_protocol import AdaptiveReservation
from backends.sequence.acp_topology import ACPRouterNetTopo
from backends.sequence.configured_topology import ConfiguredRouterNetTopo
from backends.sequence.parallel_links import (
    collect_parallel_link_diagnostics,
    expand_parallel_links,
)
from backends.sequence.qcast_topology import QCASTRouterNetTopo
from backends.sequence.workload_adapters import create_sequence_workload_adapter


class SequenceRuntime:
    """Owns SeQUeNCe construction, app attachment, and instrumentation."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_diagnostics: dict = {}

    def run(self, workload, algorithm) -> object:
        self._configure_sequence()
        if (
            isinstance(algorithm, QCAST)
            and workload.sequence_adapter not in {"concurrent_pairs", "qpq"}
        ):
            raise NotImplementedError(
                "Q-CAST-family algorithms currently support only concurrent_pairs and QPQ workloads"
            )
        adaptive_memory = getattr(algorithm, "adaptive_max_memory", 0)
        topology_config = workload.topology(adaptive_memory=adaptive_memory)
        topology_config = expand_parallel_links(
            topology_config,
            int(getattr(workload, "link_parallelism", 1)),
        )
        if isinstance(algorithm, QCAST):
            for template in topology_config.get("templates", {}).values():
                template.setdefault("EntanglementSwapping", {})[
                    "swapping_success_prob"
                ] = algorithm.swap_success_probability
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
                "acp_purify": algorithm.purify,
                "acp_execution_profile": algorithm.execution_profile,
            })
        elif isinstance(algorithm, ShortestPathOnDemand):
            network_topo = ConfiguredRouterNetTopo(topology_config)
        elif isinstance(algorithm, QCAST):
            network_topo = QCASTRouterNetTopo(topology_config)
        else:
            raise TypeError(f"Unsupported algorithm: {algorithm!r}")

        workload_adapter = create_sequence_workload_adapter(
            workload.sequence_adapter,
            network_topo,
            workload,
            algorithm.name,
            getattr(network_topo, "record_served_path", None),
            algorithm,
        )
        workload_adapter.schedule()

        tl = network_topo.get_timeline()
        tl.init()
        tl.run()
        workload_adapter.finalize()
        teardown = self._teardown_residual_memories(network_topo)
        result = workload_adapter.collect()
        self.last_diagnostics = self._collect_diagnostics(network_topo, algorithm.name)
        self.last_diagnostics["simulation_teardown"] = teardown
        self.last_diagnostics["parallel_links"] = collect_parallel_link_diagnostics(
            network_topo
        )
        self.last_diagnostics["workload"] = workload.name
        self.last_diagnostics["workload_diagnostics"] = workload_adapter.diagnostics()
        return result

    def run_single_pair(self, workload, algorithm) -> object:
        """Compatibility alias for callers created before workload adapters."""
        return self.run(workload, algorithm)

    def _configure_sequence(self) -> None:
        QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        BBPSSWProtocol.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingA.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementSwappingB.set_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementGenerationA.set_global_type("single_heralded")
        EntanglementGenerationB.set_global_type("single_heralded")

    def _teardown_residual_memories(self, network_topo) -> dict:
        """Release simulator state left by a stop-time before reservation expiry."""
        released = {}
        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            count = 0
            for info in router.resource_manager.memory_manager:
                if info.state == "RAW":
                    continue
                router.resource_manager.memory_manager.update(info.memory, "RAW")
                count += 1
            released[router.name] = count
        return {"released_memories_by_node": released}

    def _collect_diagnostics(self, network_topo, algorithm_name: str) -> dict:
        counters = Counter()
        max_memory = {}
        memory_high_watermark = {}
        probability_tables = {}
        probability_history_by_node = {}
        probability_updates_by_node = {}
        counters_by_node = {}
        background_handshake_by_node = {}
        execution_profiles = {}
        reservation_slots_at_end = {}
        accounting_consistent_at_end = {}
        cached_inventory = set()
        lifecycle_events = []
        for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            acp = getattr(router, "adaptive_continuous", None)
            if acp is None:
                continue
            counters.update(acp.counters)
            counters_by_node[router.name] = dict(acp.counters)
            max_memory[router.name] = acp.adaptive_memory_used
            memory_high_watermark[router.name] = acp.counters.get("adaptive_memory_high_watermark", 0)
            probability_updates_by_node[router.name] = acp.counters.get("probability_updates", 0)
            probability_tables[router.name] = {
                ("None" if key is None else key): value
                for key, value in acp.probability_table.items()
            }
            probability_history_by_node[router.name] = acp.probability_history
            execution_profiles[router.name] = acp.execution_profile
            active_reservations = {
                id(reservation)
                for card in router.network_manager.get_timecards()
                for reservation in card.reservations
                if isinstance(reservation, AdaptiveReservation)
                and reservation.end_time > router.timeline.now()
            }
            reservation_slots_at_end[router.name] = len(active_reservations)
            accounting_consistent_at_end[router.name] = (
                acp.adaptive_memory_used == len(active_reservations)
            )
            background_handshake_by_node[router.name] = {
                "neighbor_selections": dict(acp.neighbor_selection_counts),
                "requests_sent": dict(acp.background_requests_sent_by_neighbor),
                "requests_received": dict(acp.background_requests_received_by_neighbor),
                "requests_accepted": dict(acp.background_requests_accepted_by_neighbor),
                "requests_rejected": dict(acp.background_requests_rejected_by_neighbor),
                "responses_accepted": dict(acp.background_responses_accepted_by_neighbor),
                "responses_rejected": dict(acp.background_responses_rejected_by_neighbor),
            }
            for pair in acp.generated_entanglement_pairs:
                cached_inventory.add(self._canonical_pair(pair))
            lifecycle_events.extend(acp.lifecycle_events)
        normalized = self._normalize_acp_events(lifecycle_events)
        normalized["physical_cached_inventory_at_end"] = len(cached_inventory)
        return {
            "algorithm": algorithm_name,
            "counters": dict(counters),
            "counters_by_node": counters_by_node,
            "execution_profiles_by_node": execution_profiles,
            "adaptive_reservation_slots_at_end": reservation_slots_at_end,
            "adaptive_memory_accounting_consistent_at_end": accounting_consistent_at_end,
            "background_handshake_by_node": background_handshake_by_node,
            "adaptive_memory_at_end": max_memory,
            "adaptive_memory_high_watermark_by_node": memory_high_watermark,
            "probability_updates_by_node": probability_updates_by_node,
            "probability_tables": probability_tables,
            "probability_history_by_node": probability_history_by_node,
            "normalized_counters": normalized,
            "lifecycle_events": lifecycle_events,
        }

    def _normalize_acp_events(self, lifecycle_events: list[dict]) -> dict:
        generated = set()
        reused = set()
        reuse_tts_ps = []
        for event in lifecycle_events:
            name = event.get("event")
            if name == "background_pair_available":
                pair = event.get("pair")
                if pair:
                    generated.add((
                        event.get("time_ps"),
                        self._canonical_pair(pair),
                    ))
            elif name == "background_pair_adopted_by_application":
                pair = event.get("app_pair") or event.get("pair")
                if pair:
                    key = (
                        event.get("time_ps"),
                        event.get("reservation"),
                        self._canonical_pair(pair),
                    )
                    if key not in reused and event.get("recorded_tts_ps") is not None:
                        reuse_tts_ps.append(event["recorded_tts_ps"])
                    reused.add(key)
        return {
            "physical_background_pairs_generated": len(generated),
            "physical_background_pairs_reused": len(reused),
            "cache_reuse_tts_ps": sorted(reuse_tts_ps),
        }

    def _canonical_pair(self, pair) -> tuple:
        return tuple(sorted((tuple(pair[0]), tuple(pair[1]))))
