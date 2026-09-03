#!/usr/bin/env python3
"""Reproduce one canonical Q-GUARD condition under a selected source tree."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("QDC_SOURCE_ROOT", SCRIPT_ROOT)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import networkx as nx
import numpy as np

from algorithms.qcast import QCAST_CONTROL_PAPER_DISTRIBUTED
from algorithms.qguard import QGUARD
from backends.sequence.runtime import SequenceRuntime
from common import SECOND
from topology import _replace_router_graph, generate_hub_spoke_topology
from workloads.concurrent_pairs import ConcurrentPairSpec, ConcurrentPairWorkload


def _algorithm(window_ms: float) -> QGUARD:
    return QGUARD(
        edge_width=4,
        generation_window_ps=int(window_ms * 1e-3 * SECOND),
        control_processing_delay_ps=int(0.0001 * SECOND),
        swap_success_probability=0.9,
        link_state_hops=3,
        recovery_paths_per_segment=1,
        max_recovery_paths_per_major=12,
        max_major_paths=24,
        max_hops=12,
        max_purification_rounds=20,
        control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
        algorithm_name="qguard_reproduction",
    )


def _old_workload(seed: int) -> ConcurrentPairWorkload:
    return ConcurrentPairWorkload(
        num_requests=1,
        pair_count=1,
        seed=seed,
        num_nodes=6,
        qdc_node_index=0,
        topology_type="ring",
        extra_mesh_edges=0,
        inter_node_distance_m=10_000,
        memories_per_node=24,
        memory_fidelity=0.97,
        memory_efficiency=0.3,
        coherence_time_s=5.0,
        gate_fidelity=1.0,
        measurement_fidelity=1.0,
        swapping_success_probability=0.9,
        link_parallelism=4,
        simulation_end_time_s=1.2,
        request_override=(ConcurrentPairSpec(
            request_id=0,
            source="router_0",
            destination="router_3",
            start_time_ps=int(0.01 * SECOND),
            deadline_ps=int(1.0 * SECOND),
            pair_count=1,
            fidelity_threshold=0.9,
        ),),
    )


def _new_workload(seed: int) -> ConcurrentPairWorkload:
    kwargs = {
        "inter_node_distance_m": 1_000.0,
        "memo_size": 48,
        "adaptive_max_memory": 0,
        "memory_fidelity": 0.97,
        "memory_efficiency": 0.3,
        "coherence_time_s": 0.1,
        "gate_fidelity": 1.0,
        "measurement_fidelity": 1.0,
        "swapping_success_probability": 0.9,
        "attenuation": 0.0002,
        "stop_time_s": 0.055,
        "seed": seed,
        "extra_mesh_edges": 0,
        "qdc_node_index": 0,
        "encoding_type": "single_heralded",
        "formalism": "bell_diagonal",
    }
    topology = generate_hub_spoke_topology(num_nodes=25, **kwargs)
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5), ordering="sorted")
    topology = _replace_router_graph(topology, graph, kwargs)
    rng = np.random.default_rng(seed)
    for node in sorted(topology["nodes"], key=lambda item: item["name"]):
        node["seed"] = int(rng.integers(0, 2**31))
    request = ConcurrentPairSpec(
        request_id=0,
        source="router_0",
        destination="router_14",
        start_time_ps=int(0.01 * SECOND),
        deadline_ps=int(0.05 * SECOND),
        pair_count=1,
        fidelity_threshold=0.75,
    )
    return ConcurrentPairWorkload(
        num_requests=1,
        pair_count=1,
        seed=seed,
        num_nodes=25,
        qdc_node_index=0,
        topology_type="linear",
        inter_node_distance_m=1_000,
        memories_per_node=48,
        memory_fidelity=0.97,
        memory_efficiency=0.3,
        coherence_time_s=0.1,
        gate_fidelity=1.0,
        measurement_fidelity=1.0,
        swapping_success_probability=0.9,
        link_parallelism=4,
        simulation_end_time_s=0.055,
        request_override=(request,),
        topology_override=topology,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("old", "new"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workload = _old_workload(args.seed) if args.configuration == "old" else _new_workload(args.seed)
    window_ms = 5.0 if args.configuration == "old" else 2.0
    runtime = SequenceRuntime(args.output / "runtime")
    result = _algorithm(window_ms).run(runtime, workload)
    request = result.request_results[0]
    counters = runtime.last_diagnostics["workload_diagnostics"]["counters"]
    record = {
        "source_root": str(SOURCE_ROOT),
        "configuration": args.configuration,
        "seed": args.seed,
        "success": request.success,
        "time_to_serve_ms": request.time_to_serve_ms,
        "fidelity": request.fidelity,
        "failure_reason": request.failure_reason or "",
        "slots_started": counters.get("slots_started", 0),
        "purification_attempts": counters.get("qguard_purification_attempts", 0),
        "swap_attempts": counters.get("swaps_attempted", 0),
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
