"""Algorithms must compare on the same physical swapping hardware."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from algorithms.registry import create_algorithm
from backends.sequence.runtime import SequenceRuntime
from workloads.concurrent_pairs import ConcurrentPairWorkload


@pytest.mark.parametrize("name", [
    "odo", "acp_freshest", "qcast", "qcast_distributed", "qguard", "qcast_e2e", "dfer",
])
def test_runtime_preserves_hardware_and_resolves_planner_probability(tmp_path, name):
    workload = ConcurrentPairWorkload(
        num_requests=1,
        num_nodes=3,
        topology_type="linear",
        memories_per_node=16,
        swapping_success_probability=0.9,
        simulation_end_time_s=0.001,
    )
    algorithm = create_algorithm(name, {"swap_success_probability": 1.0})
    runtime = SequenceRuntime(tmp_path)
    runtime.run(workload, algorithm)
    diagnostic = runtime.last_diagnostics["swap_probability"]
    assert set(diagnostic["by_router"].values()) == {0.9}
    assert diagnostic["source"] == "physical_topology"
    if hasattr(algorithm, "swap_success_probability"):
        assert algorithm.swap_success_probability == 1.0  # caller is unchanged
        assert diagnostic["requested_algorithm_value"] == 1.0
        assert diagnostic["resolved_algorithm_value"] == 0.9
    topology = json.loads((tmp_path / "topologies" / f"{algorithm.name}_topology.json").read_text())
    assert topology == workload.topology(adaptive_memory=getattr(algorithm, "adaptive_max_memory", 0))


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_planner_probability_comes_from_instantiated_topology(tmp_path, heterogeneous):
    workload = ConcurrentPairWorkload(num_requests=1, simulation_end_time_s=0.001)
    topology = workload.topology(adaptive_memory=0)
    routers = [node for node in topology["nodes"] if node["type"] == "QuantumRouter"]
    for router in routers:
        topology["templates"][router["template"]]["EntanglementSwapping"]["swapping_success_prob"] = 0.7
    # A template that is not used by any router must not affect resolution.
    topology["templates"]["unused"] = deepcopy(topology["templates"][routers[0]["template"]])
    topology["templates"]["unused"]["EntanglementSwapping"]["swapping_success_prob"] = 0.2
    if heterogeneous:
        routers[0]["template"] = "unused"
    workload = replace(workload, topology_override=topology)
    runtime = SequenceRuntime(tmp_path)
    algorithm = create_algorithm("qguard", {"swap_success_probability": 1.0})
    if heterogeneous:
        with pytest.raises(ValueError, match="uniform router swap probabilities"):
            runtime.run(workload, algorithm)
    else:
        runtime.run(workload, algorithm)
        assert runtime.last_diagnostics["swap_probability"]["resolved_algorithm_value"] == 0.7
