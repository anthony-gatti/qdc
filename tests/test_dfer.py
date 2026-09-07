"""DFER paper math, locality, physical execution, and workload integration."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest
from sequence.kernel.event import Event
from sequence.kernel.process import Process

from algorithms.dfer import (
    DFER,
    bbpssw_werner_once,
    expected_distribution_rate,
    pumping_plan,
    required_link_fidelity,
    werner_parameter,
)
from algorithms.registry import algorithm_names, create_algorithm
from backends.sequence.runtime import SequenceRuntime
from backends.sequence.dfer_scheduler import DFERDemandScheduler
from common import SECOND
from workloads.concurrent_pairs import ConcurrentPairSpec, ConcurrentPairWorkload
from workloads.qpq import QPQWorkload


def _pair_workload(
    *,
    nodes: int,
    destination: int,
    threshold: float,
    distance_m: float = 1_000,
    pair_count: int = 1,
    deadline_s: float = 1.5,
) -> ConcurrentPairWorkload:
    return ConcurrentPairWorkload(
        num_requests=1,
        num_nodes=nodes,
        qdc_node_index=0,
        topology_type="linear",
        inter_node_distance_m=distance_m,
        memories_per_node=16,
        memory_fidelity=0.97,
        memory_efficiency=1.0,
        coherence_time_s=5.0,
        gate_fidelity=1.0,
        measurement_fidelity=1.0,
        swapping_success_probability=1.0,
        simulation_end_time_s=deadline_s + 0.1,
        request_override=(
            ConcurrentPairSpec(
                1,
                "router_0",
                f"router_{destination}",
                int(0.01 * SECOND),
                int(deadline_s * SECOND),
                pair_count,
                threshold,
            ),
        ),
    )


def test_dfer_is_registered_and_validates_configuration():
    assert "dfer" in algorithm_names()
    algorithm = create_algorithm("dfer", {
        "cutoff_fidelity": 0.72,
        "max_purification_rounds": 7,
        "control_processing_delay_s": 0.0002,
    })
    assert algorithm.cutoff_fidelity == pytest.approx(0.72)
    assert algorithm.max_purification_rounds == 7
    assert algorithm.control_processing_delay_ps == 200_000_000
    with pytest.raises(ValueError):
        DFER(cutoff_fidelity=0.49)


@pytest.mark.parametrize("coherence,processing,deadline", [
    (0.0001, 0.0005, 0.05),
    (5.0, 0.001, 0.013),
    (5.0, 0.0001, 0.0108),
    (0.001, 0.0005, 0.05),
])
def test_delayed_pumping_handles_expiration_and_deadlines(
    tmp_path, monkeypatch, coherence, processing, deadline,
):
    # Inspect cleanup at failure, before runtime teardown can mask leaked state.
    failures = []
    original_fail = DFERDemandScheduler._fail

    def checked_fail(self, context, *args, **kwargs):
        protocols = [
            protocol
            for operations in (self.purification_operations, self.swap_operations)
            for op in operations.values()
            for protocol in op.protocols
        ]
        original_fail(self, context, *args, **kwargs)
        reservation_id = context.demand.reservation_id
        for operations in (self.purification_operations, self.swap_operations):
            assert not any(
                op.reservation_id == reservation_id for op in operations.values()
            )
        assert all(not indices for indices in self.claimed_memories.values())
        for protocol in protocols:
            assert protocol not in protocol.owner.protocols
            assert all(protocol not in memory._observers for memory in protocol.memories)
        failures.append(reservation_id)

    monkeypatch.setattr(DFERDemandScheduler, "_fail", checked_fail)
    workload = replace(
        _pair_workload(nodes=4, destination=3, threshold=0.95, deadline_s=deadline),
        coherence_time_s=coherence,
        simulation_end_time_s=0.06,
    )
    runtime = SequenceRuntime(tmp_path)
    result = runtime.run(workload, DFER(
        swap_success_probability=1.0,
        control_processing_delay_ps=int(processing * SECOND),
    ))
    assert failures
    assert not result.request_results[0].success
    assert result.request_results[0].failure_reason == "dfer_deadline"
    diagnostics = runtime.last_diagnostics["workload_diagnostics"]
    assert all(diagnostics["all_memories_raw_at_end"].values())
    assert diagnostics["locality_invariants"]["all_claims_released"]


@pytest.mark.parametrize("phase", ["purification", "swap"])
def test_deadline_cancels_inflight_protocols_before_memory_reuse(tmp_path, monkeypatch, phase):
    method = "_begin_purification" if phase == "purification" else "_start_swap"
    original = getattr(DFERDemandScheduler, method)
    cancelled = []

    def expire_during_operation(self, *args):
        original(self, *args)
        operations = self.purification_operations if phase == "purification" else self.swap_operations
        for operation in operations.values():
            if operation.reservation_id == 1 and operation.protocols and not cancelled:
                cancelled.append(operation)
                # End the request after native start but before result messages.
                self.timeline.schedule(Event(
                    self.timeline.now() + 1, Process(self, "deadline", [1]),
                ))

    monkeypatch.setattr(DFERDemandScheduler, method, expire_during_operation)
    workload = replace(
        _pair_workload(nodes=4, destination=3, threshold=0.95, deadline_s=0.06),
        num_requests=2,
        memories_per_node=3,
        simulation_end_time_s=0.07,
        request_override=(
            ConcurrentPairSpec(1, "router_0", "router_3", int(0.01 * SECOND), int(0.06 * SECOND), 1, 0.95),
            ConcurrentPairSpec(2, "router_0", "router_3", int(0.02 * SECOND), int(0.06 * SECOND), 1, 0.95),
        ),
    )
    runtime = SequenceRuntime(tmp_path)
    result = runtime.run(workload, DFER(swap_success_probability=1.0))
    assert cancelled
    first, second = sorted(result.request_results, key=lambda request: request.request_id)
    assert not first.success
    assert second.success
    for operation in cancelled:
        for protocol in operation.protocols:
            assert protocol not in protocol.owner.protocols
            assert all(protocol not in memory._observers for memory in protocol.memories)
    assert runtime.last_diagnostics["workload_diagnostics"]["counters"]["operations_cancelled"] == 1


def test_dlfr_equalizes_the_remaining_werner_budget():
    threshold = 0.95
    target = required_link_fidelity(threshold, 3)
    assert werner_parameter(target) ** 3 == pytest.approx(
        werner_parameter(threshold)
    )

    current = 0.98
    remaining_target = required_link_fidelity(threshold, 2, current)
    assert (
        werner_parameter(current) * werner_parameter(remaining_target) ** 2
        == pytest.approx(werner_parameter(threshold))
    )


def test_dfer_uses_pumping_not_a_binary_purification_tree():
    plan = pumping_plan(0.97, required_link_fidelity(0.95, 3), 20)
    assert plan is not None
    assert plan.rounds == 3
    assert plan.output_fidelity >= plan.target_fidelity
    assert plan.success_probability < 1
    current = 0.97
    for _ in range(plan.rounds):
        current, _ = bbpssw_werner_once(current, 0.97)
    assert current == pytest.approx(plan.output_fidelity)
    # Three pumping rounds consume four raw pairs, not 2**3.
    assert plan.rounds + 1 == 4


def test_dfer_planning_matches_hardware_limited_bbpssw_ceiling():
    ideal = pumping_plan(0.97, 0.98, 20)
    hardware_limited = pumping_plan(
        0.97,
        0.98,
        20,
        kept_gate_fidelity=0.99,
        remote_gate_fidelity=0.99,
        kept_measurement_fidelity=0.99,
        remote_measurement_fidelity=0.99,
    )
    assert ideal is not None
    assert hardware_limited is None


def test_dfps_edr_penalizes_purification_and_remaining_swaps():
    no_purification = pumping_plan(0.97, 0.96, 5)
    purified = pumping_plan(0.97, 0.98, 5)
    assert no_purification is not None and purified is not None
    fast = expected_distribution_rate(20, no_purification, 0.001, 1, 0.9)
    costly = expected_distribution_rate(20, purified, 0.001, 3, 0.9)
    assert fast > costly > 0


def test_direct_dfer_models_neighbor_round_trip_and_cleans_resources(tmp_path):
    workload = _pair_workload(
        nodes=2,
        destination=1,
        threshold=0.9,
        distance_m=10_000,
    )
    runtime = SequenceRuntime(tmp_path / "direct")
    result = runtime.run(workload, DFER(swap_success_probability=1.0))
    request = result.request_results[0]
    diagnostics = runtime.last_diagnostics["workload_diagnostics"]

    assert request.success
    # 10 km router-to-router classical propagation is 50 us each way.  Two
    # 100 us endpoint processing delays plus propagation make the state
    # request/response phase at least 0.30 ms before physical generation.
    submitted = next(
        event for event in diagnostics["events"]
        if event["event"] == "demand_submitted"
    )
    generation = next(
        event for event in diagnostics["events"]
        if event["event"] == "elementary_generation_started"
    )
    assert generation["time_ps"] - submitted["time_ps"] >= 300_000_000
    assert diagnostics["counters"]["neighbor_state_queries_sent"] == 1
    assert diagnostics["counters"]["neighbor_state_responses_received"] == 1
    assert all(diagnostics["all_memories_raw_at_end"].values())
    assert diagnostics["locality_invariants"]["all_claims_released"]


def test_multihop_dfer_physically_pumps_swaps_and_guarantees_fidelity(tmp_path):
    workload = _pair_workload(nodes=4, destination=3, threshold=0.95)
    runtime = SequenceRuntime(tmp_path / "multihop")
    result = runtime.run(workload, DFER(swap_success_probability=1.0))
    request = result.request_results[0]
    diagnostics = runtime.last_diagnostics["workload_diagnostics"]
    counters = diagnostics["counters"]

    assert request.success
    assert request.fidelity >= 0.95
    assert counters["dlfr_evaluations"] == 3
    assert counters["dfps_selections"] == 3
    assert counters["pumping_attempts"] >= 3
    assert counters["pumping_successes"] >= 3
    assert counters["swaps_attempted"] == 2
    assert counters["swaps_succeeded"] == 2
    assert all(diagnostics["all_memories_raw_at_end"].values())
    assert all(
        not indices
        for indices in diagnostics["claimed_memories_at_end"].values()
    )
    assert diagnostics["locality_invariants"] == {
        "all_state_queries_one_hop": True,
        "no_global_control_messages": True,
        "all_claims_released": True,
    }


def test_dfps_selects_the_higher_edr_progress_neighbor(tmp_path):
    base = ConcurrentPairWorkload(
        num_requests=1,
        num_nodes=4,
        qdc_node_index=0,
        topology_type="ring",
        inter_node_distance_m=1_000,
        memories_per_node=12,
        memory_fidelity=0.97,
        memory_efficiency=1.0,
        coherence_time_s=5.0,
        gate_fidelity=1.0,
        measurement_fidelity=1.0,
        swapping_success_probability=1.0,
        simulation_end_time_s=1.0,
    ).topology(adaptive_memory=0)
    for channel in base["qchannels"]:
        if channel["destination"] == "BSM_0_3":
            channel["distance"] = 5_000.0
    workload = ConcurrentPairWorkload(
        num_requests=1,
        num_nodes=4,
        qdc_node_index=0,
        topology_type="ring",
        memories_per_node=12,
        simulation_end_time_s=1.0,
        topology_override=base,
        request_override=(
            ConcurrentPairSpec(
                1,
                "router_0",
                "router_2",
                int(0.01 * SECOND),
                int(0.9 * SECOND),
                1,
                0.9,
            ),
        ),
    )
    runtime = SequenceRuntime(tmp_path / "edr")
    result = runtime.run(workload, DFER(swap_success_probability=1.0))
    decisions = runtime.last_diagnostics[
        "workload_diagnostics"
    ]["dfer"]["decisions"]

    assert result.request_results[0].success
    first = decisions[0]
    assert {item["neighbor"] for item in first["candidates"]} == {
        "router_1",
        "router_3",
    }
    assert first["selected"] == "router_1"
    scores = {
        item["neighbor"]: item["expected_edr_hz"]
        for item in first["candidates"]
    }
    assert scores["router_1"] > scores["router_3"]


def test_dfer_supports_multiple_pairs_sequentially(tmp_path):
    workload = _pair_workload(
        nodes=2,
        destination=1,
        threshold=0.9,
        pair_count=3,
    )
    runtime = SequenceRuntime(tmp_path / "multiple")
    result = runtime.run(workload, DFER(swap_success_probability=1.0))
    request = result.request_results[0]
    diagnostics = runtime.last_diagnostics["workload_diagnostics"]

    assert request.success
    assert len(request.pair_arrival_ms) == 3
    assert request.pair_arrival_ms == sorted(request.pair_arrival_ms)
    assert diagnostics["counters"]["end_to_end_pairs_delivered"] == 3


def test_dfer_runs_qpq_without_changing_result_schema(tmp_path):
    workload = QPQWorkload(
        database_size_log=1,
        num_clients=1,
        queries_per_client=1,
        round_deadline_s=1.0,
        transaction_duration_s=3.0,
        request_period_s=2.0,
        seed=7,
        num_nodes=2,
        qdc_node_index=0,
        topology_type="linear",
        extra_mesh_edges=0,
        inter_node_distance_m=1_000,
        memories_per_node=20,
        memory_fidelity=0.99,
        memory_efficiency=1.0,
        coherence_time_s=5.0,
        fidelity_threshold=0.9,
        link_parallelism=1,
    )
    runtime = SequenceRuntime(tmp_path / "qpq")
    result = runtime.run(workload, DFER(swap_success_probability=1.0))
    request = result.request_results[0]
    diagnostics = runtime.last_diagnostics["workload_diagnostics"]

    assert request.success
    assert request.round1_pairs == 3
    assert request.round2_pairs == 3
    assert len(request.pair_arrival_ms) == 6
    assert diagnostics["transactions"]["0"]["success"]
    assert all(diagnostics["all_memories_raw_at_end"].values())
