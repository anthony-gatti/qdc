from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sequence.resource_management.memory_manager import MemoryInfo

from algorithms.qcast import (
    QCAST,
    QCAST_CONTROL_PAPER_DISTRIBUTED,
    QCASTDemand,
    QCASTEdge,
    QCASTPath,
    QCASTPlanner,
    expected_throughput,
)
from algorithms.odo import ShortestPathOnDemand
from algorithms.registry import create_algorithm
from backends.sequence.qcast_scheduler import (
    QCASTDemandScheduler,
    QCASTLane,
    QCASTLaneEdge,
    QCASTPathAllocation,
    single_heralded_attempt_success_probability,
)
from backends.sequence.qcast_distributed import QCASTDistributedDemandScheduler
from backends.sequence.runtime import SequenceRuntime
from backends.sequence.qcast_topology import expand_qcast_parallel_links
from common import SECOND
from topology import SPEED_OF_LIGHT, generate_linear_topology
from workloads.concurrent_pairs import (
    ConcurrentPairSpec,
    ConcurrentPairWorkload,
)


class QCASTPlannerTest(unittest.TestCase):
    def test_registry_constructs_qcast(self):
        algorithm = create_algorithm("qcast", {"edge_width": 4})
        self.assertIsInstance(algorithm, QCAST)
        self.assertEqual(algorithm.edge_width, 4)

    def test_registry_constructs_paper_distributed_qcast(self):
        algorithm = create_algorithm("qcast_distributed", {"edge_width": 2})
        self.assertIsInstance(algorithm, QCAST)
        self.assertEqual(algorithm.config.name, "qcast_distributed")
        self.assertEqual(algorithm.config.kind, "qcast")
        self.assertEqual(algorithm.control_mode, QCAST_CONTROL_PAPER_DISTRIBUTED)

    def test_qcast_rejects_unintegrated_workloads(self):
        workload = SimpleNamespace(sequence_adapter="single_pair")
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            with self.assertRaisesRegex(NotImplementedError, "concurrent_pairs and QPQ"):
                QCAST().run(runtime, workload)

    def test_ext_matches_single_hop_binomial_expectation(self):
        self.assertAlmostEqual(expected_throughput(1, [0.6], 0.9), 0.6)
        self.assertAlmostEqual(expected_throughput(2, [0.6], 0.9), 1.2)
        self.assertAlmostEqual(expected_throughput(1, [0.6, 0.6], 0.9), 0.324)

    def test_single_heralded_probability_matches_upstream_protocol(self):
        round_probability = 0.5 * (0.6 * 0.8) ** 2 * 0.9**2
        self.assertAlmostEqual(
            single_heralded_attempt_success_probability(
                0.6 * 0.8,
                0.6 * 0.8,
                (0.9, 0.9),
                0.5,
            ),
            round_probability,
        )

    def test_zero_probability_edges_are_not_reserved(self):
        planner = QCASTPlanner(link_state_hops=0)
        plan = planner.plan(
            [QCASTDemand("d0", "a", "b")],
            {"a": 2, "b": 2},
            [QCASTEdge("a", "b", 2, 0.0)],
        )
        self.assertEqual(plan.major_paths, ())

    def test_parallel_link_expansion_creates_independent_seeded_bsms(self):
        first = ConcurrentPairWorkload(seed=1, num_nodes=2).topology(0)
        second = ConcurrentPairWorkload(seed=2, num_nodes=2).topology(0)
        first = expand_qcast_parallel_links(first, 3)
        second = expand_qcast_parallel_links(second, 3)
        first_bsms = [
            node for node in first["nodes"] if node["type"] == "BSMNode"
        ]
        second_bsms = [
            node for node in second["nodes"] if node["type"] == "BSMNode"
        ]
        self.assertEqual(len(first_bsms), 3)
        self.assertEqual(len({node["seed"] for node in first_bsms}), 3)
        self.assertNotEqual(
            [node["seed"] for node in first_bsms],
            [node["seed"] for node in second_bsms],
        )

    def test_wide_reliable_path_beats_narrow_direct_path(self):
        planner = QCASTPlanner(
            swap_success_probability=0.9,
            max_major_paths=1,
            max_hops=4,
            link_state_hops=0,
        )
        plan = planner.plan(
            [QCASTDemand("d0", "a", "d")],
            {"a": 4, "b": 8, "d": 4},
            [
                QCASTEdge("a", "d", 1, 0.5),
                QCASTEdge("a", "b", 2, 0.9),
                QCASTEdge("b", "d", 2, 0.9),
            ],
        )
        self.assertEqual(len(plan.major_paths), 1)
        self.assertEqual(plan.major_paths[0].nodes, ("a", "b", "d"))
        self.assertEqual(plan.major_paths[0].width, 2)

    def test_major_paths_are_contention_free(self):
        planner = QCASTPlanner(
            max_major_paths=10,
            max_hops=4,
            link_state_hops=0,
        )
        plan = planner.plan(
            [
                QCASTDemand("d0", "a", "d"),
                QCASTDemand("d1", "b", "c"),
            ],
            {node: 2 for node in "abcd"},
            [
                QCASTEdge("a", "b", 1, 0.9),
                QCASTEdge("b", "d", 1, 0.9),
                QCASTEdge("a", "c", 1, 0.9),
                QCASTEdge("c", "d", 1, 0.9),
                QCASTEdge("b", "c", 1, 0.9),
            ],
        )
        used_edges = [edge for path in plan.major_paths for edge in path.edges]
        self.assertEqual(len(used_edges), len(set(used_edges)))
        self.assertTrue(all(value >= 0 for value in plan.residual_node_memories.values()))

    def test_recovery_selector_routes_around_failed_major_edge(self):
        major_path = QCASTPath(
            "major-0", "d0", ("a", "b", "d"), 1, 0.5,
        )
        recovery_path = QCASTPath(
            "recovery-0",
            "d0",
            ("a", "c", "b"),
            1,
            0.4,
            kind="recovery",
            parent_path_id="major-0",
            covered_segment=(0, 1),
        )

        def assignment(lane_id, left, right):
            return QCASTLaneEdge(lane_id, left, right, 0, 0, None)

        major_lane = QCASTLane(
            "major-lane",
            major_path,
            0,
            (
                assignment("m0", "a", "b"),
                assignment("m1", "b", "d"),
            ),
        )
        recovery_lane = QCASTLane(
            "recovery-lane",
            recovery_path,
            0,
            (
                assignment("r0", "a", "c"),
                assignment("r1", "c", "b"),
            ),
        )
        slot = SimpleNamespace(edge_success={
            "m0": False,
            "m1": True,
            "r0": True,
            "r1": True,
        })
        scheduler = object.__new__(QCASTDemandScheduler)
        selected = scheduler._select_physical_path(
            slot,
            major_lane,
            [QCASTPathAllocation(recovery_path, (recovery_lane,))],
        )
        self.assertIsNotNone(selected)
        nodes, _assignments, used_recovery = selected
        self.assertEqual(nodes, ("a", "c", "b", "d"))
        self.assertTrue(used_recovery)

        distributed = object.__new__(QCASTDistributedDemandScheduler)
        selected = distributed._select_xor_path(
            major_lane,
            [QCASTPathAllocation(recovery_path, (recovery_lane,))],
            slot.edge_success,
            set(),
        )
        self.assertIsNotNone(selected)
        nodes, _assignments, consumed = selected
        self.assertEqual(nodes, ("a", "c", "b", "d"))
        self.assertEqual(consumed, {"r0", "r1"})

    def test_recovery_selector_rejects_overlapping_segment_repairs(self):
        major_path = QCASTPath(
            "major-0", "d0", ("a", "b", "c", "d"), 1, 0.5,
        )

        def assignment(lane_id, left, right):
            return QCASTLaneEdge(lane_id, left, right, 0, 0, None)

        major_lane = QCASTLane(
            "major-lane",
            major_path,
            0,
            (
                assignment("m0", "a", "b"),
                assignment("m1", "b", "c"),
                assignment("m2", "c", "d"),
            ),
        )
        first_path = QCASTPath(
            "r0", "d0", ("a", "x", "c"), 1, 0.4,
            kind="recovery", parent_path_id="major-0", covered_segment=(0, 2),
        )
        second_path = QCASTPath(
            "r1", "d0", ("b", "y", "d"), 1, 0.4,
            kind="recovery", parent_path_id="major-0", covered_segment=(1, 3),
        )
        first_lane = QCASTLane(
            "first", first_path, 0,
            (assignment("r0a", "a", "x"), assignment("r0b", "x", "c")),
        )
        second_lane = QCASTLane(
            "second", second_path, 0,
            (assignment("r1a", "b", "y"), assignment("r1b", "y", "d")),
        )
        slot = SimpleNamespace(edge_success={
            "m0": False,
            "m1": True,
            "m2": False,
            "r0a": True,
            "r0b": True,
            "r1a": True,
            "r1b": True,
        })
        scheduler = object.__new__(QCASTDemandScheduler)
        selected = scheduler._select_physical_path(
            slot,
            major_lane,
            [
                QCASTPathAllocation(first_path, (first_lane,)),
                QCASTPathAllocation(second_path, (second_lane,)),
            ],
        )
        self.assertIsNone(selected)


class QCASTSequenceIntegrationTest(unittest.TestCase):
    def _run(self, workload, algorithm):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = algorithm.run(runtime, workload)
        return result, runtime.last_diagnostics["workload_diagnostics"]

    @staticmethod
    def _heterogeneous_control_workload():
        normal_distance_m = 1_000.0
        remote_distance_m = 1_000_000.0
        topology = generate_linear_topology(
            8,
            inter_node_distance_m=normal_distance_m,
            memo_size=8,
            adaptive_max_memory=0,
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            coherence_time_s=0.002,
            gate_fidelity=0.99,
            measurement_fidelity=0.99,
            swapping_success_probability=0.9,
            stop_time_s=0.08,
            qdc_node_index=0,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
            seed=31,
        )
        link_distances = [normal_distance_m] * 6 + [remote_distance_m]
        for channel in topology["qchannels"]:
            if channel["destination"] == "BSM_6_7":
                channel["distance"] = remote_distance_m / 2
        for channel in topology["cchannels"]:
            source, destination = channel["source"], channel["destination"]
            if source.startswith("router_") and destination.startswith("router_"):
                left, right = sorted((
                    int(source.split("_")[1]),
                    int(destination.split("_")[1]),
                ))
                channel["delay"] = sum(link_distances[left:right]) / SPEED_OF_LIGHT
            elif source == "BSM_6_7" or destination == "BSM_6_7":
                channel["delay"] = remote_distance_m / (2 * SPEED_OF_LIGHT)

        return ConcurrentPairWorkload(
            num_requests=1,
            seed=31,
            num_nodes=8,
            qdc_node_index=0,
            topology_type="linear",
            memories_per_node=8,
            coherence_time_s=0.002,
            simulation_end_time_s=0.08,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                int(0.01 * SECOND),
                int(0.07 * SECOND),
                fidelity_threshold=0.26,
            ),),
            topology_override=topology,
        )

    def test_direct_pair_includes_control_and_generation_phases(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=3,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=6,
            simulation_end_time_s=0.04,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.03 * SECOND),
            ),),
        )
        generation_window = int(0.003 * SECOND)
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=2,
                generation_window_ps=generation_window,
                max_major_paths=1,
                link_state_hops=1,
            ),
        )
        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertGreater(
            result.request_results[0].time_to_serve_ms,
            generation_window / 1e9,
        )
        self.assertGreater(diagnostics["counters"]["plan_messages_sent"], 0)
        self.assertGreater(diagnostics["counters"]["link_state_messages_sent"], 0)
        self.assertEqual(diagnostics["edge_models"][0]["attempt_duration_ps"], 20_000_000)
        self.assertEqual(len(diagnostics["edge_models"][0]["physical_channels"]), 1)
        self.assertEqual(diagnostics["edge_models"][0]["scheduling_width"], 1)
        self.assertEqual(
            diagnostics["edge_width_model"],
            "path_scheduling_cap_over_shared_link_parallelism",
        )
        self.assertEqual(
            diagnostics["counters"]["end_to_end_pairs_delivered_major"],
            1,
        )
        self.assertGreater(
            sum(
                diagnostics["scheduled_lane_usage_by_link"]["router_0|router_1"].values()
            ),
            0,
        )
        timing = diagnostics["timing"]
        self.assertEqual(timing["summary"]["delivered_pairs_timed"], 1)
        self.assertGreater(
            timing["deliveries"][0]["oldest_pair_age_at_generation_window_close_ps"],
            0,
        )
        self.assertGreaterEqual(
            timing["deliveries"][0]["post_generation_control_wait_ps"],
            0,
        )
        self.assertEqual(
            timing["slots"][0]["major_path_p3_readiness"][0]["path"],
            ["router_0", "router_1"],
        )
        self.assertTrue(all(
            state["plan_slots"]
            for state in diagnostics["control_state_by_node"].values()
        ))
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_qcast_scheduling_width_does_not_create_hardware(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=4,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=6,
            link_parallelism=3,
            simulation_end_time_s=0.04,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.03 * SECOND),
            ),),
        )
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=1,
                generation_window_ps=int(0.003 * SECOND),
                max_major_paths=1,
                link_state_hops=1,
            ),
        )
        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        edge = diagnostics["edge_models"][0]
        self.assertEqual(edge["physical_parallelism"], 3)
        self.assertEqual(edge["scheduling_width"], 1)
        self.assertLessEqual(
            max(diagnostics["scheduled_lane_usage_by_link"]["router_0|router_1"].values()),
            diagnostics["counters"]["slots_started"],
        )

    def test_central_p4_barrier_can_outwait_a_local_path(self):
        workload = self._heterogeneous_control_workload()
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=1,
                generation_window_ps=int(0.001 * SECOND),
                control_processing_delay_ps=int(0.0001 * SECOND),
                link_state_hops=1,
                max_major_paths=1,
                max_hops=8,
            ),
        )

        self.assertEqual((result.num_requests, result.num_success), (1, 0))
        path_timing = diagnostics["timing"]["slots"][0]["major_path_p3_readiness"][0]
        self.assertGreater(path_timing["global_barrier_excess_ps"], int(0.004 * SECOND))

    def test_paper_distributed_p4_ignores_unrelated_remote_link(self):
        workload = self._heterogeneous_control_workload()
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=1,
                generation_window_ps=int(0.001 * SECOND),
                control_processing_delay_ps=int(0.0001 * SECOND),
                link_state_hops=1,
                max_major_paths=1,
                max_hops=8,
                control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
            ),
        )

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertGreater(result.request_results[0].fidelity, 0.26)
        path_timing = diagnostics["timing"]["slots"][0]["major_path_p3_readiness"][0]
        self.assertLess(path_timing["global_barrier_excess_ps"], 10)
        self.assertEqual(diagnostics["control_mode"], "paper_distributed")
        self.assertEqual(
            diagnostics["locality_invariants"]["network_wide_p4_barriers"],
            0,
        )
        self.assertTrue(
            diagnostics["locality_invariants"]["all_decisions_path_scoped"]
        )
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_concurrent_pair_workload_remains_usable_by_odo(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=9,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=4,
            simulation_end_time_s=0.04,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.03 * SECOND),
            ),),
        )
        result, diagnostics = self._run(workload, ShortestPathOnDemand())
        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertGreater(diagnostics["counters"]["pairs_delivered"], 0)

    def test_two_pair_demand_uses_two_physical_lanes(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=12,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=4,
            simulation_end_time_s=0.04,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.03 * SECOND),
                pair_count=2,
            ),),
        )
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=2,
                generation_window_ps=int(0.003 * SECOND),
                max_major_paths=1,
                link_state_hops=1,
            ),
        )
        request = result.request_results[0]
        self.assertTrue(request.success)
        self.assertEqual(len(request.pair_arrival_ms), 2)
        self.assertEqual(diagnostics["counters"]["end_to_end_pairs_delivered"], 2)

    def test_deadline_failure_when_window_cannot_complete_an_attempt(self):
        start = int(0.001 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=13,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=20_000,
            memories_per_node=2,
            simulation_end_time_s=0.004,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.003 * SECOND),
            ),),
        )
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=1,
                generation_window_ps=int(0.0001 * SECOND),
                max_major_paths=1,
                link_state_hops=0,
            ),
        )
        request = result.request_results[0]
        self.assertFalse(request.success)
        self.assertEqual(request.failure_reason, "qcast_deadline")
        self.assertGreater(diagnostics["counters"]["slots_without_paths"], 0)

    def test_simulation_end_cleans_an_active_slot(self):
        start = int(0.001 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=14,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=20_000,
            memories_per_node=2,
            simulation_end_time_s=0.003,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.02 * SECOND),
            ),),
        )
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=1,
                generation_window_ps=int(0.01 * SECOND),
                max_major_paths=1,
                link_state_hops=0,
            ),
        )
        request = result.request_results[0]
        self.assertFalse(request.success)
        self.assertEqual(request.failure_reason, "simulation_end")
        self.assertEqual(
            diagnostics["counters"]["slots_cleaned_at_simulation_end"],
            1,
        )
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_multihop_pair_uses_official_swapping_and_respects_memory_capacity(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=5,
            num_nodes=3,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=8,
            memory_fidelity=0.99,
            swapping_success_probability=1.0,
            simulation_end_time_s=0.05,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_2",
                start,
                int(0.04 * SECOND),
            ),),
        )
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=2,
                generation_window_ps=int(0.003 * SECOND),
                swap_success_probability=1.0,
                max_major_paths=1,
                link_state_hops=2,
            ),
        )
        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertGreater(diagnostics["counters"]["swaps_attempted"], 0)
        self.assertLess(result.request_results[0].fidelity, 0.99)
        self.assertTrue(all(
            count <= workload.memories_per_node
            for count in diagnostics["max_allocated_memories_by_node"].values()
        ))
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_long_swap_chain_waits_for_distant_endpoint_update(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=15,
            num_nodes=5,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=10_000,
            memories_per_node=8,
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            swapping_success_probability=1.0,
            simulation_end_time_s=0.08,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_4",
                start,
                int(0.07 * SECOND),
                fidelity_threshold=0.5,
            ),),
        )
        result, diagnostics = self._run(
            workload,
            QCAST(
                edge_width=1,
                generation_window_ps=int(0.01 * SECOND),
                swap_success_probability=1.0,
                max_major_paths=1,
                max_hops=5,
                link_state_hops=4,
                control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
            ),
        )

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertEqual(diagnostics["counters"]["swaps_attempted"], 3)
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_physical_recovery_lane_repairs_forced_major_link_failure(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=0,
            num_nodes=5,
            qdc_node_index=2,
            topology_type="hub_spoke",
            extra_mesh_edges=1,
            inter_node_distance_m=1_000,
            memories_per_node=20,
            simulation_end_time_s=0.06,
            request_override=(ConcurrentPairSpec(
                0,
                "router_2",
                "router_0",
                start,
                int(0.05 * SECOND),
            ),),
        )
        algorithm = QCAST(
            edge_width=2,
            generation_window_ps=int(0.004 * SECOND),
            swap_success_probability=1.0,
            max_major_paths=1,
            link_state_hops=3,
        )
        original_stop_generation = QCASTDemandScheduler.stop_generation
        forced_failures = []

        def stop_with_major_failure(scheduler, slot_id):
            if not forced_failures:
                major = scheduler.slot.plan.major_paths[0]
                assignment = scheduler.slot.allocations[major.path_id].lanes[0].edges[0]
                for node_name in (assignment.left, assignment.right):
                    memory = scheduler._memory(
                        node_name,
                        assignment.memory_index(node_name),
                    )
                    scheduler.routers[node_name].resource_manager.update(
                        None,
                        memory,
                        MemoryInfo.RAW,
                    )
                forced_failures.append(assignment.lane_id)
            return original_stop_generation(scheduler, slot_id)

        with patch.object(
            QCASTDemandScheduler,
            "stop_generation",
            stop_with_major_failure,
        ):
            result, diagnostics = self._run(workload, algorithm)

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertEqual(len(forced_failures), 1)
        self.assertGreater(diagnostics["counters"]["recovery_paths_selected"], 0)
        self.assertGreater(
            diagnostics["counters"]["end_to_end_pairs_delivered_recovery"],
            0,
        )
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_distributed_recovery_repairs_forced_major_link_failure(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=0,
            num_nodes=5,
            qdc_node_index=2,
            topology_type="hub_spoke",
            extra_mesh_edges=1,
            inter_node_distance_m=1_000,
            memories_per_node=20,
            simulation_end_time_s=0.06,
            request_override=(ConcurrentPairSpec(
                0,
                "router_2",
                "router_0",
                start,
                int(0.05 * SECOND),
            ),),
        )
        algorithm = QCAST(
            edge_width=2,
            generation_window_ps=int(0.004 * SECOND),
            swap_success_probability=1.0,
            max_major_paths=1,
            link_state_hops=3,
            control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
        )
        original_stop_generation = QCASTDistributedDemandScheduler.stop_generation
        forced_failures = []

        def stop_with_major_failure(scheduler, slot_id):
            if not forced_failures:
                major = scheduler.slot.plan.major_paths[0]
                assignment = scheduler.slot.allocations[major.path_id].lanes[0].edges[0]
                for node_name in (assignment.left, assignment.right):
                    memory = scheduler._memory(
                        node_name,
                        assignment.memory_index(node_name),
                    )
                    scheduler.routers[node_name].resource_manager.update(
                        None,
                        memory,
                        MemoryInfo.RAW,
                    )
                forced_failures.append(assignment.lane_id)
            return original_stop_generation(scheduler, slot_id)

        with patch.object(
            QCASTDistributedDemandScheduler,
            "stop_generation",
            stop_with_major_failure,
        ):
            result, diagnostics = self._run(workload, algorithm)

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertEqual(len(forced_failures), 1)
        self.assertGreater(diagnostics["counters"]["major_lanes_recovered"], 0)
        self.assertGreater(
            diagnostics["counters"]["end_to_end_pairs_delivered_recovery"],
            0,
        )
        decision = diagnostics["distributed_decisions"][0]
        self.assertTrue(any(
            selection.get("used_recovery")
            for selection in decision["selections"]
        ))
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_recovery_reservations_use_residual_resources(self):
        planner = QCASTPlanner(
            max_major_paths=1,
            max_hops=5,
            link_state_hops=2,
            recovery_paths_per_segment=1,
            max_recovery_paths_per_major=4,
        )
        plan = planner.plan(
            [QCASTDemand("d0", "a", "d")],
            {"a": 5, "b": 6, "c": 6, "d": 5, "x": 8},
            [
                QCASTEdge("a", "b", 2, 0.95),
                QCASTEdge("b", "c", 2, 0.95),
                QCASTEdge("c", "d", 2, 0.95),
                QCASTEdge("a", "x", 1, 0.9),
                QCASTEdge("x", "b", 1, 0.9),
                QCASTEdge("x", "c", 1, 0.9),
                QCASTEdge("x", "d", 1, 0.9),
            ],
        )
        self.assertEqual(len(plan.major_paths), 1)
        self.assertGreater(len(plan.recovery_paths), 0)
        self.assertTrue(all(path.kind == "recovery" for path in plan.recovery_paths))
        self.assertTrue(all(path.parent_path_id == plan.major_paths[0].path_id for path in plan.recovery_paths))
        self.assertTrue(all(value >= 0 for value in plan.residual_node_memories.values()))
        self.assertTrue(all(value >= 0 for value in plan.residual_edge_widths.values()))


if __name__ == "__main__":
    unittest.main()
