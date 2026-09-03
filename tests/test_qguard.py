from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sequence.resource_management.memory_manager import MemoryInfo

from algorithms.qcast import QCAST, QCAST_CONTROL_PAPER_DISTRIBUTED
from algorithms.qcast_e2e import QCASTE2E
from algorithms.qguard import (
    QGUARD,
    QGUARDHopPlan,
    bbpssw_werner_once,
    detour_split_target,
    equal_split_target,
    expected_goodput,
    minimum_purification_rounds,
)
from algorithms.registry import create_algorithm
from backends.sequence.qguard_scheduler import QGUARDDemandScheduler
from backends.sequence.runtime import SequenceRuntime
from common import SECOND
from workloads.concurrent_pairs import ConcurrentPairSpec, ConcurrentPairWorkload
from workloads.qpq import QPQQuerySpec, QPQWorkload


class QGUARDMathTest(unittest.TestCase):
    def test_registry_constructs_base_equal_split_variant(self):
        algorithm = create_algorithm("qguard", {
            "edge_width": 4,
            "max_purification_rounds": 6,
        })
        self.assertIsInstance(algorithm, QGUARD)
        self.assertEqual(algorithm.config.kind, "qguard")
        self.assertEqual(algorithm.config.name, "qguard")
        self.assertEqual(algorithm.edge_width, 4)
        self.assertEqual(algorithm.max_purification_rounds, 6)
        self.assertEqual(algorithm.control_mode, "paper_distributed")

    def test_equal_and_detour_targets_match_paper_example(self):
        self.assertAlmostEqual(equal_split_target(0.8, 5), 0.9548903233)
        self.assertAlmostEqual(
            detour_split_target(0.8, 5, 2, 3),
            0.9696170648,
        )

    def test_bbpssw_cost_and_exg_follow_paper_equations(self):
        output, probability = bbpssw_werner_once(0.8, 0.8)
        self.assertAlmostEqual(output, 0.8381502890)
        self.assertAlmostEqual(probability, 0.7688888889)
        self.assertEqual(minimum_purification_rounds(0.8, 0.85, 5), 2)
        result = expected_goodput(
            4,
            [
                QGUARDHopPlan(0.9, 0.8, 4, 1),
                QGUARDHopPlan(0.9, 0.8, 4, 1),
            ],
            0.9,
        )
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.expected_goodput, 1.2)
        self.assertEqual(result.required_raw_pairs, (2, 2))

        underfilled = expected_goodput(
            4,
            [
                QGUARDHopPlan(0.9, 0.8, 1, 1),
                QGUARDHopPlan(0.9, 0.8, 1, 1),
            ],
            0.9,
        )
        self.assertTrue(underfilled.feasible)
        self.assertAlmostEqual(underfilled.availability, 0.5)
        self.assertAlmostEqual(underfilled.expected_goodput, 0.6)


class QGUARDSequenceTest(unittest.TestCase):
    @staticmethod
    def _run(workload, algorithm):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = algorithm.run(runtime, workload)
            return result, runtime.last_diagnostics["workload_diagnostics"]

    def test_direct_request_uses_official_purification_and_classical_delay(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=7,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=8,
            memory_fidelity=0.8,
            memory_efficiency=1.0,
            gate_fidelity=1.0,
            measurement_fidelity=1.0,
            swapping_success_probability=1.0,
            link_parallelism=4,
            simulation_end_time_s=0.08,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_1",
                start,
                int(0.07 * SECOND),
                pair_count=1,
                fidelity_threshold=0.85,
            ),),
        )
        result, diagnostics = self._run(workload, QGUARD(
            edge_width=4,
            generation_window_ps=int(0.005 * SECOND),
            control_processing_delay_ps=int(0.0001 * SECOND),
            swap_success_probability=1.0,
            link_state_hops=1,
            max_major_paths=1,
            max_hops=2,
        ))

        request = result.request_results[0]
        counters = diagnostics["counters"]
        events = diagnostics["qguard"]["purification_events"]
        self.assertTrue(request.success)
        self.assertGreaterEqual(request.fidelity, 0.85)
        self.assertEqual(counters["qguard_purification_attempts"], 2)
        self.assertEqual(counters["qguard_purification_successes"], 2)
        starts = [event for event in events if event["event"] == "purification_started"]
        self.assertEqual(len({event["attempt_id"] for event in starts}), 2)
        self.assertTrue(all(
            event["endpoint_processing_delay_ps"] == int(0.0001 * SECOND)
            and event["classical_propagation_delay_ps"] == int(0.000005 * SECOND)
            and event["classical_wait_ps"] == int(0.000105 * SECOND) + 1
            for event in starts
        ))
        self.assertTrue(diagnostics["qguard"]["uses_official_bell_diagonal_bbpssw"])
        self.assertTrue(
            diagnostics["locality_invariants"]["qguard_decisions_path_scoped"]
        )
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_multihop_request_can_use_final_end_to_end_purification(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=8,
            num_nodes=4,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=16,
            memory_fidelity=0.9,
            memory_efficiency=1.0,
            coherence_time_s=5.0,
            gate_fidelity=1.0,
            measurement_fidelity=1.0,
            swapping_success_probability=1.0,
            link_parallelism=4,
            simulation_end_time_s=0.12,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_3",
                start,
                int(0.1 * SECOND),
                pair_count=1,
                fidelity_threshold=0.8,
            ),),
        )
        result, diagnostics = self._run(workload, QGUARD(
            edge_width=4,
            generation_window_ps=int(0.005 * SECOND),
            control_processing_delay_ps=0,
            swap_success_probability=1.0,
            link_state_hops=3,
            max_major_paths=1,
            max_hops=4,
        ))
        counters = diagnostics["counters"]
        self.assertTrue(result.request_results[0].success)
        self.assertGreaterEqual(result.request_results[0].fidelity, 0.8)
        self.assertGreater(counters["qguard_end_to_end_purification_attempts"], 0)
        self.assertGreater(counters["qguard_purification_attempts"], 0)
        self.assertEqual(
            counters.get("qguard_purification_endpoint_state_mismatches", 0),
            0,
        )
        self.assertTrue(all(
            record["p4_start_ps"] is not None
            for record in diagnostics["timing"]["deliveries"]
        ))
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_qcast_e2e_preserves_qcast_decision_then_purifies(self):
        start = int(0.005 * SECOND)
        workload = ConcurrentPairWorkload(
            num_requests=1,
            seed=18,
            num_nodes=4,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=24,
            memory_fidelity=0.9,
            memory_efficiency=1.0,
            coherence_time_s=5.0,
            gate_fidelity=1.0,
            measurement_fidelity=1.0,
            swapping_success_probability=1.0,
            link_parallelism=4,
            simulation_end_time_s=0.12,
            request_override=(ConcurrentPairSpec(
                0,
                "router_0",
                "router_3",
                start,
                int(0.1 * SECOND),
                pair_count=1,
                fidelity_threshold=0.8,
            ),),
        )
        common = dict(
            edge_width=4,
            generation_window_ps=int(0.005 * SECOND),
            control_processing_delay_ps=0,
            swap_success_probability=1.0,
            link_state_hops=3,
            max_major_paths=1,
            max_hops=4,
        )
        _qcast_result, qcast_diagnostics = self._run(workload, QCAST(
            **common,
            control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
        ))
        result, diagnostics = self._run(workload, QCASTE2E(**common))

        request = result.request_results[0]
        counters = diagnostics["counters"]
        self.assertTrue(request.success)
        self.assertGreaterEqual(request.fidelity, 0.8)
        self.assertGreater(counters["qguard_end_to_end_purification_attempts"], 0)
        self.assertGreater(counters["qguard_purification_attempts"], 0)
        self.assertNotIn("qguard", diagnostics)
        self.assertFalse(diagnostics["qcast_e2e"]["per_hop_fidelity_planning"])
        self.assertTrue(all(
            event["purpose"] == "end_to_end"
            for event in diagnostics["qcast_e2e"]["purification_events"]
        ))
        qcast_first = qcast_diagnostics["distributed_decisions"][0]
        qcast_e2e_first = diagnostics["distributed_decisions"][0]
        for field in (
            "major_path_id",
            "decision_lane_ids",
            "visible_lane_ids",
            "visible_lane_states",
            "selections",
        ):
            self.assertEqual(qcast_first[field], qcast_e2e_first[field])
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_recovery_route_repairs_forced_major_link_failure(self):
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
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            gate_fidelity=1.0,
            measurement_fidelity=1.0,
            swapping_success_probability=1.0,
            link_parallelism=2,
            simulation_end_time_s=0.06,
            request_override=(ConcurrentPairSpec(
                0,
                "router_2",
                "router_0",
                start,
                int(0.05 * SECOND),
                fidelity_threshold=0.7,
            ),),
        )
        algorithm = QGUARD(
            edge_width=2,
            generation_window_ps=int(0.004 * SECOND),
            control_processing_delay_ps=0,
            swap_success_probability=1.0,
            max_major_paths=1,
            link_state_hops=3,
        )
        original_stop_generation = QGUARDDemandScheduler.stop_generation
        forced_failures = []

        def stop_with_major_failure(scheduler, slot_id):
            if not forced_failures:
                major = scheduler.slot.plan.major_paths[0]
                allocation = scheduler.slot.allocations[major.path_id]
                assignments = [lane.edges[0] for lane in allocation.lanes]
                for assignment in assignments:
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
            QGUARDDemandScheduler,
            "stop_generation",
            stop_with_major_failure,
        ):
            result, diagnostics = self._run(workload, algorithm)

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        self.assertEqual(len(forced_failures), 2)
        selected = [
            decision
            for decision in diagnostics["qguard"]["decisions"]
            if decision["event"] == "output_route_selected"
        ]
        self.assertTrue(any(decision["used_recovery"] for decision in selected))
        self.assertGreater(
            diagnostics["counters"]["end_to_end_pairs_delivered_recovery"],
            0,
        )
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_qpq_uses_common_result_schema(self):
        start = int(0.005 * SECOND)
        workload = QPQWorkload(
            database_size_log=1,
            num_clients=1,
            queries_per_client=1,
            fidelity_threshold=0.7,
            round_deadline_s=0.08,
            transaction_duration_s=0.16,
            seed=10,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=1_000,
            memories_per_node=8,
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            gate_fidelity=1.0,
            measurement_fidelity=1.0,
            swapping_success_probability=1.0,
            link_parallelism=4,
            simulation_end_time_s=0.18,
            query_override=(QPQQuerySpec(
                query_id=0,
                source="router_1",
                destination="router_0",
                start_time_ps=start,
                transaction_deadline_ps=int(0.16 * SECOND),
                database_size_log=1,
                fidelity_threshold=0.7,
                round_deadline_ps=int(0.08 * SECOND),
            ),),
        )
        result, diagnostics = self._run(workload, QGUARD(
            edge_width=4,
            generation_window_ps=int(0.003 * SECOND),
            control_processing_delay_ps=0,
            swap_success_probability=1.0,
            link_state_hops=1,
            max_major_paths=1,
            max_hops=2,
        ))
        request = result.request_results[0]
        self.assertTrue(request.success)
        self.assertEqual(request.expected_pairs, 6)
        self.assertEqual(len(request.pair_arrival_ms), 6)
        self.assertIsNotNone(request.round1_completion_ms)
        self.assertIsNotNone(request.round2_completion_ms)
        self.assertEqual(diagnostics["transactions"]["0"]["rounds_started"], [1, 2])
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))


if __name__ == "__main__":
    unittest.main()
