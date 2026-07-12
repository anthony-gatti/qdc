import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from algorithms.acp import AdaptiveContinuous
from algorithms.odo import ShortestPathOnDemand
from algorithms.qcast import QCAST, QCAST_CONTROL_PAPER_DISTRIBUTED
from algorithms.registry import create_algorithm
from backends.sequence.runtime import SequenceRuntime
from common import SECOND
from results import BackendResult, RequestResult
from topology import generate_linear_topology
from workloads.base import PairDelivery
from workloads.qpq import QPQQuerySpec, QPQTransaction, QPQWorkload


class QPQTransactionTest(unittest.TestCase):
    def test_round_two_is_submitted_only_after_round_one_completes(self):
        submitted = []
        spec = QPQQuerySpec(
            query_id=7,
            source="router_0",
            destination="router_1",
            start_time_ps=100,
            transaction_deadline_ps=1_000,
            database_size_log=1,
            fidelity_threshold=0.7,
            round_deadline_ps=400,
        )
        transaction = QPQTransaction(spec, lambda demand, callbacks: submitted.append((demand, callbacks)))

        transaction.start()
        self.assertEqual(len(submitted), 1)
        round1 = submitted[0][0]
        transaction.on_demand_accepted(round1, ("router_0", "router_1"))
        for timestamp in (120, 130, 140):
            transaction.on_pair_delivered(round1, PairDelivery(timestamp, 0.9))
        transaction.on_demand_completed(round1, 140, ("router_0", "router_1"))

        self.assertEqual(len(submitted), 2)
        round2 = submitted[1][0]
        self.assertEqual(round2.start_time_ps, 140)
        for timestamp in (160, 170, 180):
            transaction.on_pair_delivered(round2, PairDelivery(timestamp, 0.9))
        transaction.on_demand_completed(round2, 180, ("router_0", "router_1"))

        result = transaction.to_request_result()
        self.assertTrue(result.success)
        self.assertEqual((result.round1_pairs, result.round2_pairs), (3, 3))
        self.assertEqual(result.expected_pairs, 6)
        self.assertEqual(result.time_to_serve_ms, 80 / 10**9)

    def test_workload_generation_is_deterministic_and_matched(self):
        workload = QPQWorkload(num_nodes=5, qdc_node_index=2, num_clients=2, queries_per_client=2, seed=9)
        first = workload.queries()
        second = workload.queries()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(query.destination == "router_2" for query in first))
        self.assertTrue(all(query.source != query.destination for query in first))

    def test_incomplete_round_cannot_advance_the_transaction(self):
        submitted = []
        spec = QPQQuerySpec(
            query_id=8,
            source="router_0",
            destination="router_1",
            start_time_ps=100,
            transaction_deadline_ps=1_000,
            database_size_log=1,
            fidelity_threshold=0.7,
            round_deadline_ps=400,
        )
        transaction = QPQTransaction(spec, lambda demand, callbacks: submitted.append((demand, callbacks)))
        transaction.start()
        round1 = submitted[0][0]
        transaction.on_pair_delivered(round1, PairDelivery(120, 0.9))
        transaction.on_demand_completed(round1, 130, ("router_0", "router_1"))

        self.assertEqual(len(submitted), 1)
        result = transaction.to_request_result()
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "round1_incomplete_delivery")
        self.assertEqual(result.round1_pairs, 1)

    def test_registry_constructs_supported_algorithms(self):
        self.assertIsInstance(create_algorithm("odo"), ShortestPathOnDemand)
        acp = create_algorithm("acp_freshest", {"adaptive_max_memory": 3})
        self.assertIsInstance(acp, AdaptiveContinuous)
        self.assertEqual(acp.adaptive_max_memory, 3)


class QPQSequenceIntegrationTest(unittest.TestCase):
    def _workload(self) -> QPQWorkload:
        return QPQWorkload(
            database_size_log=1,
            num_clients=1,
            queries_per_client=1,
            round_deadline_s=1,
            transaction_duration_s=3,
            start_offset_s=1,
            num_nodes=2,
            qdc_node_index=1,
            extra_mesh_edges=0,
            memories_per_node=10,
            memory_efficiency=1.0,
            simulation_end_time_s=5,
            seed=0,
        )

    def _multihop_workload(self) -> QPQWorkload:
        return QPQWorkload(
            database_size_log=5,
            num_clients=2,
            queries_per_client=1,
            round_deadline_s=5,
            transaction_duration_s=12,
            start_offset_s=2,
            num_nodes=5,
            qdc_node_index=2,
            extra_mesh_edges=1,
            memories_per_node=50,
            memory_efficiency=0.5,
            gate_fidelity=0.99,
            measurement_fidelity=0.99,
            simulation_end_time_s=30,
            seed=42,
        )

    def _purification_workload(self, gate_fidelity: float) -> QPQWorkload:
        return QPQWorkload(
            database_size_log=1,
            num_clients=1,
            queries_per_client=1,
            round_deadline_s=0.05,
            transaction_duration_s=0.08,
            start_offset_s=0.01,
            num_nodes=2,
            qdc_node_index=1,
            extra_mesh_edges=0,
            inter_node_distance_m=10_000,
            memories_per_node=10,
            memory_efficiency=1.0,
            gate_fidelity=gate_fidelity,
            measurement_fidelity=gate_fidelity,
            simulation_end_time_s=0.02,
            seed=0,
        )

    def _multihop_cache_workload(self) -> QPQWorkload:
        topology = generate_linear_topology(
            3,
            inter_node_distance_m=10_000,
            memo_size=10,
            adaptive_max_memory=5,
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            coherence_time_s=5,
            gate_fidelity=0.99,
            measurement_fidelity=0.99,
            stop_time_s=0.03,
            qdc_node_index=2,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
            seed=0,
        )
        query = QPQQuerySpec(
            query_id=0,
            source="router_0",
            destination="router_2",
            start_time_ps=int(0.01 * SECOND),
            transaction_deadline_ps=int(0.09 * SECOND),
            database_size_log=1,
            fidelity_threshold=0.7,
            round_deadline_ps=int(0.05 * SECOND),
        )
        return QPQWorkload(
            database_size_log=1,
            num_clients=1,
            queries_per_client=1,
            round_deadline_s=0.05,
            transaction_duration_s=0.08,
            start_offset_s=0.01,
            num_nodes=3,
            qdc_node_index=2,
            extra_mesh_edges=0,
            inter_node_distance_m=10_000,
            memories_per_node=10,
            memory_efficiency=1.0,
            simulation_end_time_s=0.03,
            seed=0,
            topology_override=topology,
            query_override=(query,),
        )

    def _qcast_multihop_workload(self, fidelity_threshold: float = 0.7) -> QPQWorkload:
        topology = generate_linear_topology(
            3,
            inter_node_distance_m=1_000,
            memo_size=8,
            adaptive_max_memory=0,
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            coherence_time_s=5,
            gate_fidelity=0.99,
            measurement_fidelity=0.99,
            swapping_success_probability=1.0,
            stop_time_s=0.1,
            qdc_node_index=2,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
            seed=17,
        )
        query = QPQQuerySpec(
            query_id=0,
            source="router_0",
            destination="router_2",
            start_time_ps=int(0.005 * SECOND),
            transaction_deadline_ps=int(0.08 * SECOND),
            database_size_log=1,
            fidelity_threshold=fidelity_threshold,
            round_deadline_ps=int(0.03 * SECOND),
        )
        return QPQWorkload(
            database_size_log=1,
            num_clients=1,
            queries_per_client=1,
            round_deadline_s=0.03,
            transaction_duration_s=0.075,
            start_offset_s=0.005,
            num_nodes=3,
            qdc_node_index=2,
            extra_mesh_edges=0,
            inter_node_distance_m=1_000,
            memories_per_node=8,
            memory_efficiency=1.0,
            swapping_success_probability=1.0,
            simulation_end_time_s=0.1,
            seed=17,
            topology_override=topology,
            query_override=(query,),
        )

    def test_odo_completes_both_rounds_with_exact_pair_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = ShortestPathOnDemand().run(runtime, self._workload())

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        row = result.request_results[0]
        self.assertEqual((row.round1_pairs, row.round2_pairs, row.expected_pairs), (3, 3, 6))
        self.assertEqual(len(row.pair_arrival_ms), 6)
        self.assertLess(row.round1_completion_ms, row.round2_completion_ms)
        self.assertEqual(row.delivered_pairs_fully_fresh, 6)
        workload_diagnostics = runtime.last_diagnostics["workload_diagnostics"]
        self.assertEqual(workload_diagnostics["counters"]["demands_completed"], 2)
        events = workload_diagnostics["events"]
        submissions = [event for event in events if event["event"] == "demand_submitted"]
        completions = [event for event in events if event["event"] == "demand_completed"]
        self.assertEqual(submissions[0]["time_ps"], submissions[0]["application_start_ps"])
        self.assertGreater(submissions[0]["reservation_start_ps"], submissions[0]["application_start_ps"])
        self.assertEqual(submissions[1]["time_ps"], completions[0]["time_ps"])
        self.assertGreater(submissions[1]["reservation_start_ps"], submissions[1]["application_start_ps"])

    def test_qcast_completes_both_rounds_with_exact_pair_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = QCAST(
                edge_width=3,
                generation_window_ps=int(0.003 * SECOND),
                max_major_paths=1,
                link_state_hops=1,
            ).run(runtime, self._workload())

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        row = result.request_results[0]
        self.assertEqual((row.round1_pairs, row.round2_pairs, row.expected_pairs), (3, 3, 6))
        self.assertEqual(len(row.pair_arrival_ms), 6)
        self.assertLess(row.round1_completion_ms, row.round2_completion_ms)
        self.assertEqual(row.delivered_pairs_fully_fresh, 6)
        self.assertEqual(row.delivered_fresh_elementary_edges, 6)

        diagnostics = runtime.last_diagnostics["workload_diagnostics"]
        self.assertEqual(diagnostics["controller_node"], "router_1")
        self.assertEqual(diagnostics["counters"]["demands_completed"], 2)
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))
        submissions = [
            event for event in diagnostics["events"]
            if event["event"] == "demand_submitted"
        ]
        completions = [
            event for event in diagnostics["events"]
            if event["event"] == "demand_completed"
        ]
        self.assertEqual(len(submissions), 2)
        self.assertEqual(len(completions), 2)
        self.assertEqual(submissions[1]["time_ps"], completions[0]["time_ps"])

    def test_distributed_qcast_completes_both_qpq_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = QCAST(
                edge_width=3,
                generation_window_ps=int(0.003 * SECOND),
                max_major_paths=1,
                link_state_hops=1,
                control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
            ).run(runtime, self._workload())

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        row = result.request_results[0]
        self.assertEqual((row.round1_pairs, row.round2_pairs), (3, 3))
        self.assertEqual(len(row.pair_arrival_ms), 6)
        diagnostics = runtime.last_diagnostics["workload_diagnostics"]
        self.assertEqual(diagnostics["control_mode"], "paper_distributed")
        self.assertEqual(diagnostics["counters"]["demands_completed"], 2)
        self.assertGreaterEqual(len(diagnostics["distributed_decisions"]), 2)
        self.assertEqual(
            len(diagnostics["distributed_decisions"]),
            diagnostics["counters"]["distributed_major_path_decisions"],
        )
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_shared_link_parallelism_is_visible_to_all_algorithms(self):
        workload = replace(self._workload(), link_parallelism=3)
        algorithms = (
            ShortestPathOnDemand(),
            AdaptiveContinuous(adaptive_max_memory=0),
            QCAST(
                edge_width=3,
                generation_window_ps=int(0.003 * SECOND),
                max_major_paths=1,
                link_state_hops=1,
            ),
        )
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm.name), tempfile.TemporaryDirectory() as directory:
                runtime = SequenceRuntime(Path(directory))
                result = algorithm.run(runtime, workload)
            self.assertEqual((result.num_requests, result.num_success), (1, 1))
            diagnostics = runtime.last_diagnostics["parallel_links"]
            link = diagnostics["links"]["router_0|router_1"]
            self.assertEqual(link["parallelism"], 3)
            self.assertEqual(len(link["channels"]), 3)
            self.assertGreater(
                sum(item["generation_attempts"] for item in link["channels"].values()),
                0,
            )
            self.assertGreater(
                sum(item["generation_successes"] for item in link["channels"].values()),
                0,
            )
            self.assertTrue(all(
                states["final"].get("RAW", 0) == workload.memories_per_node
                for states in diagnostics["memory_occupancy"].values()
            ))

    def test_qcast_multihop_qpq_swaps_every_delivered_pair(self):
        workload = self._qcast_multihop_workload()
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = QCAST(
                edge_width=1,
                generation_window_ps=int(0.002 * SECOND),
                swap_success_probability=1.0,
                max_major_paths=1,
                link_state_hops=2,
            ).run(runtime, workload)

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        row = result.request_results[0]
        self.assertEqual((row.round1_pairs, row.round2_pairs), (3, 3))
        self.assertEqual(row.delivered_fresh_elementary_edges, 12)
        self.assertLess(row.fidelity, workload.memory_fidelity)
        diagnostics = runtime.last_diagnostics["workload_diagnostics"]
        self.assertGreater(diagnostics["counters"]["swaps_attempted"], 0)
        self.assertTrue(all(
            count <= workload.memories_per_node
            for count in diagnostics["max_allocated_memories_by_node"].values()
        ))
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_qcast_qpq_rejects_low_fidelity_swapped_pairs_until_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = QCAST(
                edge_width=1,
                generation_window_ps=int(0.002 * SECOND),
                swap_success_probability=1.0,
                max_major_paths=1,
                link_state_hops=2,
            ).run(runtime, self._qcast_multihop_workload(fidelity_threshold=0.98))

        self.assertEqual((result.num_requests, result.num_success), (1, 0))
        row = result.request_results[0]
        self.assertEqual(row.failure_reason, "round1_qcast_deadline")
        self.assertEqual(row.round2_pairs, 0)
        self.assertGreater(row.pairs_rejected_fidelity, 0)
        diagnostics = runtime.last_diagnostics["workload_diagnostics"]
        self.assertGreater(diagnostics["counters"]["pairs_rejected_fidelity"], 0)
        self.assertTrue(all(diagnostics["all_memories_raw_at_end"].values()))

    def test_acp_reuses_multiple_pairs_without_exceeding_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = AdaptiveContinuous(adaptive_max_memory=5).run(runtime, self._workload())

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        row = result.request_results[0]
        self.assertEqual((row.round1_pairs, row.round2_pairs), (3, 3))
        self.assertGreater(row.delivered_pairs_with_background_contribution, 1)
        normalized = runtime.last_diagnostics["normalized_counters"]
        self.assertGreater(normalized["physical_background_pairs_reused"], 1)
        # Background reservation contention must back off instead of filling
        # the timeline with micro-polling events while all cache slots are in use.
        self.assertLess(runtime.last_diagnostics["counters"]["start_invocations"], 2_000)
        self.assertTrue(all(runtime.last_diagnostics["adaptive_memory_accounting_consistent_at_end"].values()))
        self.assertTrue(all(
            high_water <= 5
            for high_water in runtime.last_diagnostics["adaptive_memory_high_watermark_by_node"].values()
        ))

    def test_acp_zero_memory_matches_odo_on_multihop_fidelity(self):
        workload = self._multihop_workload()
        with tempfile.TemporaryDirectory() as directory:
            odo_runtime = SequenceRuntime(Path(directory) / "odo")
            acp_runtime = SequenceRuntime(Path(directory) / "acp_m0")
            odo = ShortestPathOnDemand().run(odo_runtime, workload)
            acp = AdaptiveContinuous(adaptive_max_memory=0).run(acp_runtime, workload)

        self.assertEqual(odo.num_success, acp.num_success)
        for odo_row, acp_row in zip(odo.request_results, acp.request_results):
            self.assertEqual(odo_row.success, acp_row.success)
            self.assertEqual(odo_row.time_to_serve_ms, acp_row.time_to_serve_ms)
            self.assertEqual(odo_row.pair_arrival_ms, acp_row.pair_arrival_ms)
            self.assertEqual(odo_row.fidelity, acp_row.fidelity)
        self.assertEqual(
            acp_runtime.last_diagnostics["normalized_counters"]["physical_background_pairs_generated"],
            0,
        )
        self.assertEqual(
            acp_runtime.last_diagnostics["normalized_counters"]["physical_background_pairs_reused"],
            0,
        )

    def test_completed_round_releases_timecards_before_next_round(self):
        workload = QPQWorkload(
            database_size_log=2,
            num_clients=3,
            queries_per_client=1,
            round_deadline_s=0.8,
            transaction_duration_s=1.5,
            start_offset_s=0.03,
            num_nodes=5,
            qdc_node_index=2,
            extra_mesh_edges=3,
            inter_node_distance_m=20_000,
            memories_per_node=20,
            memory_efficiency=0.5,
            swapping_success_probability=0.9,
            link_parallelism=3,
            simulation_end_time_s=1.6,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = AdaptiveContinuous(adaptive_max_memory=4).run(runtime, workload)

        self.assertEqual((result.num_requests, result.num_success), (3, 3))
        counters = runtime.last_diagnostics["workload_diagnostics"]["counters"]
        self.assertEqual(counters.get("reservations_rejected", 0), 0)
        occupancy = runtime.last_diagnostics["parallel_links"]["memory_occupancy"]
        self.assertGreater(
            sum(node["timecard_slots_released"] for node in occupancy.values()),
            0,
        )
        self.assertTrue(all(node["final"] == {"RAW": 20} for node in occupancy.values()))

    def test_qpq_purification_skips_predicted_fidelity_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = AdaptiveContinuous(
                adaptive_max_memory=5,
                purify=True,
                algorithm_name="acp_purify",
            ).run(runtime, self._purification_workload(gate_fidelity=0.99))

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        counters = runtime.last_diagnostics["counters"]
        self.assertEqual(counters.get("purification_attempts", 0), 0)
        self.assertGreater(counters["purification_skipped_non_improving"], 0)
        self.assertTrue(all(runtime.last_diagnostics["adaptive_memory_accounting_consistent_at_end"].values()))

    def test_qpq_purification_runs_when_predicted_to_improve_fidelity(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = AdaptiveContinuous(
                adaptive_max_memory=5,
                purify=True,
                algorithm_name="acp_purify",
            ).run(runtime, self._purification_workload(gate_fidelity=1.0))

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        counters = runtime.last_diagnostics["counters"]
        self.assertGreater(counters["purification_attempts"], 0)
        self.assertGreater(counters["purification_successes"], 0)
        self.assertTrue(all(runtime.last_diagnostics["adaptive_memory_accounting_consistent_at_end"].values()))

    def test_multihop_qpq_reuses_cache_and_swaps_mixed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = AdaptiveContinuous(adaptive_max_memory=5).run(
                runtime,
                self._multihop_cache_workload(),
            )

        self.assertEqual((result.num_requests, result.num_success), (1, 1))
        row = result.request_results[0]
        self.assertEqual((row.round1_pairs, row.round2_pairs), (3, 3))
        self.assertGreater(row.delivered_pairs_fully_background_supported, 0)
        self.assertGreater(row.delivered_pairs_partially_background_supported, 0)
        self.assertGreater(row.delivered_background_elementary_edges, 0)
        self.assertGreater(row.delivered_fresh_elementary_edges, 0)
        normalized = runtime.last_diagnostics["normalized_counters"]
        self.assertGreater(normalized["physical_background_pairs_generated"], 0)
        self.assertGreater(normalized["physical_background_pairs_reused"], 0)
        self.assertTrue(all(runtime.last_diagnostics["adaptive_memory_accounting_consistent_at_end"].values()))
        self.assertTrue(all(
            high_water <= 5
            for high_water in runtime.last_diagnostics["adaptive_memory_high_watermark_by_node"].values()
        ))

    def test_unstarted_query_is_finalized_after_simulation_end(self):
        workload = replace(self._workload(), simulation_end_time_s=0.5)
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            result = ShortestPathOnDemand().run(runtime, workload)

        self.assertEqual((result.num_requests, result.num_success), (1, 0))
        row = result.request_results[0]
        self.assertEqual(row.failure_reason, "round1_simulation_end")
        self.assertEqual((row.round1_pairs, row.round2_pairs), (0, 0))

    def test_backend_result_writes_complete_qpq_schema(self):
        row = RequestResult(
            request_id=1,
            src="router_0",
            dst="router_1",
            success=True,
            start_time_ps=10,
            pair_arrival_ms=[1.25, 2.5],
            round1_pairs=1,
            round2_pairs=1,
            expected_pairs=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            BackendResult("odo", 0, 2, [row]).to_csv(str(path))
            with path.open(newline="") as handle:
                written = next(csv.DictReader(handle))
        self.assertEqual(written["pair_arrival_ms"], "1.250;2.500")
        self.assertEqual(written["expected_pairs"], "2")


class WorkloadIsolationTest(unittest.TestCase):
    def test_workloads_do_not_import_algorithm_packages(self):
        root = Path(__file__).resolve().parents[1] / "workloads"
        for source in root.glob("*.py"):
            self.assertNotIn("from algorithms", source.read_text(), source.name)
            self.assertNotIn("import algorithms", source.read_text(), source.name)


if __name__ == "__main__":
    unittest.main()
