import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from algorithms.acp import AdaptiveContinuous
from algorithms.odo import ShortestPathOnDemand
from algorithms.registry import create_algorithm
from backends.sequence.runtime import SequenceRuntime
from results import BackendResult, RequestResult
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
