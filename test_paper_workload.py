import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from algorithms.acp import (
    ACP_EXECUTION_ASYNCHRONOUS,
    ACP_EXECUTION_PAPER_LEGACY,
    AdaptiveContinuous,
)
from backends.sequence.acp_protocol import AdaptiveContinuousProtocol
from backends.sequence.runtime import SequenceRuntime
from paper_workload import SCENARIOS, generate_requests, prepare_topology, validate_paths


class PaperWorkloadTest(unittest.TestCase):
    def test_request_counts_timing_and_threshold(self):
        for name, spec in SCENARIOS.items():
            requests = generate_requests(name, 7)
            self.assertEqual(len(requests), spec["requests"])
            self.assertTrue(all(request[6] == 0.5 for request in requests))
            self.assertTrue(all(request[4] - request[3] == 80_000_000_000 for request in requests))

    def test_deterministic_matched_workload(self):
        self.assertEqual(generate_requests("bottleneck20", 3), generate_requests("bottleneck20", 3))
        self.assertNotEqual(generate_requests("bottleneck20", 3), generate_requests("bottleneck20", 4))

    def test_bottleneck_uses_archived_request_count_and_switch(self):
        requests = generate_requests("bottleneck20", 0)
        self.assertEqual(len(requests), 110)
        first_sources = {request[1] for request in requests[:55]}
        second_sources = {request[1] for request in requests[55:]}
        self.assertLessEqual(first_sources, {"router_0", "router_1"})
        self.assertLessEqual(second_sources, {"router_7", "router_8"})
        first_matrix = SCENARIOS["bottleneck20"]["matrices"][0]
        second_matrix = SCENARIOS["bottleneck20"]["matrices"][1]
        first_choices = [(request[1], request[2]) for request in requests[:55]]
        second_choices = [(request[1], request[2]) for request in requests[55:]]
        first_indices = [
            next(i for i, pair in enumerate(first_matrix) if pair[:2] == choice)
            for choice in first_choices
        ]
        second_indices = [
            next(i for i, pair in enumerate(second_matrix) if pair[:2] == choice)
            for choice in second_choices
        ]
        self.assertEqual(first_indices, second_indices)

    def test_acp_execution_profiles_are_explicit(self):
        self.assertEqual(AdaptiveContinuous().execution_profile, ACP_EXECUTION_ASYNCHRONOUS)
        self.assertEqual(
            AdaptiveContinuous(execution_profile=ACP_EXECUTION_PAPER_LEGACY).execution_profile,
            ACP_EXECUTION_PAPER_LEGACY,
        )
        with self.assertRaises(ValueError):
            AdaptiveContinuous(execution_profile="unknown")

    def test_paper_legacy_aligns_background_reservation_expiry(self):
        protocol = SimpleNamespace(period_ps=100, execution_profile=ACP_EXECUTION_PAPER_LEGACY)
        self.assertEqual(
            AdaptiveContinuousProtocol._background_reservation_end_time(protocol, 30),
            100,
        )
        protocol.execution_profile = ACP_EXECUTION_ASYNCHRONOUS
        self.assertEqual(
            AdaptiveContinuousProtocol._background_reservation_end_time(protocol, 30),
            130,
        )

    def test_reused_memory_releases_each_distinct_reservation_slot(self):
        protocol = SimpleNamespace(
            adaptive_memory_used=2,
            adaptive_memory_names={"memory_0"},
            counters=Counter(),
            generated_entanglement_pairs=set(),
        )
        memory = SimpleNamespace(name="memory_0")

        AdaptiveContinuousProtocol.adaptive_memory_used_minus_one(protocol, memory)
        AdaptiveContinuousProtocol.adaptive_memory_used_minus_one(protocol, memory)

        self.assertEqual(protocol.adaptive_memory_used, 0)
        self.assertEqual(protocol.counters["adaptive_memory_release_underflow"], 0)

    def test_physical_generation_normalization_keeps_repeated_slot_yield(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SequenceRuntime(Path(directory))
            pair = (("router_0", "memory_0"), ("router_1", "memory_1"))
            events = []
            for timestamp in (10, 20):
                events.extend([
                    {"event": "background_pair_available", "time_ps": timestamp, "pair": pair},
                    {"event": "background_pair_available", "time_ps": timestamp, "pair": pair[::-1]},
                ])
            normalized = runtime._normalize_acp_events(events)
            self.assertEqual(normalized["physical_background_pairs_generated"], 2)

    def test_paper_paths_and_neighbor_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in SCENARIOS:
                config = prepare_topology(name, 0, 5, Path(directory) / f"{name}.json")
                validate_paths(name, config, generate_requests(name, 0))
                if name == "line2":
                    channel = next(c for c in config["cchannels"]
                                   if c["source"] == "router_0" and c["destination"] == "router_1")
                    self.assertEqual(channel["delay"], 150_000_000)


if __name__ == "__main__":
    unittest.main()
