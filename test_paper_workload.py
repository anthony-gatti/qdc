import tempfile
import unittest
from pathlib import Path

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
