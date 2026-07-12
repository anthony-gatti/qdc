"""Topology-generator coverage for reusable evaluation graphs."""

from __future__ import annotations

import unittest

from topology import generate_ring_topology, validate_topology
from workloads.concurrent_pairs import ConcurrentPairWorkload
from workloads.qpq import QPQWorkload


class RingTopologyTest(unittest.TestCase):
    def test_ring_has_one_quantum_link_per_cycle_edge(self):
        topology = generate_ring_topology(
            6,
            inter_node_distance_m=20_000,
            memo_size=8,
            qdc_node_index=0,
        )

        self.assertEqual(validate_topology(topology), [])
        bsm_names = {
            node["name"]
            for node in topology["nodes"]
            if node["type"] == "BSMNode"
        }
        self.assertEqual(len(bsm_names), 6)
        self.assertIn("BSM_0_1", bsm_names)
        self.assertIn("BSM_0_5", bsm_names)

    def test_workloads_accept_ring_topology(self):
        concurrent = ConcurrentPairWorkload(
            num_nodes=6,
            qdc_node_index=0,
            topology_type="ring",
        )
        self.assertEqual(validate_topology(concurrent.topology(0)), [])

        qpq = QPQWorkload(
            num_nodes=6,
            qdc_node_index=0,
            topology_type="ring",
            num_clients=1,
            queries_per_client=1,
        )
        self.assertEqual(validate_topology(qpq.topology(0)), [])

        configured_qpq = QPQWorkload.from_config({
            "workload": {"num_clients": 1, "queries_per_client": 1},
            "topology": {
                "type": "ring",
                "num_nodes": 6,
                "qdc_node_index": 0,
            },
        })
        self.assertEqual(configured_qpq.topology_type, "ring")


if __name__ == "__main__":
    unittest.main()
