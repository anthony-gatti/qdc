"""ACP paper-style single-pair workload."""

from __future__ import annotations

from dataclasses import dataclass

from topology import generate_linear_topology

SECOND = 10**12


@dataclass(frozen=True)
class SinglePairPaperWorkload:
    """Two-node no-purification setup from the ACP paper validation case."""

    num_requests: int = 100
    link_distance_m: float = 10_000.0
    memories_per_node: int = 10
    request_rate_hz: float = 10.0
    request_lead_s: float = 0.02
    request_window_s: float = 0.08
    fidelity_threshold: float = 0.5
    seed: int = 0
    stop_margin_s: float = 1.0

    def requests(self) -> list[tuple]:
        interval = 1.0 / self.request_rate_hz
        rows = []
        for identity in range(self.num_requests):
            slot = identity * interval
            rows.append((
                identity,
                "router_0",
                "router_1",
                round((slot + self.request_lead_s) * SECOND),
                round((slot + self.request_lead_s + self.request_window_s) * SECOND),
                1,
                self.fidelity_threshold,
                1,
            ))
        return rows

    def topology(self, adaptive_memory: int) -> dict:
        stop_time_s = self.num_requests / self.request_rate_hz + self.stop_margin_s
        config = generate_linear_topology(
            num_nodes=2,
            inter_node_distance_m=self.link_distance_m,
            memo_size=self.memories_per_node,
            adaptive_max_memory=adaptive_memory,
            memory_fidelity=0.95,
            memory_efficiency=0.6,
            coherence_time_s=2.0,
            gate_fidelity=0.99,
            measurement_fidelity=0.99,
            stop_time_s=stop_time_s,
            seed=self.seed,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
        )
        for node in config["nodes"]:
            if node["type"] == "QuantumRouter":
                node["seed"] = int(node.get("seed", 0)) + self.seed
        template = config["templates"]["default_template"]
        template["SingleHeraldedBSM"] = {
            "detectors": [{"efficiency": 0.95}, {"efficiency": 0.95}],
        }
        return config
