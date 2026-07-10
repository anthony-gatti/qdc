"""ACP paper-style single-pair workload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from topology import (
    ACP_PAPER_END_NODE_PROCESSING_DELAY_PS,
    CLASSICAL_TIMING_ACP_PAPER,
    CLASSICAL_TIMING_SEQUENCE,
    generate_linear_topology,
)
from workloads.base import Workload

SECOND = 10**12


@dataclass(frozen=True)
class SinglePairPaperWorkload(Workload):
    """Two-node no-purification setup from the ACP paper validation case."""

    name: ClassVar[str] = "single_pair"
    sequence_adapter: ClassVar[str] = "single_pair"

    num_requests: int = 100
    link_distance_m: float = 10_000.0
    memories_per_node: int = 10
    request_rate_hz: float = 10.0
    request_lead_s: float = 0.02
    request_window_s: float = 0.08
    fidelity_threshold: float = 0.5
    seed: int = 0
    stop_margin_s: float = 1.0
    classical_timing_profile: str = CLASSICAL_TIMING_ACP_PAPER
    end_node_processing_delay_ps: int = ACP_PAPER_END_NODE_PROCESSING_DELAY_PS

    def __post_init__(self):
        if self.classical_timing_profile not in {CLASSICAL_TIMING_SEQUENCE, CLASSICAL_TIMING_ACP_PAPER}:
            raise ValueError(f"Unsupported classical timing profile: {self.classical_timing_profile}")

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
            classical_timing_profile=self.classical_timing_profile,
            end_node_processing_delay_ps=self.end_node_processing_delay_ps,
            stop_time_s=stop_time_s,
            seed=self.seed,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
        )
        for node in config["nodes"]:
            node["seed"] = int(node.get("seed", 0)) + self.seed
        template = config["templates"]["default_template"]
        template["SingleHeraldedBSM"] = {
            "detectors": [{"efficiency": 0.95}, {"efficiency": 0.95}],
        }
        return config
