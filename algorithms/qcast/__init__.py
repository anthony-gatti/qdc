"""Q-CAST routing algorithm plugin."""

from __future__ import annotations

from dataclasses import dataclass

from algorithms.base import AlgorithmConfig, RoutingAlgorithm
from algorithms.qcast.planner import (
    QCASTDemand,
    QCASTEdge,
    QCASTPath,
    QCASTPlan,
    QCASTPlanner,
    edge_key,
    expected_throughput,
)


@dataclass(frozen=True)
class QCAST(RoutingAlgorithm):
    """Configuration for paper-style online Q-CAST without purification."""

    edge_width: int = 3
    generation_window_ps: int = 5_000_000_000
    control_processing_delay_ps: int = 100_000_000
    swap_success_probability: float = 0.9
    link_state_hops: int = 3
    recovery_paths_per_segment: int = 1
    max_recovery_paths_per_major: int = 12
    max_major_paths: int = 200
    max_hops: int = 8
    config: AlgorithmConfig = AlgorithmConfig(name="qcast", kind="qcast")

    def __post_init__(self) -> None:
        if self.edge_width <= 0 or self.generation_window_ps <= 0:
            raise ValueError("Q-CAST width and generation window must be positive")
        if self.control_processing_delay_ps < 0:
            raise ValueError("Q-CAST processing delay cannot be negative")
        if not 0 <= self.swap_success_probability <= 1:
            raise ValueError("Q-CAST swap success probability must be in [0, 1]")
        if self.link_state_hops < 0 or self.recovery_paths_per_segment < 0:
            raise ValueError("Q-CAST recovery settings cannot be negative")
        if self.max_recovery_paths_per_major < 0:
            raise ValueError("Q-CAST recovery path limit cannot be negative")
        if self.max_major_paths <= 0 or self.max_hops <= 0:
            raise ValueError("Q-CAST path limits must be positive")

    def create_planner(self) -> QCASTPlanner:
        return QCASTPlanner(
            swap_success_probability=self.swap_success_probability,
            max_major_paths=self.max_major_paths,
            max_hops=self.max_hops,
            link_state_hops=self.link_state_hops,
            recovery_paths_per_segment=self.recovery_paths_per_segment,
            max_recovery_paths_per_major=self.max_recovery_paths_per_major,
        )

    def run(self, runtime, workload):
        return runtime.run(workload, self)


__all__ = [
    "QCAST",
    "QCASTDemand",
    "QCASTEdge",
    "QCASTPath",
    "QCASTPlan",
    "QCASTPlanner",
    "edge_key",
    "expected_throughput",
]
