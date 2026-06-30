"""Adaptive Continuous entanglement generation algorithm plugin."""

from __future__ import annotations

from dataclasses import dataclass

from algorithms.base import AlgorithmConfig, RoutingAlgorithm


@dataclass(frozen=True)
class AdaptiveContinuous(RoutingAlgorithm):
    """Paper-faithful no-purification ACP milestone configuration."""

    adaptive_max_memory: int = 5
    cache_strategy: str = "freshest"
    update_prob: bool = True
    period_ps: int = 100_000_000_000
    delta: float = 0.05
    background_enabled: bool = True
    algorithm_name: str | None = None
    config: AlgorithmConfig = None

    def __post_init__(self):
        if self.cache_strategy not in {"freshest", "random"}:
            raise ValueError(f"Unsupported ACP cache strategy: {self.cache_strategy}")
        name = self.algorithm_name or f"acp_{self.cache_strategy}"
        kind = "ucp" if name == "ucp" else "acp"
        object.__setattr__(self, "config", AlgorithmConfig(name=name, kind=kind))

    def run(self, runtime, workload):
        return runtime.run_single_pair(workload, self)
