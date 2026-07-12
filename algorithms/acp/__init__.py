"""Adaptive Continuous entanglement generation algorithm plugin."""

from __future__ import annotations

from dataclasses import dataclass

from algorithms.base import AlgorithmConfig, RoutingAlgorithm


ACP_EXECUTION_ASYNCHRONOUS = "asynchronous"
# Reproduces the archived experiment's windowed updates and aligned expiry epochs.
ACP_EXECUTION_PAPER_LEGACY = "paper_legacy"
ACP_EXECUTION_PROFILES = (
    ACP_EXECUTION_ASYNCHRONOUS,
    ACP_EXECUTION_PAPER_LEGACY,
)


@dataclass(frozen=True)
class AdaptiveContinuous(RoutingAlgorithm):
    """ACP configuration with explicit native and paper-reproduction profiles."""

    adaptive_max_memory: int = 5
    cache_strategy: str = "freshest"
    update_prob: bool = True
    period_ps: int = 100_000_000_000
    delta: float = 0.05
    background_enabled: bool = True
    purify: bool = False
    execution_profile: str = ACP_EXECUTION_ASYNCHRONOUS
    algorithm_name: str | None = None
    config: AlgorithmConfig = None

    def __post_init__(self):
        if self.cache_strategy not in {"freshest", "random"}:
            raise ValueError(f"Unsupported ACP cache strategy: {self.cache_strategy}")
        if self.execution_profile not in ACP_EXECUTION_PROFILES:
            raise ValueError(f"Unsupported ACP execution profile: {self.execution_profile}")
        name = self.algorithm_name or f"acp_{self.cache_strategy}"
        kind = "ucp" if name.startswith("ucp") else "acp"
        object.__setattr__(self, "config", AlgorithmConfig(name=name, kind=kind))

    def run(self, runtime, workload):
        return runtime.run(workload, self)
