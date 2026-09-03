"""Q-CAST with only Q-GUARD's final end-to-end purification stage."""

from __future__ import annotations

from dataclasses import dataclass, field

from algorithms.base import AlgorithmConfig
from algorithms.qcast import QCAST_CONTROL_PAPER_DISTRIBUTED
from algorithms.qguard import QGUARD


@dataclass(frozen=True)
class QCASTE2E(QGUARD):
    """Paper-distributed Q-CAST followed by end-to-end BBPSSW.

    The runtime scheduler deliberately bypasses Q-GUARD's per-hop planning and
    uses unmodified Q-CAST path selection, reservation, generation, recovery,
    and swapping. Inheriting Q-GUARD exposes the identical final-purification
    configuration without duplicating it.
    """

    control_mode: str = QCAST_CONTROL_PAPER_DISTRIBUTED
    algorithm_name: str | None = None
    config: AlgorithmConfig = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "config", AlgorithmConfig(
            name=self.algorithm_name or "qcast_e2e",
            kind="qcast_e2e",
        ))


__all__ = ["QCASTE2E"]
