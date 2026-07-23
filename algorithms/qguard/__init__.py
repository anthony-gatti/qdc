"""Q-GUARD routing configuration and fidelity-planning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from algorithms.base import AlgorithmConfig
from algorithms.qcast import QCAST, QCAST_CONTROL_PAPER_DISTRIBUTED


def werner_parameter(fidelity: float) -> float:
    """Convert Werner-state fidelity to its depolarization parameter."""
    return (4.0 * fidelity - 1.0) / 3.0


def fidelity_from_werner(parameter: float) -> float:
    """Convert a Werner depolarization parameter to fidelity."""
    return (1.0 + 3.0 * parameter) / 4.0


def equal_split_target(fidelity_threshold: float, hops: int) -> float:
    """Return the paper's equal-split per-hop fidelity target (Eq. 7)."""
    if hops <= 0:
        raise ValueError("Q-GUARD paths must contain at least one hop")
    if not 0.25 <= fidelity_threshold <= 1.0:
        raise ValueError("Q-GUARD Werner fidelity thresholds must be in [0.25, 1]")
    target_w = werner_parameter(fidelity_threshold) ** (1.0 / hops)
    return fidelity_from_werner(target_w)


def detour_split_target(
    fidelity_threshold: float,
    major_hops: int,
    replaced_hops: int,
    detour_hops: int,
) -> float:
    """Return the per-hop target for a detour replacing a major-path span."""
    if major_hops <= 0 or replaced_hops <= 0 or detour_hops <= 0:
        raise ValueError("Q-GUARD detour dimensions must be positive")
    if replaced_hops > major_hops:
        raise ValueError("A Q-GUARD detour cannot replace more than the major path")
    segment_w = werner_parameter(fidelity_threshold) ** (
        replaced_hops / major_hops
    )
    return fidelity_from_werner(segment_w ** (1.0 / detour_hops))


def bbpssw_werner_once(
    first_fidelity: float,
    second_fidelity: float,
) -> tuple[float, float]:
    """Return ideal Werner-state BBPSSW output fidelity and success chance."""
    first = min(1.0, max(0.25, first_fidelity))
    second = min(1.0, max(0.25, second_fidelity))
    first_error = 1.0 - first
    second_error = 1.0 - second
    numerator = first * second + first_error * second_error / 9.0
    denominator = (
        first * second
        + (first * second_error + first_error * second) / 3.0
        + 5.0 * first_error * second_error / 9.0
    )
    if denominator <= 0:
        return 0.25, 0.0
    return numerator / denominator, denominator


def minimum_purification_rounds(
    initial_fidelity: float,
    target_fidelity: float,
    max_rounds: int,
) -> int | None:
    """Estimate symmetric BBPSSW rounds, returning ``None`` if infeasible."""
    if max_rounds < 0:
        raise ValueError("Q-GUARD maximum purification rounds cannot be negative")
    if initial_fidelity + 1e-12 >= target_fidelity:
        return 0
    current = initial_fidelity
    for rounds in range(1, max_rounds + 1):
        updated, _success_probability = bbpssw_werner_once(current, current)
        if updated <= current + 1e-12:
            return None
        current = updated
        if current + 1e-12 >= target_fidelity:
            return rounds
    return None


@dataclass(frozen=True)
class QGUARDHopPlan:
    target_fidelity: float
    initial_fidelity: float
    available_pairs: int
    purification_rounds: int

    @property
    def required_raw_pairs(self) -> int:
        return 1 << self.purification_rounds


@dataclass(frozen=True)
class QGUARDExgResult:
    expected_goodput: float
    feasible: bool
    required_raw_pairs: tuple[int, ...]
    availability: float


def expected_goodput(
    width: int,
    hop_plans: Iterable[QGUARDHopPlan],
    swap_success_probability: float,
) -> QGUARDExgResult:
    """Evaluate Q-GUARD's realized-span EXG metric (Eq. 9)."""
    plans = tuple(hop_plans)
    if width <= 0 or not plans:
        return QGUARDExgResult(0.0, False, (), 0.0)
    if not 0 <= swap_success_probability <= 1:
        raise ValueError("Swap success probability must be in [0, 1]")

    required = tuple(plan.required_raw_pairs for plan in plans)
    # Eq. 9 treats path width as the purification-plan feasibility bound.
    # Realized availability is a separate multiplicative penalty (A_min), so
    # an underfilled route remains feasible but receives a lower EXG score.
    feasible = all(count <= width for count in required)
    if not feasible:
        return QGUARDExgResult(0.0, False, required, 0.0)

    availability = min(
        min(1.0, plan.available_pairs / count)
        for count, plan in zip(required, plans)
    )
    extra_pair_cost = sum(count - 1 for count in required)
    swaps = max(0, len(plans) - 1)
    score = (
        width
        * math.pow(swap_success_probability, swaps)
        / (1.0 + extra_pair_cost)
        * availability
    )
    return QGUARDExgResult(score, True, required, availability)


@dataclass(frozen=True)
class QGUARD(QCAST):
    """Base paper Q-GUARD with equal-split, post-generation planning."""

    max_purification_rounds: int = 20
    control_mode: str = QCAST_CONTROL_PAPER_DISTRIBUTED
    algorithm_name: str | None = None
    config: AlgorithmConfig = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_mode != QCAST_CONTROL_PAPER_DISTRIBUTED:
            raise ValueError("Q-GUARD requires paper-distributed control")
        if self.max_purification_rounds < 0:
            raise ValueError("Q-GUARD maximum purification rounds cannot be negative")
        object.__setattr__(self, "config", AlgorithmConfig(
            name=self.algorithm_name or "qguard",
            kind="qguard",
        ))


__all__ = [
    "QGUARD",
    "QGUARDExgResult",
    "QGUARDHopPlan",
    "bbpssw_werner_once",
    "detour_split_target",
    "equal_split_target",
    "expected_goodput",
    "fidelity_from_werner",
    "minimum_purification_rounds",
    "werner_parameter",
]
