"""DFER configuration and paper-level fidelity-routing helpers.

The paper contains inconsistent uses of ``l_curr`` in Algorithm 1 and a
negative exponent in Eq. 16.  This implementation uses the physically
consistent interpretation stated in the surrounding text: at the current
endpoint, divide the still-required Werner parameter equally over the
remaining shortest-path hops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from algorithms.base import AlgorithmConfig, RoutingAlgorithm


def werner_parameter(fidelity: float) -> float:
    """Convert Werner-state fidelity to its depolarization parameter."""
    return (4.0 * fidelity - 1.0) / 3.0


def fidelity_from_werner(parameter: float) -> float:
    """Convert a Werner depolarization parameter to fidelity."""
    return (1.0 + 3.0 * parameter) / 4.0


def bbpssw_werner_once(
    kept_fidelity: float,
    auxiliary_fidelity: float,
    *,
    kept_gate_fidelity: float = 1.0,
    remote_gate_fidelity: float = 1.0,
    kept_measurement_fidelity: float = 1.0,
    remote_measurement_fidelity: float = 1.0,
) -> tuple[float, float]:
    """Return Werner BBPSSW output fidelity and success probability.

    The optional local-operation parameters match upstream SeQUeNCe's
    Bell-diagonal, twirled BBPSSW equations.  Their defaults reduce exactly to
    the ideal formula used in the DFER paper.
    """
    kept = min(1.0, max(0.25, kept_fidelity))
    auxiliary = min(1.0, max(0.25, auxiliary_fidelity))
    kept_error = 1.0 - kept
    auxiliary_error = 1.0 - auxiliary
    kept_second = kept_error / 3.0
    auxiliary_second = auxiliary_error / 3.0
    kept_even = kept + kept_second
    auxiliary_even = auxiliary + auxiliary_second
    gate_product = kept_gate_fidelity * remote_gate_fidelity
    same_measurement = (
        kept_measurement_fidelity * remote_measurement_fidelity
        + (1.0 - kept_measurement_fidelity)
        * (1.0 - remote_measurement_fidelity)
    )
    different_measurement = (
        kept_measurement_fidelity * (1.0 - remote_measurement_fidelity)
        + (1.0 - kept_measurement_fidelity) * remote_measurement_fidelity
    )
    probability = (
        0.5
        + gate_product * different_measurement
        + gate_product
        * (
            kept_even * auxiliary_even
            + (1.0 - kept_even) * (1.0 - auxiliary_even)
        )
        * (same_measurement - different_measurement)
        - gate_product / 2.0
    )
    if probability <= 0:
        return 0.25, 0.0
    numerator = (
        gate_product
        * (
            same_measurement
            * (
                kept * auxiliary
                + kept_second * auxiliary_second
            )
            + different_measurement
            * (
                kept * auxiliary_second
                + kept_second * auxiliary_second
            )
        )
        + (1.0 - gate_product) / 8.0
    )
    return numerator / probability, probability


def required_link_fidelity(
    fidelity_threshold: float,
    remaining_hops: int,
    current_fidelity: float | None = None,
) -> float:
    """Compute DLFR's per-link target using the remaining fidelity budget."""
    if remaining_hops <= 0:
        raise ValueError("DFER requires at least one remaining hop")
    if not 0.25 <= fidelity_threshold <= 1.0:
        raise ValueError("DFER Werner fidelity thresholds must be in [0.25, 1]")
    required = werner_parameter(fidelity_threshold)
    if current_fidelity is not None:
        current = werner_parameter(current_fidelity)
        if current <= 0:
            return 1.0
        required /= current
    required = min(1.0, max(0.0, required))
    return fidelity_from_werner(required ** (1.0 / remaining_hops))


@dataclass(frozen=True)
class DFERPumpingPlan:
    target_fidelity: float
    output_fidelity: float
    rounds: int
    success_probability: float


def pumping_plan(
    initial_fidelity: float,
    target_fidelity: float,
    max_rounds: int,
    *,
    kept_gate_fidelity: float = 1.0,
    remote_gate_fidelity: float = 1.0,
    kept_measurement_fidelity: float = 1.0,
    remote_measurement_fidelity: float = 1.0,
) -> DFERPumpingPlan | None:
    """Plan DFER pumping with one fresh elementary pair per round."""
    if max_rounds < 0:
        raise ValueError("DFER maximum purification rounds cannot be negative")
    if initial_fidelity + 1e-12 >= target_fidelity:
        return DFERPumpingPlan(target_fidelity, initial_fidelity, 0, 1.0)
    current = initial_fidelity
    probability = 1.0
    for rounds in range(1, max_rounds + 1):
        updated, success = bbpssw_werner_once(
            current,
            initial_fidelity,
            kept_gate_fidelity=kept_gate_fidelity,
            remote_gate_fidelity=remote_gate_fidelity,
            kept_measurement_fidelity=kept_measurement_fidelity,
            remote_measurement_fidelity=remote_measurement_fidelity,
        )
        if updated <= current + 1e-12 or success <= 0:
            return None
        current = updated
        probability *= success
        if current + 1e-12 >= target_fidelity:
            return DFERPumpingPlan(
                target_fidelity,
                current,
                rounds,
                probability,
            )
    return None


def expected_distribution_rate(
    generation_rate_hz: float,
    plan: DFERPumpingPlan,
    purification_round_time_s: float,
    remaining_hops: int,
    swap_success_probability: float,
) -> float:
    """Score one DFPS candidate by expected successful end-pair rate.

    This is the paper's EDR intent expressed without its off-by-one summation
    indices.  Generation and purification costs are charged explicitly and
    the remaining swaps contribute their success probability.
    """
    if generation_rate_hz <= 0:
        return 0.0
    generated_pairs = plan.rounds + 1
    expected_time = (
        generated_pairs / generation_rate_hz
        + plan.rounds * max(0.0, purification_round_time_s)
    )
    if expected_time <= 0:
        return 0.0
    downstream_swaps = max(0, remaining_hops - 1)
    return (
        plan.success_probability
        * math.pow(swap_success_probability, downstream_swaps)
        / expected_time
    )


@dataclass(frozen=True)
class DFER(RoutingAlgorithm):
    """Distributed fidelity-guaranteed entanglement routing."""

    cutoff_fidelity: float = 0.7
    max_purification_rounds: int = 20
    control_processing_delay_ps: int = 100_000_000
    resource_retry_delay_ps: int = 100_000_000
    swap_success_probability: float = 0.9
    max_hops: int = 64
    algorithm_name: str = "dfer"
    config: AlgorithmConfig = field(init=False)

    def __post_init__(self) -> None:
        if not 0.5 <= self.cutoff_fidelity <= 1.0:
            raise ValueError("DFER cutoff fidelity must be in [0.5, 1]")
        if self.max_purification_rounds < 0:
            raise ValueError("DFER maximum purification rounds cannot be negative")
        if self.control_processing_delay_ps < 0:
            raise ValueError("DFER processing delay cannot be negative")
        if self.resource_retry_delay_ps <= 0:
            raise ValueError("DFER resource retry delay must be positive")
        if not 0 <= self.swap_success_probability <= 1:
            raise ValueError("DFER swap success probability must be in [0, 1]")
        if self.max_hops <= 0:
            raise ValueError("DFER maximum hops must be positive")
        object.__setattr__(
            self,
            "config",
            AlgorithmConfig(name=self.algorithm_name, kind="dfer"),
        )

    def run(self, runtime, workload):
        return runtime.run(workload, self)


__all__ = [
    "DFER",
    "DFERPumpingPlan",
    "bbpssw_werner_once",
    "expected_distribution_rate",
    "fidelity_from_werner",
    "pumping_plan",
    "required_link_fidelity",
    "werner_parameter",
]
