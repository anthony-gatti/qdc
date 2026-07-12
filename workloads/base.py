"""Simulator-neutral workload and entanglement-demand contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class EntanglementDemand:
    """One application stage requesting end-to-end Bell pairs."""

    reservation_id: int
    demand_id: str
    transaction_id: str
    source: str
    destination: str
    pair_count: int
    fidelity_threshold: float
    start_time_ps: int
    deadline_ps: int
    priority: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source == self.destination:
            raise ValueError("Entanglement demand endpoints must differ")
        if self.pair_count <= 0:
            raise ValueError("Entanglement demand pair_count must be positive")
        if not 0 < self.fidelity_threshold <= 1:
            raise ValueError("Entanglement demand fidelity_threshold must be in (0, 1]")
        if self.start_time_ps < 0 or self.deadline_ps <= self.start_time_ps:
            raise ValueError("Entanglement demand requires a positive service window")


@dataclass(frozen=True)
class PairDelivery:
    """One qualifying end-to-end pair delivered to an application demand."""

    timestamp_ps: int
    fidelity: float
    generation_source: str = "unknown"
    elementary_sources: tuple[Mapping[str, Any], ...] = ()


class DemandCallbacks(Protocol):
    """Callbacks exposed by a workload transaction to the backend."""

    def on_demand_accepted(self, demand: EntanglementDemand, path: tuple[str, ...]) -> None: ...

    def on_pair_delivered(self, demand: EntanglementDemand, delivery: PairDelivery) -> None: ...

    def on_pair_rejected(self, demand: EntanglementDemand, fidelity: float) -> None: ...

    def on_demand_completed(
        self,
        demand: EntanglementDemand,
        completed_at_ps: int,
        path: tuple[str, ...],
    ) -> None: ...

    def on_demand_failed(
        self,
        demand: EntanglementDemand,
        failed_at_ps: int,
        reason: str,
        path: tuple[str, ...],
        pairs_delivered: int,
    ) -> None: ...


class DemandSubmitter(Protocol):
    def __call__(self, demand: EntanglementDemand, callbacks: DemandCallbacks) -> None: ...


class Workload(ABC):
    """Algorithm-independent application workload."""

    name: str
    sequence_adapter: str

    @abstractmethod
    def topology(self, adaptive_memory: int) -> dict:
        """Return a SeQUeNCe topology config for this workload."""
