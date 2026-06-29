"""Algorithm plugin interfaces for QDC experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from results import BackendResult


AlgorithmKind = Literal["odo", "acp"]


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    kind: AlgorithmKind


class RoutingAlgorithm(ABC):
    """Algorithm plugin executed by a backend runtime."""

    config: AlgorithmConfig

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def run(self, runtime, workload) -> BackendResult:
        """Run this algorithm through the supplied runtime."""

