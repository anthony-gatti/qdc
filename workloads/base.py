"""Workload plugin interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Workload(ABC):
    @abstractmethod
    def requests(self) -> list[tuple]:
        """Return runtime-native request tuples."""

    @abstractmethod
    def topology(self, adaptive_memory: int) -> dict:
        """Return a SeQUeNCe topology config for this workload."""

