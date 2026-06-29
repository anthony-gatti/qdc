"""Shortest-path on-demand algorithm plugin."""

from __future__ import annotations

from dataclasses import dataclass

from algorithms.base import AlgorithmConfig, RoutingAlgorithm


@dataclass(frozen=True)
class ShortestPathOnDemand(RoutingAlgorithm):
    """Vanilla SeQUeNCe RSVP shortest-path on-demand baseline."""

    config: AlgorithmConfig = AlgorithmConfig(name="odo", kind="odo")

    def run(self, runtime, workload):
        return runtime.run_single_pair(workload, self)

