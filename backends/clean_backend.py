"""Compatibility backend wrapper for the clean plugin/runtime path."""

from __future__ import annotations

from pathlib import Path

from algorithms.acp import AdaptiveContinuous
from algorithms.odo import ShortestPathOnDemand
from backends.base import BackendBase
from backends.sequence.runtime import SequenceRuntime
from workloads.single_pair import SinglePairPaperWorkload


class CleanAlgorithmBackend(BackendBase):
    """Adapter for existing backend registry callers.

    This milestone supports the paper single-pair workload only.
    """

    def __init__(self, algorithm_name: str, adaptive_max_memory: int = 5, name_override: str | None = None):
        self.algorithm_name = algorithm_name
        self._adaptive_max_memory = adaptive_max_memory
        self._name_override = name_override

    @property
    def name(self) -> str:
        return self._name_override or self.algorithm_name

    @property
    def adaptive_max_memory(self) -> int:
        return self._adaptive_max_memory if self.algorithm_name.startswith("acp") else 0

    def run(self, topo_json_path: str, request_queue: list, config: dict):
        mode = config.get("workload", {}).get("mode", "single_pair")
        if mode not in {"pair", "single_pair"}:
            raise NotImplementedError("CleanAlgorithmBackend currently supports only the single-pair milestone.")
        requests = len(request_queue) if request_queue else config.get("paper_comparison", {}).get("requests", 100)
        workload = SinglePairPaperWorkload(
            num_requests=requests,
            seed=config.get("topology", {}).get("random_seed", 0),
        )
        if self.algorithm_name == "odo":
            algorithm = ShortestPathOnDemand()
        elif self.algorithm_name in {"acp", "acp_freshest"}:
            algorithm = AdaptiveContinuous(adaptive_max_memory=self._adaptive_max_memory, cache_strategy="freshest")
        elif self.algorithm_name == "acp_random":
            algorithm = AdaptiveContinuous(adaptive_max_memory=self._adaptive_max_memory, cache_strategy="random")
        else:
            raise ValueError(self.algorithm_name)
        runtime = SequenceRuntime(Path(config.get("experiment", {}).get("output_dir", "output")) / self.name)
        result = algorithm.run(runtime, workload)
        result.backend_name = self.name
        return result

