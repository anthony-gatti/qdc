"""Compatibility backend wrapper for the clean plugin/runtime path."""

from __future__ import annotations

import json
from pathlib import Path

from algorithms.registry import create_algorithm
from backends.base import BackendBase
from backends.sequence.runtime import SequenceRuntime
from workloads.qpq import QPQWorkload
from workloads.single_pair import SinglePairPaperWorkload


class CleanAlgorithmBackend(BackendBase):
    """Adapter allowing the legacy sweep shell to use the clean runtime."""

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
        seed = int(config.get("topology", {}).get("random_seed", 0))
        if mode == "qpq":
            with open(topo_json_path) as source:
                topology = json.load(source)
            workload = QPQWorkload.from_config(
                config,
                seed=seed,
                topology_override=topology,
                query_override=request_queue,
            )
        elif mode in {"pair", "single_pair"}:
            requests = len(request_queue) if request_queue else config.get("paper_comparison", {}).get("requests", 100)
            workload = SinglePairPaperWorkload(num_requests=requests, seed=seed)
        else:
            raise ValueError(f"Unsupported clean workload mode: {mode}")

        algorithm_options = dict(config.get("algorithm", {}))
        algorithm_options["adaptive_max_memory"] = self._adaptive_max_memory
        algorithm = create_algorithm(self.algorithm_name, algorithm_options)
        experiment = config.get("experiment", {})
        runtime_root = Path(experiment.get(
            "runtime_output_dir",
            Path(experiment.get("output_dir", "output")) / self.name,
        ))
        runtime = SequenceRuntime(runtime_root)
        result = algorithm.run(runtime, workload)
        (runtime_root / "diagnostics.json").write_text(
            json.dumps(runtime.last_diagnostics, indent=2, sort_keys=True) + "\n"
        )
        result.backend_name = self.name
        return result
