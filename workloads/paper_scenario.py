"""ACP paper scenario workloads backed by archived topology inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_workload import SCENARIOS, generate_requests, prepare_topology, validate_paths


@dataclass(frozen=True)
class PaperScenarioWorkload:
    """Single-pair paper scenario, e.g. the 20-node bottleneck experiment."""

    scenario: str
    seed: int = 0
    topology_output_dir: Path = Path("/tmp/qdc_paper_topologies")

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unsupported paper scenario: {self.scenario}")

    def requests(self) -> list[tuple]:
        return generate_requests(self.scenario, self.seed)

    def topology(self, adaptive_memory: int) -> dict:
        output = self.topology_output_dir / f"{self.scenario}_seed{self.seed}_ma{adaptive_memory}.json"
        config = prepare_topology(self.scenario, self.seed, adaptive_memory, output)
        validate_paths(self.scenario, config, self.requests())
        return config
