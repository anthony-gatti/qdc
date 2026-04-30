"""
Standardized result collection for QPQ evaluation backends.
"""

import csv
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RequestResult:
    """Result for a single entanglement request (or QPQ query)."""
    request_id: int
    src: str
    dst: str
    start_time_ps: int   # when request was submitted
    time_to_serve_ms: Optional[float] = None  # None if request failed/timed out
    fidelity: Optional[float] = None
    success: bool = False
    # QPQ-specific fields (optional — populated only for QPQ queries)
    failure_reason: str = ""
    # Pair arrival times (ms, relative to query start). Flat list across rounds.
    # Empty for non-QPQ workloads or failed queries with no pairs delivered.
    pair_arrival_ms: List[float] = field(default_factory=list)


@dataclass
class BackendResult:
    """Aggregated result from a single backend run."""
    backend_name: str
    seed: int
    num_nodes: int
    request_results: List[RequestResult] = field(default_factory=list)

    @property
    def num_requests(self) -> int:
        return len(self.request_results)

    @property
    def num_success(self) -> int:
        return sum(1 for r in self.request_results if r.success)

    @property
    def success_rate(self) -> float:
        if not self.request_results:
            return 0.0
        return self.num_success / self.num_requests

    @property
    def avg_tts_ms(self) -> Optional[float]:
        served = [r.time_to_serve_ms for r in self.request_results
                  if r.time_to_serve_ms is not None]
        return sum(served) / len(served) if served else None

    @property
    def avg_fidelity(self) -> Optional[float]:
        fids = [r.fidelity for r in self.request_results
                if r.fidelity is not None]
        return sum(fids) / len(fids) if fids else None

    def to_csv(self, path: str) -> None:
        """Write per-request results to CSV.

        pair_arrival_ms is serialized as a semicolon-separated string to keep
        the CSV single-row-per-request. Use parse_pair_arrivals() to read it back.
        """
        if not self.request_results:
            return
        fieldnames = ["request_id", "src", "dst", "start_time_ps",
                      "time_to_serve_ms", "fidelity", "success",
                      "failure_reason", "pair_arrival_ms"]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.request_results:
                row = asdict(r)
                # Flatten pair timestamps to semicolon-joined string
                row["pair_arrival_ms"] = ";".join(
                    f"{t:.3f}" for t in r.pair_arrival_ms
                )
                writer.writerow(row)

    def summary(self) -> str:
        """Readable summary."""
        lines = [
            f"Backend: {self.backend_name}",
            f"  Requests: {self.num_requests} total, {self.num_success} succeeded ({self.success_rate:.1%})",
        ]
        if self.avg_tts_ms is not None:
            lines.append(f"  Avg TTS: {self.avg_tts_ms:.2f} ms")
        if self.avg_fidelity is not None:
            lines.append(f"  Avg fidelity: {self.avg_fidelity:.4f}")
        return "\n".join(lines)


def parse_pair_arrivals(s: str) -> List[float]:
    """Parse a semicolon-joined pair_arrival_ms CSV field back to a list."""
    if not s:
        return []
    return [float(t) for t in s.split(";") if t]