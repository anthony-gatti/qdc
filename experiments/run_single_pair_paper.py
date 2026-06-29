#!/usr/bin/env python3
"""Run the first clean ACP milestone on the two-node paper setup."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.acp import AdaptiveContinuous
from algorithms.odo import ShortestPathOnDemand
from backends.sequence.runtime import SequenceRuntime
from workloads.single_pair import SinglePairPaperWorkload
from topology import CLASSICAL_TIMING_ACP_PAPER, CLASSICAL_TIMING_SEQUENCE


def build_algorithm(name: str):
    if name == "odo":
        return ShortestPathOnDemand()
    if name == "acp_freshest":
        return AdaptiveContinuous(cache_strategy="freshest")
    if name == "acp_random":
        return AdaptiveContinuous(cache_strategy="random")
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", default=["odo", "acp_freshest", "acp_random"])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classical-timing-profile",
        choices=[CLASSICAL_TIMING_SEQUENCE, CLASSICAL_TIMING_ACP_PAPER],
        default=CLASSICAL_TIMING_ACP_PAPER,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    workload = SinglePairPaperWorkload(
        num_requests=args.requests,
        seed=args.seed,
        classical_timing_profile=args.classical_timing_profile,
    )
    summaries = []
    for algorithm_name in args.algorithms:
        algorithm = build_algorithm(algorithm_name)
        runtime = SequenceRuntime(args.output / algorithm.name)
        started = time.perf_counter()
        result = algorithm.run(runtime, workload)
        elapsed = time.perf_counter() - started

        rows_path = args.output / f"{algorithm.name}.csv"
        with rows_path.open("w", newline="") as handle:
            fieldnames = ["request_id", "src", "dst", "start_time_ps", "time_to_serve_ms", "fidelity", "success", "failure_reason"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in result.request_results:
                data = asdict(row)
                writer.writerow({key: data[key] for key in fieldnames})

        diag_path = args.output / f"{algorithm.name}_diagnostics.json"
        diag_path.write_text(json.dumps(runtime.last_diagnostics, indent=2, sort_keys=True) + "\n")
        summary = {
            "algorithm": algorithm.name,
            "requests": result.num_requests,
            "successes": result.num_success,
            "success_rate": result.success_rate,
            "mean_tts_ms": result.avg_tts_ms,
            "mean_fidelity": result.avg_fidelity,
            "runtime_s": elapsed,
            "rows": str(rows_path),
            "diagnostics": str(diag_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))

    manifest = {
        "study": "clean ACP rebuild: two-node no-purification single-pair validation",
        "python": sys.version,
        "conditions": {
            "link_distance_km": 10,
            "memories_per_node": 10,
            "acp_max_memory_per_node": 5,
            "request_rate_hz": 10,
            "request_window_s": 0.08,
            "pairs_per_request": 1,
            "application_purification": False,
            "background_purification": False,
            "classical_timing_profile": args.classical_timing_profile,
        },
        "runs": summaries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
