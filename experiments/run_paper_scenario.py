#!/usr/bin/env python3
"""Run archived ACP paper scenarios through the clean SeQUeNCe runtime."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from paper_workload import SCENARIOS
from workloads.paper_scenario import PaperScenarioWorkload


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
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="bottleneck20")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", default=["odo", "acp_freshest", "acp_random"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        for algorithm_name in args.algorithms:
            algorithm = build_algorithm(algorithm_name)
            run_dir = args.output / f"seed_{seed}" / algorithm.name
            workload = PaperScenarioWorkload(args.scenario, seed=seed, topology_output_dir=run_dir / "paper_topologies")
            runtime = SequenceRuntime(run_dir)
            started = time.perf_counter()
            result = algorithm.run(runtime, workload)
            elapsed = time.perf_counter() - started

            rows_path = args.output / f"{args.scenario}_seed{seed}_{algorithm.name}.csv"
            with rows_path.open("w", newline="") as handle:
                fieldnames = ["request_id", "src", "dst", "start_time_ps", "time_to_serve_ms", "fidelity", "success", "failure_reason"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in result.request_results:
                    data = asdict(row)
                    writer.writerow({key: data[key] for key in fieldnames})

            diag_path = args.output / f"{args.scenario}_seed{seed}_{algorithm.name}_diagnostics.json"
            diag_path.write_text(json.dumps(runtime.last_diagnostics, indent=2, sort_keys=True) + "\n")
            summary = {
                "scenario": args.scenario,
                "seed": seed,
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

    grouped = {}
    for summary in summaries:
        grouped.setdefault(summary["algorithm"], []).append(summary)
    aggregate = {}
    for algorithm_name, rows in grouped.items():
        aggregate[algorithm_name] = {
            "runs": len(rows),
            "mean_success_rate": statistics.mean(row["success_rate"] for row in rows),
            "mean_tts_ms": statistics.mean(row["mean_tts_ms"] for row in rows if row["mean_tts_ms"] is not None),
            "mean_fidelity": statistics.mean(row["mean_fidelity"] for row in rows if row["mean_fidelity"] is not None),
        }

    manifest = {
        "study": "clean ACP rebuild: archived ACP paper scenario",
        "scenario": args.scenario,
        "seeds": list(range(args.seed_offset, args.seed_offset + args.seeds)),
        "runs": summaries,
        "aggregate": aggregate,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
