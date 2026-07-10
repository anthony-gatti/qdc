#!/usr/bin/env python3
"""Run matched workload/algorithm experiments through the common runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.registry import algorithm_names, create_algorithm
from backends.sequence.runtime import SequenceRuntime
from workloads.registry import create_workload, workload_names


def _git_state(path: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", choices=workload_names())
    parser.add_argument("--algorithms", nargs="+", choices=algorithm_names())
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    with args.config.open() as handle:
        config = yaml.safe_load(handle) or {}
    workload_config = config.get("workload", {})
    experiment_config = config.get("experiment", {})
    algorithm_config = config.get("algorithm", {})
    workload_name = args.workload or workload_config.get("name") or workload_config.get("mode")
    if not workload_name:
        raise ValueError("Configuration must select workload.name")
    selected_algorithms = (
        args.algorithms
        or experiment_config.get("algorithms")
        or experiment_config.get("backends")
        or [algorithm_config.get("name", "odo")]
    )
    selected_seeds = args.seeds or experiment_config.get("seeds") or [
        int(config.get("topology", {}).get("random_seed", 0))
    ]
    args.output.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in selected_seeds:
        workload = create_workload(workload_name, config, int(seed))
        for algorithm_name in selected_algorithms:
            algorithm = create_algorithm(algorithm_name, algorithm_config)
            run_dir = args.output / f"seed_{seed}" / algorithm.name
            runtime = SequenceRuntime(run_dir)
            started = time.perf_counter()
            result = algorithm.run(runtime, workload)
            elapsed = time.perf_counter() - started
            results_path = run_dir / "results.csv"
            result.to_csv(str(results_path))
            diagnostics_path = run_dir / "diagnostics.json"
            diagnostics_path.write_text(
                json.dumps(runtime.last_diagnostics, indent=2, sort_keys=True) + "\n"
            )
            summary = {
                "seed": seed,
                "workload": workload_name,
                "algorithm": algorithm.name,
                "requests": result.num_requests,
                "successes": result.num_success,
                "success_rate": result.success_rate,
                "mean_tts_ms": result.avg_tts_ms,
                "mean_fidelity": result.avg_fidelity,
                "runtime_s": elapsed,
                "results": str(results_path),
                "diagnostics": str(diagnostics_path),
            }
            runs.append(summary)
            print(json.dumps(summary, sort_keys=True))

    sequence_root = ROOT.parent / "SeQUeNCe"
    manifest = {
        "schema_version": 1,
        "command": [sys.executable, *sys.argv],
        "python": {"executable": sys.executable, "version": sys.version},
        "resolved_config": config,
        "workload": workload_name,
        "algorithms": list(selected_algorithms),
        "seeds": list(selected_seeds),
        "qdc": _git_state(ROOT),
        "sequence": _git_state(sequence_root),
        "runs": runs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
