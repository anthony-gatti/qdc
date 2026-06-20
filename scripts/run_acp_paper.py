#!/usr/bin/env python3
"""Run the ACP paper's no-purification workload on SeQUeNCe v1.0.0."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backends.acp_backend import ACPBackend
from backends.odo_backend import ODOBackend
from paper_workload import SCENARIOS, generate_requests, prepare_topology, validate_paths


def backend_for(name):
    if name == "odo":
        return ODOBackend()
    if name in {"acp_freshest", "acp_random"}:
        strategy = name.removeprefix("acp_")
        return ACPBackend(
            adaptive_max_memory=5, update_prob=True,
            application_priority=False, cache_strategy=strategy,
            name_override=name,
        )
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["line2"])
    parser.add_argument("--backends", nargs="+", choices=["odo", "acp_freshest", "acp_random"],
                        default=["odo", "acp_freshest", "acp_random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            requests = generate_requests(scenario, seed)
            for backend_name in args.backends:
                backend = backend_for(backend_name)
                topology_path = args.output / "topologies" / f"{scenario}_seed{seed}_{backend_name}.json"
                config_data = prepare_topology(
                    scenario, seed, backend.adaptive_max_memory, topology_path,
                )
                validate_paths(scenario, config_data, requests)
                config = {
                    "workload": {"mode": "pair"},
                    "topology": {"random_seed": seed},
                    "acp": {"update_period_s": 0.1},
                    "paper_comparison": {
                        "fidelity_threshold": 0.5,
                        "purification": False,
                        "application_priority": False,
                        "request_interval_s": 0.1,
                        "request_lead_s": 0.02,
                    },
                }
                started = time.perf_counter()
                result = backend.run(str(topology_path), requests, config)
                runtime = time.perf_counter() - started
                csv_path = args.output / f"{scenario}_seed{seed}_{backend_name}.csv"
                with csv_path.open("w", newline="") as handle:
                    fieldnames = ["request_id", "src", "dst", "start_time_ps", "time_to_serve_ms",
                                  "fidelity", "success", "failure_reason"]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in result.request_results:
                        data = asdict(row)
                        writer.writerow({key: data[key] for key in fieldnames})
                summary = {
                    "scenario": scenario, "seed": seed, "backend": backend_name,
                    "requests": result.num_requests, "successes": result.num_success,
                    "success_rate": result.success_rate, "mean_tts_ms": result.avg_tts_ms,
                    "mean_fidelity": result.avg_fidelity, "runtime_s": runtime,
                    "topology": str(topology_path), "results": str(csv_path),
                }
                summaries.append(summary)
                print(json.dumps(summary, sort_keys=True))

    manifest = {
        "study": "ACP paper comparison, no purification",
        "sequence_version": "1.0.0", "sequence_commit": "ffd7c837",
        "python": sys.version, "conditions": {
            "fidelity_threshold": 0.5, "memory_per_node": 10,
            "acp_memory_per_node": 5, "link_distance_km": 10,
            "request_rate_hz": 10, "request_window_ms": 80,
            "purification": False, "application_priority": False,
            "acp_update_period_s": 0.1,
            "seed_policy": "same workload seed and component seed offset for every backend",
        },
        "runs": summaries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
