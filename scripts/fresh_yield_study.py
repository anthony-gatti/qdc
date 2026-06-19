#!/usr/bin/env python3
"""Run and summarize the gated five-seed fresh-yield QPQ study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep2d import run_one_cell


def _sum_edges(diagnostic: dict) -> tuple[dict, dict]:
    by_edge = defaultdict(lambda: defaultdict(int))
    by_hop = defaultdict(lambda: defaultdict(int))
    fields = (
        "elementary_pair_demand",
        "cache_adoption_succeeded",
        "successful_physical_pairs",
        "bsm_attempts",
        "completed_physical_attempts",
        "demand_unsatisfied_at_deadline",
    )
    for reservation in diagnostic["reservations"]:
        hop = len(reservation["path"]) - 1
        for edge in reservation["edges"].values():
            edge_name = "|".join(edge["edge"])
            for field in fields:
                by_edge[edge_name][field] += edge.get(field, 0)
                by_hop[hop][field] += edge.get(field, 0)
    return ({key: dict(value) for key, value in by_edge.items()},
            {str(key): dict(value) for key, value in by_hop.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="diagnostics/fresh_yield_5seed_20260619")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--backends", default="acp_no_bg,acp_m1,acp_m6")
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as source:
        config = yaml.safe_load(source)
    config.setdefault("workload", {})["mode"] = "qpq"
    config.setdefault("diagnostics", {})["application_demand"] = True
    config["diagnostics"]["application_demand_events"] = False
    config["diagnostics"]["cache_lifecycle"] = True

    selected_seeds = [int(value) for value in args.seeds.split(",") if value]
    selected_backends = [value for value in args.backends.split(",") if value]
    rows = []
    cells = []
    for seed in selected_seeds:
        for backend in selected_backends:
            checkpoint = outdir / f"cell_s{seed}_{backend}.json"
            if checkpoint.exists():
                cells.append(json.loads(checkpoint.read_text()))
                continue
            started = time.monotonic()
            cell_rows = run_one_cell(config, 40.0, 10, 25, seed, backend, str(outdir))
            runtime = time.monotonic() - started
            rows.extend(cell_rows)
            stem = f"d40_n25_nb10_s{seed}_{backend}.json"
            demand = json.loads((outdir / "application_demand" / f"demand_{stem}").read_text())
            cache = json.loads((outdir / "cache_lifecycle" / f"cache_{stem}").read_text())
            by_edge, by_hop = _sum_edges(demand)
            successes = sum(bool(row["success"]) for row in cell_rows)
            failure_reasons = defaultdict(int)
            for row in cell_rows:
                if not row["success"]:
                    failure_reasons[row["failure_reason"]] += 1
            cache_counters = cache["counters"]
            useful = max(
                (snapshot.get("useful_unique_pair_records", 0) for snapshot in cache["snapshots"]),
                default=0,
            )
            cell = {
                "seed": seed,
                "backend": backend,
                "runtime_s": runtime,
                "query_successes": successes,
                "query_failures": len(cell_rows) - successes,
                "failure_reasons": dict(failure_reasons),
                "application_demand": demand["totals"].get("elementary_pair_demand", 0),
                "demand_satisfied_cache": demand["totals"].get("cache_adoption_succeeded", 0),
                "demand_satisfied_fresh": demand["totals"].get("successful_physical_pairs", 0),
                "deadline_deficit": demand["totals"].get("demand_unsatisfied_at_deadline", 0),
                "protocol_request_objects": demand["totals"].get("protocol_request_objects", 0),
                "protocol_start_calls": demand["totals"].get("protocol_start_calls", 0),
                "endpoint_emissions": demand["totals"].get("endpoint_emissions", 0),
                "bsm_attempts": demand["totals"].get("bsm_attempts", 0),
                "completed_physical_attempts": demand["totals"].get("completed_physical_attempts", 0),
                "physical_elementary_successes": demand["totals"].get("successful_physical_pairs", 0),
                "background_physical_pairs_generated": cache_counters.get("background_physical_pairs_generated", 0),
                "background_physical_pairs_adopted": cache_counters.get("background_physical_pairs_adopted", 0),
                "max_useful_inventory": useful,
                "background_cost_per_adopted_pair": (
                    cache_counters.get("background_physical_pairs_generated", 0)
                    / cache_counters["background_physical_pairs_adopted"]
                    if cache_counters.get("background_physical_pairs_adopted", 0) else None
                ),
                "input_fingerprints": demand["input_fingerprints"],
                "by_edge": by_edge,
                "by_hop": by_hop,
            }
            cells.append(cell)
            checkpoint.write_text(json.dumps(cell, indent=2, sort_keys=True))
            with open(outdir / f"query_rows_s{seed}_{backend}.csv", "w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(cell_rows[0]))
                writer.writeheader()
                writer.writerows(cell_rows)

    if rows:
        with open(outdir / "query_rows.csv", "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "scope": {"nodes": 25, "distance_km": 40, "database_size_log": 10, "seeds": selected_seeds},
        "cells": cells,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
