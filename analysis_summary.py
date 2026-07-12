#!/usr/bin/env python3
"""Write post-port comparison summaries from sweep CSVs."""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from statistics import median
from typing import Iterable, List

from results import parse_pair_arrivals


NUMERIC_FIELDS = {
    "inter_node_distance_km": float,
    "tts_ms": float,
    "fidelity": float,
    "first_pair_arrival_ms": float,
    "round1_completion_ms": float,
    "round2_completion_ms": float,
    "walltime_s": float,
}
INT_FIELDS = {
    "seed",
    "database_size_log",
    "num_nodes",
    "query_id",
    "hop_distance",
    "acp_memory_budget",
    "round1_pairs",
    "round2_pairs",
    "expected_pairs",
    "pairs_rejected_fidelity",
    "delivered_background_pairs",
    "delivered_application_pairs",
    "delivered_pairs_with_background_contribution",
    "delivered_pairs_fully_background_supported",
    "delivered_pairs_partially_background_supported",
    "delivered_pairs_fully_fresh",
    "delivered_background_elementary_edges",
    "delivered_fresh_elementary_edges",
}


def load_rows(path: str) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_reject_field = "pairs_rejected_fidelity" in (reader.fieldnames or [])
        for row in reader:
            row["_has_pairs_rejected_fidelity"] = has_reject_field
            row["success"] = row.get("success") in ("True", "true", "1")
            for field, caster in NUMERIC_FIELDS.items():
                value = row.get(field, "")
                row[field] = caster(value) if value not in ("", None) else None
            for field in INT_FIELDS:
                value = row.get(field, "")
                row[field] = int(value) if value not in ("", None) else 0
            if row["acp_memory_budget"] == 0:
                backend = row.get("backend", "")
                if backend.startswith("acp_m") and backend[5:].isdigit():
                    row["acp_memory_budget"] = int(backend[5:])
            row["pair_arrival_list"] = parse_pair_arrivals(row.get("pair_arrival_ms", ""))
            rows.append(row)
    return rows


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(path: str, fieldnames: List[str], rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fieldnames})


def group_rows(rows: List[dict], keys: List[str]):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def summarize_groups(rows: List[dict], keys: List[str], min_success: int) -> List[dict]:
    out = []
    for key, recs in sorted(group_rows(rows, keys).items()):
        successes = [r for r in recs if r["success"]]
        tts = [r["tts_ms"] for r in successes if r["tts_ms"] is not None and r["tts_ms"] > 0]
        fids = [r["fidelity"] for r in successes if r["fidelity"] is not None and r["fidelity"] > 0]
        first_pairs = [
            r["first_pair_arrival_ms"] for r in recs
            if r["first_pair_arrival_ms"] is not None
        ]
        r1 = [r["round1_completion_ms"] for r in recs if r["round1_completion_ms"] is not None]
        r2 = [r["round2_completion_ms"] for r in recs if r["round2_completion_ms"] is not None]
        reject_rows = [r for r in recs if r.get("_has_pairs_rejected_fidelity")]
        rejected = (
            sum(r.get("pairs_rejected_fidelity", 0) for r in reject_rows)
            if reject_rows else None
        )
        expected = (
            sum(r.get("expected_pairs", 0) for r in reject_rows)
            if reject_rows else None
        )
        delivered_with_bg = sum(r.get("delivered_pairs_with_background_contribution", 0) for r in recs)
        delivered_full_bg = sum(r.get("delivered_pairs_fully_background_supported", 0) for r in recs)
        delivered_partial_bg = sum(r.get("delivered_pairs_partially_background_supported", 0) for r in recs)
        delivered_fresh = sum(r.get("delivered_pairs_fully_fresh", 0) for r in recs)
        delivered_bg_edges = sum(r.get("delivered_background_elementary_edges", 0) for r in recs)
        delivered_fresh_edges = sum(r.get("delivered_fresh_elementary_edges", 0) for r in recs)
        walltimes = [r["walltime_s"] for r in recs if r.get("walltime_s") is not None]
        result = dict(zip(keys, key))
        result.update({
            "n_queries": len(recs),
            "n_success": len(successes),
            "success_rate": len(successes) / len(recs) if recs else math.nan,
            "n_tts": len(tts),
            "tts_mean_ms": sum(tts) / len(tts) if len(tts) >= min_success else math.nan,
            "tts_p50_ms": median(tts) if len(tts) >= min_success else math.nan,
            "tts_p95_ms": percentile(tts, 0.95) if len(tts) >= min_success else math.nan,
            "first_pair_p50_ms": median(first_pairs) if len(first_pairs) >= min_success else math.nan,
            "round1_completion_p50_ms": median(r1) if len(r1) >= min_success else math.nan,
            "round2_completion_p50_ms": median(r2) if len(r2) >= min_success else math.nan,
            "delivered_fidelity_mean": sum(fids) / len(fids) if len(fids) >= min_success else math.nan,
            "pairs_rejected_fidelity": rejected,
            "expected_pairs": expected,
            "delivered_pairs_with_background_contribution": delivered_with_bg,
            "delivered_pairs_fully_background_supported": delivered_full_bg,
            "delivered_pairs_partially_background_supported": delivered_partial_bg,
            "delivered_pairs_fully_fresh": delivered_fresh,
            "delivered_background_elementary_edges": delivered_bg_edges,
            "delivered_fresh_elementary_edges": delivered_fresh_edges,
            "walltime_s": max(walltimes) if walltimes else math.nan,
            "fraction_pairs_rejected_fidelity": (
                rejected / expected if expected else math.nan
            ),
            "sample_status": "ok" if len(tts) >= min_success else "insufficient_success_samples",
        })
        out.append(result)
    return out


def outcome_rows(rows: List[dict], keys: List[str]) -> List[dict]:
    out = []
    for key, recs in sorted(group_rows(rows, keys).items()):
        counts = Counter("success" if r["success"] else (r.get("failure_reason") or "failed") for r in recs)
        total = len(recs)
        for outcome, count in sorted(counts.items()):
            row = dict(zip(keys, key))
            row.update({
                "outcome": outcome,
                "n_queries": count,
                "fraction": count / total if total else math.nan,
            })
            out.append(row)
    return out


def matched_seed_differences(rows: List[dict]) -> List[dict]:
    keys = ["seed", "inter_node_distance_km", "database_size_log", "hop_distance", "backend"]
    grouped = summarize_groups(rows, keys, min_success=1)
    by_cell = defaultdict(dict)
    for row in grouped:
        cell = (
            row["seed"],
            row["inter_node_distance_km"],
            row["database_size_log"],
            row["hop_distance"],
        )
        by_cell[cell][row["backend"]] = row

    out = []
    for cell, backend_rows in sorted(by_cell.items()):
        if "odo" not in backend_rows:
            continue
        odo = backend_rows["odo"]
        for backend, current in sorted(backend_rows.items()):
            if backend == "odo":
                continue
            out.append({
                "seed": cell[0],
                "inter_node_distance_km": cell[1],
                "database_size_log": cell[2],
                "hop_distance": cell[3],
                "comparison_backend": backend,
                "odo_n_queries": odo["n_queries"],
                "comparison_n_queries": current["n_queries"],
                "success_rate_delta_vs_odo": current["success_rate"] - odo["success_rate"],
                "tts_p50_delta_ms_vs_odo": (
                    current["tts_p50_ms"] - odo["tts_p50_ms"]
                    if not math.isnan(current["tts_p50_ms"]) and not math.isnan(odo["tts_p50_ms"])
                    else math.nan
                ),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write post-port QPQ comparison summaries.")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-success", type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.sweep_dir, "summaries")
    rows = []
    for name in ("primary_sweep.csv", "dbsize_sweep.csv"):
        rows.extend(load_rows(os.path.join(args.sweep_dir, name)))

    if not rows:
        raise SystemExit(f"No sweep rows found under {args.sweep_dir}")

    summary_fields = [
        "backend", "acp_memory_budget", "inter_node_distance_km", "hop_distance",
        "database_size_log", "n_queries", "n_success", "success_rate", "n_tts",
        "tts_mean_ms", "tts_p50_ms", "tts_p95_ms", "first_pair_p50_ms",
        "round1_completion_p50_ms", "round2_completion_p50_ms",
        "delivered_fidelity_mean", "pairs_rejected_fidelity", "expected_pairs",
        "fraction_pairs_rejected_fidelity", "sample_status",
        "delivered_pairs_with_background_contribution",
        "delivered_pairs_fully_background_supported",
        "delivered_pairs_partially_background_supported",
        "delivered_pairs_fully_fresh",
        "delivered_background_elementary_edges",
        "delivered_fresh_elementary_edges",
        "walltime_s",
    ]
    summary = summarize_groups(
        rows,
        ["backend", "acp_memory_budget", "inter_node_distance_km", "hop_distance", "database_size_log"],
        args.min_success,
    )
    write_csv(os.path.join(output_dir, "metrics_by_backend_distance_hop_db.csv"), summary_fields, summary)

    outcome = outcome_rows(
        rows,
        ["backend", "acp_memory_budget", "inter_node_distance_km", "database_size_log"],
    )
    write_csv(
        os.path.join(output_dir, "outcomes_by_backend_distance_db.csv"),
        ["backend", "acp_memory_budget", "inter_node_distance_km", "database_size_log", "outcome", "n_queries", "fraction"],
        outcome,
    )

    matched = matched_seed_differences(rows)
    write_csv(
        os.path.join(output_dir, "matched_seed_differences_vs_odo.csv"),
        [
            "seed", "inter_node_distance_km", "database_size_log", "hop_distance",
            "comparison_backend", "odo_n_queries", "comparison_n_queries",
            "success_rate_delta_vs_odo", "tts_p50_delta_ms_vs_odo",
        ],
        matched,
    )

    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
