#!/usr/bin/env python3
"""Run small, matched regimes that isolate routing-algorithm tradeoffs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.acp import AdaptiveContinuous
from algorithms.odo import ShortestPathOnDemand
from algorithms.qcast import QCAST, QCAST_CONTROL_PAPER_DISTRIBUTED
from backends.sequence.runtime import SequenceRuntime
from common import SECOND
from workloads.concurrent_pairs import ConcurrentPairSpec, ConcurrentPairWorkload


def _ring_requests(destinations: tuple[int, ...], start_ps: int, end_ps: int):
    return tuple(
        ConcurrentPairSpec(
            request_id=index,
            source="router_0",
            destination=f"router_{destination}",
            start_time_ps=start_ps,
            deadline_ps=end_ps,
            fidelity_threshold=0.7,
        )
        for index, destination in enumerate(destinations)
    )


def build_workload(case: str, seed: int) -> ConcurrentPairWorkload:
    if case == "ring_single_opposite":
        start = int(0.020 * SECOND)
        return ConcurrentPairWorkload(
            num_requests=1,
            seed=seed,
            num_nodes=6,
            qdc_node_index=0,
            topology_type="ring",
            inter_node_distance_m=20_000,
            memories_per_node=12,
            memory_efficiency=0.5,
            coherence_time_s=5.0,
            simulation_end_time_s=0.20,
            request_override=_ring_requests((3,), start, int(0.160 * SECOND)),
        )
    if case == "ring_fanout":
        start = int(0.030 * SECOND)
        return ConcurrentPairWorkload(
            num_requests=5,
            seed=seed,
            num_nodes=10,
            qdc_node_index=0,
            topology_type="ring",
            inter_node_distance_m=20_000,
            memories_per_node=30,
            memory_efficiency=0.3,
            coherence_time_s=5.0,
            simulation_end_time_s=0.30,
            request_override=_ring_requests((3, 4, 5, 6, 7), start, int(0.250 * SECOND)),
        )
    if case == "direct_short_coherence":
        start = int(0.050 * SECOND)
        return ConcurrentPairWorkload(
            num_requests=1,
            seed=seed,
            num_nodes=2,
            qdc_node_index=0,
            topology_type="linear",
            inter_node_distance_m=10_000,
            memories_per_node=12,
            memory_efficiency=0.5,
            coherence_time_s=0.0005,
            simulation_end_time_s=0.20,
            request_override=_ring_requests((1,), start, int(0.150 * SECOND)),
        )
    raise ValueError(f"Unknown regime {case!r}")


def build_algorithm(name: str, case: str):
    if name == "odo":
        return ShortestPathOnDemand()
    if name == "acp_freshest":
        cache_cap = 1 if case == "ring_single_opposite" else 4
        return AdaptiveContinuous(
            adaptive_max_memory=cache_cap,
            cache_strategy="freshest",
        )
    if name == "qcast_distributed":
        generation_window_ps = int(0.020 * SECOND)
        if case == "direct_short_coherence":
            generation_window_ps = int(0.001 * SECOND)
        return QCAST(
            edge_width=1,
            generation_window_ps=generation_window_ps,
            control_processing_delay_ps=int(0.0001 * SECOND),
            link_state_hops=1,
            max_major_paths=100,
            max_hops=10,
            control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
        )
    raise ValueError(f"Unknown algorithm {name!r}")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("ring_single_opposite", "ring_fanout", "direct_short_coherence"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        for algorithm_name in ("odo", "acp_freshest", "qcast_distributed"):
            algorithm = build_algorithm(algorithm_name, args.case)
            workload = build_workload(args.case, seed)
            run_dir = args.output / f"seed_{seed}" / algorithm.name
            runtime = SequenceRuntime(run_dir)
            started = time.perf_counter()
            result = algorithm.run(runtime, workload)
            elapsed = time.perf_counter() - started
            result.to_csv(str(run_dir / "results.csv"))
            (run_dir / "diagnostics.json").write_text(
                json.dumps(runtime.last_diagnostics, indent=2, sort_keys=True) + "\n"
            )
            for row in result.request_results:
                runs.append({
                    "seed": seed,
                    "algorithm": algorithm.name,
                    "request_id": row.request_id,
                    "success": row.success,
                    "time_to_serve_ms": row.time_to_serve_ms,
                    "fidelity": row.fidelity,
                    "failure_reason": row.failure_reason,
                    "runtime_s": elapsed,
                })

    summary = {}
    for algorithm_name in ("odo", "acp_freshest", "qcast_distributed"):
        rows = [row for row in runs if row["algorithm"] == algorithm_name]
        successes = [row for row in rows if row["success"]]
        tts = [row["time_to_serve_ms"] for row in successes]
        fidelity = [row["fidelity"] for row in successes]
        summary[algorithm_name] = {
            "attempted": len(rows),
            "succeeded": len(successes),
            "success_rate": len(successes) / len(rows) if rows else 0.0,
            "mean_tts_ms": statistics.fmean(tts) if tts else None,
            "median_tts_ms": statistics.median(tts) if tts else None,
            "p95_tts_ms": _percentile(tts, 0.95),
            "mean_fidelity": statistics.fmean(fidelity) if fidelity else None,
            "failure_reasons": dict(Counter(
                row["failure_reason"] for row in rows if not row["success"]
            )),
        }

    report = {
        "case": args.case,
        "seeds": args.seeds,
        "qdc_commit": _git_revision(ROOT),
        "sequence_commit": _git_revision(ROOT.parent / "SeQUeNCe"),
        "summary": summary,
        "runs": runs,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
