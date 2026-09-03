#!/usr/bin/env python3
"""Run matched Q-GUARD evaluation regimes and aggregate audit diagnostics."""

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
from algorithms.qguard import QGUARD
from backends.sequence.runtime import SequenceRuntime
from common import SECOND
from workloads.concurrent_pairs import ConcurrentPairSpec, ConcurrentPairWorkload
from workloads.qpq import QPQQuerySpec, QPQWorkload


ALGORITHMS = ("odo", "acp_freshest", "qcast_distributed", "qguard")


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


def _qcast_options(link_parallelism: int) -> dict:
    return {
        "edge_width": link_parallelism,
        "generation_window_ps": int(0.005 * SECOND),
        "control_processing_delay_ps": int(0.0001 * SECOND),
        "swap_success_probability": 1.0,
        "link_state_hops": 3,
        "recovery_paths_per_segment": 1,
        "max_recovery_paths_per_major": 12,
        "max_major_paths": 8,
        "max_hops": 10,
    }


def build_algorithm(name: str, link_parallelism: int):
    if name == "odo":
        return ShortestPathOnDemand()
    if name == "acp_freshest":
        return AdaptiveContinuous(
            adaptive_max_memory=min(4, link_parallelism),
            cache_strategy="freshest",
        )
    options = _qcast_options(link_parallelism)
    if name == "qcast_distributed":
        return QCAST(**options, control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED)
    if name == "qguard":
        return QGUARD(**options, max_purification_rounds=20)
    raise ValueError(f"Unknown algorithm {name!r}")


def _linear_qpq(seed: int, threshold: float, link_parallelism: int) -> QPQWorkload:
    start = int(0.005 * SECOND)
    return QPQWorkload(
        database_size_log=1,
        num_clients=1,
        queries_per_client=1,
        fidelity_threshold=threshold,
        round_deadline_s=0.5,
        transaction_duration_s=1.0,
        request_period_s=1.0,
        start_offset_s=0.005,
        seed=seed,
        num_nodes=4,
        qdc_node_index=0,
        topology_type="linear",
        inter_node_distance_m=1_000,
        memories_per_node=16,
        memory_fidelity=0.9,
        memory_efficiency=1.0,
        coherence_time_s=5.0,
        gate_fidelity=1.0,
        measurement_fidelity=1.0,
        swapping_success_probability=1.0,
        link_parallelism=link_parallelism,
        simulation_end_time_s=1.2,
        query_override=(QPQQuerySpec(
            query_id=0,
            source="router_3",
            destination="router_0",
            start_time_ps=start,
            transaction_deadline_ps=int(1.0 * SECOND),
            database_size_log=1,
            fidelity_threshold=threshold,
            round_deadline_ps=int(0.5 * SECOND),
        ),),
    )


def _ring_single(
    seed: int,
    threshold: float,
    link_parallelism: int,
    *,
    fidelity_bound: bool,
) -> ConcurrentPairWorkload:
    start = int(0.020 * SECOND)
    return ConcurrentPairWorkload(
        num_requests=1,
        seed=seed,
        num_nodes=6,
        qdc_node_index=0,
        topology_type="ring",
        inter_node_distance_m=10_000 if fidelity_bound else 20_000,
        memories_per_node=20 if fidelity_bound else 12,
        memory_fidelity=0.9 if fidelity_bound else 0.99,
        memory_efficiency=1.0 if fidelity_bound else 0.5,
        coherence_time_s=5.0,
        gate_fidelity=1.0 if fidelity_bound else 0.99,
        measurement_fidelity=1.0 if fidelity_bound else 0.99,
        swapping_success_probability=1.0 if fidelity_bound else 0.9,
        link_parallelism=link_parallelism,
        simulation_end_time_s=0.40 if fidelity_bound else 0.20,
        request_override=(ConcurrentPairSpec(
            request_id=0,
            source="router_0",
            destination="router_3",
            start_time_ps=start,
            deadline_ps=int(0.35 * SECOND) if fidelity_bound else int(0.160 * SECOND),
            fidelity_threshold=threshold,
        ),),
    )


def _ring_fanout(seed: int, threshold: float, link_parallelism: int) -> ConcurrentPairWorkload:
    start = int(0.030 * SECOND)
    requests = tuple(
        ConcurrentPairSpec(
            request_id=index,
            source="router_0",
            destination=f"router_{destination}",
            start_time_ps=start,
            deadline_ps=int(0.250 * SECOND),
            fidelity_threshold=threshold,
        )
        for index, destination in enumerate((3, 4, 5, 6, 7))
    )
    return ConcurrentPairWorkload(
        num_requests=len(requests),
        seed=seed,
        num_nodes=10,
        qdc_node_index=0,
        topology_type="ring",
        inter_node_distance_m=20_000,
        memories_per_node=30,
        memory_fidelity=0.99,
        memory_efficiency=0.3,
        coherence_time_s=5.0,
        link_parallelism=link_parallelism,
        simulation_end_time_s=0.30,
        request_override=requests,
    )


def build_workload(case: str, seed: int, threshold: float, link_parallelism: int):
    if case == "linear_qpq":
        return _linear_qpq(seed, threshold, link_parallelism)
    if case == "ring_fidelity_bound":
        return _ring_single(
            seed,
            threshold,
            link_parallelism,
            fidelity_bound=True,
        )
    if case == "ring_nonbinding":
        return _ring_single(
            seed,
            threshold,
            link_parallelism,
            fidelity_bound=False,
        )
    if case == "ring_fanout":
        return _ring_fanout(seed, threshold, link_parallelism)
    raise ValueError(f"Unknown evaluation case {case!r}")


def _run_diagnostics(diagnostics: dict, memory_cap: int) -> dict:
    workload = diagnostics["workload_diagnostics"]
    counters = workload.get("counters", {})
    locality = workload.get("locality_invariants", {})
    return {
        "all_memories_raw": all(
            workload.get("all_memories_raw_at_end", {}).values()
        ),
        "memory_cap_ok": all(
            count <= memory_cap
            for count in workload.get("max_allocated_memories_by_node", {}).values()
        ),
        "locality_ok": locality.get("qguard_decisions_path_scoped", True),
        "purification_attempts": counters.get("qguard_purification_attempts", 0),
        "purification_successes": counters.get("qguard_purification_successes", 0),
        "purification_failures": counters.get("qguard_purification_failures", 0),
        "purification_endpoint_state_mismatches": counters.get(
            "qguard_purification_endpoint_state_mismatches",
            0,
        ),
        "end_to_end_purification_attempts": counters.get(
            "qguard_end_to_end_purification_attempts",
            0,
        ),
        "recovery_deliveries": counters.get(
            "end_to_end_pairs_delivered_recovery",
            0,
        ),
        "major_deliveries": counters.get(
            "end_to_end_pairs_delivered_major",
            0,
        ),
        "slots_completed": counters.get("slots_completed", 0),
    }


def _summarize(rows: list[dict]) -> dict:
    successes = [row for row in rows if row["success"]]
    tts = [row["time_to_serve_ms"] for row in successes]
    fidelities = [row["fidelity"] for row in successes]
    diagnostics = [row["run_diagnostics"] for row in rows]
    return {
        "attempted": len(rows),
        "succeeded": len(successes),
        "success_rate": len(successes) / len(rows) if rows else 0.0,
        "mean_tts_ms": statistics.fmean(tts) if tts else None,
        "median_tts_ms": statistics.median(tts) if tts else None,
        "p95_tts_ms": _percentile(tts, 0.95),
        "mean_fidelity": statistics.fmean(fidelities) if fidelities else None,
        "failure_reasons": dict(Counter(
            row["failure_reason"] for row in rows if not row["success"]
        )),
        "purification_attempts": sum(
            item["purification_attempts"] for item in diagnostics
        ),
        "purification_successes": sum(
            item["purification_successes"] for item in diagnostics
        ),
        "purification_failures": sum(
            item["purification_failures"] for item in diagnostics
        ),
        "purification_endpoint_state_mismatches": sum(
            item["purification_endpoint_state_mismatches"]
            for item in diagnostics
        ),
        "end_to_end_purification_attempts": sum(
            item["end_to_end_purification_attempts"] for item in diagnostics
        ),
        "recovery_deliveries": sum(
            item["recovery_deliveries"] for item in diagnostics
        ),
        "major_deliveries": sum(
            item["major_deliveries"] for item in diagnostics
        ),
        "slots_completed": sum(item["slots_completed"] for item in diagnostics),
        "all_memories_raw": all(item["all_memories_raw"] for item in diagnostics),
        "memory_cap_ok": all(item["memory_cap_ok"] for item in diagnostics),
        "locality_ok": all(item["locality_ok"] for item in diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "linear_qpq",
            "ring_fidelity_bound",
            "ring_nonbinding",
            "ring_fanout",
        ),
        required=True,
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--link-parallelism", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    args = parser.parse_args()

    if not 0.25 <= args.threshold <= 1:
        raise ValueError("Q-GUARD evaluation threshold must be in [0.25, 1]")
    if args.link_parallelism <= 0:
        raise ValueError("link parallelism must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for algorithm_name in args.algorithms:
            workload = build_workload(
                args.case,
                seed,
                args.threshold,
                args.link_parallelism,
            )
            algorithm = build_algorithm(algorithm_name, args.link_parallelism)
            run_dir = args.output / f"seed_{seed}" / algorithm.name
            runtime = SequenceRuntime(run_dir)
            started = time.perf_counter()
            result = algorithm.run(runtime, workload)
            elapsed = time.perf_counter() - started
            result.to_csv(str(run_dir / "results.csv"))
            (run_dir / "diagnostics.json").write_text(
                json.dumps(runtime.last_diagnostics, indent=2, sort_keys=True) + "\n"
            )
            run_diagnostics = _run_diagnostics(
                runtime.last_diagnostics,
                workload.memories_per_node,
            )
            for request in result.request_results:
                rows.append({
                    "seed": seed,
                    "algorithm": algorithm.name,
                    "request_id": request.request_id,
                    "success": request.success,
                    "time_to_serve_ms": request.time_to_serve_ms,
                    "fidelity": request.fidelity,
                    "failure_reason": request.failure_reason,
                    "runtime_s": elapsed,
                    "run_diagnostics": run_diagnostics,
                })

    summary = {
        algorithm_name: _summarize([
            row for row in rows if row["algorithm"] == algorithm_name
        ])
        for algorithm_name in args.algorithms
    }
    report = {
        "case": args.case,
        "threshold": args.threshold,
        "link_parallelism": args.link_parallelism,
        "seeds": args.seeds,
        "algorithms": args.algorithms,
        "qdc_commit": _git_revision(ROOT),
        "sequence_commit": _git_revision(ROOT.parent / "SeQUeNCe"),
        "summary": summary,
        "runs": rows,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
