#!/usr/bin/env python3
"""Tiny deterministic ACP cache-consumption diagnostics."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.acp_backend import ACPBackend
from common import SECOND
from topology import generate_hub_spoke_topology, generate_linear_topology, save_topology


def query(src: str, dst: str, start_s: float, duration_s: float = 5.0, query_id: int = 0, database_size_log: int = 0) -> list[dict]:
    start = int(start_s * SECOND)
    return [{
        "query_id": query_id,
        "src": src,
        "dst": dst,
        "start_time": start,
        "end_time": start + int(duration_s * SECOND),
        "database_size_log": database_size_log,
        "fidelity": 0.7,
        "round_deadline_ps": int(duration_s * SECOND),
    }]


def base_config(output_path: str, force_probability_table: dict) -> dict:
    return {
        "hardware": {
            "link_fidelity": 0.99,
            "memory_efficiency": 1.0,
            "memory_coherence_time_s": 100.0,
            "memories_per_node": 12,
            "gate_fidelity": 0.99,
            "measurement_fidelity": 0.99,
            "purify": False,
            "formalism": "bell_diagonal",
            "encoding_type": "single_heralded",
            "acp_formalism": "bell_diagonal",
            "acp_encoding_type": "single_heralded",
        },
        "workload": {
            "mode": "qpq",
            "database_size_log": 0,
            "num_clients": 1,
            "queries_per_client": 1,
            "round_deadline_s": 5.0,
            "reservation_duration_s": 5.0,
            "request_period_s": 1.0,
            "start_offset_s": 1.5,
            "fidelity_threshold": 0.7,
        },
        "experiment": {
            "simulation_end_time_s": 8.0,
        },
        "diagnostics": {
            "cache_lifecycle": True,
            "cache_lifecycle_output": output_path,
            "force_probability_table": force_probability_table,
        },
        "topology": {
            "random_seed": 7,
        },
    }


def summarize(result, diag_path: str) -> dict:
    diag = json.load(open(diag_path))
    request_results = result.request_results
    first = request_results[0] if request_results else None
    snapshots = diag.get("snapshots", [])
    max_available = max((s["local_pair_records"] for s in snapshots), default=0)
    max_useful = max((s["useful_local_pair_records"] for s in snapshots), default=0)
    max_memory_used = max(
        (
            max(s["adaptive_memory_by_node"].values())
            for s in snapshots
            if s["adaptive_memory_by_node"]
        ),
        default=0,
    )
    return {
        "backend": result.backend_name,
        "request_count": len(request_results),
        "success_count": sum(1 for r in request_results if r.success),
        "success": first.success if first else False,
        "failure_reason": first.failure_reason if first else "no_result",
        "first_pair_arrival_ms": first.first_pair_arrival_ms if first else None,
        "round1_completion_ms": first.round1_completion_ms if first else None,
        "round2_completion_ms": first.round2_completion_ms if first else None,
        "delivered_background_pairs": sum(r.delivered_background_pairs for r in request_results),
        "delivered_application_pairs": sum(r.delivered_application_pairs for r in request_results),
        "delivered_pairs_with_background_contribution": sum(
            r.delivered_pairs_with_background_contribution for r in request_results
        ),
        "delivered_pairs_fully_background_supported": sum(
            r.delivered_pairs_fully_background_supported for r in request_results
        ),
        "delivered_pairs_partially_background_supported": sum(
            r.delivered_pairs_partially_background_supported for r in request_results
        ),
        "delivered_pairs_fully_fresh": sum(r.delivered_pairs_fully_fresh for r in request_results),
        "delivered_background_elementary_edges": sum(
            r.delivered_background_elementary_edges for r in request_results
        ),
        "delivered_fresh_elementary_edges": sum(
            r.delivered_fresh_elementary_edges for r in request_results
        ),
        "max_pregenerated_pair_records_at_request_start": max_available,
        "max_useful_pair_records_at_request_start": max_useful,
        "max_adaptive_memory_used_at_snapshot": max_memory_used,
        "counters": diag.get("counters", {}),
        "diagnostic_path": diag_path,
    }


def run_case(
    name: str,
    outdir: Path,
    topo: dict,
    query_specs: list[dict],
    backend: ACPBackend,
    force_probability_table: dict,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    topo_path = outdir / f"{name}.topology.json"
    diag_path = outdir / f"{name}.cache_lifecycle.json"
    save_topology(topo, str(topo_path))
    config = base_config(str(diag_path), force_probability_table)
    config["topology"]["random_seed"] = 7
    result = backend.run(str(topo_path), query_specs, config)
    summary = summarize(result, str(diag_path))
    summary["scenario"] = name
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/qdc_cache_consumption_diagnostic")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    summaries = []

    useful_topo = generate_hub_spoke_topology(
        num_nodes=2,
        inter_node_distance_m=100.0,
        memo_size=12,
        adaptive_max_memory=1,
        memory_fidelity=0.99,
        memory_efficiency=1.0,
        coherence_time_s=100.0,
        stop_time_s=8.0,
        seed=7,
        extra_mesh_edges=0,
        qdc_node_index=1,
        encoding_type="single_heralded",
        formalism="bell_diagonal",
    )
    useful_force = {
        "router_0": {"router_1": 1.0, "": 0.0},
        "router_1": {"router_0": 1.0, "": 0.0},
    }
    summaries.append(run_case(
        "useful_acp_m1",
        outdir,
        useful_topo,
        query("router_0", "router_1", start_s=1.5),
        ACPBackend(adaptive_max_memory=1, name_override="acp_m1"),
        useful_force,
    ))
    summaries.append(run_case(
        "no_background",
        outdir,
        useful_topo,
        query("router_0", "router_1", start_s=1.5),
        ACPBackend(adaptive_max_memory=1, background_enabled=False, name_override="acp_no_bg"),
        useful_force,
    ))
    summaries.append(run_case(
        "mixed_cached_fresh",
        outdir,
        useful_topo,
        query("router_0", "router_1", start_s=1.5, database_size_log=1),
        ACPBackend(adaptive_max_memory=1, name_override="acp_m1"),
        useful_force,
    ))

    mismatch_topo = generate_linear_topology(
        num_nodes=3,
        inter_node_distance_m=100.0,
        memo_size=12,
        adaptive_max_memory=1,
        memory_fidelity=0.99,
        memory_efficiency=1.0,
        coherence_time_s=100.0,
        stop_time_s=8.0,
        seed=7,
        extra_mesh_edges=0,
        qdc_node_index=1,
        encoding_type="single_heralded",
        formalism="bell_diagonal",
    )
    mismatch_force = {
        "router_0": {"": 1.0},
        "router_1": {"router_2": 1.0, "": 0.0},
        "router_2": {"router_1": 1.0, "": 0.0},
    }
    summaries.append(run_case(
        "mismatched_demand",
        outdir,
        mismatch_topo,
        query("router_0", "router_1", start_s=1.5),
        ACPBackend(adaptive_max_memory=1, name_override="acp_m1"),
        mismatch_force,
    ))

    swap_topo_all_cached = generate_linear_topology(
        num_nodes=3,
        inter_node_distance_m=100.0,
        memo_size=12,
        adaptive_max_memory=6,
        memory_fidelity=0.99,
        memory_efficiency=1.0,
        coherence_time_s=100.0,
        stop_time_s=8.0,
        seed=7,
        extra_mesh_edges=0,
        qdc_node_index=2,
        encoding_type="single_heralded",
        formalism="bell_diagonal",
    )
    swap_topo_one_edge = generate_linear_topology(
        num_nodes=3,
        inter_node_distance_m=100.0,
        memo_size=12,
        adaptive_max_memory=1,
        memory_fidelity=0.99,
        memory_efficiency=1.0,
        coherence_time_s=100.0,
        stop_time_s=8.0,
        seed=7,
        extra_mesh_edges=0,
        qdc_node_index=2,
        encoding_type="single_heralded",
        formalism="bell_diagonal",
    )
    swap_force = {
        "router_0": {"router_1": 1.0, "": 0.0},
        "router_1": {"router_0": 0.5, "router_2": 0.5, "": 0.0},
        "router_2": {"router_1": 1.0, "": 0.0},
    }
    summaries.append(run_case(
        "three_node_swap_all_cached",
        outdir,
        swap_topo_all_cached,
        query("router_0", "router_2", start_s=4.0),
        ACPBackend(adaptive_max_memory=6, name_override="acp_m6"),
        swap_force,
    ))

    one_edge_force = {
        "router_0": {"router_1": 1.0, "": 0.0},
        "router_1": {"router_0": 1.0, "": 0.0},
        "router_2": {"": 1.0},
    }
    summaries.append(run_case(
        "three_node_one_cached_edge",
        outdir,
        swap_topo_one_edge,
        query("router_0", "router_2", start_s=1.5),
        ACPBackend(adaptive_max_memory=1, name_override="acp_m1"),
        one_edge_force,
    ))

    contention_queries = (
        query("router_0", "router_1", start_s=1.5, query_id=0)
        + query("router_0", "router_1", start_s=1.500000001, query_id=1)
    )
    summaries.append(run_case(
        "contention_one_cached_pair",
        outdir,
        useful_topo,
        contention_queries,
        ACPBackend(adaptive_max_memory=1, name_override="acp_m1"),
        useful_force,
    ))

    budget_force = useful_force
    for budget in (1, 6):
        budget_topo = generate_hub_spoke_topology(
            num_nodes=2,
            inter_node_distance_m=100.0,
            memo_size=12,
            adaptive_max_memory=budget,
            memory_fidelity=0.99,
            memory_efficiency=1.0,
            coherence_time_s=100.0,
            stop_time_s=8.0,
            seed=7,
            extra_mesh_edges=0,
            qdc_node_index=1,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
        )
        summaries.append(run_case(
            f"budget_m{budget}",
            outdir,
            budget_topo,
            query("router_0", "router_1", start_s=4.0),
            ACPBackend(adaptive_max_memory=budget, name_override=f"acp_m{budget}"),
            budget_force,
        ))

    by_name = {summary["scenario"]: summary for summary in summaries}
    checks = {
        "useful_cache_inventory_present": (
            by_name["useful_acp_m1"]["max_useful_pair_records_at_request_start"] > 0
        ),
        "useful_cache_consumed_by_application": (
            by_name["useful_acp_m1"]["delivered_pairs_with_background_contribution"] > 0
            and by_name["useful_acp_m1"]["counters"]["background_pairs_consumed_by_app"] > 0
        ),
        "no_background_has_no_background_pairs": (
            by_name["no_background"]["counters"]["background_generation_successes"] == 0
            and by_name["no_background"]["delivered_pairs_with_background_contribution"] == 0
        ),
        "mixed_cached_and_fresh_delivery": (
            by_name["mixed_cached_fresh"]["delivered_pairs_with_background_contribution"] > 0
            and by_name["mixed_cached_fresh"]["delivered_pairs_fully_fresh"] > 0
        ),
        "mismatched_inventory_not_useful": (
            by_name["mismatched_demand"]["max_pregenerated_pair_records_at_request_start"] > 0
            and by_name["mismatched_demand"]["max_useful_pair_records_at_request_start"] == 0
            and by_name["mismatched_demand"]["delivered_pairs_with_background_contribution"] == 0
        ),
        "three_node_swap_delivers_background": (
            by_name["three_node_swap_all_cached"]["success"]
            and by_name["three_node_swap_all_cached"]["delivered_pairs_with_background_contribution"] > 0
            and by_name["three_node_swap_all_cached"]["delivered_pairs_fully_background_supported"] > 0
            and by_name["three_node_swap_all_cached"]["counters"]["background_pairs_consumed_by_app"] >= 2
        ),
        "one_cached_edge_swapping_succeeds": (
            by_name["three_node_one_cached_edge"]["success"]
            and by_name["three_node_one_cached_edge"]["delivered_pairs_with_background_contribution"] > 0
            and by_name["three_node_one_cached_edge"]["delivered_pairs_partially_background_supported"] > 0
            and by_name["three_node_one_cached_edge"]["delivered_pairs_fully_fresh"] > 0
        ),
        "contention_consumes_at_most_one_cached_pair": (
            by_name["contention_one_cached_pair"]["counters"]["background_pairs_consumed_by_app"] <= 1
            and by_name["contention_one_cached_pair"]["success_count"] >= 1
        ),
        "m6_uses_more_adaptive_memory_than_m1": (
            by_name["budget_m6"]["max_adaptive_memory_used_at_snapshot"]
            > by_name["budget_m1"]["max_adaptive_memory_used_at_snapshot"]
        ),
    }
    output = {
        "checks": checks,
        "scenarios": summaries,
    }
    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
