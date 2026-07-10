#!/usr/bin/env python3
"""Controlled fresh-generation yield comparison across supported backends."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.registry import get_backend
from common import SECOND
from topology import generate_hub_spoke_topology, generate_linear_topology, save_topology


def _config(diagnostic_path: Path, budget: int, router_names: list[str]) -> dict:
    return {
        "hardware": {
            "link_fidelity": 0.99,
            "memory_efficiency": 1.0,
            "memory_coherence_time_s": 100.0,
            "memories_per_node": 40,
            "gate_fidelity": 1.0,
            "measurement_fidelity": 1.0,
            "formalism": "bell_diagonal",
            "encoding_type": "single_heralded",
            "acp_formalism": "bell_diagonal",
            "acp_encoding_type": "single_heralded",
            "acp_memory": budget,
        },
        "workload": {"mode": "qpq"},
        "experiment": {"simulation_end_time_s": 45.0},
        "diagnostics": {
            "application_demand": True,
            "application_demand_output": str(diagnostic_path),
            # ACP still executes its control loop, but cannot create a cached pair.
            "force_probability_table": {name: {"": 1.0} for name in router_names},
        },
        "topology": {"random_seed": 19},
    }


def _query(destination: str, database_size_log: int) -> list[dict]:
    start = int(0.5 * SECOND)
    return [{
        "query_id": 0,
        "src": "router_0",
        "dst": destination,
        "start_time": start,
        "end_time": start + int(40 * SECOND),
        "database_size_log": database_size_log,
        "fidelity": 0.5,
        "round_deadline_ps": int(20 * SECOND),
    }]


def _run_case(outdir: Path, node_count: int, backend_name: str, database_size_log: int) -> dict:
    backend = get_backend(backend_name, {"hardware": {"acp_memory": 1 if backend_name == "acp_m1" else 6}})
    budget = backend.adaptive_max_memory
    generator = generate_hub_spoke_topology if node_count == 2 else generate_linear_topology
    topology = generator(
        num_nodes=node_count,
        inter_node_distance_m=40_000,
        memo_size=40,
        adaptive_max_memory=budget,
        memory_fidelity=0.99,
        memory_efficiency=1.0,
        coherence_time_s=100.0,
        gate_fidelity=1.0,
        measurement_fidelity=1.0,
        stop_time_s=45.0,
        seed=19,
        extra_mesh_edges=0,
        qdc_node_index=node_count - 1,
        encoding_type="single_heralded",
        formalism="bell_diagonal",
    )
    case = f"{node_count}node_{backend_name}"
    topology_path = outdir / f"{case}.topology.json"
    diagnostic_path = outdir / f"{case}.demand.json"
    save_topology(topology, str(topology_path))
    config = _config(diagnostic_path, budget, [f"router_{i}" for i in range(node_count)])
    started = time.monotonic()
    result = backend.run(
        str(topology_path),
        _query(f"router_{node_count - 1}", database_size_log),
        config,
    )
    runtime = time.monotonic() - started
    diagnostic = json.loads(diagnostic_path.read_text())
    totals = diagnostic["totals"]
    attempts = totals.get("bsm_attempts", 0)
    successes = totals.get("successful_physical_pairs", 0)
    completed = totals.get("completed_physical_attempts", 0)
    return {
        "case": case,
        "backend": backend.name,
        "nodes": node_count,
        "application_edge_demand": totals.get("elementary_pair_demand", 0),
        "protocol_request_objects": totals.get("protocol_request_objects", 0),
        "unique_protocol_instances": totals.get("unique_protocol_instances", 0),
        "protocol_start_calls": totals.get("protocol_start_calls", 0),
        "starts_that_schedule_an_emission": totals.get("starts_that_schedule_an_emission", 0),
        "protocol_callbacks_that_schedule_emission": totals.get("protocol_callbacks_that_schedule_emission", 0),
        "endpoint_emissions": totals.get("endpoint_emissions", 0),
        "bsm_attempts": attempts,
        "completed_physical_attempts": completed,
        "successful_physical_pairs": successes,
        "success_per_bsm_attempt": successes / attempts if attempts else None,
        "success_per_completed_attempt": successes / completed if completed else None,
        "query_success": bool(result.request_results and result.request_results[0].success),
        "runtime_s": runtime,
        "input_fingerprints": diagnostic["input_fingerprints"],
        "physical_snapshots": diagnostic["physical_snapshots"],
        "diagnostic_path": str(diagnostic_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/qdc_fresh_yield")
    parser.add_argument("--database-size-log", type=int, default=5)
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(outdir, nodes, backend, args.database_size_log)
        for nodes in (2, 3)
        for backend in ("acp_no_bg", "acp_m1", "acp_m6")
    ]
    summary = {"results": results}
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
