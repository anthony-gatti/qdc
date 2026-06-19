"""
2D sweep for QPQ frontier characterization.

Runs the full (distance x seed) matrix for ODO and ACP, producing one unified
CSV that feeds charts 1, 2, 3, 5, 6. Also runs a separate db-size sub-sweep
that feeds chart 4.

Usage:
    python sweep2d.py --config config/default.yaml --output sweep2d_output
    python sweep2d.py --config config/default.yaml --output sweep2d_output --pilot

--pilot uses fewer seeds / distances for a fast sanity check before the full run.
"""

import os
import csv
import copy
import argparse
import json
import platform
import subprocess
import sys
import time as walltime
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict

import networkx as nx

from topology import (
    generate_hub_spoke_topology, save_topology, validate_topology
)
from workload import generate_qpq_queries
from backends.registry import get_backend
from results import OUTPUT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Hop distance computation (same as sweep.py — kept local for clarity)
# ---------------------------------------------------------------------------

def compute_hop_distances(topo_config: dict, qdc_name: str) -> Dict[str, int]:
    router_names = {n["name"] for n in topo_config["nodes"]
                    if n["type"] == "QuantumRouter"}
    bsm_to_routers = defaultdict(list)
    for qc in topo_config["qchannels"]:
        src, dst = qc["source"], qc["destination"]
        if dst.startswith("BSM"):
            bsm_to_routers[dst].append(src)

    graph = nx.Graph()
    graph.add_nodes_from(router_names)
    for bsm, routers in bsm_to_routers.items():
        if len(routers) == 2:
            graph.add_edge(routers[0], routers[1])

    if qdc_name not in graph:
        return {r: -1 for r in router_names}
    distances = nx.single_source_shortest_path_length(graph, qdc_name)
    return {r: distances.get(r, -1) for r in router_names}


# ---------------------------------------------------------------------------
# Build a single-cell config and run one backend
# ---------------------------------------------------------------------------

def run_one_cell(
    base_config: dict,
    distance_km: float,
    database_size_log: int,
    num_nodes: int,
    seed: int,
    backend_name: str,
    output_dir: str,
) -> List[dict]:
    """Run one (distance, db_size, num_nodes, seed, backend) cell.

    Returns a list of flat dict records, one per query.
    """
    config = copy.deepcopy(base_config)
    config["topology"]["num_nodes"] = num_nodes
    config["topology"]["inter_node_distance_m"] = distance_km * 1000.0
    config["topology"]["qdc_node_index"] = num_nodes // 2
    config["topology"]["random_seed"] = seed
    config["workload"]["database_size_log"] = database_size_log
    if config.get("diagnostics", {}).get("cache_lifecycle", False):
        diag_dir = os.path.join(output_dir, "cache_lifecycle")
        os.makedirs(diag_dir, exist_ok=True)
        config.setdefault("diagnostics", {})["cache_lifecycle_output"] = os.path.join(
            diag_dir,
            f"cache_d{int(distance_km)}_n{num_nodes}_nb{database_size_log}_s{seed}_{backend_name}.json",
        )

    hw = config.get("hardware", {})
    exp = config.get("experiment", {})
    wl = config.get("workload", {})
    topo_cfg = config["topology"]

    qdc_name = f"router_{topo_cfg['qdc_node_index']}"

    # Backend
    backend = get_backend(backend_name, config)

    # Backend-specific physical/topology mode.
    # For the current upstream SeQUeNCe ODO spike, use Bell diagonal + single heralded.
    if backend_name in ("odo", "sequence_bd", "odo_vanilla"):
        formalism = hw.get("formalism", "bell_diagonal")
        encoding_type = hw.get("encoding_type", "single_heralded")
    elif backend_name == "acp":
        formalism = hw.get("acp_formalism", "bell_diagonal")
        encoding_type = hw.get("acp_encoding_type", "single_heralded")
    else:
        formalism = hw.get("formalism", "bell_diagonal")
        encoding_type = hw.get("encoding_type", "single_heralded")

    # Build topology
    topo_config = generate_hub_spoke_topology(
        num_nodes=num_nodes,
        inter_node_distance_m=distance_km * 1000.0,
        memo_size=hw.get("memories_per_node", 50),
        adaptive_max_memory=backend.adaptive_max_memory,
        memory_fidelity=hw.get("link_fidelity", 0.99),
        memory_efficiency=hw.get("memory_efficiency", 0.5),
        coherence_time_s=hw.get("memory_coherence_time_s", 5.0),
        gate_fidelity=hw.get("gate_fidelity", 0.99),
        measurement_fidelity=hw.get("measurement_fidelity", 0.99),
        stop_time_s=exp.get("simulation_end_time_s", 180.0),
        seed=seed,
        extra_mesh_edges=topo_cfg.get("extra_mesh_edges", 3),
        qdc_node_index=topo_cfg["qdc_node_index"],
        encoding_type=encoding_type,
        formalism=formalism,
    )

    issues = validate_topology(topo_config)
    if issues:
        print(f"    WARNING: {len(issues)} topology issues")

    hop_distances = compute_hop_distances(topo_config, qdc_name)

    # Workload
    router_names = [n["name"] for n in topo_config["nodes"]
                    if n["type"] == "QuantumRouter"]

    # Need enough clients to get coverage across hop depths.
    # Default to all non-QDC routers.
    num_clients_target = wl.get("num_clients", 10)
    num_clients = min(num_clients_target, len(router_names) - 1)

    query_specs = generate_qpq_queries(
        router_names=router_names,
        qdc_name=qdc_name,
        num_clients=num_clients,
        queries_per_client=wl.get("queries_per_client", 3),
        database_size_log=database_size_log,
        fidelity_threshold=wl.get("fidelity_threshold", 0.7),
        round_deadline_s=wl.get("round_deadline_s", 5.0),
        request_period_s=wl.get("request_period_s", 6.0),
        start_offset_s=wl.get("start_offset_s", 2.0),
        reservation_duration_s=wl.get("reservation_duration_s", 5.0),
        seed=seed,
    )

    # Topology JSON
    backend_topo = copy.deepcopy(topo_config)
    for tmpl in backend_topo["templates"].values():
        tmpl["adaptive_max_memory"] = backend.adaptive_max_memory
    topo_path = os.path.join(
        output_dir,
        f"topo_d{int(distance_km)}_n{num_nodes}_nb{database_size_log}_s{seed}_{backend.name}.json"
    )
    save_topology(backend_topo, topo_path)

    # Run
    t0 = walltime.time()
    result = backend.run(topo_path, query_specs, config)
    elapsed = walltime.time() - t0

    # Flatten to records
    records = []
    for rr in result.request_results:
        hop = hop_distances.get(rr.src, -1)
        records.append({
            "backend": backend.name,
            "acp_memory_budget": backend.adaptive_max_memory,
            "seed": seed,
            "inter_node_distance_km": distance_km,
            "database_size_log": database_size_log,
            "num_nodes": num_nodes,
            "query_id": rr.request_id,
            "src": rr.src,
            "dst": rr.dst,
            "hop_distance": hop,
            "success": rr.success,
            "failure_reason": rr.failure_reason or "",
            "tts_ms": rr.time_to_serve_ms or 0.0,
            "fidelity": rr.fidelity if rr.fidelity is not None else 0.0,
            "first_pair_arrival_ms": (
                f"{rr.first_pair_arrival_ms:.3f}"
                if rr.first_pair_arrival_ms is not None else ""
            ),
            "round1_completion_ms": (
                f"{rr.round1_completion_ms:.3f}"
                if rr.round1_completion_ms is not None else ""
            ),
            "round2_completion_ms": (
                f"{rr.round2_completion_ms:.3f}"
                if rr.round2_completion_ms is not None else ""
            ),
            "round1_pairs": rr.round1_pairs,
            "round2_pairs": rr.round2_pairs,
            "expected_pairs": rr.expected_pairs,
            "pairs_rejected_fidelity": rr.pairs_rejected_fidelity,
            "delivered_background_pairs": rr.delivered_background_pairs,
            "delivered_application_pairs": rr.delivered_application_pairs,
            "delivered_pairs_with_background_contribution": rr.delivered_pairs_with_background_contribution,
            "delivered_pairs_fully_background_supported": rr.delivered_pairs_fully_background_supported,
            "delivered_pairs_partially_background_supported": rr.delivered_pairs_partially_background_supported,
            "delivered_pairs_fully_fresh": rr.delivered_pairs_fully_fresh,
            "delivered_background_elementary_edges": rr.delivered_background_elementary_edges,
            "delivered_fresh_elementary_edges": rr.delivered_fresh_elementary_edges,
            "pair_arrival_ms": ";".join(f"{t:.3f}" for t in rr.pair_arrival_ms),
            "walltime_s": elapsed,
        })

    n_success = sum(1 for r in records if r["success"])
    print(f"    {backend.name}: {n_success}/{len(records)} ok, walltime {elapsed:.1f}s")
    return records


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "backend", "acp_memory_budget", "seed", "inter_node_distance_km", "database_size_log",
    "num_nodes", "query_id", "src", "dst", "hop_distance",
    "success", "failure_reason", "tts_ms", "fidelity",
    "first_pair_arrival_ms", "round1_completion_ms", "round2_completion_ms",
    "round1_pairs", "round2_pairs", "expected_pairs",
    "pairs_rejected_fidelity", "delivered_background_pairs",
    "delivered_application_pairs",
    "delivered_pairs_with_background_contribution",
    "delivered_pairs_fully_background_supported",
    "delivered_pairs_partially_background_supported",
    "delivered_pairs_fully_fresh",
    "delivered_background_elementary_edges",
    "delivered_fresh_elementary_edges",
    "pair_arrival_ms",
    "walltime_s",
]


def _git_commit(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _resolve_sequence_commit() -> str:
    candidates = [
        os.environ.get("QDC_SEQUENCE_DIR"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SeQUeNCe")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SeQUeNCe")),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, ".git")):
            return _git_commit(candidate)
    return "unknown"


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _backend_budget(backend_name: str, config: dict) -> int:
    name = backend_name.lower()
    if name in ("odo", "sequence_bd", "odo_vanilla"):
        return 0
    if name.startswith("acp_m") and name[5:].isdigit():
        return int(name[5:])
    if name in ("acp", "acp_no_bg"):
        return int(config.get("hardware", {}).get("acp_memory", 8))
    return 0


def write_manifest(
    output_dir: str,
    config_path: str,
    base_config: dict,
    distances_km: List[float],
    seeds: List[int],
    db_sizes: List[int],
    num_nodes: int,
    backends: List[str],
    skip_primary: bool,
    skip_dbsize: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "schema_version": "1",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "config_path": config_path,
        "resolved_config": base_config,
        "resolved_sweep": {
            "distances_km": distances_km,
            "seeds": seeds,
            "db_sizes": db_sizes,
            "num_nodes": num_nodes,
            "backends": [
                {
                    "requested_name": name,
                    "resolved_memory_budget": _backend_budget(name, base_config),
                }
                for name in backends
            ],
            "skip_primary": skip_primary,
            "skip_dbsize": skip_dbsize,
        },
        "git": {
            "qdc_commit": _git_commit(os.path.dirname(__file__)),
            "sequence_commit": _resolve_sequence_commit(),
        },
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "qdc_acp_dir": os.environ.get("QDC_ACP_DIR", ""),
            "qdc_sequence_dir": os.environ.get("QDC_SEQUENCE_DIR", ""),
        },
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Manifest wrote {path}")
    return path


def run_primary_sweep(
    base_config: dict,
    distances_km: List[float],
    seeds: List[int],
    num_nodes: int,
    database_size_log: int,
    backends: List[str],
    output_dir: str,
) -> str:
    """Primary (distance x seed) sweep. Returns path to the unified CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "primary_sweep.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        total_cells = len(distances_km) * len(seeds) * len(backends)
        cell_idx = 0

        for distance_km in distances_km:
            for seed in seeds:
                for backend_name in backends:
                    cell_idx += 1
                    print(f"[{cell_idx}/{total_cells}] dist={distance_km}km "
                          f"seed={seed} backend={backend_name}")
                    try:
                        records = run_one_cell(
                            base_config=base_config,
                            distance_km=distance_km,
                            database_size_log=database_size_log,
                            num_nodes=num_nodes,
                            seed=seed,
                            backend_name=backend_name,
                            output_dir=output_dir,
                        )
                        for r in records:
                            writer.writerow(r)
                        f.flush()
                    except Exception as e:
                        print(f"    FAILED: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\nPrimary sweep wrote {csv_path}")
    return csv_path


def run_dbsize_sweep(
    base_config: dict,
    db_sizes: List[int],
    seeds: List[int],
    distance_km: float,
    num_nodes: int,
    backends: List[str],
    output_dir: str,
) -> str:
    """Database-size sub-sweep. Returns path to its CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dbsize_sweep.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        total_cells = len(db_sizes) * len(seeds) * len(backends)
        cell_idx = 0

        for db_size in db_sizes:
            for seed in seeds:
                for backend_name in backends:
                    cell_idx += 1
                    print(f"[{cell_idx}/{total_cells}] db_log={db_size} "
                          f"seed={seed} backend={backend_name}")
                    try:
                        records = run_one_cell(
                            base_config=base_config,
                            distance_km=distance_km,
                            database_size_log=db_size,
                            num_nodes=num_nodes,
                            seed=seed,
                            backend_name=backend_name,
                            output_dir=output_dir,
                        )
                        for r in records:
                            writer.writerow(r)
                        f.flush()
                    except Exception as e:
                        print(f"    FAILED: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\nDB-size sweep wrote {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import yaml

    parser = argparse.ArgumentParser(description="QPQ 2D sweep")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output", type=str, default="sweep2d_output")
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated backend list. Defaults to experiment.backends in the config.",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default=None,
        help="Comma-separated primary-sweep distances in km.",
    )
    parser.add_argument(
        "--db-sizes",
        type=str,
        default=None,
        help="Comma-separated database-size sweep n values.",
    )
    parser.add_argument("--pilot", action="store_true",
                        help="Fast pilot: 3 seeds, 2 distances")
    parser.add_argument("--skip-primary", action="store_true")
    parser.add_argument("--skip-dbsize", action="store_true")
    parser.add_argument(
        "--cache-diagnostics",
        action="store_true",
        help="Write per-cell ACP cache lifecycle diagnostics under the output directory.",
    )
    parser.add_argument("--num-nodes", type=int, default=25,
                        help="Nodes for primary sweep (default: 25, gives hops 1-7)")
    parser.add_argument("--seeds", type=int, default=15,
                        help="Number of seeds per cell (default: 15)")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="First seed value (default: 42)")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    base_config.setdefault("workload", {})["mode"] = "qpq"
    if args.cache_diagnostics:
        base_config.setdefault("diagnostics", {})["cache_lifecycle"] = True

    exp_cfg = base_config.get("experiment", {})
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    else:
        backends = exp_cfg.get("backends", ["odo"])

    if args.pilot:
        distances_km = [10.0, 40.0]
        seeds = list(range(args.seed_start, args.seed_start + 3))  # 3 seeds
        db_sizes = [10, 20]
        print("=== PILOT MODE ===")
    else:
        distances_km = exp_cfg.get("distances_km", [10.0, 20.0, 30.0, 40.0])
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        db_sizes = exp_cfg.get("db_sizes", [5, 10, 15, 20])

    if args.distances:
        distances_km = _parse_float_list(args.distances)
    if args.db_sizes:
        db_sizes = _parse_int_list(args.db_sizes)

    print(f"Distances: {distances_km}")
    print(f"Seeds: {seeds}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"DB sizes (sub-sweep): {db_sizes}")
    print(f"Output dir: {args.output}")
    print()

    write_manifest(
        output_dir=args.output,
        config_path=args.config,
        base_config=base_config,
        distances_km=distances_km,
        seeds=seeds,
        db_sizes=db_sizes,
        num_nodes=args.num_nodes,
        backends=backends,
        skip_primary=args.skip_primary,
        skip_dbsize=args.skip_dbsize,
    )

    total_t0 = walltime.time()

    if not args.skip_primary:
        print("=" * 70)
        print("PRIMARY SWEEP (distance x seed)")
        print("=" * 70)
        run_primary_sweep(
            base_config=base_config,
            distances_km=distances_km,
            seeds=seeds,
            num_nodes=args.num_nodes,
            database_size_log=base_config.get("workload", {}).get(
                "database_size_log", 10),
            backends=backends,
            output_dir=args.output,
        )

    if not args.skip_dbsize:
        print("\n" + "=" * 70)
        print("DB-SIZE SUB-SWEEP (fixed 20km, sweep n)")
        print("=" * 70)
        run_dbsize_sweep(
            base_config=base_config,
            db_sizes=db_sizes,
            seeds=seeds,
            distance_km=20.0,
            num_nodes=args.num_nodes,
            backends=backends,
            output_dir=args.output,
        )

    elapsed = walltime.time() - total_t0
    print(f"\nTotal walltime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
