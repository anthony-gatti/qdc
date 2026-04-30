"""
Generates SeQUeNCe-compatible JSON topology configs for QPQ evaluation.

Produces hub-and-spoke topologies with a QDC server at the center,
compatible with both RouterNetTopo and RouterNetTopoAdaptive.

Output JSON matches the format used by the ACP paper's line_5-m4.json:
explicit nodes, qchannels, cchannels.
"""

import json
import math
import itertools
from typing import Optional
import networkx as nx
import numpy as np


# sequence uses picoseconds
SECOND = int(1e12)
MILLISECOND = int(1e9)
SPEED_OF_LIGHT = 2e-4  #km per picosecond in fiber

def generate_hub_spoke_topology(
    num_nodes: int,
    inter_node_distance_m: float = 1000.0,
    memo_size: int = 10,
    adaptive_max_memory: int = 0,
    memory_fidelity: float = 0.99,
    memory_efficiency: float = 0.5,
    coherence_time_s: float = 5.0,
    gate_fidelity: float = 0.99,
    measurement_fidelity: float = 0.99,
    attenuation: float = 0.0002,
    stop_time_s: float = 60.0,
    seed: int = 42,
    extra_mesh_edges: int = 0,
    qdc_node_index: Optional[int] = None,
) -> dict:
    """Generate a hub-and-spoke topology with optional extra mesh edges.

    The QDC server sits at the center. Clients are arranged in a tree/mesh
    expanding outward. Every adjacent router pair gets a BSM node between them.

    Args:
        num_nodes: total number of quantum router nodes (including QDC)
        inter_node_distance_m: physical distance between adjacent nodes (meters)
            Each router is inter_node_distance_m/2 from the BSM between them.
        memo_size: quantum memories per router (total, shared between app + ACP)
        adaptive_max_memory: memories reserved for ACP (0 = ODO baseline)
        memory_fidelity: raw fidelity of generated entanglement
        memory_efficiency: probability of successful photon emission
        coherence_time_s: memory coherence time in seconds
        gate_fidelity: gate operation fidelity
        measurement_fidelity: measurement fidelity
        attenuation: fiber attenuation (loss per m)
        stop_time_s: simulation stop time in seconds
        seed: random seed for topology generation
        extra_mesh_edges: additional random edges beyond the spanning tree
        qdc_node_index: which node is the QDC hub (default: center node)

    Returns:
        dict: SeQUeNCe-compatible topology config
    """
    rng = np.random.default_rng(seed)

    # Build graph
    if num_nodes <= 1:
        raise ValueError("Need at least 2 nodes")

    # start with a spanning tree rooted at the QDC hub
    if qdc_node_index is None:
        qdc_node_index = num_nodes // 2

    graph = _build_hub_spoke_tree(num_nodes, qdc_node_index, rng)

    # add extra mesh edges for richer connectivity
    _add_random_edges(graph, extra_mesh_edges, rng)

    # Node names
    router_names = [f"router_{i}" for i in range(num_nodes)]

    # Template
    template = {
        "MemoryArray": {
            "fidelity": memory_fidelity,
            "efficiency": memory_efficiency,
            "coherence_time": coherence_time_s,
        },
        "adaptive_max_memory": adaptive_max_memory,
        "encoding_type": "single_heralded",
        "decoherence_errors": [1/3, 1/3, 1/3],
    }

    # Router nodes
    nodes = []
    for i in range(num_nodes):
        nodes.append({
            "name": router_names[i],
            "type": "QuantumRouter",
            "seed": int(rng.integers(0, 2**31)),
            "memo_size": memo_size,
            "template": "default_template",
            "gate_fidelity": gate_fidelity,
            "measurement_fidelity": measurement_fidelity,
        })

    # BSM nodes + quantum channels
    bsm_nodes = []
    qchannels = []
    half_distance = inter_node_distance_m / 2.0

    for u, v in graph.edges():
        # ordering so BSM name is deterministic
        a, b = min(u, v), max(u, v)
        bsm_name = f"BSM_{a}_{b}"

        bsm_nodes.append({
            "name": bsm_name,
            "type": "BSMNode",
            "seed": int(rng.integers(0, 2**31)),
            "template": "default_template",
        })

        # Two quantum channels per link: each router to BSM
        for router_idx in [a, b]:
            qchannels.append({
                "source": router_names[router_idx],
                "destination": bsm_name,
                "distance": half_distance,
                "attenuation": attenuation,
            })

    # --- Classical channels ---
    # Need bidirectional between each router and adjacent BSMs + full mesh between all router pairs
    cchannels = []

    # 1) Router + BSM channels
    for u, v in graph.edges():
        a, b = min(u, v), max(u, v)
        bsm_name = f"BSM_{a}_{b}"
        cc_delay = half_distance / SPEED_OF_LIGHT  # propagation delay

        for router_idx in [a, b]:
            rname = router_names[router_idx]
            # router to BSM
            cchannels.append({
                "source": rname,
                "destination": bsm_name,
                "delay": cc_delay,
            })
            # BSM to router
            cchannels.append({
                "source": bsm_name,
                "destination": rname,
                "delay": cc_delay,
            })

    # 2) Full mesh between all routers
    # Uses distance / speed_of_light
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(
        graph, weight=lambda u, v, d: inter_node_distance_m
    ))

    for i, j in itertools.permutations(range(num_nodes), 2):
        dist = shortest_paths[i].get(j, num_nodes * inter_node_distance_m)
        delay = dist / SPEED_OF_LIGHT
        cchannels.append({
            "source": router_names[i],
            "destination": router_names[j],
            "delay": delay,
        })

    # Assemble config
    all_nodes = nodes + bsm_nodes

    config = {
        "templates": {
            "default_template": template,
        },
        "nodes": all_nodes,
        "qchannels": qchannels,
        "cchannels": cchannels,
        "stop_time": stop_time_s * SECOND,
        "is_parallel": False,
    }

    return config


def _build_hub_spoke_tree(
    num_nodes: int,
    hub_index: int,
    rng: np.random.Generator,
) -> nx.Graph:
    """Build a tree rooted at hub_index using random BFS-like expansion.

    Produces a tree where the hub has higher degree and nodes further
    from the hub have lower degree — represents a QDC at the center
    with clients expanding outward.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    if num_nodes == 2:
        graph.add_edge(0, 1)
        return graph

    # BFS-like tree: each layer adds children to frontier nodes
    connected = {hub_index}
    frontier = [hub_index]
    remaining = set(range(num_nodes)) - connected

    while remaining:
        next_frontier = []
        for parent in frontier:
            if not remaining:
                break
            # Each frontier node gets 1-3 children
            n_children = min(
                int(rng.integers(1, 4)),
                len(remaining)
            )
            children = list(rng.choice(list(remaining), size=n_children, replace=False))
            for child in children:
                graph.add_edge(parent, child)
                connected.add(child)
                remaining.discard(child)
                next_frontier.append(child)
        frontier = next_frontier if next_frontier else list(connected)

    return graph


def _add_random_edges(
    graph: nx.Graph,
    num_edges: int,
    rng: np.random.Generator,
) -> None:
    """Add random edges to an existing graph (for mesh connectivity)."""
    nodes = list(graph.nodes())
    n = len(nodes)
    added = 0
    attempts = 0
    max_attempts = num_edges * 10

    while added < num_edges and attempts < max_attempts:
        u, v = rng.choice(nodes, size=2, replace=False)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            added += 1
        attempts += 1


def generate_linear_topology(
    num_nodes: int, **kwargs
) -> dict:
    """Generate a simple linear chain topology.

    Good for comparing against ACP paper's 5-node linear results.
    """
    kwargs.setdefault("inter_node_distance_m", 1000.0)
    kwargs.setdefault("extra_mesh_edges", 0)
    kwargs.setdefault("qdc_node_index", num_nodes // 2)

    config = generate_hub_spoke_topology(num_nodes=num_nodes, **kwargs)

    # rebuild as simple linear chain
    # a little hacky but fine
    rng = np.random.default_rng(kwargs.get("seed", 42))

    # Clear existing BSM/qchannel/cchannel data and rebuild
    router_names = [f"router_{i}" for i in range(num_nodes)]
    template = config["templates"]["default_template"]
    inter_node_distance_m = kwargs["inter_node_distance_m"]
    half_distance = inter_node_distance_m / 2.0
    attenuation = kwargs.get("attenuation", 0.0002)

    # Router nodes (keep from config)
    router_nodes = [n for n in config["nodes"] if n["type"] == "QuantumRouter"]

    # BSM nodes: one between each adjacent pair
    bsm_nodes = []
    qchannels = []
    for i in range(num_nodes - 1):
        bsm_name = f"BSM_{i}_{i+1}"
        bsm_nodes.append({
            "name": bsm_name,
            "type": "BSMNode",
            "seed": i,
            "template": "default_template",
        })
        for router_idx in [i, i + 1]:
            qchannels.append({
                "source": router_names[router_idx],
                "destination": bsm_name,
                "distance": half_distance,
                "attenuation": attenuation,
            })

    # Classical channels: full mesh + router to BSMs
    cchannels = []
    cc_delay = half_distance / SPEED_OF_LIGHT

    for i in range(num_nodes - 1):
        bsm_name = f"BSM_{i}_{i+1}"
        for router_idx in [i, i + 1]:
            rname = router_names[router_idx]
            cchannels.append({"source": rname, "destination": bsm_name, "delay": cc_delay})
            cchannels.append({"source": bsm_name, "destination": rname, "delay": cc_delay})

    # Full router mesh
    for i, j in itertools.permutations(range(num_nodes), 2):
        hop_dist = abs(i - j) * inter_node_distance_m
        delay = hop_dist / SPEED_OF_LIGHT
        cchannels.append({
            "source": router_names[i],
            "destination": router_names[j],
            "delay": delay,
        })

    config["nodes"] = router_nodes + bsm_nodes
    config["qchannels"] = qchannels
    config["cchannels"] = cchannels

    return config


def save_topology(config: dict, path: str) -> None:
    """Write topology config to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def validate_topology(config: dict) -> list:
    """Basic validation of a topology config. Returns list of issues."""
    issues = []

    router_names = {n["name"] for n in config["nodes"] if n["type"] == "QuantumRouter"}
    bsm_names = {n["name"] for n in config["nodes"] if n["type"] == "BSMNode"}
    all_names = router_names | bsm_names

    # Check quantum channels: source must be router, dest must be BSM
    for qc in config.get("qchannels", []):
        if qc["source"] not in router_names:
            issues.append(f"qchannel source '{qc['source']}' is not a router")
        if qc["destination"] not in bsm_names:
            issues.append(f"qchannel dest '{qc['destination']}' is not a BSM node")

    # Check classical channels: all endpoints must exist
    for cc in config.get("cchannels", []):
        if cc["source"] not in all_names:
            issues.append(f"cchannel source '{cc['source']}' not found")
        if cc["destination"] not in all_names:
            issues.append(f"cchannel dest '{cc['destination']}' not found")

    # Check full mesh: every router pair should have bidirectional cchannels
    cc_pairs = {(cc["source"], cc["destination"]) for cc in config.get("cchannels", [])}
    for r1, r2 in itertools.permutations(router_names, 2):
        if (r1, r2) not in cc_pairs:
            issues.append(f"Missing cchannel: {r1} → {r2}")

    # Check BSM connectivity: each BSM should have exactly 2 qchannels pointing to it
    for bsm in bsm_names:
        count = sum(1 for qc in config["qchannels"] if qc["destination"] == bsm)
        if count != 2:
            issues.append(f"BSM '{bsm}' has {count} incoming qchannels (expected 2)")

    return issues


# main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QPQ evaluation topology")
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--linear", action="store_true", help="Linear chain instead of hub-spoke")
    parser.add_argument("--distance", type=float, default=1000.0, help="Inter-node distance (meters)")
    parser.add_argument("--memo-size", type=int, default=10)
    parser.add_argument("--acp-memory", type=int, default=0, help="ACP memory budget (0=ODO)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="topology.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    gen_func = generate_linear_topology if args.linear else generate_hub_spoke_topology
    config = gen_func(
        num_nodes=args.nodes,
        inter_node_distance_m=args.distance,
        memo_size=args.memo_size,
        adaptive_max_memory=args.acp_memory,
        seed=args.seed,
    )

    if args.validate:
        issues = validate_topology(config)
        if issues:
            print(f"Validation found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Topology valid.")

    save_topology(config, args.output)
    n_routers = sum(1 for n in config["nodes"] if n["type"] == "QuantumRouter")
    n_bsm = sum(1 for n in config["nodes"] if n["type"] == "BSMNode")
    n_qc = len(config["qchannels"])
    n_cc = len(config["cchannels"])
    print(f"Wrote {args.output}: {n_routers} routers, {n_bsm} BSMs, {n_qc} qchannels, {n_cc} cchannels")