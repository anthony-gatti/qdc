"""Exact, versioned inputs for comparing SeQUeNCe v1.0.0 with the ACP paper."""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx

SECOND = 10**12
MICROSECOND = 10**6
LIGHT_SPEED_M_PER_PS = 2e-4

SCENARIOS = {
    "line2": {
        "topology": "line_2.json", "nodes": 2, "requests": 100,
        "matrices": [[("router_0", "router_1", 1.0)]],
        "expected_intermediate_hops": 0,
    },
    "bottleneck20": {
        "topology": "bottleneck_20.json", "nodes": 20, "requests": 100,
        "matrices": [
            [("router_0", "router_11", .25), ("router_0", "router_12", .25),
             ("router_1", "router_11", .25), ("router_1", "router_12", .25)],
            [("router_7", "router_18", .25), ("router_7", "router_19", .25),
             ("router_8", "router_18", .25), ("router_8", "router_19", .25)],
        ],
        "expected_intermediate_hops": 2,
    },
    "as200": {
        "topology": "as_200.json", "nodes": 200, "requests": 200,
        "matrices": [
            [("router_99", "router_50", .25), ("router_148", "router_154", .25),
             ("router_189", "router_49", .25), ("router_186", "router_98", .25)],
            [("router_176", "router_195", .25), ("router_199", "router_181", .25),
             ("router_79", "router_82", .25), ("router_94", "router_160", .25)],
        ],
        "expected_intermediate_hops": 4,
    },
}


def generate_requests(scenario: str, seed: int) -> list[tuple]:
    """Generate matched paper requests: 10 Hz, 20 ms lead, 80 ms window."""
    spec = SCENARIOS[scenario]
    rng = random.Random(seed)
    requests = []
    half = spec["requests"] // 2
    for identity in range(spec["requests"]):
        matrix = spec["matrices"][0 if len(spec["matrices"]) == 1 or identity < half else 1]
        value = rng.random()
        cumulative = 0.0
        src = dst = None
        for candidate_src, candidate_dst, probability in matrix:
            cumulative += probability
            if value <= cumulative:
                src, dst = candidate_src, candidate_dst
                break
        slot_start = identity * 0.1
        requests.append((
            identity, src, dst,
            round((slot_start + 0.02) * SECOND),
            round((slot_start + 0.1) * SECOND),
            1, 0.5, 1,
        ))
    return requests


def prepare_topology(scenario: str, seed: int, adaptive_memory: int,
                     output: Path) -> dict:
    """Patch the archived paper topology without modifying the ACP checkout."""
    spec = SCENARIOS[scenario]
    source = _topology_source(spec["topology"])
    config = json.loads(source.read_text())
    config["formalism"] = "bell_diagonal"
    config["encoding_type"] = "single_heralded"
    config["stop_time"] = spec["requests"] * 0.1 * SECOND
    config["is_parallel"] = False

    for template in config["templates"].values():
        template["adaptive_max_memory"] = adaptive_memory
        template["encoding_type"] = "single_heralded"
    for node in config["nodes"]:
        node["seed"] = int(node.get("seed", 0)) + seed

    _set_paper_classical_delays(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    return config


def _topology_source(filename: str) -> Path:
    candidates = [
        Path(__file__).parent / "external" / "acp" / "config" / filename,
        Path(__file__).resolve().parents[1] / "docs" / "adaptive-continuous" / "config" / filename,
        Path(__file__).resolve().parents[1] / "docs" / "acp_modified" / "config" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find archived paper topology {filename}")


def router_graph(config: dict) -> nx.Graph:
    graph = nx.Graph()
    bsm_links = {}
    for channel in config["qchannels"]:
        bsm_links.setdefault(channel["destination"], []).append(
            (channel["source"], float(channel["distance"]))
        )
    for endpoints in bsm_links.values():
        if len(endpoints) == 2:
            (left, dl), (right, dr) = endpoints
            graph.add_edge(left, right, distance=dl + dr)
    return graph


def _set_paper_classical_delays(config: dict) -> None:
    graph = router_graph(config)
    bsm_distance = {}
    for channel in config["qchannels"]:
        bsm_distance[(channel["source"], channel["destination"])] = float(channel["distance"])
        bsm_distance[(channel["destination"], channel["source"])] = float(channel["distance"])

    for channel in config["cchannels"]:
        src, dst = channel["source"], channel["destination"]
        if (src, dst) in bsm_distance:
            distance, forwarding_hops = bsm_distance[(src, dst)], 0
        elif src in graph and dst in graph:
            path = nx.shortest_path(graph, src, dst, weight="distance")
            distance = nx.path_weight(graph, path, weight="distance")
            forwarding_hops = len(path) - 2
        else:
            continue
        channel["distance"] = distance
        channel["delay"] = (
            distance / LIGHT_SPEED_M_PER_PS
            + forwarding_hops * 20 * MICROSECOND
            + 100 * MICROSECOND
        )


def validate_paths(scenario: str, config: dict, requests: list[tuple]) -> None:
    graph = router_graph(config)
    expected = SCENARIOS[scenario]["expected_intermediate_hops"]
    for request in requests:
        path = nx.shortest_path(graph, request[1], request[2], weight="distance")
        actual = len(path) - 2
        if actual != expected:
            raise ValueError(f"{scenario}: path {path} has {actual} intermediate hops, expected {expected}")
