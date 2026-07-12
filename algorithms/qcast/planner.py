"""Paper-style Q-CAST path planning independent of SeQUeNCe execution."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping


EdgeKey = tuple[str, str]


def edge_key(left: str, right: str) -> EdgeKey:
    return (left, right) if left < right else (right, left)


@dataclass(frozen=True)
class QCASTEdge:
    left: str
    right: str
    width: int
    success_probability: float

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise ValueError("Q-CAST edges require distinct endpoints")
        if self.width <= 0:
            raise ValueError("Q-CAST edge width must be positive")
        if not 0 <= self.success_probability <= 1:
            raise ValueError("Q-CAST edge success probability must be in [0, 1]")

    @property
    def key(self) -> EdgeKey:
        return edge_key(self.left, self.right)


@dataclass(frozen=True)
class QCASTDemand:
    demand_id: str
    source: str
    destination: str


@dataclass(frozen=True)
class QCASTPath:
    path_id: str
    demand_id: str
    nodes: tuple[str, ...]
    width: int
    ext: float
    kind: str = "major"
    parent_path_id: str | None = None
    covered_segment: tuple[int, int] | None = None

    @property
    def edges(self) -> tuple[EdgeKey, ...]:
        return tuple(edge_key(left, right) for left, right in zip(self.nodes, self.nodes[1:]))


@dataclass(frozen=True)
class QCASTPlan:
    major_paths: tuple[QCASTPath, ...]
    recovery_paths: tuple[QCASTPath, ...]
    residual_node_memories: Mapping[str, int]
    residual_edge_widths: Mapping[EdgeKey, int]


def expected_throughput(
    width: int,
    hop_probabilities: Iterable[float],
    swap_success_probability: float,
) -> float:
    """Return Equation 2 EXT for a fixed-width path.

    The expected minimum number of successful links across all hops is
    ``sum_i P(all hops have at least i successes)``.  This is equivalent to
    the paper's recursive distribution and is numerically simpler.
    """
    probabilities = tuple(hop_probabilities)
    if width <= 0 or not probabilities:
        return 0.0
    if not 0 <= swap_success_probability <= 1:
        raise ValueError("Swap success probability must be in [0, 1]")

    expected_minimum = 0.0
    for threshold in range(1, width + 1):
        all_hops = 1.0
        for probability in probabilities:
            if not 0 <= probability <= 1:
                raise ValueError("Link success probability must be in [0, 1]")
            tail = sum(
                math.comb(width, successes)
                * probability**successes
                * (1 - probability) ** (width - successes)
                for successes in range(threshold, width + 1)
            )
            all_hops *= tail
        expected_minimum += all_hops

    swaps = max(0, len(probabilities) - 1)
    return expected_minimum * swap_success_probability**swaps


@dataclass
class _ResidualNetwork:
    node_memories: dict[str, int]
    edge_widths: dict[EdgeKey, int]
    edge_probabilities: dict[EdgeKey, float]
    neighbors: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        node_memories: Mapping[str, int],
        edges: Iterable[QCASTEdge],
    ) -> "_ResidualNetwork":
        widths: dict[EdgeKey, int] = {}
        probabilities: dict[EdgeKey, float] = {}
        neighbors = {node: set() for node in node_memories}
        for edge in edges:
            if edge.left not in node_memories or edge.right not in node_memories:
                raise ValueError(f"Q-CAST edge references unknown node: {edge}")
            if edge.key in widths:
                raise ValueError(f"Duplicate Q-CAST edge {edge.key}")
            widths[edge.key] = edge.width
            probabilities[edge.key] = edge.success_probability
            neighbors[edge.left].add(edge.right)
            neighbors[edge.right].add(edge.left)
        return cls(dict(node_memories), widths, probabilities, neighbors)

    def reserve(self, nodes: tuple[str, ...], width: int) -> None:
        required: dict[str, int] = {}
        for index, node in enumerate(nodes):
            required[node] = required.get(node, 0) + (
                width if index in (0, len(nodes) - 1) else 2 * width
            )
        for node, amount in required.items():
            if self.node_memories[node] < amount:
                raise RuntimeError(f"Q-CAST node over-allocation on {node}")
        for edge in (edge_key(left, right) for left, right in zip(nodes, nodes[1:])):
            if self.edge_widths.get(edge, 0) < width:
                raise RuntimeError(f"Q-CAST edge over-allocation on {edge}")
        for node, amount in required.items():
            self.node_memories[node] -= amount
        for edge in (edge_key(left, right) for left, right in zip(nodes, nodes[1:])):
            self.edge_widths[edge] -= width


class QCASTPlanner:
    """Greedy EDA planner from Q-CAST P2 with contention-free recovery paths."""

    def __init__(
        self,
        *,
        swap_success_probability: float = 0.9,
        max_major_paths: int = 200,
        max_hops: int = 8,
        link_state_hops: int = 3,
        recovery_paths_per_segment: int = 1,
        max_recovery_paths_per_major: int = 12,
    ) -> None:
        if max_major_paths <= 0 or max_hops <= 0:
            raise ValueError("Q-CAST path limits must be positive")
        if link_state_hops < 0 or recovery_paths_per_segment < 0:
            raise ValueError("Q-CAST recovery limits cannot be negative")
        self.swap_success_probability = swap_success_probability
        self.max_major_paths = max_major_paths
        self.max_hops = max_hops
        self.link_state_hops = link_state_hops
        self.recovery_paths_per_segment = recovery_paths_per_segment
        self.max_recovery_paths_per_major = max_recovery_paths_per_major

    def plan(
        self,
        demands: Iterable[QCASTDemand],
        node_memories: Mapping[str, int],
        edges: Iterable[QCASTEdge],
    ) -> QCASTPlan:
        ordered_demands = tuple(sorted(demands, key=lambda item: item.demand_id))
        residual = _ResidualNetwork.create(node_memories, edges)
        major_paths: list[QCASTPath] = []

        while len(major_paths) < self.max_major_paths:
            candidates = []
            for demand in ordered_demands:
                candidate = self._best_path(
                    residual,
                    demand.source,
                    demand.destination,
                )
                if candidate is not None:
                    ext, width, nodes = candidate
                    candidates.append((ext, width, nodes, demand))
            if not candidates:
                break
            ext, width, nodes, demand = max(
                candidates,
                key=lambda item: (item[0], item[1], tuple(reversed(item[2])), item[3].demand_id),
            )
            if ext <= 0:
                break
            path = QCASTPath(
                path_id=f"major-{len(major_paths)}",
                demand_id=demand.demand_id,
                nodes=nodes,
                width=width,
                ext=ext,
            )
            residual.reserve(nodes, width)
            major_paths.append(path)

        recovery_paths: list[QCASTPath] = []
        if self.link_state_hops and self.recovery_paths_per_segment:
            for major in major_paths:
                self._reserve_recovery_paths(residual, major, recovery_paths)

        return QCASTPlan(
            tuple(major_paths),
            tuple(recovery_paths),
            dict(residual.node_memories),
            dict(residual.edge_widths),
        )

    def _reserve_recovery_paths(
        self,
        residual: _ResidualNetwork,
        major: QCASTPath,
        output: list[QCASTPath],
    ) -> None:
        count = 0
        major_nodes = set(major.nodes)
        max_segment = min(self.link_state_hops, len(major.nodes) - 1)
        for segment_hops in range(1, max_segment + 1):
            for start in range(0, len(major.nodes) - segment_hops):
                end = start + segment_hops
                forbidden = major_nodes - {major.nodes[start], major.nodes[end]}
                for _ in range(self.recovery_paths_per_segment):
                    if count >= self.max_recovery_paths_per_major:
                        return
                    candidate = self._best_path(
                        residual,
                        major.nodes[start],
                        major.nodes[end],
                        forbidden_nodes=forbidden,
                    )
                    if candidate is None:
                        break
                    ext, width, nodes = candidate
                    if ext <= 0:
                        break
                    recovery = QCASTPath(
                        path_id=f"recovery-{major.path_id}-{count}",
                        demand_id=major.demand_id,
                        nodes=nodes,
                        width=width,
                        ext=ext,
                        kind="recovery",
                        parent_path_id=major.path_id,
                        covered_segment=(start, end),
                    )
                    residual.reserve(nodes, width)
                    output.append(recovery)
                    count += 1

    def _best_path(
        self,
        residual: _ResidualNetwork,
        source: str,
        destination: str,
        *,
        forbidden_nodes: set[str] | None = None,
    ) -> tuple[float, int, tuple[str, ...]] | None:
        if source not in residual.node_memories or destination not in residual.node_memories:
            return None
        maximum_width = min(
            residual.node_memories[source],
            residual.node_memories[destination],
            max(residual.edge_widths.values(), default=0),
        )
        best = None
        for width in range(maximum_width, 0, -1):
            path = self._extended_dijkstra(
                residual,
                source,
                destination,
                width,
                forbidden_nodes or set(),
            )
            if path is None:
                continue
            probabilities = [residual.edge_probabilities[edge] for edge in path_edges(path)]
            ext = expected_throughput(width, probabilities, self.swap_success_probability)
            candidate = (ext, width, path)
            if best is None or (candidate[0], candidate[1], tuple(reversed(candidate[2]))) > (
                best[0], best[1], tuple(reversed(best[2]))
            ):
                best = candidate
        return best

    def _extended_dijkstra(
        self,
        residual: _ResidualNetwork,
        source: str,
        destination: str,
        width: int,
        forbidden_nodes: set[str],
    ) -> tuple[str, ...] | None:
        if source in forbidden_nodes or destination in forbidden_nodes:
            return None
        queue: list[tuple[float, tuple[str, ...], str]] = [(-math.inf, (source,), source)]
        best_score = {source: math.inf}
        best_path = {source: (source,)}

        while queue:
            negative_score, path, node = heapq.heappop(queue)
            score = -negative_score
            if score + 1e-15 < best_score.get(node, -math.inf) or path != best_path.get(node):
                continue
            if node == destination:
                return path
            if len(path) - 1 >= self.max_hops:
                continue
            for neighbor in sorted(residual.neighbors.get(node, ())):
                if neighbor in path or neighbor in forbidden_nodes:
                    continue
                edge = edge_key(node, neighbor)
                if residual.edge_widths.get(edge, 0) < width:
                    continue
                if neighbor not in (source, destination) and residual.node_memories[neighbor] < 2 * width:
                    continue
                new_path = path + (neighbor,)
                probabilities = [residual.edge_probabilities[item] for item in path_edges(new_path)]
                new_score = expected_throughput(width, probabilities, self.swap_success_probability)
                old_score = best_score.get(neighbor, -math.inf)
                old_path = best_path.get(neighbor)
                if new_score > old_score + 1e-15 or (
                    abs(new_score - old_score) <= 1e-15
                    and (old_path is None or new_path < old_path)
                ):
                    best_score[neighbor] = new_score
                    best_path[neighbor] = new_path
                    heapq.heappush(queue, (-new_score, new_path, neighbor))
        return None


def path_edges(nodes: tuple[str, ...]) -> tuple[EdgeKey, ...]:
    return tuple(edge_key(left, right) for left, right in zip(nodes, nodes[1:]))
