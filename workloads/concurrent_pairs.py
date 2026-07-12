"""Small concurrent-pair workload for slot-based routing algorithms."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Any

import numpy as np

from common import MILLISECOND, SECOND
from results import RequestResult
from topology import generate_hub_spoke_topology, generate_linear_topology
from workloads.base import EntanglementDemand, PairDelivery, Workload


@dataclass(frozen=True)
class ConcurrentPairSpec:
    request_id: int
    source: str
    destination: str
    start_time_ps: int
    deadline_ps: int
    pair_count: int = 1
    fidelity_threshold: float = 0.7

    def demand(self) -> EntanglementDemand:
        return EntanglementDemand(
            reservation_id=self.request_id,
            demand_id=f"pair-{self.request_id}",
            transaction_id=f"pair-{self.request_id}",
            source=self.source,
            destination=self.destination,
            pair_count=self.pair_count,
            fidelity_threshold=self.fidelity_threshold,
            start_time_ps=self.start_time_ps,
            deadline_ps=self.deadline_ps,
        )


@dataclass
class ConcurrentPairTransaction:
    spec: ConcurrentPairSpec
    accepted_path: tuple[str, ...] = ()
    deliveries: list[PairDelivery] = field(default_factory=list)
    rejected_fidelities: list[float] = field(default_factory=list)
    completed_at_ps: int | None = None
    failed_at_ps: int | None = None
    failure_reason: str = ""

    @property
    def terminal(self) -> bool:
        return self.completed_at_ps is not None or self.failed_at_ps is not None

    def on_demand_accepted(self, _demand, path: tuple[str, ...]) -> None:
        self.accepted_path = path

    def on_pair_delivered(self, _demand, delivery: PairDelivery) -> None:
        if not self.terminal:
            self.deliveries.append(delivery)

    def on_pair_rejected(self, _demand, fidelity: float) -> None:
        if not self.terminal:
            self.rejected_fidelities.append(fidelity)

    def on_demand_completed(self, _demand, completed_at_ps: int, path: tuple[str, ...]) -> None:
        if not self.terminal:
            self.completed_at_ps = completed_at_ps
            self.accepted_path = path

    def on_demand_failed(
        self,
        _demand,
        failed_at_ps: int,
        reason: str,
        path: tuple[str, ...],
        _pairs_delivered: int,
    ) -> None:
        if not self.terminal:
            self.failed_at_ps = failed_at_ps
            self.failure_reason = reason
            self.accepted_path = path

    def finalize(self, now_ps: int) -> None:
        if not self.terminal:
            self.failed_at_ps = now_ps
            self.failure_reason = "simulation_end"

    def to_request_result(self) -> RequestResult:
        success = self.completed_at_ps is not None
        completion = self.completed_at_ps if success else None
        arrivals = [
            (delivery.timestamp_ps - self.spec.start_time_ps) / MILLISECOND
            for delivery in self.deliveries
        ]
        return RequestResult(
            request_id=self.spec.request_id,
            src=self.spec.source,
            dst=self.spec.destination,
            start_time_ps=self.spec.start_time_ps,
            time_to_serve_ms=(
                (completion - self.spec.start_time_ps) / MILLISECOND
                if completion is not None else None
            ),
            fidelity=(
                sum(delivery.fidelity for delivery in self.deliveries) / len(self.deliveries)
                if self.deliveries else None
            ),
            success=success,
            failure_reason="" if success else self.failure_reason,
            pair_arrival_ms=arrivals,
            first_pair_arrival_ms=arrivals[0] if arrivals else None,
            expected_pairs=self.spec.pair_count,
            pairs_rejected_fidelity=len(self.rejected_fidelities),
            delivered_pairs_fully_fresh=len(self.deliveries),
            delivered_application_pairs=len(self.deliveries),
        )


@dataclass(frozen=True)
class ConcurrentPairWorkload(Workload):
    """A deterministic batch of requests sharing one QDC controller."""

    name: ClassVar[str] = "concurrent_pairs"
    sequence_adapter: ClassVar[str] = "concurrent_pairs"

    num_requests: int = 2
    pair_count: int = 1
    start_offset_s: float = 0.01
    request_window_s: float = 0.05
    fidelity_threshold: float = 0.7
    seed: int = 0
    num_nodes: int = 3
    qdc_node_index: int = 0
    topology_type: str = "linear"
    extra_mesh_edges: int = 0
    inter_node_distance_m: float = 1_000.0
    memories_per_node: int = 12
    memory_fidelity: float = 0.99
    memory_efficiency: float = 1.0
    coherence_time_s: float = 5.0
    gate_fidelity: float = 0.99
    measurement_fidelity: float = 0.99
    swapping_success_probability: float = 0.9
    link_parallelism: int = 1
    simulation_end_time_s: float = 0.1
    request_override: tuple[ConcurrentPairSpec, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    topology_override: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.num_nodes < 2 or not 0 <= self.qdc_node_index < self.num_nodes:
            raise ValueError("Concurrent-pair workload requires a valid QDC node")
        if self.topology_type not in {"linear", "hub_spoke"}:
            raise ValueError(f"Unsupported concurrent-pair topology {self.topology_type!r}")
        if self.num_requests <= 0 or self.pair_count <= 0:
            raise ValueError("Concurrent-pair request and pair counts must be positive")
        if self.start_offset_s < 0 or self.request_window_s <= 0:
            raise ValueError("Concurrent-pair request timing is invalid")
        if self.inter_node_distance_m <= 0 or self.memories_per_node <= 0:
            raise ValueError("Concurrent-pair topology resources must be positive")
        if self.link_parallelism <= 0:
            raise ValueError("Concurrent-pair link_parallelism must be positive")
        for name, value in (
            ("fidelity_threshold", self.fidelity_threshold),
            ("memory_fidelity", self.memory_fidelity),
            ("memory_efficiency", self.memory_efficiency),
            ("gate_fidelity", self.gate_fidelity),
            ("measurement_fidelity", self.measurement_fidelity),
            ("swapping_success_probability", self.swapping_success_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"Concurrent-pair {name} must be in [0, 1]")

    @property
    def controller_node(self) -> str:
        return f"router_{self.qdc_node_index}"

    def requests(self) -> list[ConcurrentPairSpec]:
        if self.request_override is not None:
            requests = list(self.request_override)
            if any(request.source != self.controller_node for request in requests):
                raise ValueError("Concurrent-pair requests must originate at the QDC node")
            if any(
                request.source == request.destination
                or request.pair_count <= 0
                or request.deadline_ps <= request.start_time_ps
                for request in requests
            ):
                raise ValueError("Concurrent-pair request override is invalid")
            return requests
        destinations = [
            f"router_{index}"
            for index in range(self.num_nodes)
            if index != self.qdc_node_index
        ]
        rng = np.random.default_rng(self.seed)
        chosen = list(rng.choice(
            destinations,
            size=self.num_requests,
            replace=self.num_requests > len(destinations),
        ))
        start = int(self.start_offset_s * SECOND)
        deadline = start + int(self.request_window_s * SECOND)
        return [
            ConcurrentPairSpec(
                request_id=index,
                source=self.controller_node,
                destination=str(destination),
                start_time_ps=start,
                deadline_ps=deadline,
                pair_count=self.pair_count,
                fidelity_threshold=self.fidelity_threshold,
            )
            for index, destination in enumerate(chosen)
        ]

    def topology(self, adaptive_memory: int) -> dict:
        if self.topology_override is not None:
            config = copy.deepcopy(dict(self.topology_override))
            for template in config.get("templates", {}).values():
                template["adaptive_max_memory"] = adaptive_memory
            return config
        generator = (
            generate_linear_topology
            if self.topology_type == "linear"
            else generate_hub_spoke_topology
        )
        return generator(
            num_nodes=self.num_nodes,
            inter_node_distance_m=self.inter_node_distance_m,
            memo_size=self.memories_per_node,
            adaptive_max_memory=adaptive_memory,
            memory_fidelity=self.memory_fidelity,
            memory_efficiency=self.memory_efficiency,
            coherence_time_s=self.coherence_time_s,
            gate_fidelity=self.gate_fidelity,
            measurement_fidelity=self.measurement_fidelity,
            swapping_success_probability=self.swapping_success_probability,
            stop_time_s=self.simulation_end_time_s,
            seed=self.seed,
            extra_mesh_edges=self.extra_mesh_edges,
            qdc_node_index=self.qdc_node_index,
            encoding_type="single_heralded",
            formalism="bell_diagonal",
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], seed: int) -> "ConcurrentPairWorkload":
        workload = config.get("workload", {})
        topology = config.get("topology", {})
        hardware = config.get("hardware", {})
        experiment = config.get("experiment", {})
        num_nodes = int(topology.get("num_nodes", 3))
        return cls(
            num_requests=int(workload.get("request_count", workload.get("requests", 2))),
            pair_count=int(workload.get("pair_count", 1)),
            start_offset_s=float(workload.get("start_offset_s", 0.01)),
            request_window_s=float(workload.get("request_window_s", 0.05)),
            fidelity_threshold=float(workload.get("fidelity_threshold", 0.7)),
            seed=seed,
            num_nodes=num_nodes,
            qdc_node_index=int(topology.get("qdc_node_index", 0)),
            topology_type=str(topology.get("type", "linear")),
            extra_mesh_edges=int(topology.get("extra_mesh_edges", 0)),
            inter_node_distance_m=float(topology.get("inter_node_distance_m", 1_000.0)),
            memories_per_node=int(hardware.get("memories_per_node", 12)),
            memory_fidelity=float(hardware.get("link_fidelity", 0.99)),
            memory_efficiency=float(hardware.get("memory_efficiency", 1.0)),
            coherence_time_s=float(hardware.get("memory_coherence_time_s", 5.0)),
            gate_fidelity=float(hardware.get("gate_fidelity", 0.99)),
            measurement_fidelity=float(hardware.get("measurement_fidelity", 0.99)),
            swapping_success_probability=float(hardware.get("swapping_success_probability", 0.9)),
            link_parallelism=int(hardware.get(
                "link_parallelism",
                hardware.get("bsm_lanes_per_link", 1),
            )),
            simulation_end_time_s=float(experiment.get("simulation_end_time_s", 0.1)),
        )
