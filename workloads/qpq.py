"""Quantum Private Query workload and two-stage transaction state machine."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

from common import MILLISECOND, SECOND, pairs_per_round
from results import RequestResult
from topology import generate_hub_spoke_topology
from workloads.base import (
    DemandCallbacks,
    DemandSubmitter,
    EntanglementDemand,
    PairDelivery,
    Workload,
)


@dataclass(frozen=True)
class QPQQuerySpec:
    query_id: int
    source: str
    destination: str
    start_time_ps: int
    transaction_deadline_ps: int
    database_size_log: int
    fidelity_threshold: float
    round_deadline_ps: int


@dataclass
class QPQRoundState:
    round_number: int
    demand: EntanglementDemand
    accepted: bool = False
    path: tuple[str, ...] = ()
    deliveries: list[PairDelivery] = field(default_factory=list)
    rejected_pairs: int = 0
    completed_at_ps: int | None = None
    failure_reason: str = ""


class QPQTransaction(DemandCallbacks):
    """Backend-neutral state machine for one two-round QPQ query."""

    def __init__(self, spec: QPQQuerySpec, submit: DemandSubmitter):
        self.spec = spec
        self._submit = submit
        self.rounds: dict[int, QPQRoundState] = {}
        self.success = False
        self.failure_reason = ""
        self.completed_at_ps: int | None = None

    @property
    def transaction_id(self) -> str:
        return f"qpq:{self.spec.query_id}"

    def start(self) -> None:
        self._submit_round(1, self.spec.start_time_ps)

    def _submit_round(self, round_number: int, activation_time_ps: int) -> None:
        deadline = min(
            self.spec.transaction_deadline_ps,
            activation_time_ps + self.spec.round_deadline_ps,
        )
        if deadline <= activation_time_ps:
            self.failure_reason = f"round{round_number}_no_time"
            self.completed_at_ps = activation_time_ps
            return

        demand = EntanglementDemand(
            reservation_id=self.spec.query_id * 2 + round_number - 1,
            demand_id=f"{self.transaction_id}:round:{round_number}",
            transaction_id=self.transaction_id,
            source=self.spec.source,
            destination=self.spec.destination,
            pair_count=pairs_per_round(self.spec.database_size_log),
            fidelity_threshold=self.spec.fidelity_threshold,
            start_time_ps=activation_time_ps,
            deadline_ps=deadline,
            metadata={
                "workload": "qpq",
                "round": round_number,
                "database_size_log": self.spec.database_size_log,
            },
        )
        self.rounds[round_number] = QPQRoundState(round_number, demand)
        self._submit(demand, self)

    def on_demand_accepted(self, demand: EntanglementDemand, path: tuple[str, ...]) -> None:
        state = self._state_for(demand)
        if state is None:
            return
        state.accepted = True
        state.path = path

    def on_pair_delivered(self, demand: EntanglementDemand, delivery: PairDelivery) -> None:
        state = self._state_for(demand)
        if state is not None and not state.failure_reason and state.completed_at_ps is None:
            state.deliveries.append(delivery)

    def on_pair_rejected(self, demand: EntanglementDemand, fidelity: float) -> None:
        del fidelity
        state = self._state_for(demand)
        if state is not None and not state.failure_reason and state.completed_at_ps is None:
            state.rejected_pairs += 1

    def on_demand_completed(
        self,
        demand: EntanglementDemand,
        completed_at_ps: int,
        path: tuple[str, ...],
    ) -> None:
        state = self._state_for(demand)
        if state is None or self._is_terminal:
            return
        state.completed_at_ps = completed_at_ps
        state.path = path
        if len(state.deliveries) != demand.pair_count:
            state.failure_reason = "incomplete_delivery"
            self.failure_reason = f"round{state.round_number}_incomplete_delivery"
            self.completed_at_ps = completed_at_ps
            return
        if state.round_number == 1:
            self._submit_round(2, completed_at_ps)
        else:
            self.success = True
            self.completed_at_ps = completed_at_ps

    def on_demand_failed(
        self,
        demand: EntanglementDemand,
        failed_at_ps: int,
        reason: str,
        path: tuple[str, ...],
        pairs_delivered: int,
    ) -> None:
        del pairs_delivered
        state = self._state_for(demand)
        if state is None or self._is_terminal:
            return
        state.failure_reason = reason
        state.path = path
        self.failure_reason = f"round{state.round_number}_{reason}"
        self.completed_at_ps = failed_at_ps

    @property
    def _is_terminal(self) -> bool:
        return self.success or bool(self.failure_reason)

    def _state_for(self, demand: EntanglementDemand) -> QPQRoundState | None:
        round_number = int(demand.metadata.get("round", 0))
        state = self.rounds.get(round_number)
        if state is None or state.demand.demand_id != demand.demand_id:
            return None
        return state

    def finalize(self, now_ps: int) -> None:
        if self._is_terminal:
            return
        active_round = max(self.rounds, default=1)
        self.failure_reason = f"round{active_round}_simulation_end"
        self.completed_at_ps = now_ps

    def to_request_result(self) -> RequestResult:
        all_deliveries = [
            delivery
            for round_number in sorted(self.rounds)
            for delivery in self.rounds[round_number].deliveries
        ]
        arrivals_ms = [
            (delivery.timestamp_ps - self.spec.start_time_ps) / MILLISECOND
            for delivery in all_deliveries
        ]
        expected_per_round = pairs_per_round(self.spec.database_size_log)
        round1 = self.rounds.get(1)
        round2 = self.rounds.get(2)
        background_pairs = 0
        partial_background_pairs = 0
        background_edges = 0
        fresh_edges = 0
        for delivery in all_deliveries:
            sources = list(delivery.elementary_sources)
            if sources:
                pair_background_edges = sum(1 for item in sources if item.get("source") == "background")
                pair_fresh_edges = sum(1 for item in sources if item.get("source") == "application")
            else:
                pair_background_edges = int(delivery.generation_source.startswith("background"))
                pair_fresh_edges = int(not pair_background_edges)
            background_edges += pair_background_edges
            fresh_edges += pair_fresh_edges
            if pair_background_edges:
                background_pairs += 1
                if pair_fresh_edges:
                    partial_background_pairs += 1

        fidelity = None
        if self.success and all_deliveries:
            fidelity = sum(delivery.fidelity for delivery in all_deliveries) / len(all_deliveries)
        tts_ms = None
        if self.success and self.completed_at_ps is not None:
            tts_ms = (self.completed_at_ps - self.spec.start_time_ps) / MILLISECOND

        return RequestResult(
            request_id=self.spec.query_id,
            src=self.spec.source,
            dst=self.spec.destination,
            start_time_ps=self.spec.start_time_ps,
            time_to_serve_ms=tts_ms,
            fidelity=fidelity,
            success=self.success,
            failure_reason=self.failure_reason,
            pair_arrival_ms=arrivals_ms,
            first_pair_arrival_ms=min(arrivals_ms) if arrivals_ms else None,
            round1_completion_ms=self._completion_ms(round1),
            round2_completion_ms=self._completion_ms(round2),
            round1_pairs=len(round1.deliveries) if round1 else 0,
            round2_pairs=len(round2.deliveries) if round2 else 0,
            expected_pairs=2 * expected_per_round,
            pairs_rejected_fidelity=sum(state.rejected_pairs for state in self.rounds.values()),
            delivered_background_pairs=background_pairs,
            delivered_application_pairs=len(all_deliveries) - background_pairs,
            delivered_pairs_with_background_contribution=background_pairs,
            delivered_pairs_fully_background_supported=background_pairs - partial_background_pairs,
            delivered_pairs_partially_background_supported=partial_background_pairs,
            delivered_pairs_fully_fresh=len(all_deliveries) - background_pairs,
            delivered_background_elementary_edges=background_edges,
            delivered_fresh_elementary_edges=fresh_edges,
        )

    def _completion_ms(self, state: QPQRoundState | None) -> float | None:
        if state is None or state.completed_at_ps is None:
            return None
        return (state.completed_at_ps - self.spec.start_time_ps) / MILLISECOND


@dataclass(frozen=True)
class QPQWorkload(Workload):
    """Matched QPQ transactions over a configurable QDC topology."""

    name: ClassVar[str] = "qpq"
    sequence_adapter: ClassVar[str] = "qpq"

    database_size_log: int = 10
    num_clients: int = 10
    queries_per_client: int = 3
    fidelity_threshold: float = 0.7
    round_deadline_s: float = 5.0
    request_period_s: float = 6.0
    start_offset_s: float = 2.0
    transaction_duration_s: float = 5.0
    seed: int = 42
    num_nodes: int = 25
    inter_node_distance_m: float = 20_000.0
    qdc_node_index: int = 12
    extra_mesh_edges: int = 3
    memories_per_node: int = 50
    memory_fidelity: float = 0.99
    memory_efficiency: float = 0.5
    coherence_time_s: float = 5.0
    gate_fidelity: float = 0.99
    measurement_fidelity: float = 0.99
    swapping_success_probability: float = 1.0
    simulation_end_time_s: float = 180.0
    topology_override: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    query_override: tuple[QPQQuerySpec, ...] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.database_size_log < 1:
            raise ValueError("QPQ database_size_log must be at least one")
        if self.num_clients < 1 or self.queries_per_client < 1:
            raise ValueError("QPQ requires at least one client and one query")
        if self.round_deadline_s <= 0 or self.transaction_duration_s <= 0:
            raise ValueError("QPQ deadlines must be positive")
        if self.num_nodes < 2:
            raise ValueError("QPQ requires at least two quantum routers")
        if not 0 <= self.qdc_node_index < self.num_nodes:
            raise ValueError("QPQ qdc_node_index must identify a quantum router")

    def queries(self) -> list[QPQQuerySpec]:
        if self.query_override is not None:
            return list(self.query_override)
        qdc_name = f"router_{self.qdc_node_index}"
        available_clients = [f"router_{index}" for index in range(self.num_nodes) if index != self.qdc_node_index]
        rng = np.random.default_rng(self.seed)
        client_count = min(self.num_clients, len(available_clients))
        clients = list(rng.choice(available_clients, size=client_count, replace=False))
        queries = []
        query_id = 0
        for client in clients:
            for query_index in range(self.queries_per_client):
                start = int((self.start_offset_s + query_index * self.request_period_s) * SECOND)
                queries.append(QPQQuerySpec(
                    query_id=query_id,
                    source=str(client),
                    destination=qdc_name,
                    start_time_ps=start,
                    transaction_deadline_ps=start + int(self.transaction_duration_s * SECOND),
                    database_size_log=self.database_size_log,
                    fidelity_threshold=self.fidelity_threshold,
                    round_deadline_ps=int(self.round_deadline_s * SECOND),
                ))
                query_id += 1
        return sorted(queries, key=lambda query: (query.start_time_ps, query.query_id))

    @property
    def controller_node(self) -> str:
        return f"router_{self.qdc_node_index}"

    def topology(self, adaptive_memory: int) -> dict:
        if self.topology_override is not None:
            config = copy.deepcopy(dict(self.topology_override))
            for template in config.get("templates", {}).values():
                template["adaptive_max_memory"] = adaptive_memory
            return config
        return generate_hub_spoke_topology(
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
    def from_config(
        cls,
        config: Mapping[str, Any],
        seed: int | None = None,
        topology_override: Mapping[str, Any] | None = None,
        query_override: list[Mapping[str, Any]] | None = None,
    ) -> "QPQWorkload":
        workload = config.get("workload", {})
        topology = config.get("topology", {})
        hardware = config.get("hardware", {})
        experiment = config.get("experiment", {})
        resolved_seed = int(topology.get("random_seed", 42) if seed is None else seed)
        resolved_queries = None
        if query_override is not None:
            resolved_queries = tuple(
                QPQQuerySpec(
                    query_id=int(item["query_id"]),
                    source=str(item["src"]),
                    destination=str(item["dst"]),
                    start_time_ps=int(item["start_time"]),
                    transaction_deadline_ps=int(item["end_time"]),
                    database_size_log=int(item["database_size_log"]),
                    fidelity_threshold=float(item["fidelity"]),
                    round_deadline_ps=int(item["round_deadline_ps"]),
                )
                for item in query_override
            )
        num_nodes = int(topology.get("num_nodes", 25))
        return cls(
            database_size_log=int(workload.get("database_size_log", 10)),
            num_clients=int(workload.get("num_clients", 10)),
            queries_per_client=int(workload.get("queries_per_client", 3)),
            fidelity_threshold=float(workload.get("fidelity_threshold", 0.7)),
            round_deadline_s=float(workload.get("round_deadline_s", 5.0)),
            request_period_s=float(workload.get("request_period_s", 6.0)),
            start_offset_s=float(workload.get("start_offset_s", 2.0)),
            transaction_duration_s=float(workload.get(
                "transaction_duration_s",
                workload.get("reservation_duration_s", 5.0),
            )),
            seed=resolved_seed,
            num_nodes=num_nodes,
            inter_node_distance_m=float(topology.get("inter_node_distance_m", 20_000.0)),
            qdc_node_index=int(topology.get("qdc_node_index", num_nodes // 2)),
            extra_mesh_edges=int(topology.get("extra_mesh_edges", 3)),
            memories_per_node=int(hardware.get("memories_per_node", 50)),
            memory_fidelity=float(hardware.get("link_fidelity", 0.99)),
            memory_efficiency=float(hardware.get("memory_efficiency", 0.5)),
            coherence_time_s=float(hardware.get("memory_coherence_time_s", 5.0)),
            gate_fidelity=float(hardware.get("gate_fidelity", 0.99)),
            measurement_fidelity=float(hardware.get("measurement_fidelity", 0.99)),
            swapping_success_probability=float(hardware.get(
                "swapping_success_probability",
                1.0,
            )),
            simulation_end_time_s=float(experiment.get("simulation_end_time_s", 180.0)),
            topology_override=topology_override,
            query_override=resolved_queries,
        )
