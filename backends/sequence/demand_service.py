"""Reusable SeQUeNCe reservation service for application workload demands."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from sequence.app.request_app import RequestApp
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.resource_management.memory_manager import MemoryInfo

from workloads.base import DemandCallbacks, EntanglementDemand, PairDelivery


@dataclass
class _DemandContext:
    demand: EntanglementDemand
    callbacks: DemandCallbacks
    reservation_start_ps: int
    reservation: object | None = None
    path: tuple[str, ...] = ()
    deliveries: list[PairDelivery] = field(default_factory=list)
    terminal: bool = False


class SequenceDemandService(RequestApp):
    """Translate algorithm-independent demands into native RSVP reservations."""

    def __init__(
        self,
        node,
        served_path_observer: Callable[[tuple[str, ...], int], None] | None = None,
    ):
        super().__init__(node)
        self._contexts: dict[int, _DemandContext] = {}
        self._reservation_contexts: dict[int, _DemandContext] = {}
        self._delivered_events: set[tuple] = set()
        self._served_path_observer = served_path_observer
        self.counters = Counter()
        self.events: list[dict] = []

    def submit(self, demand: EntanglementDemand, callbacks: DemandCallbacks) -> None:
        if demand.source != self.node.name:
            raise ValueError(f"Demand source {demand.source} does not match service node {self.node.name}")
        if demand.reservation_id in self._contexts:
            raise ValueError(f"Duplicate reservation id {demand.reservation_id} on {self.node.name}")

        now = self.node.timeline.now()
        reservation_start = max(demand.start_time_ps, now + self._reservation_setup_delay(demand.destination))
        context = _DemandContext(demand, callbacks, reservation_start)
        self._contexts[demand.reservation_id] = context
        self.counters["demands_submitted"] += 1
        self.events.append({
            "event": "demand_submitted",
            "time_ps": now,
            "demand_id": demand.demand_id,
            "reservation_id": demand.reservation_id,
            "application_start_ps": demand.start_time_ps,
            "reservation_start_ps": reservation_start,
            "deadline_ps": demand.deadline_ps,
        })

        if reservation_start >= demand.deadline_ps:
            self._fail(context, "setup_deadline", now)
            return

        self.node.reserve_net_resource(
            demand.destination,
            reservation_start,
            demand.deadline_ps,
            min(
                demand.pair_count,
                max(1, int(getattr(self.node, "link_parallelism", 1))),
            ),
            demand.fidelity_threshold,
            demand.pair_count,
            demand.reservation_id,
        )
        self.node.timeline.schedule(Event(
            demand.deadline_ps,
            Process(self, "deadline", [demand.reservation_id]),
            -10,
        ))

    def get_reservation_result(self, reservation, result: bool) -> None:
        context = self._contexts.get(reservation.identity)
        if context is None:
            self.counters["unknown_reservation_results"] += 1
            return
        context.reservation = reservation
        context.path = tuple(getattr(reservation, "path", ()))
        self._reservation_contexts[id(reservation)] = context
        if not result:
            self.counters["reservations_rejected"] += 1
            self._fail(context, "reservation_rejected", self.node.timeline.now())
            return
        super().get_reservation_result(reservation, result)
        self.counters["reservations_accepted"] += 1
        context.callbacks.on_demand_accepted(context.demand, context.path)
        self.events.append({
            "event": "demand_accepted",
            "time_ps": self.node.timeline.now(),
            "demand_id": context.demand.demand_id,
            "path": list(context.path),
        })

    def get_memory(self, info) -> None:
        if info.state != MemoryInfo.ENTANGLED:
            return
        reservation = self.memo_to_reservation.get(info.index)
        if reservation is None:
            # Intermediate and background pairs also notify the node app; they
            # are outside this endpoint demand service.
            return

        if info.remote_node == reservation.initiator:
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
            self.counters["responder_pair_callbacks"] += 1
            return
        if info.remote_node != reservation.responder:
            return

        context = self._reservation_contexts.get(id(reservation))
        if context is None:
            self.counters["unknown_initiator_pair_callbacks"] += 1
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
            return
        delivery_key = (
            id(reservation),
            info.index,
            info.remote_node,
            info.remote_memo,
            self.node.timeline.now(),
        )
        if delivery_key in self._delivered_events:
            self.counters["duplicate_pair_callbacks"] += 1
            return
        self._delivered_events.add(delivery_key)
        if context.terminal:
            self.counters["post_terminal_pair_callbacks"] += 1
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
            return
        if info.fidelity < context.demand.fidelity_threshold:
            self.counters["pairs_rejected_fidelity"] += 1
            context.callbacks.on_pair_rejected(context.demand, info.fidelity)
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
            return

        elementary_sources = tuple(
            dict(source)
            for source in getattr(info.memory, "qdc_elementary_sources", ())
        )
        if not elementary_sources and not hasattr(self.node, "adaptive_continuous"):
            elementary_sources = tuple(
                {"source": "application", "link": tuple(sorted(edge))}
                for edge in zip(context.path, context.path[1:])
            )
        delivery = PairDelivery(
            timestamp_ps=self.node.timeline.now(),
            fidelity=info.fidelity,
            generation_source=getattr(info.memory, "qdc_generation_source", "application"),
            elementary_sources=elementary_sources,
        )
        context.deliveries.append(delivery)
        self.counters["pairs_delivered"] += 1
        context.callbacks.on_pair_delivered(context.demand, delivery)
        self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)

        if len(context.deliveries) >= context.demand.pair_count:
            self._complete(context)

    def deadline(self, reservation_id: int) -> None:
        context = self._contexts.get(reservation_id)
        if context is not None and not context.terminal:
            self.counters["demand_deadlines"] += 1
            self._fail(context, "deadline", self.node.timeline.now())

    def finalize(self, now_ps: int) -> None:
        for context in self._contexts.values():
            if not context.terminal:
                self._fail(context, "simulation_end", now_ps)

    def _complete(self, context: _DemandContext) -> None:
        if context.terminal:
            return
        context.terminal = True
        now = self.node.timeline.now()
        self.counters["demands_completed"] += 1
        if self._served_path_observer is not None:
            self._served_path_observer(context.path, now)
        self._expire_reservation(context)
        self.events.append({
            "event": "demand_completed",
            "time_ps": now,
            "demand_id": context.demand.demand_id,
            "pairs_delivered": len(context.deliveries),
            "path": list(context.path),
        })
        context.callbacks.on_demand_completed(context.demand, now, context.path)

    def _fail(self, context: _DemandContext, reason: str, now_ps: int) -> None:
        if context.terminal:
            return
        context.terminal = True
        self.counters[f"demands_failed_{reason}"] += 1
        self._expire_reservation(context)
        self.events.append({
            "event": "demand_failed",
            "time_ps": now_ps,
            "demand_id": context.demand.demand_id,
            "reason": reason,
            "pairs_delivered": len(context.deliveries),
            "path": list(context.path),
        })
        context.callbacks.on_demand_failed(
            context.demand,
            now_ps,
            reason,
            context.path,
            len(context.deliveries),
        )

    def _expire_reservation(self, context: _DemandContext) -> None:
        reservation = context.reservation
        if reservation is None:
            return
        self.node.resource_manager.expire_rules_by_reservation(reservation)
        for node_name in context.path:
            if node_name != self.node.name:
                self.node.resource_manager.expire_remote_rules(node_name, reservation)

    def _reservation_setup_delay(self, destination: str) -> int:
        path = self._forwarding_path(destination)
        if len(path) < 2:
            return 1
        one_way = 0
        for left_name, right_name in zip(path, path[1:]):
            left = self.node.timeline.get_entity_by_name(left_name)
            channel = getattr(left, "cchannels", {}).get(right_name)
            if channel is None:
                return 1
            one_way += int(channel.delay)
        return max(1, 2 * one_way + 1)

    def _forwarding_path(self, destination: str) -> tuple[str, ...]:
        current = self.node
        path = [current.name]
        seen = {current.name}
        while current.name != destination:
            next_hop = current.network_manager.get_forwarding_table().get(destination)
            if next_hop is None or next_hop in seen:
                return ()
            path.append(next_hop)
            seen.add(next_hop)
            current = self.node.timeline.get_entity_by_name(next_hop)
            if current is None:
                return ()
        return tuple(path)
