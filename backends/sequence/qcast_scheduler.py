"""SeQUeNCe-native slot execution for Q-CAST major and recovery paths."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import combinations

from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.resource_management.action_condition_set import (
    eg_rule_action_await,
    eg_rule_action_request,
    eg_rule_condition,
)
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.resource_management.rule_manager import Rule
from sequence.topology.router_net_topo import RouterNetTopo

from algorithms.qcast import QCASTDemand, QCASTEdge, QCASTPath, edge_key
from backends.sequence.qcast_protocol import (
    QCASTMessage,
    QCASTMessageType,
    QCASTReservation,
)
from workloads.base import PairDelivery


def single_heralded_attempt_success_probability(
    left_emission_and_transmission: float,
    right_emission_and_transmission: float,
    detector_efficiencies: tuple[float, float],
    bsm_success_probability: float,
) -> float:
    """Model one complete SeQUeNCe v1.0.0 single-heralded attempt.

    The protocol emits one photon from each endpoint. Its second state-machine
    update evaluates that BSM result; it does not initiate a second optical
    round despite stale comments in the upstream implementation.
    """
    return (
        bsm_success_probability
        * left_emission_and_transmission
        * right_emission_and_transmission
        * math.prod(detector_efficiencies)
    )


@dataclass
class _DemandContext:
    demand: object
    callbacks: object
    deliveries: list[PairDelivery] = field(default_factory=list)
    paths: list[tuple[str, ...]] = field(default_factory=list)
    terminal: bool = False


@dataclass(frozen=True)
class QCASTLaneEdge:
    lane_id: str
    left: str
    right: str
    left_memory_index: int
    right_memory_index: int
    reservation: QCASTReservation
    middle: str | None = None

    @property
    def key(self):
        return edge_key(self.left, self.right)

    def memory_index(self, node: str) -> int:
        if node == self.left:
            return self.left_memory_index
        if node == self.right:
            return self.right_memory_index
        raise KeyError(node)


@dataclass
class QCASTLane:
    lane_id: str
    path: QCASTPath
    lane_index: int
    edges: tuple[QCASTLaneEdge, ...]


@dataclass
class QCASTPathAllocation:
    path: QCASTPath
    lanes: tuple[QCASTLane, ...]


@dataclass
class _Slot:
    slot_id: int
    started_at_ps: int
    generation_start_ps: int
    generation_end_ps: int
    plan: object
    allocations: dict[str, QCASTPathAllocation]
    allocated_memories: dict[str, set[int]]
    rules: list[tuple[object, Rule]] = field(default_factory=list)
    edge_success: dict[str, bool] = field(default_factory=dict)
    edge_fidelity: dict[str, float] = field(default_factory=dict)
    plan_receipts: set[str] = field(default_factory=set)
    link_state_receipts: dict[str, set[str]] = field(default_factory=dict)
    aborted_deliveries: set[str] = field(default_factory=set)
    cleanup_time_ps: int = 0


class QCASTDemandScheduler:
    """Batch active demands into Q-CAST slots and execute them physically."""

    def __init__(self, network_topology, algorithm, controller_node: str):
        self.network_topology = network_topology
        self.algorithm = algorithm
        self.controller = network_topology.get_timeline().get_entity_by_name(controller_node)
        if self.controller is None:
            raise ValueError(f"Unknown Q-CAST controller node {controller_node!r}")
        self.timeline = network_topology.get_timeline()
        self.routers = {
            router.name: router
            for router in network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        }
        self.contexts: dict[int, _DemandContext] = {}
        self.slot: _Slot | None = None
        self.slot_counter = 0
        self.slot_event_pending = False
        self.counters = Counter()
        self.events: list[dict] = []
        self.max_allocated_by_node = Counter()
        self._edge_middle_nodes = self._middle_nodes_by_edge()
        self._graph = self._router_graph()
        self._edge_model_details = {}
        self._edge_models = self._physical_edge_models()
        for router in self.routers.values():
            router.qcast_control.configure(self)

    def submit(self, demand, callbacks) -> None:
        if demand.reservation_id in self.contexts:
            raise ValueError(f"Duplicate Q-CAST demand id {demand.reservation_id}")
        context = _DemandContext(demand, callbacks)
        self.contexts[demand.reservation_id] = context
        self.counters["demands_submitted"] += 1
        self.events.append({
            "event": "demand_submitted",
            "time_ps": self.timeline.now(),
            "demand_id": demand.demand_id,
            "source": demand.source,
            "destination": demand.destination,
        })
        self.timeline.schedule(Event(
            demand.deadline_ps,
            Process(self, "deadline", [demand.reservation_id]),
            -10,
        ))
        self._schedule_slot(self.timeline.now() + 1)

    def deadline(self, reservation_id: int) -> None:
        context = self.contexts.get(reservation_id)
        if context is None or context.terminal:
            return
        context.terminal = True
        self.counters["demands_failed_deadline"] += 1
        self.events.append({
            "event": "demand_failed",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "reason": "qcast_deadline",
            "pairs_delivered": len(context.deliveries),
        })
        context.callbacks.on_demand_failed(
            context.demand,
            self.timeline.now(),
            "qcast_deadline",
            context.paths[0] if context.paths else (),
            len(context.deliveries),
        )

    def run_slot(self) -> None:
        self.slot_event_pending = False
        if self.slot is not None:
            return
        active = [
            context
            for context in self.contexts.values()
            if not context.terminal
            and context.demand.start_time_ps <= self.timeline.now() < context.demand.deadline_ps
        ]
        if not active:
            return

        free_memories = self._free_memory_indices()
        demands = [
            QCASTDemand(
                context.demand.demand_id,
                context.demand.source,
                context.demand.destination,
            )
            for context in active
        ]
        plan = self.algorithm.create_planner().plan(
            demands,
            {node: len(indices) for node, indices in free_memories.items()},
            self._edge_models,
        )
        self.counters["slots_started"] += 1
        self.counters["major_paths_selected"] += len(plan.major_paths)
        self.counters["recovery_paths_selected"] += len(plan.recovery_paths)
        if not plan.major_paths:
            self.counters["slots_without_paths"] += 1
            self._schedule_slot(self.timeline.now() + self.algorithm.generation_window_ps)
            return

        slot_id = self.slot_counter
        self.slot_counter += 1
        allocations, allocated = self._allocate_memories(slot_id, plan, free_memories)
        all_router_names = tuple(sorted(self.routers))
        maximum_plan_delay = max(
            [self.algorithm.control_processing_delay_ps]
            + [
                self.algorithm.control_processing_delay_ps
                + int(self.controller.cchannels[name].delay)
                for name in all_router_names
                if name != self.controller.name
            ]
        )
        generation_start = self.timeline.now() + maximum_plan_delay + 1
        generation_end = generation_start + self.algorithm.generation_window_ps
        slot = _Slot(
            slot_id,
            self.timeline.now(),
            generation_start,
            generation_end,
            plan,
            allocations,
            allocated,
        )
        self.slot = slot
        for node, indices in allocated.items():
            self.max_allocated_by_node[node] = max(self.max_allocated_by_node[node], len(indices))

        paths_by_demand = {}
        for path in plan.major_paths:
            paths_by_demand.setdefault(path.demand_id, []).append(path.nodes)
        for context in active:
            paths = paths_by_demand.get(context.demand.demand_id, [])
            if paths:
                context.paths.extend(paths)
                context.callbacks.on_demand_accepted(context.demand, paths[0])

        self.events.append({
            "event": "slot_planned",
            "slot_id": slot_id,
            "time_ps": self.timeline.now(),
            "generation_start_ps": generation_start,
            "generation_end_ps": generation_end,
            "major_paths": [self._path_record(path) for path in plan.major_paths],
            "recovery_paths": [self._path_record(path) for path in plan.recovery_paths],
        })
        for name in all_router_names:
            message = QCASTMessage(
                QCASTMessageType.PLAN,
                slot_id,
                self._local_plan_payload(slot, name),
            )
            if name == self.controller.name:
                self.timeline.schedule(Event(
                    self.timeline.now() + self.algorithm.control_processing_delay_ps,
                    Process(
                        self.controller.qcast_control,
                        "received_message",
                        [name, message],
                    ),
                ))
            else:
                self.controller.send_message(
                    name,
                    message,
                    sender_delay=self.algorithm.control_processing_delay_ps,
                )
                self.counters["plan_messages_sent"] += 1

        self.timeline.schedule(Event(
            generation_end,
            Process(self, "stop_generation", [slot_id]),
        ))

    def received_control_message(self, receiver: str, src: str, msg: QCASTMessage) -> None:
        slot = self.slot
        if slot is None or msg.slot_id != slot.slot_id:
            self.counters["stale_control_messages"] += 1
            return
        if msg.msg_type is QCASTMessageType.PLAN:
            slot.plan_receipts.add(receiver)
            self.counters["plan_messages_received"] += 1
            self.timeline.schedule(Event(
                slot.generation_start_ps,
                Process(self, "activate_node", [slot.slot_id, receiver]),
            ))
        elif msg.msg_type is QCASTMessageType.LINK_STATE:
            slot.link_state_receipts.setdefault(receiver, set()).add(src)
            self.counters["link_state_messages_received"] += 1

    def activate_node(self, slot_id: int, node_name: str) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        node = self.routers[node_name]
        payload = node.qcast_control.plan_by_slot.get(slot_id)
        if payload is None:
            self.counters["plan_activations_missing_payload"] += 1
            return
        assignments_by_id = self._assignments_by_id(slot)
        for record in payload["assignments"]:
            assignment = assignments_by_id[record["lane_edge_id"]]
            is_requester = record["role"] == "request"
            memory_index = record["memory_index"]
            condition_args = {"memory_indices": [memory_index]}
            if is_requester:
                action = eg_rule_action_request
                action_args = {
                    "mid": assignment.middle,
                    "path": [assignment.left, assignment.right],
                    "index": 0,
                    "name": assignment.left,
                    "reservation": assignment.reservation,
                }
            else:
                action = eg_rule_action_await
                action_args = {
                    "mid": assignment.middle,
                    "path": [assignment.left, assignment.right],
                    "index": 1,
                }
            rule = Rule(10, action, eg_rule_condition, action_args, condition_args)
            rule.set_reservation(assignment.reservation)
            node.resource_manager.load(rule)
            slot.rules.append((node, rule))
        self.counters["plan_activations"] += 1

    def stop_generation(self, slot_id: int) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        self._expire_slot_rules(slot)

        for allocation in slot.allocations.values():
            for lane in allocation.lanes:
                for assignment in lane.edges:
                    success, fidelity = self._assignment_state(assignment)
                    slot.edge_success[assignment.lane_id] = success
                    slot.edge_fidelity[assignment.lane_id] = fidelity
                    self.counters["elementary_lanes_succeeded" if success else "elementary_lanes_failed"] += 1

        maximum_state_delay = self.algorithm.control_processing_delay_ps
        for sender in sorted(self.routers):
            payload = {
                lane_id: success
                for lane_id, success in slot.edge_success.items()
                if self._lane_touches_node(slot, lane_id, sender)
            }
            for receiver in self._nodes_within_hops(sender, self.algorithm.link_state_hops):
                if receiver == sender:
                    continue
                delay = (
                    self.algorithm.control_processing_delay_ps
                    + int(self.routers[sender].cchannels[receiver].delay)
                )
                maximum_state_delay = max(maximum_state_delay, delay)
                self.routers[sender].send_message(
                    receiver,
                    QCASTMessage(QCASTMessageType.LINK_STATE, slot_id, payload),
                    sender_delay=self.algorithm.control_processing_delay_ps,
                )
                self.counters["link_state_messages_sent"] += 1

        p4_start = self.timeline.now() + maximum_state_delay + 1
        self.timeline.schedule(Event(p4_start, Process(self, "start_swapping", [slot_id])))
        self.events.append({
            "event": "generation_window_closed",
            "slot_id": slot_id,
            "time_ps": self.timeline.now(),
            "successful_lanes": sum(slot.edge_success.values()),
            "total_lanes": len(slot.edge_success),
            "p4_start_ps": p4_start,
        })

    def start_swapping(self, slot_id: int) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        expected_receipts = sum(
            len(self._nodes_within_hops(sender, self.algorithm.link_state_hops)) - 1
            for sender in self.routers
        )
        received_receipts = sum(
            len(senders) for senders in slot.link_state_receipts.values()
        )
        if received_receipts != expected_receipts:
            self.counters["slots_with_incomplete_link_state"] += 1
        self.events.append({
            "event": "recovery_decision_started",
            "slot_id": slot_id,
            "time_ps": self.timeline.now(),
            "link_state_receipts_expected": expected_receipts,
            "link_state_receipts_received": received_receipts,
        })
        latest_check = self.timeline.now()
        recovery_allocations = [
            allocation
            for allocation in slot.allocations.values()
            if allocation.path.kind == "recovery"
        ]
        recovery_by_parent = {}
        for allocation in recovery_allocations:
            recovery_by_parent.setdefault(allocation.path.parent_path_id, []).append(allocation)

        for major in slot.plan.major_paths:
            context = self._context_by_demand_id(major.demand_id)
            if context is None or context.terminal:
                continue
            allocation = slot.allocations[major.path_id]
            for lane in allocation.lanes:
                selected = self._select_physical_path(
                    slot,
                    lane,
                    recovery_by_parent.get(major.path_id, []),
                )
                if selected is None:
                    self.counters["major_lanes_unrecoverable"] += 1
                    continue
                nodes, assignments, used_recovery = selected
                if used_recovery:
                    major_assignment_ids = {edge.lane_id for edge in lane.edges}
                    for assignment in assignments:
                        if assignment.lane_id not in major_assignment_ids:
                            # A physical recovery lane can support only one
                            # final path. Marking one of its edges consumed
                            # makes that lane unavailable to later major lanes.
                            slot.edge_success[assignment.lane_id] = False
                delivery_id = f"slot-{slot_id}:{lane.lane_id}"
                if used_recovery:
                    self.counters["major_lanes_recovered"] += 1
                check_time = self._schedule_swap_chain(
                    slot,
                    delivery_id,
                    context.demand.reservation_id,
                    nodes,
                    assignments,
                    self.timeline.now(),
                    used_recovery,
                )
                latest_check = max(latest_check, check_time)

        slot.cleanup_time_ps = latest_check + 2
        self.timeline.schedule(Event(
            slot.cleanup_time_ps,
            Process(self, "finish_slot", [slot_id]),
        ))

    def perform_swap(
        self,
        slot_id: int,
        delivery_id: str,
        nodes: tuple[str, ...],
        assignments: tuple[QCASTLaneEdge, ...],
        stage: int,
    ) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id or delivery_id in slot.aborted_deliveries:
            return
        middle_name = nodes[stage]
        middle = self.routers[middle_name]
        left_memory = self._memory(middle_name, assignments[stage - 1].memory_index(middle_name))
        right_memory = self._memory(middle_name, assignments[stage].memory_index(middle_name))
        left_info = middle.resource_manager.memory_manager.get_info_by_memory(left_memory)
        right_info = middle.resource_manager.memory_manager.get_info_by_memory(right_memory)
        if left_info.state != MemoryInfo.ENTANGLED or right_info.state != MemoryInfo.ENTANGLED:
            slot.aborted_deliveries.add(delivery_id)
            self.counters["swap_chains_aborted_missing_link"] += 1
            return

        left_remote = self.routers[left_memory.entangled_memory["node_id"]]
        right_remote = self.routers[right_memory.entangled_memory["node_id"]]
        left_hold = self.timeline.get_entity_by_name(left_memory.entangled_memory["memo_id"])
        right_hold = self.timeline.get_entity_by_name(right_memory.entangled_memory["memo_id"])
        name = f"QCAST.SWAP.{delivery_id}.{stage}"
        protocol_a = EntanglementSwappingA.create(
            middle,
            name + ".A",
            left_memory,
            right_memory,
            success_prob=self.algorithm.swap_success_probability,
        )
        protocol_left = EntanglementSwappingB.create(left_remote, name + ".L", left_hold)
        protocol_right = EntanglementSwappingB.create(right_remote, name + ".R", right_hold)
        protocol_a.set_others(protocol_left.name, left_remote.name, [left_hold.name])
        protocol_a.set_others(protocol_right.name, right_remote.name, [right_hold.name])
        protocol_left.set_others(protocol_a.name, middle.name, [left_memory.name, right_memory.name])
        protocol_right.set_others(protocol_a.name, middle.name, [left_memory.name, right_memory.name])
        for owner, protocol in (
            (middle, protocol_a),
            (left_remote, protocol_left),
            (right_remote, protocol_right),
        ):
            owner.protocols.append(protocol)
            for memory in protocol.memories:
                memory.detach(memory.memory_array)
                memory.attach(protocol)
                owner.resource_manager.memory_manager.get_info_by_memory(memory).to_occupied()
        protocol_left.start()
        protocol_right.start()
        self.counters["swaps_attempted"] += 1
        protocol_a.start()

    def check_delivery(
        self,
        slot_id: int,
        delivery_id: str,
        reservation_id: int,
        nodes: tuple[str, ...],
        assignments: tuple[QCASTLaneEdge, ...],
        used_recovery: bool,
    ) -> None:
        slot = self.slot
        context = self.contexts.get(reservation_id)
        if slot is None or slot.slot_id != slot_id or context is None or context.terminal:
            return
        source_memory = self._memory(nodes[0], assignments[0].memory_index(nodes[0]))
        destination_memory = self._memory(nodes[-1], assignments[-1].memory_index(nodes[-1]))
        source_info = self.routers[nodes[0]].resource_manager.memory_manager.get_info_by_memory(source_memory)
        destination_info = self.routers[nodes[-1]].resource_manager.memory_manager.get_info_by_memory(destination_memory)
        delivered = (
            delivery_id not in slot.aborted_deliveries
            and source_info.state == MemoryInfo.ENTANGLED
            and destination_info.state == MemoryInfo.ENTANGLED
            and source_info.remote_node == nodes[-1]
            and destination_info.remote_node == nodes[0]
        )
        if not delivered:
            self.counters["end_to_end_pairs_failed"] += 1
            return
        fidelity = min(source_info.fidelity, destination_info.fidelity)
        if fidelity < context.demand.fidelity_threshold:
            self.counters["pairs_rejected_fidelity"] += 1
            context.callbacks.on_pair_rejected(context.demand, fidelity)
            return
        qcast_role = "recovery" if used_recovery else "major"
        elementary_sources = tuple({
            "source": "application",
            "qcast_role": qcast_role,
            "link": assignment.key,
        } for assignment in assignments)
        delivery = PairDelivery(
            timestamp_ps=self.timeline.now(),
            fidelity=fidelity,
            generation_source="application",
            elementary_sources=elementary_sources,
        )
        context.deliveries.append(delivery)
        context.callbacks.on_pair_delivered(context.demand, delivery)
        self.counters["end_to_end_pairs_delivered"] += 1
        if used_recovery:
            self.counters["end_to_end_pairs_delivered_recovery"] += 1
        self.events.append({
            "event": "pair_delivered",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "fidelity": fidelity,
            "path": list(nodes),
            "qcast_role": qcast_role,
        })
        if len(context.deliveries) >= context.demand.pair_count:
            context.terminal = True
            self.counters["demands_completed"] += 1
            self.events.append({
                "event": "demand_completed",
                "time_ps": self.timeline.now(),
                "demand_id": context.demand.demand_id,
                "pairs_delivered": len(context.deliveries),
                "path": list(nodes),
            })
            context.callbacks.on_demand_completed(
                context.demand,
                self.timeline.now(),
                nodes,
            )

    def finish_slot(self, slot_id: int) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        self._reset_slot_memories(slot)
        self.counters["slots_completed"] += 1
        self.events.append({
            "event": "slot_completed",
            "slot_id": slot_id,
            "time_ps": self.timeline.now(),
        })
        self.slot = None
        if any(not context.terminal for context in self.contexts.values()):
            self._schedule_slot(self.timeline.now() + 1)

    def finalize(self, now_ps: int) -> None:
        if self.slot is not None:
            self._expire_slot_rules(self.slot)
            self._reset_slot_memories(self.slot)
            self.counters["slots_cleaned_at_simulation_end"] += 1
            self.slot = None
        for context in self.contexts.values():
            if context.terminal:
                continue
            context.terminal = True
            self.events.append({
                "event": "demand_failed",
                "time_ps": now_ps,
                "demand_id": context.demand.demand_id,
                "reason": "simulation_end",
                "pairs_delivered": len(context.deliveries),
            })
            context.callbacks.on_demand_failed(
                context.demand,
                now_ps,
                "simulation_end",
                context.paths[0] if context.paths else (),
                len(context.deliveries),
            )

    def diagnostics(self) -> dict:
        all_raw = {}
        for name, router in self.routers.items():
            all_raw[name] = all(
                info.state == MemoryInfo.RAW
                for info in router.resource_manager.memory_manager
            )
        return {
            "controller_node": self.controller.name,
            "counters": dict(self.counters),
            "events": self.events,
            "max_allocated_memories_by_node": dict(self.max_allocated_by_node),
            "all_memories_raw_at_end": all_raw,
            "edge_models": [
                {
                    "edge": list(edge.key),
                    "width": edge.width,
                    "success_probability": edge.success_probability,
                    **self._edge_model_details[edge.key],
                }
                for edge in self._edge_models
            ],
            "edge_width_model": "independent_midpoint_bsm_channels",
            "control_state_by_node": {
                name: {
                    "plan_slots": sorted(router.qcast_control.plan_by_slot),
                    "link_state_senders_by_slot": {
                        str(slot_id): sorted(messages)
                        for slot_id, messages in router.qcast_control.link_state_by_slot.items()
                    },
                }
                for name, router in self.routers.items()
            },
        }

    def _schedule_slot(self, time_ps: int) -> None:
        if self.slot_event_pending:
            return
        self.slot_event_pending = True
        self.timeline.schedule(Event(time_ps, Process(self, "run_slot", []), 1))

    def _expire_slot_rules(self, slot: _Slot) -> None:
        for node, rule in list(slot.rules):
            for protocol in list(rule.protocols):
                for event in list(getattr(protocol, "scheduled_events", ())):
                    if event.time >= self.timeline.now():
                        self.timeline.remove_event(event)
            if rule in node.resource_manager.rule_manager.rules:
                node.resource_manager.expire(rule)
        slot.rules.clear()

    def _reset_slot_memories(self, slot: _Slot) -> None:
        for node_name, indices in slot.allocated_memories.items():
            node = self.routers[node_name]
            for index in indices:
                memory = self._memory(node_name, index)
                info = node.resource_manager.memory_manager.get_info_by_memory(memory)
                if info.state != MemoryInfo.RAW:
                    node.resource_manager.update(None, memory, MemoryInfo.RAW)

    def _free_memory_indices(self) -> dict[str, list[int]]:
        return {
            name: [
                info.index
                for info in router.resource_manager.memory_manager
                if info.state == MemoryInfo.RAW
            ]
            for name, router in self.routers.items()
        }

    def _allocate_memories(self, slot_id: int, plan, free_memories):
        pools = {node: deque(sorted(indices)) for node, indices in free_memories.items()}
        middle_pools = {
            edge: deque(sorted(middles))
            for edge, middles in self._edge_middle_nodes.items()
        }
        allocations = {}
        allocated = {node: set() for node in self.routers}
        identity = slot_id * 1_000_000
        for path in (*plan.major_paths, *plan.recovery_paths):
            lanes = []
            for lane_index in range(path.width):
                assignments = []
                for edge_index, (left, right) in enumerate(zip(path.nodes, path.nodes[1:])):
                    middle = middle_pools[edge_key(left, right)].popleft()
                    left_index = pools[left].popleft()
                    right_index = pools[right].popleft()
                    allocated[left].add(left_index)
                    allocated[right].add(right_index)
                    lane_id = f"slot-{slot_id}:{path.path_id}:{lane_index}:{edge_index}"
                    reservation = QCASTReservation(
                        left,
                        right,
                        self.timeline.now() + 1,
                        self.timeline.now() + self.algorithm.generation_window_ps + 10**10,
                        1,
                        0.5,
                        1,
                        identity,
                        slot_id=slot_id,
                        lane_id=lane_id,
                    )
                    reservation.set_path([left, right])
                    identity += 1
                    assignments.append(QCASTLaneEdge(
                        lane_id,
                        left,
                        right,
                        left_index,
                        right_index,
                        reservation,
                        middle,
                    ))
                lane_name = f"slot-{slot_id}:{path.path_id}:{lane_index}"
                lanes.append(QCASTLane(lane_name, path, lane_index, tuple(assignments)))
            allocations[path.path_id] = QCASTPathAllocation(path, tuple(lanes))
        return allocations, allocated

    def _assignment_state(self, assignment: QCASTLaneEdge) -> tuple[bool, float]:
        left_memory = self._memory(assignment.left, assignment.left_memory_index)
        right_memory = self._memory(assignment.right, assignment.right_memory_index)
        left_info = self.routers[assignment.left].resource_manager.memory_manager.get_info_by_memory(left_memory)
        right_info = self.routers[assignment.right].resource_manager.memory_manager.get_info_by_memory(right_memory)
        success = (
            left_info.state == MemoryInfo.ENTANGLED
            and right_info.state == MemoryInfo.ENTANGLED
            and left_info.remote_node == assignment.right
            and right_info.remote_node == assignment.left
            and left_memory.entangled_memory["memo_id"] == right_memory.name
            and right_memory.entangled_memory["memo_id"] == left_memory.name
        )
        return success, min(left_info.fidelity, right_info.fidelity) if success else 0.0

    def _select_physical_path(self, slot: _Slot, major_lane: QCASTLane, recoveries):
        major_assignments = {
            assignment.key: assignment
            for assignment in major_lane.edges
            if slot.edge_success.get(assignment.lane_id, False)
        }
        failed_edge_indices = {
            index
            for index, assignment in enumerate(major_lane.edges)
            if not slot.edge_success.get(assignment.lane_id, False)
        }
        if not failed_edge_indices:
            return major_lane.path.nodes, major_lane.edges, False

        candidates = []
        for allocation in recoveries:
            segment = allocation.path.covered_segment
            if segment is None:
                continue
            start, end = segment
            if not any(start <= index < end for index in failed_edge_indices):
                continue
            lane = next(
                (
                    candidate
                    for candidate in allocation.lanes
                    if all(
                        slot.edge_success.get(edge.lane_id, False)
                        for edge in candidate.edges
                    )
                ),
                None,
            )
            if lane is not None:
                candidates.append((start, end, lane))

        source = major_lane.path.nodes[0]
        destination = major_lane.path.nodes[-1]
        candidates.sort(key=lambda item: (item[1] - item[0], item[0], item[2].lane_id))
        for count in range(1, len(candidates) + 1):
            candidate_sets = sorted(
                combinations(candidates, count),
                key=lambda items: (
                    sum(len(item[2].edges) for item in items),
                    tuple(item[2].lane_id for item in items),
                ),
            )
            for selected in candidate_sets:
                intervals = sorted((start, end) for start, end, _lane in selected)
                if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                    continue
                covered = {
                    index
                    for start, end, _lane in selected
                    for index in range(start, end)
                }
                if not failed_edge_indices.issubset(covered):
                    continue

                available = dict(major_assignments)
                recovery_lane_ids = set()
                for _start, _end, lane in selected:
                    for assignment in lane.edges:
                        available.setdefault(assignment.key, assignment)
                        recovery_lane_ids.add(assignment.lane_id)
                path = self._bfs_path(source, destination, set(available))
                if path is None:
                    continue
                assignments = tuple(
                    available[edge_key(left, right)]
                    for left, right in zip(path, path[1:])
                )
                if not any(
                    assignment.lane_id in recovery_lane_ids
                    for assignment in assignments
                ):
                    continue
                return path, assignments, True
        return None

    def _schedule_swap_chain(
        self,
        slot: _Slot,
        delivery_id: str,
        reservation_id: int,
        nodes: tuple[str, ...],
        assignments: tuple[QCASTLaneEdge, ...],
        start_time: int,
        used_recovery: bool,
    ) -> int:
        if len(nodes) == 2:
            check_time = start_time + 1
        else:
            swap_time = start_time
            for stage in range(1, len(nodes) - 1):
                self.timeline.schedule(Event(
                    swap_time,
                    Process(self, "perform_swap", [
                        slot.slot_id,
                        delivery_id,
                        nodes,
                        assignments,
                        stage,
                    ]),
                ))
                if stage < len(nodes) - 2:
                    swap_time += int(self.routers[nodes[stage]].cchannels[nodes[stage + 1]].delay) + 1
            final_middle = self.routers[nodes[-2]]
            check_time = swap_time + max(
                int(final_middle.cchannels[nodes[0]].delay),
                int(final_middle.cchannels[nodes[-1]].delay),
            ) + 1
        self.timeline.schedule(Event(
            check_time,
            Process(self, "check_delivery", [
                slot.slot_id,
                delivery_id,
                reservation_id,
                nodes,
                assignments,
                used_recovery,
            ]),
        ))
        return check_time

    def _router_graph(self) -> dict[str, set[str]]:
        graph = {name: set() for name in self.routers}
        for left, right in self._edge_middle_nodes:
            graph[left].add(right)
            graph[right].add(left)
        return graph

    def _middle_nodes_by_edge(self) -> dict[tuple[str, str], tuple[str, ...]]:
        middles: dict[tuple[str, str], list[str]] = {}
        for middle, endpoints in self.network_topology.bsm_to_router_map.items():
            if len(endpoints) != 2:
                raise ValueError(f"Q-CAST BSM {middle!r} does not have two endpoints")
            key = edge_key(*endpoints)
            middles.setdefault(key, []).append(middle)
        return {
            key: tuple(sorted(names))
            for key, names in middles.items()
        }

    def _physical_edge_models(self) -> tuple[QCASTEdge, ...]:
        models = []
        for left in sorted(self.routers):
            for right in sorted(self._graph[left]):
                if left >= right:
                    continue
                left_node = self.routers[left]
                right_node = self.routers[right]
                middle_names = self._edge_middle_nodes[edge_key(left, right)]
                channel_details = []
                for middle_name in middle_names:
                    channel_details.append(self._physical_channel_model(
                        left_node,
                        right_node,
                        middle_name,
                    ))
                slot_probability = sum(
                    detail["success_probability"] for detail in channel_details
                ) / len(channel_details)
                key = edge_key(left, right)
                first = channel_details[0]
                self._edge_model_details[key] = {
                    "attempt_success_probability": first["attempt_success_probability"],
                    "attempt_duration_ps": first["attempt_duration_ps"],
                    "attempts_per_generation_window": first["attempts_per_generation_window"],
                    "bsm_success_probability": first["bsm_success_probability"],
                    "detector_efficiencies": first["detector_efficiencies"],
                    "physical_channels": list(middle_names),
                }
                models.append(QCASTEdge(
                    left,
                    right,
                    len(middle_names),
                    min(1.0, max(0.0, slot_probability)),
                ))
        return tuple(models)

    def _physical_channel_model(self, left_node, right_node, middle_name: str) -> dict:
        left_channel = left_node.qchannels[middle_name]
        right_channel = right_node.qchannels[middle_name]
        left_transmittance = 10 ** (
            -left_channel.distance * left_channel.attenuation / 10
        )
        right_transmittance = 10 ** (
            -right_channel.distance * right_channel.attenuation / 10
        )
        left_memory = left_node.components[left_node.memo_arr_name][0]
        right_memory = right_node.components[right_node.memo_arr_name][0]
        bsm_node = self.timeline.get_entity_by_name(middle_name)
        bsm = next(iter(bsm_node.components.values()))
        detector_efficiencies = tuple(
            detector.efficiency for detector in bsm.detectors
        )
        attempt_probability = single_heralded_attempt_success_probability(
            left_memory.efficiency * left_transmittance,
            right_memory.efficiency * right_transmittance,
            detector_efficiencies,
            bsm.success_rate,
        )

        router_delay = max(
            int(left_node.cchannels[right_node.name].delay),
            int(right_node.cchannels[left_node.name].delay),
        )
        bsm_delay = max(
            int(bsm_node.cchannels[left_node.name].delay),
            int(bsm_node.cchannels[right_node.name].delay),
        )
        quantum_delay = max(
            round(left_channel.distance / left_channel.light_speed),
            round(right_channel.distance / right_channel.light_speed),
        )
        # Resource-manager pairing, emission negotiation, midpoint
        # propagation, and the herald response determine retry pace.
        # This matches the native v1.0.0 trace for symmetric links.
        attempt_duration = max(
            1,
            2 * router_delay + 2 * (quantum_delay + bsm_delay),
        )
        attempts = self.algorithm.generation_window_ps // attempt_duration
        slot_probability = 1 - (1 - attempt_probability) ** attempts
        return {
            "attempt_success_probability": attempt_probability,
            "attempt_duration_ps": attempt_duration,
            "attempts_per_generation_window": attempts,
            "bsm_success_probability": bsm.success_rate,
            "detector_efficiencies": list(detector_efficiencies),
            "success_probability": slot_probability,
        }

    def _nodes_within_hops(self, source: str, hops: int) -> set[str]:
        if hops <= 0:
            return {source}
        reached = {source}
        frontier = {source}
        for _ in range(hops):
            frontier = {
                neighbor
                for node in frontier
                for neighbor in self._graph[node]
                if neighbor not in reached
            }
            reached.update(frontier)
        return reached

    def _lane_touches_node(self, slot: _Slot, lane_id: str, node: str) -> bool:
        for allocation in slot.allocations.values():
            for lane in allocation.lanes:
                for assignment in lane.edges:
                    if assignment.lane_id == lane_id:
                        return node in (assignment.left, assignment.right)
        return False

    def _memory(self, node_name: str, index: int):
        node = self.routers[node_name]
        return node.components[node.memo_arr_name][index]

    def _local_plan_payload(self, slot: _Slot, node_name: str) -> dict:
        assignments = []
        for allocation in slot.allocations.values():
            for lane in allocation.lanes:
                for assignment in lane.edges:
                    if node_name not in (assignment.left, assignment.right):
                        continue
                    assignments.append({
                        "path_id": allocation.path.path_id,
                        "path_kind": allocation.path.kind,
                        "lane_id": lane.lane_id,
                        "lane_edge_id": assignment.lane_id,
                        "neighbor": (
                            assignment.right
                            if node_name == assignment.left
                            else assignment.left
                        ),
                        "memory_index": assignment.memory_index(node_name),
                        "middle": assignment.middle,
                        "role": (
                            "request" if node_name == assignment.left else "await"
                        ),
                    })
        return {"assignments": assignments}

    @staticmethod
    def _assignments_by_id(slot: _Slot) -> dict[str, QCASTLaneEdge]:
        return {
            assignment.lane_id: assignment
            for allocation in slot.allocations.values()
            for lane in allocation.lanes
            for assignment in lane.edges
        }

    def _context_by_demand_id(self, demand_id: str):
        for context in self.contexts.values():
            if context.demand.demand_id == demand_id:
                return context
        return None

    @staticmethod
    def _path_record(path: QCASTPath) -> dict:
        return {
            "path_id": path.path_id,
            "demand_id": path.demand_id,
            "nodes": list(path.nodes),
            "width": path.width,
            "ext": path.ext,
            "kind": path.kind,
            "parent_path_id": path.parent_path_id,
            "covered_segment": list(path.covered_segment) if path.covered_segment else None,
        }

    @staticmethod
    def _bfs_path(source: str, destination: str, edges: set[tuple[str, str]]):
        neighbors: dict[str, set[str]] = {}
        for left, right in edges:
            neighbors.setdefault(left, set()).add(right)
            neighbors.setdefault(right, set()).add(left)
        queue = deque([(source, (source,))])
        seen = {source}
        while queue:
            node, path = queue.popleft()
            if node == destination:
                return path
            for neighbor in sorted(neighbors.get(node, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + (neighbor,)))
        return None
