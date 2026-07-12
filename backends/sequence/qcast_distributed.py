"""Paper-local Q-CAST P3/P4 execution over SeQUeNCe resources."""

from __future__ import annotations

from collections import deque
from itertools import combinations

from sequence.kernel.event import Event
from sequence.kernel.process import Process

from backends.sequence.qcast_protocol import QCASTMessageType
from backends.sequence.qcast_scheduler import (
    QCASTDemandScheduler,
    QCASTLane,
    QCASTLaneEdge,
    QCASTPathAllocation,
    _Slot,
)


class QCASTDistributedDemandScheduler(QCASTDemandScheduler):
    """Execute Q-CAST P4 when each selected path's local P3 view is ready.

    The class remains a simulation coordinator, but it does not wait for or
    inspect network-wide link state when making recovery decisions. A path's
    decision input is assembled only from the local views stored at routers on
    that major path and its pre-reserved recovery paths.
    """

    def __init__(self, network_topology, algorithm, controller_node: str):
        super().__init__(network_topology, algorithm, controller_node)
        self._started_major_paths: set[tuple[int, str]] = set()
        self._major_path_check_ps: dict[tuple[int, str], int] = {}
        self._finish_scheduled_slots: set[int] = set()
        self._consumed_recovery_lanes: dict[int, set[str]] = {}
        self.distributed_decisions: list[dict] = []

    def received_control_message(self, receiver: str, src: str, msg) -> None:
        super().received_control_message(receiver, src, msg)
        if msg.msg_type is QCASTMessageType.LINK_STATE:
            self._try_start_ready_major_paths(msg.slot_id)

    def stop_generation(self, slot_id: int) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        self._close_generation(slot)
        self._broadcast_link_state(slot)
        self._consumed_recovery_lanes[slot_id] = set()
        self.events.append({
            "event": "generation_window_closed",
            "slot_id": slot_id,
            "time_ps": self.timeline.now(),
            "successful_lanes": sum(slot.edge_success.values()),
            "total_lanes": len(slot.edge_success),
            "p4_start_ps": None,
            "control_mode": self.algorithm.control_mode,
        })
        self._try_start_ready_major_paths(slot_id)

    def _try_start_ready_major_paths(self, slot_id: int) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id or not slot.edge_success:
            return
        for major in slot.plan.major_paths:
            key = (slot_id, major.path_id)
            if key in self._started_major_paths:
                continue
            scope = self._decision_scope(slot, major.path_id)
            if not scope.issubset(slot.local_p3_ready_ps):
                continue
            self._started_major_paths.add(key)
            ready_time = max(
                (slot.local_p3_ready_ps[node] for node in scope),
                default=self.timeline.now(),
            )
            start_time = max(self.timeline.now(), ready_time) + 1
            slot.path_p4_start_ps[major.path_id] = start_time
            self.timeline.schedule(Event(
                start_time,
                Process(self, "start_distributed_major_path", [
                    slot_id,
                    major.path_id,
                ]),
            ))

    def start_distributed_major_path(self, slot_id: int, major_path_id: str) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        major = next(
            path for path in slot.plan.major_paths
            if path.path_id == major_path_id
        )
        scope = self._decision_scope(slot, major_path_id)
        recovery_allocations = [
            allocation
            for allocation in slot.allocations.values()
            if allocation.path.kind == "recovery"
            and allocation.path.parent_path_id == major_path_id
        ]
        decision_lane_ids = {
            edge.lane_id
            for allocation in (
                slot.allocations[major_path_id],
                *recovery_allocations,
            )
            for lane in allocation.lanes
            for edge in lane.edges
        }
        visible_state, witnesses = self._path_local_link_state(
            slot,
            scope,
            decision_lane_ids,
        )
        context = self._context_by_demand_id(major.demand_id)
        latest_check = self.timeline.now()
        selected_records = []
        if context is not None and not context.terminal:
            allocation = slot.allocations[major_path_id]
            for lane in allocation.lanes:
                selected = self._select_xor_path(
                    lane,
                    recovery_allocations,
                    visible_state,
                    self._consumed_recovery_lanes[slot_id],
                )
                if selected is None:
                    self.counters["major_lanes_unrecoverable"] += 1
                    selected_records.append({
                        "major_lane": lane.lane_id,
                        "selected_path": None,
                    })
                    continue
                nodes, assignments, consumed = selected
                used_recovery = bool(consumed)
                self._consumed_recovery_lanes[slot_id].update(consumed)
                if used_recovery:
                    self.counters["major_lanes_recovered"] += 1
                delivery_id = f"slot-{slot_id}:{lane.lane_id}"
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
                selected_records.append({
                    "major_lane": lane.lane_id,
                    "selected_path": list(nodes),
                    "used_recovery": used_recovery,
                    "consumed_recovery_lane_edges": sorted(consumed),
                })

        self._major_path_check_ps[(slot_id, major_path_id)] = latest_check
        self.distributed_decisions.append({
            "slot_id": slot_id,
            "major_path_id": major_path_id,
            "demand_id": major.demand_id,
            "decision_ps": self.timeline.now(),
            "scope_nodes": sorted(scope),
            "decision_lane_ids": sorted(decision_lane_ids),
            "visible_lane_ids": sorted(visible_state),
            "visible_lane_states": len(visible_state),
            "decision_lane_states": len(decision_lane_ids),
            "global_allocated_lane_states": len(slot.edge_success),
            "witnesses_by_lane": {
                lane_id: sorted(nodes)
                for lane_id, nodes in sorted(witnesses.items())
            },
            "selections": selected_records,
        })
        self.counters["distributed_major_path_decisions"] += 1
        self._schedule_finish_when_all_paths_started(slot)

    def _schedule_finish_when_all_paths_started(self, slot: _Slot) -> None:
        expected = {
            (slot.slot_id, path.path_id)
            for path in slot.plan.major_paths
        }
        if (
            slot.slot_id in self._finish_scheduled_slots
            or not expected.issubset(self._major_path_check_ps)
        ):
            return
        finish_time = max(
            self._major_path_check_ps[key]
            for key in expected
        ) + 2
        slot.cleanup_time_ps = finish_time
        self._finish_scheduled_slots.add(slot.slot_id)
        self.timeline.schedule(Event(
            finish_time,
            Process(self, "finish_slot", [slot.slot_id]),
        ))

    def _decision_scope(self, slot: _Slot, major_path_id: str) -> set[str]:
        scope = set(slot.allocations[major_path_id].path.nodes)
        for allocation in slot.allocations.values():
            if allocation.path.parent_path_id == major_path_id:
                scope.update(allocation.path.nodes)
        return scope

    def _path_local_link_state(
        self,
        slot: _Slot,
        scope: set[str],
        decision_lane_ids: set[str],
    ) -> tuple[dict[str, bool], dict[str, set[str]]]:
        visible: dict[str, bool] = {}
        witnesses: dict[str, set[str]] = {}
        for observer in sorted(scope):
            local = {
                lane_id: success
                for lane_id, success in slot.edge_success.items()
                if lane_id in decision_lane_ids
                and self._lane_touches_node(slot, lane_id, observer)
            }
            received = self.routers[observer].qcast_control.link_state_by_slot.get(
                slot.slot_id,
                {},
            )
            for payload in received.values():
                local.update({
                    lane_id: success
                    for lane_id, success in payload.items()
                    if lane_id in decision_lane_ids
                })
            for lane_id, success in local.items():
                if lane_id in visible and visible[lane_id] != success:
                    raise RuntimeError(
                        f"Conflicting local Q-CAST state for {lane_id}"
                    )
                visible[lane_id] = success
                witnesses.setdefault(lane_id, set()).add(observer)
        return visible, witnesses

    def _select_xor_path(
        self,
        major_lane: QCASTLane,
        recovery_allocations: list[QCASTPathAllocation],
        visible_state: dict[str, bool],
        consumed_recovery_lanes: set[str],
    ) -> tuple[tuple[str, ...], tuple[QCASTLaneEdge, ...], set[str]] | None:
        candidates: list[tuple[QCASTPathAllocation, QCASTLane]] = []
        failed_major_edges = {
            index
            for index, assignment in enumerate(major_lane.edges)
            if not visible_state.get(assignment.lane_id, False)
        }
        for allocation in recovery_allocations:
            segment = allocation.path.covered_segment
            if segment is None:
                continue
            start, end = segment
            if failed_major_edges and not any(
                start <= index < end for index in failed_major_edges
            ):
                continue
            lane = next((
                candidate
                for candidate in allocation.lanes
                if not any(
                    edge.lane_id in consumed_recovery_lanes
                    for edge in candidate.edges
                )
                and all(
                    visible_state.get(edge.lane_id, False)
                    for edge in candidate.edges
                )
            ), None)
            if lane is not None:
                candidates.append((allocation, lane))
        candidates.sort(key=lambda item: (
            len(item[0].path.nodes),
            item[0].path.covered_segment,
            item[0].path.path_id,
            item[1].lane_id,
        ))

        for count in range(len(candidates) + 1):
            for selected in combinations(candidates, count):
                active = {
                    assignment.lane_id: assignment
                    for assignment in major_lane.edges
                }
                consumed = set()
                for allocation, lane in selected:
                    start, end = allocation.path.covered_segment
                    for assignment in major_lane.edges[start:end]:
                        self._xor_toggle(active, assignment)
                    for assignment in lane.edges:
                        self._xor_toggle(active, assignment)
                        consumed.add(assignment.lane_id)
                if not all(
                    visible_state.get(assignment.lane_id, False)
                    for assignment in active.values()
                ):
                    continue
                path = self._assignment_path(
                    major_lane.path.nodes[0],
                    major_lane.path.nodes[-1],
                    tuple(active.values()),
                )
                if path is not None:
                    nodes, assignments = path
                    return nodes, assignments, consumed
        return None

    @staticmethod
    def _xor_toggle(
        active: dict[str, QCASTLaneEdge],
        assignment: QCASTLaneEdge,
    ) -> None:
        if assignment.lane_id in active:
            del active[assignment.lane_id]
        else:
            active[assignment.lane_id] = assignment

    @staticmethod
    def _assignment_path(
        source: str,
        destination: str,
        assignments: tuple[QCASTLaneEdge, ...],
    ) -> tuple[tuple[str, ...], tuple[QCASTLaneEdge, ...]] | None:
        adjacency: dict[str, list[tuple[str, QCASTLaneEdge]]] = {}
        for assignment in assignments:
            adjacency.setdefault(assignment.left, []).append((
                assignment.right,
                assignment,
            ))
            adjacency.setdefault(assignment.right, []).append((
                assignment.left,
                assignment,
            ))
        queue = deque([source])
        previous: dict[str, tuple[str, QCASTLaneEdge] | None] = {source: None}
        while queue:
            node = queue.popleft()
            if node == destination:
                break
            for neighbor, assignment in sorted(
                adjacency.get(node, ()),
                key=lambda item: (item[0], item[1].lane_id),
            ):
                if neighbor in previous:
                    continue
                previous[neighbor] = (node, assignment)
                queue.append(neighbor)
        if destination not in previous:
            return None
        nodes = [destination]
        edges = []
        while nodes[-1] != source:
            parent, assignment = previous[nodes[-1]]
            edges.append(assignment)
            nodes.append(parent)
        nodes.reverse()
        edges.reverse()
        return tuple(nodes), tuple(edges)

    def diagnostics(self) -> dict:
        diagnostics = super().diagnostics()
        diagnostics["control_mode"] = self.algorithm.control_mode
        diagnostics["distributed_decisions"] = self.distributed_decisions
        diagnostics["locality_invariants"] = {
            "network_wide_p4_barriers": 0,
            "decisions": len(self.distributed_decisions),
            "all_decisions_path_scoped": all(
                set(decision["visible_lane_ids"]).issubset(
                    decision["decision_lane_ids"]
                )
                for decision in self.distributed_decisions
            ),
        }
        return diagnostics
