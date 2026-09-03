"""Paper-local Q-GUARD fidelity planning and physical execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from sequence.entanglement_management.purification.bbpssw_protocol import (
    BBPSSWProtocol,
)
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.resource_management.memory_manager import MemoryInfo

from algorithms.qcast import edge_key
from algorithms.qguard import (
    QGUARDHopPlan,
    detour_split_target,
    equal_split_target,
    expected_goodput,
    minimum_purification_rounds,
)
from backends.sequence.qcast_distributed import QCASTDistributedDemandScheduler
from backends.sequence.qcast_protocol import QCASTMessage, QCASTMessageType
from backends.sequence.qcast_scheduler import (
    QCASTLaneEdge,
    QCASTPathAllocation,
    _Slot,
)
from workloads.base import PairDelivery


@dataclass
class _PairRef:
    pair_id: str
    left: str
    right: str
    left_memory_index: int
    right_memory_index: int
    carrier: QCASTLaneEdge | None
    path: tuple[str, ...]
    provenance: tuple[QCASTLaneEdge, ...]
    used_recovery: bool = False
    p4_start_ps: int | None = None

    def memory_index(self, node: str) -> int:
        if node == self.left:
            return self.left_memory_index
        if node == self.right:
            return self.right_memory_index
        raise KeyError(node)


@dataclass
class _RouteCandidate:
    nodes: tuple[str, ...]
    assignment_ids_by_edge: dict[tuple[str, str], tuple[str, ...]]
    targets_by_edge: dict[tuple[str, str], float]
    required_by_edge: dict[tuple[str, str], int]
    score: float
    feasible: bool
    used_recovery: bool
    recovery_path_ids: tuple[str, ...]
    exg_record: dict


@dataclass
class _MajorState:
    slot_id: int
    major_path_id: str
    demand_id: str
    assignment_by_id: dict[str, QCASTLaneEdge]
    allocation_assignment_ids: dict[str, set[str]]
    available_assignment_ids: set[str]
    visible_success: dict[str, bool]
    visible_fidelity: dict[str, float]
    output_attempts: int = 0
    active_output: bool = False
    finished: bool = False


@dataclass
class _EdgeWorkflow:
    workflow_id: str
    major_key: tuple[int, str]
    output_id: str
    edge: tuple[str, str]
    target_fidelity: float
    pool: list[_PairRef]
    complete: bool = False


@dataclass
class _OutputState:
    output_id: str
    major_key: tuple[int, str]
    reservation_id: int
    nodes: tuple[str, ...]
    targets_by_edge: dict[tuple[str, str], float]
    used_recovery: bool
    recovery_path_ids: tuple[str, ...]
    workflow_ids: set[str] = field(default_factory=set)
    results_by_edge: dict[tuple[str, str], _PairRef | None] = field(default_factory=dict)


@dataclass
class _PurificationAttempt:
    attempt_id: str
    slot_id: int
    purpose: str
    owner_id: str | int
    kept: _PairRef
    measured: _PairRef
    started_at_ps: int
    completion_ps: int


class QGUARDDemandScheduler(QCASTDistributedDemandScheduler):
    """Add Q-GUARD Phase 4/5 to the validated paper-local Q-CAST runtime.

    Phase 1 path dissemination, Phase 2 EXT planning/reservation, and Phase 3
    elementary generation/local link-state exchange remain Q-CAST behavior.
    This class consumes only path-scoped Phase 3 state and performs Q-GUARD's
    equal-split planning, EXG recovery ranking, BBPSSW, swapping, and final
    request-fidelity qualification.
    """

    def __init__(self, network_topology, algorithm, controller_node: str):
        super().__init__(network_topology, algorithm, controller_node)
        self._major_states: dict[tuple[int, str], _MajorState] = {}
        self._outputs: dict[str, _OutputState] = {}
        self._edge_workflows: dict[str, _EdgeWorkflow] = {}
        self._purification_attempts: dict[str, _PurificationAttempt] = {}
        self._delivery_metadata: dict[str, dict] = {}
        self._slot_pairs: dict[int, dict[int, list[_PairRef]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._slot_active_delivery_checks: dict[int, int] = defaultdict(int)
        self._slot_finished_majors: dict[int, set[str]] = defaultdict(set)
        self._slot_qualification_started: set[int] = set()
        self._slot_qualification_pending: dict[int, set[int]] = defaultdict(set)
        self._slot_finish_scheduled: set[int] = set()
        self._purification_counter = 0
        self.qguard_decisions: list[dict] = []
        self.purification_events: list[dict] = []

    def _broadcast_link_state(self, slot: _Slot) -> int:
        """Exchange success and realized fidelity in the existing P3 round."""
        maximum_state_delay = self.algorithm.control_processing_delay_ps
        for router_name in self.routers:
            expected = set(self._nodes_within_hops(
                router_name,
                self.algorithm.link_state_hops,
            )) - {router_name}
            if not expected:
                slot.local_p3_ready_ps.setdefault(router_name, self.timeline.now())
        for sender in sorted(self.routers):
            payload = {
                lane_id: {
                    "success": success,
                    "fidelity": slot.edge_fidelity.get(lane_id, 0.0),
                }
                for lane_id, success in slot.edge_success.items()
                if self._lane_touches_node(slot, lane_id, sender)
            }
            for receiver in self._nodes_within_hops(
                sender,
                self.algorithm.link_state_hops,
            ):
                if receiver == sender:
                    continue
                delay = (
                    self.algorithm.control_processing_delay_ps
                    + int(self.routers[sender].cchannels[receiver].delay)
                )
                maximum_state_delay = max(maximum_state_delay, delay)
                self.routers[sender].send_message(
                    receiver,
                    QCASTMessage(QCASTMessageType.LINK_STATE, slot.slot_id, payload),
                    sender_delay=self.algorithm.control_processing_delay_ps,
                )
                self.counters["link_state_messages_sent"] += 1
                self.counters["qguard_fidelity_records_sent"] += len(payload)
        return maximum_state_delay

    def start_distributed_major_path(self, slot_id: int, major_path_id: str) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        major = next(
            path for path in slot.plan.major_paths
            if path.path_id == major_path_id
        )
        scope = self._decision_scope(slot, major_path_id)
        allocations = [
            allocation
            for allocation in slot.allocations.values()
            if allocation.path.path_id == major_path_id
            or allocation.path.parent_path_id == major_path_id
        ]
        decision_lane_ids = {
            assignment.lane_id
            for allocation in allocations
            for lane in allocation.lanes
            for assignment in lane.edges
        }
        visible_success, visible_fidelity, witnesses = self._path_local_qguard_state(
            slot,
            scope,
            decision_lane_ids,
        )
        context = self._context_by_demand_id(major.demand_id)
        key = (slot_id, major_path_id)
        assignment_by_id = {
            assignment.lane_id: assignment
            for allocation in allocations
            for lane in allocation.lanes
            for assignment in lane.edges
        }
        allocation_assignment_ids = {
            allocation.path.path_id: {
                assignment.lane_id
                for lane in allocation.lanes
                for assignment in lane.edges
            }
            for allocation in allocations
        }
        self._major_states[key] = _MajorState(
            slot_id=slot_id,
            major_path_id=major_path_id,
            demand_id=major.demand_id,
            assignment_by_id=assignment_by_id,
            allocation_assignment_ids=allocation_assignment_ids,
            available_assignment_ids={
                lane_id
                for lane_id in decision_lane_ids
                if visible_success.get(lane_id, False)
            },
            visible_success=visible_success,
            visible_fidelity=visible_fidelity,
        )
        self.qguard_decisions.append({
            "event": "major_path_phase4_started",
            "slot_id": slot_id,
            "major_path_id": major_path_id,
            "demand_id": major.demand_id,
            "decision_ps": self.timeline.now(),
            "scope_nodes": sorted(scope),
            "decision_lane_ids": sorted(decision_lane_ids),
            "visible_lane_ids": sorted(visible_success),
            "visible_fidelity_lane_ids": sorted(visible_fidelity),
            "global_allocated_lane_states": len(slot.edge_success),
            "witnesses_by_lane": {
                lane_id: sorted(nodes)
                for lane_id, nodes in sorted(witnesses.items())
            },
        })
        self.counters["distributed_major_path_decisions"] += 1
        self.counters["qguard_phase4_decisions"] += 1
        if context is None or context.terminal:
            self._finish_major(key)
            return
        self._start_next_output(key)

    def _path_local_qguard_state(
        self,
        slot: _Slot,
        scope: set[str],
        decision_lane_ids: set[str],
    ) -> tuple[dict[str, bool], dict[str, float], dict[str, set[str]]]:
        visible_success: dict[str, bool] = {}
        visible_fidelity: dict[str, float] = {}
        witnesses: dict[str, set[str]] = {}
        for observer in sorted(scope):
            local = {
                lane_id: {
                    "success": success,
                    "fidelity": slot.edge_fidelity.get(lane_id, 0.0),
                }
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
                    lane_id: record
                    for lane_id, record in payload.items()
                    if lane_id in decision_lane_ids
                })
            for lane_id, record in local.items():
                success = bool(record["success"])
                fidelity = float(record["fidelity"])
                if (
                    lane_id in visible_success
                    and visible_success[lane_id] != success
                ):
                    raise RuntimeError(
                        f"Conflicting local Q-GUARD state for {lane_id}"
                    )
                if (
                    lane_id in visible_fidelity
                    and abs(visible_fidelity[lane_id] - fidelity) > 1e-12
                ):
                    raise RuntimeError(
                        f"Conflicting local Q-GUARD fidelity for {lane_id}"
                    )
                visible_success[lane_id] = success
                visible_fidelity[lane_id] = fidelity
                witnesses.setdefault(lane_id, set()).add(observer)
        return visible_success, visible_fidelity, witnesses

    def _start_next_output(self, key: tuple[int, str]) -> None:
        state = self._major_states[key]
        slot = self.slot
        if slot is None or slot.slot_id != state.slot_id or state.finished:
            return
        context = self._context_by_demand_id(state.demand_id)
        major = next(
            path for path in slot.plan.major_paths
            if path.path_id == state.major_path_id
        )
        if (
            context is None
            or context.terminal
            or state.output_attempts >= major.width
        ):
            self._finish_major(key)
            return

        candidate = self._select_route(slot, state, major)
        if candidate is None:
            self.counters["qguard_outputs_without_feasible_route"] += 1
            self._finish_major(key)
            return

        output_id = (
            f"slot-{state.slot_id}:{state.major_path_id}:qguard-"
            f"{state.output_attempts}"
        )
        state.output_attempts += 1
        state.active_output = True
        output = _OutputState(
            output_id=output_id,
            major_key=key,
            reservation_id=context.demand.reservation_id,
            nodes=candidate.nodes,
            targets_by_edge=candidate.targets_by_edge,
            used_recovery=candidate.used_recovery,
            recovery_path_ids=candidate.recovery_path_ids,
        )
        self._outputs[output_id] = output

        workflows = []
        for edge in (
            edge_key(left, right)
            for left, right in zip(candidate.nodes, candidate.nodes[1:])
        ):
            assignment_ids = list(candidate.assignment_ids_by_edge[edge])
            assignment_ids.sort(key=lambda lane_id: (
                -state.visible_fidelity.get(lane_id, 0.0),
                lane_id,
            ))
            take = min(candidate.required_by_edge[edge], len(assignment_ids))
            selected_ids = assignment_ids[:take]
            state.available_assignment_ids.difference_update(selected_ids)
            pool = [
                self._elementary_pair(state.assignment_by_id[lane_id], output)
                for lane_id in selected_ids
            ]
            workflow_id = f"{output_id}:{edge[0]}|{edge[1]}"
            workflow = _EdgeWorkflow(
                workflow_id,
                key,
                output_id,
                edge,
                candidate.targets_by_edge[edge],
                pool,
            )
            self._edge_workflows[workflow_id] = workflow
            output.workflow_ids.add(workflow_id)
            workflows.append(workflow_id)

        self.qguard_decisions.append({
            "event": "output_route_selected",
            "slot_id": state.slot_id,
            "major_path_id": state.major_path_id,
            "output_id": output_id,
            "time_ps": self.timeline.now(),
            "path": list(candidate.nodes),
            "used_recovery": candidate.used_recovery,
            "recovery_path_ids": list(candidate.recovery_path_ids),
            "targets_by_edge": {
                "|".join(edge): target
                for edge, target in candidate.targets_by_edge.items()
            },
            "required_raw_pairs_by_edge": {
                "|".join(edge): count
                for edge, count in candidate.required_by_edge.items()
            },
            "exg": candidate.exg_record,
        })
        for workflow_id in workflows:
            self._continue_edge_workflow(workflow_id)

    def _select_route(self, slot: _Slot, state: _MajorState, major) -> _RouteCandidate | None:
        major_allocation = slot.allocations[major.path_id]
        recoveries = [
            allocation
            for allocation in slot.allocations.values()
            if allocation.path.parent_path_id == major.path_id
        ]
        failed_indices = {
            index
            for index, (left, right) in enumerate(zip(major.nodes, major.nodes[1:]))
            if not self._available_ids(
                state,
                major.path_id,
                edge_key(left, right),
            )
        }
        if not failed_indices:
            return self._evaluate_route(
                state,
                major,
                major_allocation,
                (),
                allow_unqualified_major=True,
            )

        candidates = [
            allocation
            for allocation in recoveries
            if allocation.path.covered_segment is not None
            and any(
                allocation.path.covered_segment[0] <= index
                < allocation.path.covered_segment[1]
                for index in failed_indices
            )
        ]
        evaluated = []
        for count in range(1, len(candidates) + 1):
            for selected in combinations(candidates, count):
                intervals = sorted(
                    allocation.path.covered_segment
                    for allocation in selected
                )
                if any(
                    left[1] > right[0]
                    for left, right in zip(intervals, intervals[1:])
                ):
                    continue
                covered = {
                    index
                    for start, end in intervals
                    for index in range(start, end)
                }
                if not failed_indices.issubset(covered):
                    continue
                candidate = self._evaluate_route(
                    state,
                    major,
                    major_allocation,
                    selected,
                    allow_unqualified_major=False,
                )
                if candidate is not None and candidate.feasible:
                    evaluated.append(candidate)
        if not evaluated:
            return None
        return max(evaluated, key=lambda item: (
            item.score,
            -len(item.nodes),
            tuple(reversed(item.recovery_path_ids)),
        ))

    def _evaluate_route(
        self,
        state: _MajorState,
        major,
        major_allocation: QCASTPathAllocation,
        recoveries: tuple[QCASTPathAllocation, ...],
        *,
        allow_unqualified_major: bool,
    ) -> _RouteCandidate | None:
        edge_sources = {
            edge_key(left, right): major.path_id
            for left, right in zip(major.nodes, major.nodes[1:])
        }
        targets = {
            edge: equal_split_target(
                self._context_by_demand_id(major.demand_id).demand.fidelity_threshold,
                len(major.nodes) - 1,
            )
            for edge in edge_sources
        }
        threshold = self._context_by_demand_id(
            major.demand_id
        ).demand.fidelity_threshold
        for recovery in recoveries:
            start, end = recovery.path.covered_segment
            for left, right in zip(major.nodes[start:end], major.nodes[start + 1:end + 1]):
                edge_sources.pop(edge_key(left, right), None)
                targets.pop(edge_key(left, right), None)
            target = detour_split_target(
                threshold,
                len(major.nodes) - 1,
                end - start,
                len(recovery.path.nodes) - 1,
            )
            for left, right in zip(recovery.path.nodes, recovery.path.nodes[1:]):
                edge = edge_key(left, right)
                edge_sources[edge] = recovery.path.path_id
                targets[edge] = target

        nodes = self._bfs_path(
            major.nodes[0],
            major.nodes[-1],
            set(edge_sources),
        )
        if nodes is None:
            return None
        ordered_edges = tuple(
            edge_key(left, right)
            for left, right in zip(nodes, nodes[1:])
        )
        assignment_ids_by_edge = {}
        hop_plans = []
        required_by_edge = {}
        fallback_used = False
        width = min(
            [major_allocation.path.width]
            + [allocation.path.width for allocation in recoveries]
        )
        for edge in ordered_edges:
            source_path_id = edge_sources[edge]
            available = self._available_ids(state, source_path_id, edge)
            if not available:
                return None
            assignment_ids_by_edge[edge] = tuple(available)
            best_fidelity = max(
                state.visible_fidelity.get(lane_id, 0.0)
                for lane_id in available
            )
            rounds = minimum_purification_rounds(
                best_fidelity,
                targets[edge],
                self.algorithm.max_purification_rounds,
            )
            required = (1 << rounds) if rounds is not None else width + 1
            width_feasible = required <= width
            if not width_feasible and allow_unqualified_major:
                fallback_used = True
                rounds = 0
                required = 1
            elif not width_feasible:
                return None
            required_by_edge[edge] = required
            hop_plans.append(QGUARDHopPlan(
                target_fidelity=targets[edge],
                initial_fidelity=best_fidelity,
                available_pairs=len(available),
                purification_rounds=rounds,
            ))

        exg = expected_goodput(
            width,
            hop_plans,
            self.algorithm.swap_success_probability,
        )
        feasible = exg.feasible and not fallback_used
        if fallback_used:
            feasible = True
        return _RouteCandidate(
            nodes=tuple(nodes),
            assignment_ids_by_edge=assignment_ids_by_edge,
            targets_by_edge={edge: targets[edge] for edge in ordered_edges},
            required_by_edge=required_by_edge,
            score=exg.expected_goodput,
            feasible=feasible,
            used_recovery=bool(recoveries),
            recovery_path_ids=tuple(
                allocation.path.path_id for allocation in recoveries
            ),
            exg_record={
                "score": exg.expected_goodput,
                "feasible": exg.feasible,
                "availability": exg.availability,
                "fallback_to_unqualified_major": fallback_used,
            },
        )

    def _available_ids(
        self,
        state: _MajorState,
        path_id: str,
        edge: tuple[str, str],
    ) -> list[str]:
        return sorted(
            lane_id
            for lane_id in state.allocation_assignment_ids.get(path_id, set())
            if lane_id in state.available_assignment_ids
            and state.assignment_by_id[lane_id].key == edge
            and state.visible_success.get(lane_id, False)
        )

    @staticmethod
    def _elementary_pair(
        assignment: QCASTLaneEdge,
        output: _OutputState,
    ) -> _PairRef:
        return _PairRef(
            pair_id=assignment.lane_id,
            left=assignment.left,
            right=assignment.right,
            left_memory_index=assignment.left_memory_index,
            right_memory_index=assignment.right_memory_index,
            carrier=assignment,
            path=(assignment.left, assignment.right),
            provenance=(assignment,),
            used_recovery=output.used_recovery,
        )

    def _continue_edge_workflow(self, workflow_id: str) -> None:
        workflow = self._edge_workflows[workflow_id]
        if workflow.complete:
            return
        valid = [pair for pair in workflow.pool if self._pair_is_entangled(pair)]
        workflow.pool[:] = valid
        workflow.pool.sort(key=lambda pair: (
            -self._refresh_pair_fidelity(pair),
            pair.pair_id,
        ))
        if not workflow.pool:
            self._complete_edge_workflow(workflow, None)
            return
        best = workflow.pool[0]
        best_fidelity = self._refresh_pair_fidelity(best)
        if best_fidelity + 1e-12 >= workflow.target_fidelity:
            workflow.pool.pop(0)
            self.counters["qguard_hops_meeting_target"] += 1
            self._return_unused_elementary_pairs(workflow, workflow.pool)
            self._complete_edge_workflow(workflow, best)
            return
        if len(workflow.pool) < 2:
            workflow.pool.pop(0)
            self.counters["qguard_hops_below_target"] += 1
            self._complete_edge_workflow(workflow, best)
            return

        kept = workflow.pool.pop(0)
        measured = workflow.pool.pop(0)
        self._launch_purification(
            workflow.major_key[0],
            "edge",
            workflow_id,
            kept,
            measured,
        )

    def _return_unused_elementary_pairs(
        self,
        workflow: _EdgeWorkflow,
        pairs: list[_PairRef],
    ) -> None:
        state = self._major_states[workflow.major_key]
        state.available_assignment_ids.update(
            pair.carrier.lane_id
            for pair in pairs
            if pair.carrier is not None and self._pair_is_entangled(pair)
        )

    def _complete_edge_workflow(
        self,
        workflow: _EdgeWorkflow,
        pair: _PairRef | None,
    ) -> None:
        workflow.complete = True
        output = self._outputs[workflow.output_id]
        output.results_by_edge[workflow.edge] = pair
        if not all(
            self._edge_workflows[item].complete
            for item in output.workflow_ids
        ):
            return
        state = self._major_states[output.major_key]
        if all(result is not None for result in output.results_by_edge.values()):
            carriers = tuple(
                output.results_by_edge[edge_key(left, right)].carrier
                for left, right in zip(output.nodes, output.nodes[1:])
            )
            provenance = tuple({
                assignment.lane_id: assignment
                for result in output.results_by_edge.values()
                for assignment in result.provenance
            }.values())
            self._delivery_metadata[output.output_id] = {
                "provenance": provenance,
                "used_recovery": output.used_recovery,
                "recovery_path_ids": output.recovery_path_ids,
            }
            self._slot_active_delivery_checks[state.slot_id] += 1
            self._schedule_swap_chain(
                self.slot,
                output.output_id,
                output.reservation_id,
                output.nodes,
                carriers,
                self.timeline.now() + 1,
                output.used_recovery,
            )
            self.counters["qguard_paths_sent_to_swapping"] += 1
        else:
            self.counters["qguard_paths_failed_purification"] += 1
        state.active_output = False
        self._start_next_output(output.major_key)

    def _launch_purification(
        self,
        slot_id: int,
        purpose: str,
        owner_id: str | int,
        kept: _PairRef,
        measured: _PairRef,
    ) -> None:
        if {kept.left, kept.right} != {measured.left, measured.right}:
            raise RuntimeError("Q-GUARD purification inputs have different endpoints")
        left = kept.left
        right = kept.right
        kept_left = self._memory(left, kept.memory_index(left))
        kept_right = self._memory(right, kept.memory_index(right))
        measured_left = self._memory(left, measured.memory_index(left))
        measured_right = self._memory(right, measured.memory_index(right))
        attempt_id = f"qguard-pur-{self._purification_counter}"
        self._purification_counter += 1
        left_protocol = BBPSSWProtocol.create(
            self.routers[left],
            f"{attempt_id}.L",
            kept_left,
            measured_left,
        )
        right_protocol = BBPSSWProtocol.create(
            self.routers[right],
            f"{attempt_id}.R",
            kept_right,
            measured_right,
        )
        left_protocol.set_others(
            right_protocol.name,
            right,
            [kept_right.name, measured_right.name],
        )
        right_protocol.set_others(
            left_protocol.name,
            left,
            [kept_left.name, measured_left.name],
        )
        input_fidelities = [
            self._refresh_pair_fidelity(kept),
            self._refresh_pair_fidelity(measured),
        ]
        for router, protocol in (
            (self.routers[left], left_protocol),
            (self.routers[right], right_protocol),
        ):
            router.protocols.append(protocol)
            for memory in protocol.memories:
                memory.detach(memory.memory_array)
                memory.attach(protocol)
                router.resource_manager.memory_manager.get_info_by_memory(
                    memory
                ).to_occupied()

        propagation_delay_ps = max(
            int(self.routers[left].cchannels[right].delay),
            int(self.routers[right].cchannels[left].delay),
        )
        protocol_start_ps = (
            self.timeline.now() + self.algorithm.control_processing_delay_ps
        )
        completion_ps = protocol_start_ps + propagation_delay_ps + 1
        self._purification_attempts[attempt_id] = _PurificationAttempt(
            attempt_id,
            slot_id,
            purpose,
            owner_id,
            kept,
            measured,
            self.timeline.now(),
            completion_ps,
        )
        self.counters["qguard_purification_attempts"] += 1
        self.purification_events.append({
            "event": "purification_started",
            "attempt_id": attempt_id,
            "purpose": purpose,
            "time_ps": self.timeline.now(),
            "protocol_start_ps": protocol_start_ps,
            "completion_check_ps": completion_ps,
            "classical_wait_ps": completion_ps - self.timeline.now(),
            "endpoint_processing_delay_ps": (
                self.algorithm.control_processing_delay_ps
            ),
            "classical_propagation_delay_ps": propagation_delay_ps,
            "endpoints": [left, right],
            "input_pair_ids": [kept.pair_id, measured.pair_id],
            "input_fidelities": input_fidelities,
        })
        self.timeline.schedule(Event(
            protocol_start_ps,
            Process(left_protocol, "start", []),
        ))
        self.timeline.schedule(Event(
            protocol_start_ps,
            Process(right_protocol, "start", []),
        ))
        self.timeline.schedule(Event(
            completion_ps,
            Process(self, "complete_purification", [attempt_id]),
        ))

    def complete_purification(self, attempt_id: str) -> None:
        attempt = self._purification_attempts.pop(attempt_id)
        success = self._pair_is_purified(attempt.kept)
        if success:
            self._normalize_purified_pair(attempt.kept)
            fidelity = self._refresh_pair_fidelity(attempt.kept)
            self.counters["qguard_purification_successes"] += 1
            result = attempt.kept
        else:
            fidelity = 0.0
            self.counters["qguard_purification_failures"] += 1
            result = None
        self.counters["qguard_pair_resources_consumed_by_purification"] += (
            1 if success else 2
        )
        self.purification_events.append({
            "event": "purification_completed",
            "attempt_id": attempt_id,
            "purpose": attempt.purpose,
            "time_ps": self.timeline.now(),
            "success": success,
            "output_fidelity": fidelity,
            "classical_wait_ps": self.timeline.now() - attempt.started_at_ps,
        })
        if attempt.purpose == "edge":
            workflow = self._edge_workflows[str(attempt.owner_id)]
            if result is not None:
                workflow.pool.append(result)
            self._continue_edge_workflow(workflow.workflow_id)
        elif attempt.purpose == "end_to_end":
            reservation_id = int(attempt.owner_id)
            if result is not None:
                self._slot_pairs[attempt.slot_id][reservation_id].append(result)
            self._continue_end_to_end_qualification(
                attempt.slot_id,
                reservation_id,
            )
        else:
            raise RuntimeError(f"Unknown Q-GUARD purification purpose {attempt.purpose}")

    def _pair_is_entangled(self, pair: _PairRef) -> bool:
        left_memory = self._memory(pair.left, pair.left_memory_index)
        right_memory = self._memory(pair.right, pair.right_memory_index)
        left_info = self.routers[pair.left].resource_manager.memory_manager.get_info_by_memory(
            left_memory
        )
        right_info = self.routers[pair.right].resource_manager.memory_manager.get_info_by_memory(
            right_memory
        )
        return (
            left_info.state in {MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED}
            and right_info.state in {MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED}
            and left_info.remote_node == pair.right
            and right_info.remote_node == pair.left
            and left_memory.entangled_memory["memo_id"] == right_memory.name
            and right_memory.entangled_memory["memo_id"] == left_memory.name
        )

    def _pair_is_purified(self, pair: _PairRef) -> bool:
        left_memory = self._memory(pair.left, pair.left_memory_index)
        right_memory = self._memory(pair.right, pair.right_memory_index)
        left_info = self.routers[pair.left].resource_manager.memory_manager.get_info_by_memory(
            left_memory
        )
        right_info = self.routers[pair.right].resource_manager.memory_manager.get_info_by_memory(
            right_memory
        )
        states = {left_info.state, right_info.state}
        if states == {MemoryInfo.PURIFIED}:
            return self._pair_is_entangled(pair)
        if MemoryInfo.PURIFIED in states:
            self.counters["qguard_purification_endpoint_state_mismatches"] += 1
        return False

    def _normalize_purified_pair(self, pair: _PairRef) -> None:
        for node, index in (
            (pair.left, pair.left_memory_index),
            (pair.right, pair.right_memory_index),
        ):
            memory = self._memory(node, index)
            self.routers[node].resource_manager.memory_manager.update(
                memory,
                MemoryInfo.ENTANGLED,
            )

    def _refresh_pair_fidelity(self, pair: _PairRef) -> float:
        if not self._pair_is_entangled(pair):
            return 0.0
        fidelities = []
        for node, index in (
            (pair.left, pair.left_memory_index),
            (pair.right, pair.right_memory_index),
        ):
            memory = self._memory(node, index)
            memory.bds_decohere()
            fidelity = memory.get_bds_fidelity()
            memory.fidelity = fidelity
            info = self.routers[node].resource_manager.memory_manager.get_info_by_memory(
                memory
            )
            info.fidelity = fidelity
            fidelities.append(fidelity)
        return min(fidelities)

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
        try:
            if slot is None or slot.slot_id != slot_id:
                return
            source_memory = self._memory(
                nodes[0],
                assignments[0].memory_index(nodes[0]),
            )
            destination_memory = self._memory(
                nodes[-1],
                assignments[-1].memory_index(nodes[-1]),
            )
            source_info = self.routers[nodes[0]].resource_manager.memory_manager.get_info_by_memory(
                source_memory
            )
            destination_info = self.routers[nodes[-1]].resource_manager.memory_manager.get_info_by_memory(
                destination_memory
            )
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
            metadata = self._delivery_metadata[delivery_id]
            pair = _PairRef(
                pair_id=delivery_id,
                left=nodes[0],
                right=nodes[-1],
                left_memory_index=assignments[0].memory_index(nodes[0]),
                right_memory_index=assignments[-1].memory_index(nodes[-1]),
                carrier=None,
                path=nodes,
                provenance=metadata["provenance"],
                used_recovery=used_recovery,
                p4_start_ps=slot.delivery_p4_start_ps.get(delivery_id),
            )
            self._slot_pairs[slot_id][reservation_id].append(pair)
            self.counters["qguard_end_to_end_pairs_assembled"] += 1
        finally:
            if self._slot_active_delivery_checks[slot_id] > 0:
                self._slot_active_delivery_checks[slot_id] -= 1
            self._maybe_begin_final_qualification(slot_id)

    def _finish_major(self, key: tuple[int, str]) -> None:
        state = self._major_states[key]
        if state.finished:
            return
        state.finished = True
        self._slot_finished_majors[state.slot_id].add(state.major_path_id)
        self._major_path_check_ps[key] = self.timeline.now()
        self._maybe_begin_final_qualification(state.slot_id)

    def _maybe_begin_final_qualification(self, slot_id: int) -> None:
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        expected = {path.path_id for path in slot.plan.major_paths}
        if not expected.issubset(self._slot_finished_majors[slot_id]):
            return
        if self._slot_active_delivery_checks[slot_id] != 0:
            return
        if slot_id in self._slot_qualification_started:
            return
        self._slot_qualification_started.add(slot_id)
        reservation_ids = {
            context.demand.reservation_id
            for context in self.contexts.values()
            if not context.terminal
            and any(
                path.demand_id == context.demand.demand_id
                for path in slot.plan.major_paths
            )
        }
        self._slot_qualification_pending[slot_id] = set(reservation_ids)
        self.counters["qguard_final_qualification_phases"] += 1
        if not reservation_ids:
            self._schedule_slot_finish(slot_id)
            return
        for reservation_id in sorted(reservation_ids):
            self._continue_end_to_end_qualification(slot_id, reservation_id)

    def _continue_end_to_end_qualification(
        self,
        slot_id: int,
        reservation_id: int,
    ) -> None:
        context = self.contexts.get(reservation_id)
        if context is None or context.terminal:
            self._finish_qualification_for_demand(slot_id, reservation_id)
            return
        pool = self._slot_pairs[slot_id][reservation_id]
        pool[:] = [pair for pair in pool if self._pair_is_entangled(pair)]
        qualifying = []
        unqualified = []
        for pair in pool:
            fidelity = self._refresh_pair_fidelity(pair)
            if fidelity + 1e-12 >= context.demand.fidelity_threshold:
                qualifying.append((fidelity, pair))
            else:
                unqualified.append((fidelity, pair))

        qualifying.sort(key=lambda item: (-item[0], item[1].pair_id))
        for fidelity, pair in qualifying:
            if context.terminal:
                break
            pool.remove(pair)
            self._deliver_qualified_pair(context, pair, fidelity)
        if context.terminal:
            self._finish_qualification_for_demand(slot_id, reservation_id)
            return

        unqualified = [
            (self._refresh_pair_fidelity(pair), pair)
            for pair in pool
            if self._pair_is_entangled(pair)
        ]
        unqualified.sort(key=lambda item: (item[0], item[1].pair_id))
        if len(unqualified) >= 2:
            first = unqualified[0][1]
            second = unqualified[1][1]
            pool.remove(first)
            pool.remove(second)
            combined = _PairRef(
                pair_id=f"{first.pair_id}+{second.pair_id}",
                left=first.left,
                right=first.right,
                left_memory_index=first.left_memory_index,
                right_memory_index=first.right_memory_index,
                carrier=None,
                path=first.path,
                provenance=tuple({
                    assignment.lane_id: assignment
                    for assignment in (*first.provenance, *second.provenance)
                }.values()),
                used_recovery=first.used_recovery or second.used_recovery,
                p4_start_ps=min((
                    value
                    for value in (first.p4_start_ps, second.p4_start_ps)
                    if value is not None
                ), default=None),
            )
            self._launch_purification(
                slot_id,
                "end_to_end",
                reservation_id,
                combined,
                second,
            )
            self.counters["qguard_end_to_end_purification_attempts"] += 1
            return

        for _fidelity, pair in unqualified:
            if pair in pool:
                pool.remove(pair)
            context.callbacks.on_pair_rejected(
                context.demand,
                self._refresh_pair_fidelity(pair),
            )
            self.counters["pairs_rejected_fidelity"] += 1
        self._finish_qualification_for_demand(slot_id, reservation_id)

    def _deliver_qualified_pair(self, context, pair: _PairRef, fidelity: float) -> None:
        role = "recovery" if pair.used_recovery else "major"
        elementary_sources = tuple({
            "source": "application",
            "qcast_role": role,
            "link": assignment.key,
            "lane_id": assignment.lane_id,
        } for assignment in pair.provenance)
        delivery = PairDelivery(
            timestamp_ps=self.timeline.now(),
            fidelity=fidelity,
            generation_source="application",
            elementary_sources=elementary_sources,
        )
        context.deliveries.append(delivery)
        context.callbacks.on_pair_delivered(context.demand, delivery)
        carriers = tuple(
            assignment
            for assignment in pair.provenance
            if assignment.key in {
                edge_key(left, right)
                for left, right in zip(pair.path, pair.path[1:])
            }
        )
        if pair.p4_start_ps is not None:
            self.slot.delivery_p4_start_ps[pair.pair_id] = pair.p4_start_ps
        self._record_delivery_timing(
            self.slot,
            pair.pair_id,
            context.demand.demand_id,
            pair.path,
            carriers,
            role,
        )
        self.counters["end_to_end_pairs_delivered"] += 1
        self.counters[f"end_to_end_pairs_delivered_{role}"] += 1
        self.counters["qguard_pairs_qualified"] += 1
        self.events.append({
            "event": "pair_delivered",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "fidelity": fidelity,
            "path": list(pair.path),
            "qcast_role": role,
            "qguard_qualified": True,
        })
        if len(context.deliveries) >= context.demand.pair_count:
            context.terminal = True
            self.counters["demands_completed"] += 1
            self.events.append({
                "event": "demand_completed",
                "time_ps": self.timeline.now(),
                "demand_id": context.demand.demand_id,
                "pairs_delivered": len(context.deliveries),
                "path": list(pair.path),
            })
            context.callbacks.on_demand_completed(
                context.demand,
                self.timeline.now(),
                pair.path,
            )

    def _finish_qualification_for_demand(
        self,
        slot_id: int,
        reservation_id: int,
    ) -> None:
        self._slot_qualification_pending[slot_id].discard(reservation_id)
        if not self._slot_qualification_pending[slot_id]:
            self._schedule_slot_finish(slot_id)

    def _schedule_slot_finish(self, slot_id: int) -> None:
        if slot_id in self._slot_finish_scheduled:
            return
        slot = self.slot
        if slot is None or slot.slot_id != slot_id:
            return
        finish_time = self.timeline.now() + 2
        slot.cleanup_time_ps = finish_time
        self._slot_finish_scheduled.add(slot_id)
        self.timeline.schedule(Event(
            finish_time,
            Process(self, "finish_slot", [slot_id]),
        ))

    def finish_slot(self, slot_id: int) -> None:
        super().finish_slot(slot_id)
        for key in [key for key in self._major_states if key[0] == slot_id]:
            self._major_states.pop(key, None)
        self._slot_pairs.pop(slot_id, None)

    def diagnostics(self) -> dict:
        diagnostics = super().diagnostics()
        diagnostics["qguard"] = {
            "variant": "equal_split",
            "max_purification_rounds": self.algorithm.max_purification_rounds,
            "decisions": self.qguard_decisions,
            "purification_events": self.purification_events,
            "uses_official_bell_diagonal_bbpssw": True,
            "final_end_to_end_purification": True,
            "link_state_includes_realized_fidelity": True,
            "new_global_control_messages": 0,
        }
        diagnostics["locality_invariants"]["qguard_decisions_path_scoped"] = all(
            set(decision.get("visible_lane_ids", ())).issubset(
                decision.get("decision_lane_ids", ())
            )
            for decision in self.qguard_decisions
            if decision["event"] == "major_path_phase4_started"
        )
        return diagnostics
