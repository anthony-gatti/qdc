"""Paper-local, asynchronous DFER execution on pristine SeQUeNCe."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from sequence.entanglement_management.purification.bbpssw_protocol import (
    BBPSSWProtocol,
)
from sequence.entanglement_management.swapping import (
    EntanglementSwappingA,
    EntanglementSwappingB,
)
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.resource_management.action_condition_set import (
    eg_rule_action_await,
    eg_rule_condition,
)
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.resource_management.rule_manager import Rule
from sequence.topology.router_net_topo import RouterNetTopo

from algorithms.dfer import (
    expected_distribution_rate,
    pumping_plan,
    required_link_fidelity,
)
from algorithms.qcast import edge_key
from backends.sequence.dfer_protocol import (
    DFERMessage,
    DFERMessageType,
    DFERReservation,
)
from backends.sequence.parallel_links import (
    parallel_eg_rule_action_request,
    record_generation_success,
    record_memory_occupancy,
)
from backends.sequence.qcast_scheduler import (
    single_heralded_attempt_success_probability,
)
from workloads.base import PairDelivery


@dataclass
class _Pair:
    pair_id: str
    left: str
    right: str
    left_index: int
    right_index: int
    path: tuple[str, ...]
    elementary_links: tuple[tuple[str, str], ...]
    created_at_ps: int

    def memory_index(self, node: str) -> int:
        if node == self.left:
            return self.left_index
        if node == self.right:
            return self.right_index
        raise KeyError(node)


@dataclass
class _GenerationOperation:
    operation_id: str
    reservation_id: int
    left: str
    right: str
    left_index: int
    right_index: int
    middle: str
    purpose: str
    rules: list[tuple[object, Rule]] = field(default_factory=list)
    terminal: bool = False


@dataclass
class _PurificationOperation:
    operation_id: str
    reservation_id: int
    kept: _Pair
    measured: _Pair
    started_at_ps: int
    events: list[Event] = field(default_factory=list)
    protocols: list[object] = field(default_factory=list)


@dataclass
class _SwapOperation:
    operation_id: str
    reservation_id: int
    long_pair: _Pair
    link_pair: _Pair
    current: str
    next_hop: str
    started_at_ps: int
    events: list[Event] = field(default_factory=list)
    protocols: list[object] = field(default_factory=list)


@dataclass
class _Context:
    demand: object
    callbacks: object
    current: str
    path: list[str]
    deliveries: list[PairDelivery] = field(default_factory=list)
    terminal: bool = False
    accepted: bool = False
    epoch: int = 0
    expected_responses: set[str] = field(default_factory=set)
    responses: dict[str, dict] = field(default_factory=dict)
    long_pair: _Pair | None = None
    hop_pair: _Pair | None = None
    selected_neighbor: str | None = None
    target_fidelity: float = 0.0
    planned_rounds: int = 0
    completed_rounds: int = 0
    last_path: tuple[str, ...] = ()


class DFERDemandScheduler:
    """Execute DLFR and DFPS at each current entanglement endpoint."""

    def __init__(self, network_topology, algorithm, controller_node: str):
        del controller_node
        self.network_topology = network_topology
        self.algorithm = algorithm
        self.timeline = network_topology.get_timeline()
        self.routers = {
            router.name: router
            for router in network_topology.get_nodes_by_type(
                RouterNetTopo.QUANTUM_ROUTER
            )
        }
        self.graph = self._router_graph()
        self.edge_middles = self._middle_nodes_by_edge()
        self.contexts: dict[int, _Context] = {}
        self.generation_operations: dict[str, _GenerationOperation] = {}
        self.purification_operations: dict[str, _PurificationOperation] = {}
        self.swap_operations: dict[str, _SwapOperation] = {}
        self.claimed_memories = {name: set() for name in self.routers}
        self.claimed_middles: set[str] = set()
        self.memory_high_watermark = Counter()
        self.channel_usage = Counter()
        self.counters = Counter()
        self.events: list[dict] = []
        self.decisions: list[dict] = []
        self._operation_counter = 0
        self._edge_models = self._physical_edge_models()
        for router in self.routers.values():
            router.dfer_control.configure(self)

    def submit(self, demand, callbacks) -> None:
        if demand.reservation_id in self.contexts:
            raise ValueError(f"Duplicate DFER demand id {demand.reservation_id}")
        context = _Context(demand, callbacks, demand.source, [demand.source])
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
        self._schedule_local_hop(context, self.timeline.now() + 1)

    def begin_hop(self, owner: str, reservation_id: int) -> None:
        context = self.contexts.get(reservation_id)
        if context is None or context.terminal or owner != context.current:
            return
        if self.timeline.now() >= context.demand.deadline_ps:
            self.deadline(reservation_id)
            return
        if context.current == context.demand.destination:
            self._deliver_or_restart(context)
            return
        if len(context.path) - 1 >= self.algorithm.max_hops:
            self._fail(context, "dfer_max_hops")
            return

        current_distance = self._distance(
            context.current,
            context.demand.destination,
        )
        if current_distance is None:
            self._fail(context, "dfer_unreachable")
            return
        candidates = {
            neighbor
            for neighbor in self.graph[context.current]
            if (
                self._distance(neighbor, context.demand.destination) is not None
                and self._distance(neighbor, context.demand.destination)
                < current_distance
            )
        }
        if not candidates:
            self._fail(context, "dfer_no_progress_neighbor")
            return

        context.epoch += 1
        context.expected_responses = set(candidates)
        context.responses.clear()
        self.counters["dlfr_evaluations"] += 1
        self.events.append({
            "event": "neighbor_state_query_started",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "current": context.current,
            "candidates": sorted(candidates),
            "epoch": context.epoch,
        })
        for neighbor in sorted(candidates):
            self.routers[context.current].send_message(
                neighbor,
                DFERMessage(
                    DFERMessageType.STATE_QUERY,
                    reservation_id,
                    context.epoch,
                    {"destination": context.demand.destination},
                ),
                sender_delay=self.algorithm.control_processing_delay_ps,
            )
            self.counters["neighbor_state_queries_sent"] += 1

    def received_control_message(
        self,
        receiver: str,
        src: str,
        message: DFERMessage,
    ) -> None:
        context = self.contexts.get(message.reservation_id)
        if (
            context is None
            or context.terminal
            or message.epoch != context.epoch
        ):
            self.counters["stale_control_messages"] += 1
            return
        if message.msg_type is DFERMessageType.STATE_QUERY:
            if src not in self.graph[receiver]:
                self.counters["nonlocal_state_queries"] += 1
                return
            model = self._edge_models[edge_key(src, receiver)]
            payload = {
                "base_fidelity": self._base_link_fidelity(src, receiver),
                "generation_rate_hz": model["generation_rate_hz"],
                "attempt_success_probability": model[
                    "attempt_success_probability"
                ],
                "attempt_duration_ps": model["attempt_duration_ps"],
                "physical_channels": list(model["physical_channels"]),
                "remote_gate_fidelity": self.routers[receiver].gate_fid,
                "remote_measurement_fidelity": self.routers[receiver].meas_fid,
            }
            self.routers[receiver].send_message(
                src,
                DFERMessage(
                    DFERMessageType.STATE_RESPONSE,
                    message.reservation_id,
                    message.epoch,
                    payload,
                ),
                sender_delay=self.algorithm.control_processing_delay_ps,
            )
            self.counters["neighbor_state_queries_received"] += 1
            self.counters["neighbor_state_responses_sent"] += 1
            return

        if message.msg_type is DFERMessageType.STATE_RESPONSE:
            if receiver != context.current or src not in context.expected_responses:
                self.counters["unexpected_state_responses"] += 1
                return
            context.responses[src] = dict(message.payload)
            self.counters["neighbor_state_responses_received"] += 1
            if context.expected_responses.issubset(context.responses):
                self._select_next_hop(context)

    def _select_next_hop(self, context: _Context) -> None:
        current_fidelity = (
            self._pair_fidelity(context.long_pair)
            if context.long_pair is not None else None
        )
        remaining_hops = self._distance(
            context.current,
            context.demand.destination,
        )
        if remaining_hops is None:
            self._fail(context, "dfer_unreachable")
            return
        target = required_link_fidelity(
            context.demand.fidelity_threshold,
            remaining_hops,
            current_fidelity,
        )
        candidates = []
        records = []
        for neighbor, response in sorted(context.responses.items()):
            initial = response["base_fidelity"]
            plan = (
                pumping_plan(
                    initial,
                    target,
                    self.algorithm.max_purification_rounds,
                    kept_gate_fidelity=self.routers[
                        context.current
                    ].gate_fid,
                    remote_gate_fidelity=response[
                        "remote_gate_fidelity"
                    ],
                    kept_measurement_fidelity=self.routers[
                        context.current
                    ].meas_fid,
                    remote_measurement_fidelity=response[
                        "remote_measurement_fidelity"
                    ],
                )
                if initial + 1e-12 >= self.algorithm.cutoff_fidelity
                else None
            )
            score = 0.0
            if plan is not None:
                propagation = max(
                    int(self.routers[context.current].cchannels[neighbor].delay),
                    int(self.routers[neighbor].cchannels[context.current].delay),
                )
                score = expected_distribution_rate(
                    response["generation_rate_hz"],
                    plan,
                    (
                        self.algorithm.control_processing_delay_ps
                        + propagation
                    )
                    / 10**12,
                    remaining_hops,
                    self.algorithm.swap_success_probability,
                )
                candidates.append((neighbor, score, plan))
            records.append({
                "neighbor": neighbor,
                "base_fidelity": initial,
                "target_fidelity": target,
                "pumping_rounds": None if plan is None else plan.rounds,
                "expected_edr_hz": score,
                "feasible": plan is not None,
            })

        if not candidates:
            self.decisions.append({
                "time_ps": self.timeline.now(),
                "demand_id": context.demand.demand_id,
                "current": context.current,
                "selected": None,
                "candidates": records,
            })
            self._fail(context, "dfer_no_fidelity_feasible_neighbor")
            return
        destination = context.demand.destination
        direct = [candidate for candidate in candidates if candidate[0] == destination]
        pool = direct or candidates
        neighbor, score, plan = max(
            pool,
            key=lambda item: (item[1], item[0]),
        )
        context.selected_neighbor = neighbor
        context.target_fidelity = target
        context.planned_rounds = plan.rounds
        context.completed_rounds = 0
        context.hop_pair = None
        self.counters["dfps_selections"] += 1
        self.decisions.append({
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "current": context.current,
            "destination": destination,
            "current_fidelity": current_fidelity,
            "remaining_hops": remaining_hops,
            "selected": neighbor,
            "selected_expected_edr_hz": score,
            "candidates": records,
        })
        if not context.accepted:
            context.accepted = True
            context.callbacks.on_demand_accepted(
                context.demand,
                self._shortest_path(
                    context.demand.source,
                    context.demand.destination,
                ),
            )
        self._start_elementary_generation(context, "initial")

    def _start_elementary_generation(
        self,
        context: _Context,
        purpose: str,
    ) -> None:
        if context.terminal or context.selected_neighbor is None:
            return
        allocation = self._claim_link_resources(
            context.current,
            context.selected_neighbor,
        )
        if allocation is None:
            self.counters["resource_waits"] += 1
            self.timeline.schedule(Event(
                min(
                    context.demand.deadline_ps,
                    self.timeline.now()
                    + self.algorithm.resource_retry_delay_ps,
                ),
                Process(
                    self,
                    "_start_elementary_generation",
                    [context, purpose],
                ),
            ))
            return
        left_index, right_index, middle = allocation
        operation_id = self._next_operation_id("gen")
        reservation = DFERReservation(
            context.current,
            context.selected_neighbor,
            self.timeline.now(),
            context.demand.deadline_ps,
            1,
            0.5,
            1,
            self._operation_counter,
            operation_id=operation_id,
        )
        reservation.set_path([context.current, context.selected_neighbor])
        operation = _GenerationOperation(
            operation_id,
            context.demand.reservation_id,
            context.current,
            context.selected_neighbor,
            left_index,
            right_index,
            middle,
            purpose,
        )
        self.generation_operations[operation_id] = operation
        requester = self.routers[context.current]
        responder = self.routers[context.selected_neighbor]
        for node, index, action, action_args in (
            (
                requester,
                left_index,
                parallel_eg_rule_action_request,
                {
                    "mid": middle,
                    "path": [context.current, context.selected_neighbor],
                    "index": 0,
                    "name": context.current,
                    "reservation": reservation,
                },
            ),
            (
                responder,
                right_index,
                eg_rule_action_await,
                {
                    "mid": middle,
                    "path": [context.current, context.selected_neighbor],
                    "index": 1,
                },
            ),
        ):
            rule = Rule(
                10,
                action,
                eg_rule_condition,
                action_args,
                {"memory_indices": [index]},
            )
            rule.set_reservation(reservation)
            memory = self._memory(node.name, index)
            memory.dfer_operation_id = operation_id
            node.resource_manager.load(rule)
            operation.rules.append((node, rule))
            record_memory_occupancy(node)
        self.channel_usage[(edge_key(operation.left, operation.right), middle)] += 1
        self.counters["elementary_generation_operations_started"] += 1
        self.counters[f"elementary_generation_{purpose}_started"] += 1
        self.events.append({
            "event": "elementary_generation_started",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "operation_id": operation_id,
            "purpose": purpose,
            "link": [operation.left, operation.right],
            "middle": middle,
        })

    def record_elementary_success(
        self,
        operation_id: str,
        time_ps: int,
        fidelity: float,
    ) -> None:
        operation = self.generation_operations.get(operation_id)
        if operation is None or operation.terminal:
            self.counters["duplicate_elementary_success_callbacks"] += 1
            return
        operation.terminal = True
        self.timeline.schedule(Event(
            time_ps + 1,
            Process(
                self,
                "_complete_elementary_generation",
                [operation_id, fidelity],
            ),
        ))

    def _complete_elementary_generation(
        self,
        operation_id: str,
        observed_fidelity: float,
    ) -> None:
        operation = self.generation_operations.pop(operation_id, None)
        if operation is None:
            return
        self._expire_rules(operation.rules)
        self.claimed_middles.discard(operation.middle)
        context = self.contexts.get(operation.reservation_id)
        if context is None or context.terminal:
            self._release_indices(
                operation.left,
                operation.left_index,
                operation.right,
                operation.right_index,
            )
            return
        pair = _Pair(
            operation_id,
            operation.left,
            operation.right,
            operation.left_index,
            operation.right_index,
            (operation.left, operation.right),
            (edge_key(operation.left, operation.right),),
            self.timeline.now(),
        )
        fidelity = self._pair_fidelity(pair)
        record_generation_success(
            self.routers[operation.left],
            operation.right,
            operation.middle,
        )
        self.counters["elementary_pairs_generated"] += 1
        self.events.append({
            "event": "elementary_pair_generated",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "operation_id": operation_id,
            "purpose": operation.purpose,
            "fidelity": fidelity,
            "observer_fidelity": observed_fidelity,
        })
        if operation.purpose == "initial":
            context.hop_pair = pair
            self._continue_hop_pumping(context)
        elif operation.purpose == "pump":
            if context.hop_pair is None:
                self._release_pair(pair)
                self._restart_pair(context, "missing_pumping_kept_pair")
            else:
                self._start_purification(context, context.hop_pair, pair)
        else:
            raise RuntimeError(f"Unknown DFER generation purpose {operation.purpose}")

    def _continue_hop_pumping(self, context: _Context) -> None:
        pair = context.hop_pair
        if pair is None:
            self._restart_pair(context, "missing_hop_pair")
            return
        fidelity = self._pair_fidelity(pair)
        if fidelity + 1e-12 >= context.target_fidelity:
            self.counters["links_meeting_dlfr_target"] += 1
            self._advance_with_link_pair(context)
            return
        if (
            fidelity + 1e-12 < self.algorithm.cutoff_fidelity
            or context.completed_rounds >= self.algorithm.max_purification_rounds
        ):
            self.counters["links_failing_dlfr_target"] += 1
            self._release_pair(pair)
            context.hop_pair = None
            self._restart_pair(context, "dfer_link_target_unmet")
            return
        self._start_elementary_generation(context, "pump")

    def _start_purification(
        self,
        context: _Context,
        kept: _Pair,
        measured: _Pair,
    ) -> None:
        operation_id = self._next_operation_id("pur")
        operation = _PurificationOperation(
            operation_id, context.demand.reservation_id, kept, measured,
            self.timeline.now(),
        )
        self.purification_operations[operation_id] = operation
        event = Event(
            self.timeline.now() + self.algorithm.control_processing_delay_ps,
            Process(self, "_begin_purification", [operation_id]),
        )
        operation.events.append(event)
        self.timeline.schedule(event)

    def _begin_purification(self, operation_id: str) -> None:
        """Validate at execution time, before attaching native protocols."""
        operation = self.purification_operations.get(operation_id)
        if operation is None:
            return
        context = self.contexts[operation.reservation_id]
        if context.terminal or self.timeline.now() >= context.demand.deadline_ps:
            self._fail(context, "dfer_deadline")
            return
        kept, measured = operation.kept, operation.measured
        fidelities = [self._pair_fidelity(kept), self._pair_fidelity(measured)]
        if min(fidelities) <= 0.5:
            self.purification_operations.pop(operation_id)
            self._release_pair(measured)
            self.counters["pumping_invalid_inputs"] += 1
            self._restart_pair(context, "pumping_inputs_expired")
            return
        left, right = kept.left, kept.right
        left_protocol = BBPSSWProtocol.create(
            self.routers[left],
            operation_id + ".L",
            self._memory(left, kept.memory_index(left)),
            self._memory(left, measured.memory_index(left)),
        )
        right_protocol = BBPSSWProtocol.create(
            self.routers[right],
            operation_id + ".R",
            self._memory(right, kept.memory_index(right)),
            self._memory(right, measured.memory_index(right)),
        )
        left_protocol.set_others(
            right_protocol.name,
            right,
            [
                self._memory(right, kept.memory_index(right)).name,
                self._memory(right, measured.memory_index(right)).name,
            ],
        )
        right_protocol.set_others(
            left_protocol.name,
            left,
            [
                self._memory(left, kept.memory_index(left)).name,
                self._memory(left, measured.memory_index(left)).name,
            ],
        )
        for router, protocol in (
            (self.routers[left], left_protocol),
            (self.routers[right], right_protocol),
        ):
            operation.protocols.append(protocol)
            router.protocols.append(protocol)
            for memory in protocol.memories:
                memory.detach(memory.memory_array)
                memory.attach(protocol)
                router.resource_manager.memory_manager.get_info_by_memory(
                    memory
                ).to_occupied()
        propagation = max(
            int(self.routers[left].cchannels[right].delay),
            int(self.routers[right].cchannels[left].delay),
        )
        completion = self.timeline.now() + propagation + 1
        self.counters["pumping_attempts"] += 1
        self.events.append({
            "event": "pumping_started",
            "time_ps": self.timeline.now(),
            "operation_id": operation_id,
            "demand_id": context.demand.demand_id,
            "input_fidelities": fidelities,
            "completion_ps": completion,
        })
        left_protocol.start()
        right_protocol.start()
        event = Event(
            completion,
            Process(self, "_complete_purification", [operation_id]),
        )
        operation.events.append(event)
        self.timeline.schedule(event)

    def _complete_purification(self, operation_id: str) -> None:
        operation = self.purification_operations.pop(operation_id, None)
        if operation is None:
            return
        context = self.contexts.get(operation.reservation_id)
        success = self._pair_is_purified(operation.kept)
        if success:
            self._normalize_pair(operation.kept)
            self._release_pair(operation.measured)
            self.counters["pumping_successes"] += 1
        else:
            self._release_pair(operation.kept)
            self._release_pair(operation.measured)
            self.counters["pumping_failures"] += 1
        self.events.append({
            "event": "pumping_completed",
            "time_ps": self.timeline.now(),
            "operation_id": operation_id,
            "success": success,
            "output_fidelity": (
                self._pair_fidelity(operation.kept) if success else 0.0
            ),
        })
        if context is None or context.terminal:
            if success:
                self._release_pair(operation.kept)
            return
        if not success:
            context.hop_pair = None
            self._start_elementary_generation(context, "initial")
            return
        context.hop_pair = operation.kept
        context.completed_rounds += 1
        self._continue_hop_pumping(context)

    def _advance_with_link_pair(self, context: _Context) -> None:
        link_pair = context.hop_pair
        if link_pair is None or context.selected_neighbor is None:
            self._restart_pair(context, "missing_qualified_link")
            return
        if context.long_pair is None:
            context.long_pair = link_pair
            context.hop_pair = None
            context.current = context.selected_neighbor
            context.path.append(context.current)
            self._schedule_local_hop(context, self.timeline.now() + 1)
            return
        self._start_swap(context, context.long_pair, link_pair)

    def _start_swap(
        self,
        context: _Context,
        long_pair: _Pair,
        link_pair: _Pair,
    ) -> None:
        if not self._pair_is_entangled(long_pair) or not self._pair_is_entangled(link_pair):
            self._restart_pair(context, "swap_inputs_expired")
            return
        current = context.current
        next_hop = context.selected_neighbor
        if next_hop is None:
            self._restart_pair(context, "missing_swap_neighbor")
            return
        operation_id = self._next_operation_id("swap")
        middle = self.routers[current]
        left_memory = self._memory(current, long_pair.memory_index(current))
        right_memory = self._memory(current, link_pair.memory_index(current))
        left_remote = self.routers[long_pair.left]
        right_remote = self.routers[next_hop]
        left_hold = self._memory(long_pair.left, long_pair.memory_index(long_pair.left))
        right_hold = self._memory(next_hop, link_pair.memory_index(next_hop))
        protocol_a = EntanglementSwappingA.create(
            middle,
            operation_id + ".A",
            left_memory,
            right_memory,
            success_prob=self.algorithm.swap_success_probability,
        )
        protocol_left = EntanglementSwappingB.create(
            left_remote,
            operation_id + ".L",
            left_hold,
        )
        protocol_right = EntanglementSwappingB.create(
            right_remote,
            operation_id + ".R",
            right_hold,
        )
        protocol_a.set_others(
            protocol_left.name,
            left_remote.name,
            [left_hold.name],
        )
        protocol_a.set_others(
            protocol_right.name,
            right_remote.name,
            [right_hold.name],
        )
        protocol_left.set_others(
            protocol_a.name,
            middle.name,
            [left_memory.name, right_memory.name],
        )
        protocol_right.set_others(
            protocol_a.name,
            middle.name,
            [left_memory.name, right_memory.name],
        )
        for owner, protocol in (
            (middle, protocol_a),
            (left_remote, protocol_left),
            (right_remote, protocol_right),
        ):
            owner.protocols.append(protocol)
            for memory in protocol.memories:
                memory.detach(memory.memory_array)
                memory.attach(protocol)
                owner.resource_manager.memory_manager.get_info_by_memory(
                    memory
                ).to_occupied()
        operation = _SwapOperation(
            operation_id,
            context.demand.reservation_id,
            long_pair,
            link_pair,
            current,
            next_hop,
            self.timeline.now(),
            protocols=[protocol_a, protocol_left, protocol_right],
        )
        self.swap_operations[operation_id] = operation
        self.counters["swaps_attempted"] += 1
        self.events.append({
            "event": "swap_started",
            "time_ps": self.timeline.now(),
            "operation_id": operation_id,
            "demand_id": context.demand.demand_id,
            "current": current,
            "next": next_hop,
        })
        protocol_left.start()
        protocol_right.start()
        protocol_a.start()
        completion = self.timeline.now() + max(
            int(middle.cchannels[left_remote.name].delay),
            int(middle.cchannels[right_remote.name].delay),
        ) + 1
        event = Event(
            completion,
            Process(self, "_complete_swap", [operation_id]),
        )
        operation.events.append(event)
        self.timeline.schedule(event)

    def _complete_swap(self, operation_id: str) -> None:
        operation = self.swap_operations.pop(operation_id, None)
        if operation is None:
            return
        context = self.contexts.get(operation.reservation_id)
        source = operation.long_pair.left
        source_index = operation.long_pair.memory_index(source)
        next_index = operation.link_pair.memory_index(operation.next_hop)
        success = self._memories_entangled(
            source,
            source_index,
            operation.next_hop,
            next_index,
        )
        self._unclaim(operation.current, operation.long_pair.memory_index(operation.current))
        self._unclaim(operation.current, operation.link_pair.memory_index(operation.current))
        if success:
            combined = _Pair(
                operation_id,
                source,
                operation.next_hop,
                source_index,
                next_index,
                operation.long_pair.path + (operation.next_hop,),
                (
                    operation.long_pair.elementary_links
                    + operation.link_pair.elementary_links
                ),
                min(
                    operation.long_pair.created_at_ps,
                    operation.link_pair.created_at_ps,
                ),
            )
            self.counters["swaps_succeeded"] += 1
        else:
            self._release_pair(operation.long_pair)
            self._release_pair(operation.link_pair)
            combined = None
            self.counters["swaps_failed"] += 1
        self.events.append({
            "event": "swap_completed",
            "time_ps": self.timeline.now(),
            "operation_id": operation_id,
            "success": success,
            "output_fidelity": (
                self._pair_fidelity(combined) if combined is not None else 0.0
            ),
        })
        if context is None or context.terminal:
            if combined is not None:
                self._release_pair(combined)
            return
        context.hop_pair = None
        if combined is None:
            context.long_pair = None
            self._restart_pair(context, "swap_failure")
            return
        context.long_pair = combined
        context.current = operation.next_hop
        context.path.append(operation.next_hop)
        self._schedule_local_hop(context, self.timeline.now() + 1)

    def _deliver_or_restart(self, context: _Context) -> None:
        pair = context.long_pair
        if pair is None:
            self._restart_pair(context, "missing_destination_pair")
            return
        fidelity = self._pair_fidelity(pair)
        if fidelity + 1e-12 < context.demand.fidelity_threshold:
            self.counters["pairs_rejected_fidelity"] += 1
            context.callbacks.on_pair_rejected(context.demand, fidelity)
            self._release_pair(pair)
            context.long_pair = None
            self._restart_pair(context, "final_fidelity_below_threshold")
            return
        delivery = PairDelivery(
            timestamp_ps=self.timeline.now(),
            fidelity=fidelity,
            generation_source="application",
            elementary_sources=tuple(
                {
                    "source": "application",
                    "algorithm": "dfer",
                    "link": link,
                }
                for link in pair.elementary_links
            ),
        )
        context.deliveries.append(delivery)
        context.last_path = tuple(context.path)
        context.callbacks.on_pair_delivered(context.demand, delivery)
        self.counters["end_to_end_pairs_delivered"] += 1
        self.events.append({
            "event": "pair_delivered",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "fidelity": fidelity,
            "path": list(context.path),
            "oldest_pair_age_ps": self.timeline.now() - pair.created_at_ps,
        })
        self._release_pair(pair)
        context.long_pair = None
        if len(context.deliveries) >= context.demand.pair_count:
            context.terminal = True
            self.counters["demands_completed"] += 1
            context.callbacks.on_demand_completed(
                context.demand,
                self.timeline.now(),
                context.last_path,
            )
            return
        self._reset_route(context)
        self._schedule_local_hop(context, self.timeline.now() + 1)

    def _restart_pair(self, context: _Context, reason: str) -> None:
        if context.terminal:
            return
        if context.long_pair is not None:
            self._release_pair(context.long_pair)
        if context.hop_pair is not None:
            self._release_pair(context.hop_pair)
        context.long_pair = None
        context.hop_pair = None
        self.counters["pair_route_restarts"] += 1
        self.counters[f"pair_route_restarts_{reason}"] += 1
        self.events.append({
            "event": "pair_route_restarted",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "reason": reason,
            "failed_path": list(context.path),
        })
        self._reset_route(context)
        self._schedule_local_hop(
            context,
            self.timeline.now() + self.algorithm.resource_retry_delay_ps,
        )

    def deadline(self, reservation_id: int) -> None:
        context = self.contexts.get(reservation_id)
        if context is not None and not context.terminal:
            self._fail(context, "dfer_deadline")

    def finalize(self, now_ps: int) -> None:
        for context in self.contexts.values():
            if not context.terminal:
                self._fail(context, "simulation_end", now_ps)
        for operation in list(self.generation_operations.values()):
            self._expire_rules(operation.rules)
            self.claimed_middles.discard(operation.middle)
            self._release_indices(
                operation.left,
                operation.left_index,
                operation.right,
                operation.right_index,
            )
        self.generation_operations.clear()
        for operation in list(self.purification_operations.values()):
            self._release_pair(operation.kept)
            self._release_pair(operation.measured)
        self.purification_operations.clear()
        for operation in list(self.swap_operations.values()):
            self._release_pair(operation.long_pair)
            self._release_pair(operation.link_pair)
        self.swap_operations.clear()
        self.claimed_middles.clear()

    def diagnostics(self) -> dict:
        all_raw = {
            name: all(
                info.state == MemoryInfo.RAW
                for info in router.resource_manager.memory_manager
            )
            for name, router in self.routers.items()
        }
        return {
            "counters": dict(self.counters),
            "events": self.events,
            "dfer": {
                "paper_profile": "physically_consistent_remaining_hops",
                "cutoff_fidelity": self.algorithm.cutoff_fidelity,
                "max_purification_rounds": (
                    self.algorithm.max_purification_rounds
                ),
                "decisions": self.decisions,
                "uses_official_bell_diagonal_bbpssw": True,
                "uses_official_bell_diagonal_swapping": True,
                "pumping_pair_cost": "one_fresh_elementary_pair_per_round",
                "explicit_neighbor_request_response": True,
            },
            "all_memories_raw_at_end": all_raw,
            "memory_high_watermark_by_node": dict(self.memory_high_watermark),
            "claimed_memories_at_end": {
                name: sorted(indices)
                for name, indices in self.claimed_memories.items()
            },
            "scheduled_lane_usage_by_link": {
                "|".join(link): {
                    middle: self.channel_usage[(link, middle)]
                    for middle in self.edge_middles[link]
                }
                for link in sorted(self.edge_middles)
            },
            "edge_models": [
                {"edge": list(link), **model}
                for link, model in sorted(self._edge_models.items())
            ],
            "locality_invariants": {
                "all_state_queries_one_hop": (
                    self.counters["nonlocal_state_queries"] == 0
                ),
                "no_global_control_messages": True,
                "all_claims_released": all(
                    not indices for indices in self.claimed_memories.values()
                ),
            },
        }

    def _fail(
        self,
        context: _Context,
        reason: str,
        now_ps: int | None = None,
    ) -> None:
        if context.terminal:
            return
        context.terminal = True
        self._cancel_pending_operations(context.demand.reservation_id)
        if context.long_pair is not None:
            self._release_pair(context.long_pair)
            context.long_pair = None
        if context.hop_pair is not None:
            self._release_pair(context.hop_pair)
            context.hop_pair = None
        for operation_id, operation in list(self.generation_operations.items()):
            if operation.reservation_id != context.demand.reservation_id:
                continue
            self._expire_rules(operation.rules)
            self.claimed_middles.discard(operation.middle)
            self._release_indices(
                operation.left,
                operation.left_index,
                operation.right,
                operation.right_index,
            )
            self.generation_operations.pop(operation_id, None)
        timestamp = self.timeline.now() if now_ps is None else now_ps
        self.counters[f"demands_failed_{reason}"] += 1
        self.events.append({
            "event": "demand_failed",
            "time_ps": timestamp,
            "demand_id": context.demand.demand_id,
            "reason": reason,
            "pairs_delivered": len(context.deliveries),
            "path": list(context.path),
        })
        context.callbacks.on_demand_failed(
            context.demand,
            timestamp,
            reason,
            tuple(context.path),
            len(context.deliveries),
        )

    def _cancel_pending_operations(self, reservation_id: int) -> None:
        """Retire callbacks/protocols before releasing their memory claims.

        Native result messages address uniquely named protocols. Removing those
        protocols also prevents late messages from updating a reused memory.
        """
        for operations in (self.purification_operations, self.swap_operations):
            for operation_id, operation in list(operations.items()):
                if operation.reservation_id != reservation_id:
                    continue
                operations.pop(operation_id)
                for event in operation.events:
                    self.timeline.remove_event(event)
                for protocol in operation.protocols:
                    if protocol in protocol.owner.protocols:
                        protocol.owner.protocols.remove(protocol)
                    for memory in protocol.memories:
                        if protocol in memory._observers:
                            memory.detach(protocol)
                            memory.attach(memory.memory_array)
                pairs = (
                    (operation.kept, operation.measured)
                    if isinstance(operation, _PurificationOperation)
                    else (operation.long_pair, operation.link_pair)
                )
                for pair in pairs:
                    self._release_pair(pair)
                self.counters["operations_cancelled"] += 1

    def _schedule_local_hop(self, context: _Context, time_ps: int) -> None:
        if context.terminal or time_ps >= context.demand.deadline_ps:
            return
        self.timeline.schedule(Event(
            time_ps,
            Process(
                self.routers[context.current].dfer_control,
                "begin_hop",
                [context.demand.reservation_id],
            ),
        ))

    def _reset_route(self, context: _Context) -> None:
        context.current = context.demand.source
        context.path = [context.demand.source]
        context.selected_neighbor = None
        context.target_fidelity = 0.0
        context.planned_rounds = 0
        context.completed_rounds = 0

    def _claim_link_resources(
        self,
        left: str,
        right: str,
    ) -> tuple[int, int, str] | None:
        left_free = self._free_indices(left)
        right_free = self._free_indices(right)
        middles = [
            middle
            for middle in self.edge_middles[edge_key(left, right)]
            if middle not in self.claimed_middles
        ]
        if not left_free or not right_free or not middles:
            return None
        left_index = left_free[0]
        right_index = right_free[0]
        middle = min(
            middles,
            key=lambda item: (
                self.channel_usage[(edge_key(left, right), item)],
                item,
            ),
        )
        self.claimed_memories[left].add(left_index)
        self.claimed_memories[right].add(right_index)
        self.claimed_middles.add(middle)
        self.memory_high_watermark[left] = max(
            self.memory_high_watermark[left],
            len(self.claimed_memories[left]),
        )
        self.memory_high_watermark[right] = max(
            self.memory_high_watermark[right],
            len(self.claimed_memories[right]),
        )
        return left_index, right_index, middle

    def _free_indices(self, node: str) -> list[int]:
        return [
            info.index
            for info in self.routers[node].resource_manager.memory_manager
            if (
                info.state == MemoryInfo.RAW
                and info.index not in self.claimed_memories[node]
            )
        ]

    def _release_pair(self, pair: _Pair) -> None:
        self._release_indices(
            pair.left,
            pair.left_index,
            pair.right,
            pair.right_index,
        )

    def _release_indices(
        self,
        left: str,
        left_index: int,
        right: str,
        right_index: int,
    ) -> None:
        self._release_index(left, left_index)
        self._release_index(right, right_index)

    def _release_index(self, node: str, index: int) -> None:
        memory = self._memory(node, index)
        info = self.routers[node].resource_manager.memory_manager.get_info_by_memory(
            memory
        )
        if info.state != MemoryInfo.RAW:
            self.routers[node].resource_manager.update(
                None,
                memory,
                MemoryInfo.RAW,
            )
        self._unclaim(node, index)

    def _unclaim(self, node: str, index: int) -> None:
        self.claimed_memories[node].discard(index)
        memory = self._memory(node, index)
        if hasattr(memory, "dfer_operation_id"):
            delattr(memory, "dfer_operation_id")

    @staticmethod
    def _expire_rules(rules: list[tuple[object, Rule]]) -> None:
        for node, rule in list(rules):
            if rule in node.resource_manager.rule_manager.rules:
                node.resource_manager.expire(rule)
        rules.clear()

    def _pair_fidelity(self, pair: _Pair | None) -> float:
        if pair is None or not self._pair_is_entangled(pair):
            return 0.0
        fidelities = []
        for node, index in (
            (pair.left, pair.left_index),
            (pair.right, pair.right_index),
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

    def _pair_is_entangled(self, pair: _Pair) -> bool:
        return self._memories_entangled(
            pair.left,
            pair.left_index,
            pair.right,
            pair.right_index,
        )

    def _memories_entangled(
        self,
        left: str,
        left_index: int,
        right: str,
        right_index: int,
    ) -> bool:
        left_memory = self._memory(left, left_index)
        right_memory = self._memory(right, right_index)
        left_info = self.routers[left].resource_manager.memory_manager.get_info_by_memory(
            left_memory
        )
        right_info = self.routers[right].resource_manager.memory_manager.get_info_by_memory(
            right_memory
        )
        return (
            left_info.state in {MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED}
            and right_info.state in {MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED}
            and left_info.remote_node == right
            and right_info.remote_node == left
            and left_memory.entangled_memory["memo_id"] == right_memory.name
            and right_memory.entangled_memory["memo_id"] == left_memory.name
        )

    def _pair_is_purified(self, pair: _Pair) -> bool:
        states = {
            self.routers[node].resource_manager.memory_manager.get_info_by_memory(
                self._memory(node, index)
            ).state
            for node, index in (
                (pair.left, pair.left_index),
                (pair.right, pair.right_index),
            )
        }
        return states == {MemoryInfo.PURIFIED} and self._pair_is_entangled(pair)

    def _normalize_pair(self, pair: _Pair) -> None:
        for node, index in (
            (pair.left, pair.left_index),
            (pair.right, pair.right_index),
        ):
            memory = self._memory(node, index)
            self.routers[node].resource_manager.memory_manager.update(
                memory,
                MemoryInfo.ENTANGLED,
            )

    def _base_link_fidelity(self, left: str, right: str) -> float:
        return min(
            self._memory(left, 0).raw_fidelity,
            self._memory(right, 0).raw_fidelity,
        )

    def _router_graph(self) -> dict[str, set[str]]:
        graph = {name: set() for name in self.routers}
        for endpoints in self.network_topology.bsm_to_router_map.values():
            left, right = endpoints
            graph[left].add(right)
            graph[right].add(left)
        return graph

    def _middle_nodes_by_edge(self) -> dict[tuple[str, str], tuple[str, ...]]:
        middles: dict[tuple[str, str], list[str]] = {}
        for middle, endpoints in self.network_topology.bsm_to_router_map.items():
            middles.setdefault(edge_key(*endpoints), []).append(middle)
        return {
            link: tuple(sorted(names))
            for link, names in middles.items()
        }

    def _physical_edge_models(self) -> dict[tuple[str, str], dict]:
        models = {}
        for link, middles in self.edge_middles.items():
            left, right = link
            channel_models = [
                self._physical_channel_model(left, right, middle)
                for middle in middles
            ]
            probability = sum(
                model["attempt_success_probability"]
                for model in channel_models
            ) / len(channel_models)
            duration = max(
                model["attempt_duration_ps"] for model in channel_models
            )
            models[link] = {
                "attempt_success_probability": probability,
                "attempt_duration_ps": duration,
                "generation_rate_hz": probability * 10**12 / duration,
                "physical_channels": list(middles),
            }
        return models

    def _physical_channel_model(
        self,
        left: str,
        right: str,
        middle: str,
    ) -> dict:
        left_node = self.routers[left]
        right_node = self.routers[right]
        left_channel = left_node.qchannels[middle]
        right_channel = right_node.qchannels[middle]
        left_transmittance = 10 ** (
            -left_channel.distance * left_channel.attenuation / 10
        )
        right_transmittance = 10 ** (
            -right_channel.distance * right_channel.attenuation / 10
        )
        bsm_node = self.timeline.get_entity_by_name(middle)
        bsm = next(iter(bsm_node.components.values()))
        detector_efficiencies = tuple(
            detector.efficiency for detector in bsm.detectors
        )
        probability = single_heralded_attempt_success_probability(
            self._memory(left, 0).efficiency * left_transmittance,
            self._memory(right, 0).efficiency * right_transmittance,
            detector_efficiencies,
            bsm.success_rate,
        )
        router_delay = max(
            int(left_node.cchannels[right].delay),
            int(right_node.cchannels[left].delay),
        )
        bsm_delay = max(
            int(bsm_node.cchannels[left].delay),
            int(bsm_node.cchannels[right].delay),
        )
        quantum_delay = max(
            round(left_channel.distance / left_channel.light_speed),
            round(right_channel.distance / right_channel.light_speed),
        )
        return {
            "attempt_success_probability": probability,
            "attempt_duration_ps": max(
                1,
                2 * router_delay + 2 * (quantum_delay + bsm_delay),
            ),
        }

    def _distance(self, source: str, destination: str) -> int | None:
        path = self._shortest_path(source, destination)
        return len(path) - 1 if path else None

    def _shortest_path(
        self,
        source: str,
        destination: str,
    ) -> tuple[str, ...]:
        queue = deque([(source, (source,))])
        seen = {source}
        while queue:
            node, path = queue.popleft()
            if node == destination:
                return path
            for neighbor in sorted(self.graph[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + (neighbor,)))
        return ()

    def _memory(self, node: str, index: int):
        router = self.routers[node]
        return router.components[router.memo_arr_name][index]

    def _next_operation_id(self, prefix: str) -> str:
        operation_id = f"dfer-{prefix}-{self._operation_counter}"
        self._operation_counter += 1
        return operation_id
