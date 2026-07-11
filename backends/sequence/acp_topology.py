"""ACP-aware SeQUeNCe topology/node integration."""

from __future__ import annotations

from typing import Any

from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.entanglement_management.generation.single_heralded import SingleHeraldedA
from sequence.entanglement_management.purification.bbpssw_bds import BBPSSW_BDS
from sequence.entanglement_management.purification.bbpssw_protocol import BBPSSWProtocol
from sequence.entanglement_management.purification.bbpssw_protocol import BBPSSWMsgType
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.network_management.rsvp import RSVPProtocol
from sequence.network_management.reservation import Reservation
from sequence.resource_management.action_condition_set import (
    eg_match_func,
    eg_rule_action_await,
    eg_rule_condition,
    es_rule_action_A,
    es_rule_action_B,
    es_rule_condition_A,
    es_rule_condition_B,
    es_rule_condition_B_end,
)
from sequence.resource_management.memory_manager import MemoryInfo, MemoryManager
from sequence.resource_management.resource_manager import ResourceManager
from sequence.resource_management.rule_manager import Arguments, Rule
from sequence.topology.node import BSMNode, QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.topology import Topology as Topo

from backends.sequence.acp_protocol import ACPMessage, ACPMsgType, AdaptiveContinuousProtocol, AdaptiveReservation
from backends.sequence.parallel_links import (
    ParallelResourceManager,
    configure_parallel_middle_nodes,
    ensure_parallel_metrics,
    middle_nodes,
    parallel_eg_match_func,
    parallel_eg_rule_action_request,
    parallel_eg_rule_condition,
    record_generation_attempt,
    record_generation_success,
    record_memory_occupancy,
)


class ACPReuseSingleHeraldedA(SingleHeraldedA):
    """Single-heralded EG that lets ACP cache replace app photon generation."""

    def start(self) -> None:
        if self._try_application_cache_reuse():
            return
        super().start()

    def _try_application_cache_reuse(self) -> bool:
        if not self.primary or self.ent_round != 0:
            return False
        if getattr(self, "cache_reuse_attempted", False):
            return False
        reservation = getattr(getattr(self, "rule", None), "reservation", None)
        if reservation is None or isinstance(reservation, AdaptiveReservation):
            return False
        resource_manager = getattr(self.owner, "resource_manager", None)
        acp = getattr(self.owner, "adaptive_continuous", None)
        if resource_manager is None or acp is None:
            return False
        right_node = self.owner.timeline.get_entity_by_name(self.remote_node_name)
        if right_node is None or not hasattr(right_node, "adaptive_continuous"):
            return False
        if resource_manager._cache_edge_satisfied(reservation, self.owner.name, right_node.name):
            acp.counters["cache_protocol_boundary_edge_already_satisfied"] += 1
            return True
        if resource_manager._cache_edge_pending(reservation, self.owner.name, right_node.name):
            self.cache_reuse_attempted = True
            acp.counters["cache_protocol_boundary_waits"] += 1
            return True
        try:
            left_target_index = self.owner.components[self.owner.memo_arr_name].memory_name_to_index[self.memory.name]
            right_target_index = right_node.components[right_node.memo_arr_name].memory_name_to_index[self.remote_memo_id]
        except (AttributeError, KeyError):
            return False

        acp.counters["cache_protocol_boundary_checks"] += 1
        requested = resource_manager._request_cached_pair(
            right_node,
            left_target_index,
            right_target_index,
            reservation,
        )
        if requested:
            self.cache_reuse_attempted = True
            acp.counters["cache_protocol_boundary_requests"] += 1
            acp.lifecycle_events.append({
                "event": "cache_protocol_boundary_request",
                "time_ps": self.owner.timeline.now(),
                "reservation": reservation.identity,
                "left": self.owner.name,
                "right": right_node.name,
                "left_target_index": left_target_index,
                "right_target_index": right_target_index,
            })
            return True
        acp.counters["cache_protocol_boundary_misses"] += 1
        return False


class ACPBackgroundPurification(BBPSSW_BDS):
    """BBPSSW BDS with stale-state handling for ACP background cache pairs."""

    def received_message(self, src: str, msg) -> None:
        if msg.msg_type is not BBPSSWMsgType.PURIFICATION_RES:
            return super().received_message(src, msg)
        purification_success = self.meas_res == msg.meas_res
        self.update_resource_manager(self.meas_memo, MemoryInfo.RAW)
        if not purification_success:
            self.update_resource_manager(self.kept_memo, MemoryInfo.RAW)
            return
        try:
            remote_kept_memory = self.owner.timeline.get_entity_by_name(self.remote_memories[0])
            remote_kept_memory.bds_decohere()
            self.kept_memo.bds_decohere()
            self.kept_memo.fidelity = self.kept_memo.get_bds_fidelity()
        except Exception:
            self.update_resource_manager(self.kept_memo, MemoryInfo.RAW)
            return
        self.update_resource_manager(self.kept_memo, state=MemoryInfo.PURIFIED)


def eg_rule_action_await_adaptive(memories_info: list[MemoryInfo], args: Arguments):
    return eg_rule_action_await(memories_info, args)


def eg_rule_action_request_adaptive(memories_info: list[MemoryInfo], args: Arguments):
    return parallel_eg_rule_action_request(memories_info, args)


def eg_rule_action_await_acp_app(memories_info: list[MemoryInfo], args: Arguments):
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    mid = args["mid"]
    path = args["path"]
    index = args["index"]
    protocol = ACPReuseSingleHeraldedA(None, f"EGA.{memory.name}", mid, path[index - 1], memory)
    return protocol, [None], [None], [None]


def eg_rule_action_request_acp_app(memories_info: list[MemoryInfo], args: Arguments):
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    mid = args["mid"]
    path = args["path"]
    index = args["index"]
    protocol = ACPReuseSingleHeraldedA(None, f"EGA.{memory.name}", mid, path[index + 1], memory)
    owner = memory.memory_array.owner
    record_generation_attempt(owner, path[index + 1], mid)
    req_args = {"name": args["name"], "reservation": args["reservation"]}
    req_args["mid"] = mid
    return protocol, [path[index + 1]], [parallel_eg_match_func], [req_args]


def eg_match_func_adaptive(protocols, args):
    return eg_match_func(protocols, args)


class ACPMemoryManager(MemoryManager):
    """Memory manager with explicit two-memory state transfer."""

    def swap_two_memory(self, memory1_name: str, memory2_name: str) -> None:
        i = self.memory_array.memory_name_to_index[memory1_name]
        j = self.memory_array.memory_name_to_index[memory2_name]
        left = self.memory_array[i]
        right = self.memory_array[j]
        fields = (
            "fidelity", "raw_fidelity", "frequency", "efficiency", "coherence_time",
            "wavelength", "qstate_key", "encoding", "previous_bsm",
            "entangled_memory", "expiration_event", "excited_photon",
            "next_excite_time", "decoherence_errors", "cutoff_ratio",
            "generation_time", "last_update_time", "is_in_application",
        )
        for field in fields:
            if hasattr(left, field) and hasattr(right, field):
                left_value = getattr(left, field)
                right_value = getattr(right, field)
                setattr(left, field, right_value)
                setattr(right, field, left_value)
        self.memory_map[i].state, self.memory_map[j].state = self.memory_map[j].state, self.memory_map[i].state
        self.memory_map[i].remote_node, self.memory_map[j].remote_node = self.memory_map[j].remote_node, self.memory_map[i].remote_node
        self.memory_map[i].remote_memo, self.memory_map[j].remote_memo = self.memory_map[j].remote_memo, self.memory_map[i].remote_memo
        self.memory_map[i].fidelity, self.memory_map[j].fidelity = self.memory_map[j].fidelity, self.memory_map[i].fidelity
        self.memory_map[i].expire_event, self.memory_map[j].expire_event = self.memory_map[j].expire_event, self.memory_map[i].expire_event
        self.memory_map[i].entangle_time, self.memory_map[j].entangle_time = self.memory_map[j].entangle_time, self.memory_map[i].entangle_time


class ACPResourceManager(ParallelResourceManager):
    """Resource manager extension for ACP inventory and cache adoption."""

    def __init__(self, owner: QuantumRouter, memory_array_name: str):
        super().__init__(owner, memory_array_name)
        self.memory_manager = ACPMemoryManager(owner.components[memory_array_name])
        self.memory_manager.set_resource_manager(self)
        self.cache_satisfied_reservations: set[int] = set()
        self.cache_satisfied_edges: set[tuple[int, tuple[str, str]]] = set()
        self.cache_pending_edges: set[tuple[int, tuple[str, str]]] = set()

    def generate_load_rules(self, path: list[str], reservation: Reservation, timecards: list, memory_array_name: str):
        if isinstance(reservation, AdaptiveReservation):
            return super().generate_load_rules(path, reservation, timecards, memory_array_name)

        if getattr(self.owner, "adaptive_continuous", None) is not None:
            self.owner.timeline.schedule(Event(
                reservation.start_time,
                Process(self, "initiate_cached_pairs_for_reservation", [path, reservation, timecards, memory_array_name]),
                -10,
            ))
        self._generate_no_purification_rules(
            path,
            reservation,
            timecards,
            memory_array_name,
            reservation.start_time,
        )

    def _cache_endpoint_processing_delay(self) -> int:
        acp = getattr(self.owner, "adaptive_continuous", None)
        if acp is None:
            return 0
        return int(getattr(acp, "cache_endpoint_processing_delay_ps", 0))

    def _generate_no_purification_rules(self, path, reservation, timecards, memory_array_name, activation_time):
        memory_indices = [card.memory_index for card in timecards if reservation in card.reservations]
        index = path.index(self.owner.name)
        rules = []
        if index > 0:
            rules.extend(self._application_generation_rules(
                path[index - 1],
                path,
                index,
                memory_indices[:reservation.memory_size],
                reservation,
                requester=False,
            ))
        if index < len(path) - 1:
            selected = memory_indices[:reservation.memory_size] if index == 0 else memory_indices[reservation.memory_size:]
            rules.extend(self._application_generation_rules(
                path[index + 1],
                path,
                index,
                selected,
                reservation,
                requester=True,
            ))
        if index == 0:
            rules.append(Rule(10, es_rule_action_B, es_rule_condition_B_end, {}, {
                "memory_indices": memory_indices,
                "target_remote": path[-1],
                "fidelity": reservation.fidelity,
            }))
        elif index == len(path) - 1:
            rules.append(Rule(10, es_rule_action_B, es_rule_condition_B_end, {}, {
                "memory_indices": memory_indices,
                "target_remote": path[0],
                "fidelity": reservation.fidelity,
            }))
        else:
            _path = path[:]
            while _path.index(self.owner.name) % 2 == 0:
                _path = [node for i, node in enumerate(_path) if i % 2 == 0 or i == len(_path) - 1]
            _index = _path.index(self.owner.name)
            left, right = _path[_index - 1], _path[_index + 1]
            condition_args = {
                "memory_indices": memory_indices,
                "left": left,
                "right": right,
                "fidelity": reservation.fidelity,
            }
            rules.append(Rule(10, es_rule_action_A, es_rule_condition_A, {
                "swapping_success_prob": self.owner.swapping_success_prob,
                "swapping_degradation": self.owner.swapping_degradation,
            }, condition_args))
            rules.append(Rule(10, es_rule_action_B, es_rule_condition_B, {}, condition_args))

        for rule in rules:
            rule.set_reservation(reservation)
            self.owner.timeline.schedule(Event(activation_time, Process(self, "load_application_rule", [rule, reservation]), self.owner.timeline.schedule_counter))
            self.owner.timeline.schedule(Event(reservation.end_time, Process(self, "expire", [rule]), self.owner.timeline.schedule_counter))
        for card in timecards:
            if reservation in card.reservations:
                memory = self.owner.components[memory_array_name][card.memory_index]
                self.owner.timeline.schedule(Event(reservation.end_time, Process(self, "update", [None, memory, MemoryInfo.RAW]), self.owner.timeline.schedule_counter))

    def _application_generation_rules(
        self,
        neighbor,
        path,
        index,
        memory_indices,
        reservation,
        *,
        requester,
    ):
        lanes = middle_nodes(self.owner, neighbor)
        rules = []
        for lane_index, middle in enumerate(lanes):
            parallel = len(lanes) > 1
            action_args = {"mid": middle, "path": path, "index": index}
            if requester:
                action_args.update({
                    "name": self.owner.name,
                    "reservation": reservation,
                    "parallel_lane": parallel,
                })
                action = eg_rule_action_request_acp_app
            else:
                action = eg_rule_action_await_acp_app
            condition_args = {"memory_indices": memory_indices}
            if parallel:
                condition_args.update({
                    "lane_index": lane_index,
                    "lane_count": len(lanes),
                })
            rules.append(Rule(
                10,
                action,
                parallel_eg_rule_condition if parallel else eg_rule_condition,
                action_args,
                condition_args,
            ))
        return rules

    def load_application_rule(self, rule: Rule, reservation: Reservation) -> None:
        if len(getattr(reservation, "path", [])) <= 2 and id(reservation) in self.cache_satisfied_reservations:
            return
        self.load(rule)

    def initiate_cached_pairs_for_reservation(self, path: list[str], reservation: Reservation, timecards: list, memory_array_name: str) -> int:
        if self.owner.name not in path or not hasattr(self.owner, "adaptive_continuous"):
            return 0
        index = path.index(self.owner.name)
        requested = 0
        if index < len(path) - 1:
            right_name = path[index + 1]
            right_node = self.owner.timeline.get_entity_by_name(right_name)
            left_indices = self._edge_indices(self.owner, reservation, timecards, path, "right")
            right_indices = self._edge_indices(right_node, reservation, right_node.network_manager.get_timecards(), path, "left")
            for left_idx, right_idx in zip(left_indices, right_indices):
                requested += self._request_cached_pair(right_node, left_idx, right_idx, reservation)
        return requested

    def _edge_indices(self, node, reservation, timecards, path, side: str) -> list[int]:
        indices = [card.memory_index for card in timecards if reservation in card.reservations]
        index = path.index(node.name)
        size = reservation.memory_size
        if side == "left":
            return [] if index <= 0 else indices[:size]
        if side == "right":
            if index >= len(path) - 1:
                return []
            return indices[:size] if index == 0 else indices[size:]
        raise ValueError(side)

    def _request_cached_pair(self, right_node, left_target_index: int, right_target_index: int, reservation: Reservation) -> int:
        left_acp = self.owner.adaptive_continuous
        now = self.owner.timeline.now()
        if self._cache_edge_satisfied(reservation, self.owner.name, right_node.name):
            left_acp.counters["cache_edge_already_satisfied"] += 1
            return 0
        if self._cache_edge_pending(reservation, self.owner.name, right_node.name):
            left_acp.counters["cache_edge_pending"] += 1
            return 0
        left_acp.lifecycle_events.append({
            "event": "cache_check",
            "time_ps": now,
            "reservation": reservation.identity,
            "request_start_ps": reservation.start_time,
            "left": self.owner.name,
            "right": right_node.name,
        })
        pair = self._select_cached_pair(self.owner, right_node, reservation)
        if pair is None:
            return 0
        left_target = self.owner.components[self.owner.memo_arr_name][left_target_index]
        if not self._target_cache_adoptable(self.owner, left_target, reservation):
            return 0
        left_acp.lifecycle_events.append({
            "event": "cache_pair_selected",
            "time_ps": now,
            "reservation": reservation.identity,
            "pair": pair,
            "left_target_index": left_target_index,
            "right_target_index": right_target_index,
        })
        message = ACPMessage(
            ACPMsgType.CACHE_REQUEST,
            reservation,
            pair=pair,
            left_target_index=left_target_index,
            right_target_index=right_target_index,
        )
        self.cache_pending_edges.add(self._cache_edge_key(reservation, self.owner.name, right_node.name))
        self.owner.send_message(right_node.name, message)
        left_acp.counters["cache_coordination_requests_sent"] += 1
        left_acp.lifecycle_events.append({
            "event": "cache_request_sent",
            "time_ps": now,
            "reservation": reservation.identity,
            "pair": pair,
            "to": right_node.name,
        })
        return 1

    def _select_cached_pair(self, left_node, right_node, reservation: Reservation):
        left_acp = left_node.adaptive_continuous
        candidates = [
            pair for pair in left_acp.generated_entanglement_pairs
            if pair[0][0] == left_node.name and pair[1][0] == right_node.name
        ]
        left_acp.counters["cache_checks"] += 1
        if not candidates:
            left_acp.counters["cache_misses"] += 1
            return None
        left_acp.counters["cache_candidates_found"] += 1
        if left_acp.strategy == "random":
            ordered = list(sorted(candidates))
            left_node.get_generator().shuffle(ordered)
        else:
            ordered = sorted(candidates, key=left_acp.cache_candidate_key, reverse=True)

        right_acp = getattr(right_node, "adaptive_continuous", None)
        for pair in ordered:
            reverse_pair = (pair[1], pair[0])
            if right_acp is None or reverse_pair not in right_acp.generated_entanglement_pairs:
                left_acp.remove_entanglement_pair(pair, reason="stale")
                left_acp.counters["cache_candidates_stale"] += 1
                continue
            if self._candidate_valid(pair, left_node, right_node, reservation):
                return pair
            left_acp.remove_entanglement_pair(pair, reason="stale")
            left_acp.counters["cache_candidates_stale"] += 1
        left_acp.counters["cache_misses"] += 1
        return None

    def handle_cache_request(self, src: str, msg: ACPMessage) -> None:
        acp = getattr(self.owner, "adaptive_continuous", None)
        if acp is None or msg.reservation is None or msg.pair is None:
            return
        reservation = msg.reservation
        pair = msg.pair
        reverse_pair = (pair[1], pair[0])
        left_node = self.owner.timeline.get_entity_by_name(src)
        answer = (
            left_node is not None
            and reverse_pair in acp.generated_entanglement_pairs
            and self._candidate_valid(pair, left_node, self.owner, reservation)
        )
        if answer:
            right_target = self.owner.components[self.owner.memo_arr_name][msg.right_target_index]
            answer = self._target_cache_adoptable(self.owner, right_target, reservation)
        if not answer:
            acp.counters["cache_coordination_requests_rejected"] += 1
        acp.lifecycle_events.append({
            "event": "cache_remote_checked",
            "time_ps": self.owner.timeline.now(),
            "reservation": reservation.identity,
            "pair": reverse_pair,
            "answer": answer,
            "from": src,
        })
        response = ACPMessage(
            ACPMsgType.CACHE_RESPONSE,
            reservation,
            answer=answer,
            pair=pair,
            left_target_index=msg.left_target_index,
            right_target_index=msg.right_target_index,
        )
        self.owner.send_message(src, response, priority=0, sender_delay=self._cache_endpoint_processing_delay())
        acp.counters["cache_coordination_responses_sent"] += 1

    def handle_cache_response(self, src: str, msg: ACPMessage) -> None:
        acp = getattr(self.owner, "adaptive_continuous", None)
        if acp is None or msg.reservation is None or msg.pair is None:
            return
        acp.counters["cache_coordination_responses_received"] += 1
        acp.lifecycle_events.append({
            "event": "cache_remote_confirmed",
            "time_ps": self.owner.timeline.now(),
            "reservation": msg.reservation.identity,
            "pair": msg.pair,
            "answer": bool(msg.answer),
            "from": src,
        })
        if not msg.answer:
            self._resume_generation_after_cache_response(src, msg)
            return
        self.owner.timeline.schedule(Event(
            self.owner.timeline.now() + self._cache_endpoint_processing_delay(),
            Process(self, "complete_cache_adoption", [src, msg]),
            -10,
        ))

    def complete_cache_adoption(self, right_name: str, msg: ACPMessage) -> int:
        reservation = msg.reservation
        pair = msg.pair
        if reservation is None or pair is None:
            return 0
        left_node = self.owner
        right_node = self.owner.timeline.get_entity_by_name(right_name)
        if right_node is None:
            self._resume_generation_after_cache_response(right_name, msg)
            return 0
        left_acp = left_node.adaptive_continuous
        right_acp = getattr(right_node, "adaptive_continuous", None)
        right_pair = (pair[1], pair[0])
        if pair not in left_acp.generated_entanglement_pairs:
            self._resume_generation_after_cache_response(right_name, msg)
            return 0
        if right_acp is None or right_pair not in right_acp.generated_entanglement_pairs:
            self._resume_generation_after_cache_response(right_name, msg)
            return 0
        if not self._candidate_valid(pair, left_node, right_node, reservation):
            self._resume_generation_after_cache_response(right_name, msg)
            return 0
        if self._cache_edge_satisfied(reservation, left_node.name, right_node.name):
            self._clear_cache_edge_pending(reservation, left_node.name, right_node.name)
            return 0
        left_bg = left_node.timeline.get_entity_by_name(pair[0][1])
        right_bg = right_node.timeline.get_entity_by_name(pair[1][1])
        left_target = left_node.components[left_node.memo_arr_name][msg.left_target_index]
        right_target = right_node.components[right_node.memo_arr_name][msg.right_target_index]
        if (
            not self._target_cache_adoptable(left_node, left_target, reservation)
            or not self._target_cache_adoptable(right_node, right_target, reservation)
        ):
            self._resume_generation_after_cache_response(right_name, msg)
            return 0

        left_meta = dict(left_acp.generated_pair_metadata.get(pair, {}))
        if left_target is not left_bg:
            left_node.resource_manager.memory_manager.swap_two_memory(left_target.name, left_bg.name)
        if right_target is not right_bg:
            right_node.resource_manager.memory_manager.swap_two_memory(right_target.name, right_bg.name)
        self._mark_adopted(left_node, left_target, right_node.name, right_target.name, reservation, left_meta)
        self._mark_adopted(right_node, right_target, left_node.name, left_target.name, reservation, left_meta)
        left_node.resource_manager.cache_satisfied_reservations.add(id(reservation))
        right_node.resource_manager.cache_satisfied_reservations.add(id(reservation))
        left_node.resource_manager._mark_cache_edge_satisfied(reservation, left_node.name, right_node.name)
        right_node.resource_manager._mark_cache_edge_satisfied(reservation, left_node.name, right_node.name)
        left_node.resource_manager._clear_cache_edge_pending(reservation, left_node.name, right_node.name)
        retired_protocols = self._retire_application_generation_protocols(
            left_node,
            left_target,
            right_node,
            right_target,
            reservation,
        )
        left_acp.lifecycle_events.append({
            "event": "cache_ownership_transferred",
            "time_ps": left_node.timeline.now(),
            "reservation": reservation.identity,
            "pair": pair,
            "app_pair": ((left_node.name, left_target.name), (right_node.name, right_target.name)),
        })
        if len(getattr(reservation, "path", [])) <= 2:
            left_node.get_idle_memory(left_node.resource_manager.memory_manager.get_info_by_memory(left_target))
            right_node.get_idle_memory(right_node.resource_manager.memory_manager.get_info_by_memory(right_target))
        elif retired_protocols:
            left_node.resource_manager._activate_adopted_application_memory(left_target)
            right_node.resource_manager._activate_adopted_application_memory(right_target)
        left_node.resource_manager.update(None, left_bg, MemoryInfo.RAW)
        right_node.resource_manager.update(None, right_bg, MemoryInfo.RAW)
        left_acp.remove_entanglement_pair(pair, reason="application")
        left_acp.counters["adaptive_memory_slots_recycled_application"] += 1
        left_acp.counters["cache_hits"] += 1
        left_acp.counters["background_pairs_reused"] += 1
        if right_acp is not None:
            right_acp.remove_entanglement_pair(right_pair, reason="application")
            right_acp.counters["adaptive_memory_slots_recycled_application"] += 1
            right_acp.counters["background_pairs_reused"] += 1
        one_way_delay = int(left_node.cchannels[right_node.name].delay)
        event = {
            "event": "background_pair_adopted_by_application",
            "time_ps": left_node.timeline.now(),
            "reservation": reservation.identity,
            "pair": pair,
            "app_pair": ((left_node.name, left_target.name), (right_node.name, right_target.name)),
            "classical_one_way_delay_ps": one_way_delay,
            "endpoint_processing_delay_ps": self._cache_endpoint_processing_delay(),
            "request_start_ps": reservation.start_time,
            "recorded_tts_ps": left_node.timeline.now() - reservation.start_time,
        }
        left_acp.lifecycle_events.append(event)
        if right_acp is not None:
            right_acp.lifecycle_events.append(dict(event, node=right_node.name))
        return 1

    def _resume_generation_after_cache_response(self, right_name: str, msg: ACPMessage) -> None:
        reservation = msg.reservation
        if reservation is None or msg.left_target_index is None:
            return
        self._clear_cache_edge_pending(reservation, self.owner.name, right_name)
        try:
            left_target = self.owner.components[self.owner.memo_arr_name][msg.left_target_index]
        except (KeyError, IndexError, TypeError):
            return
        for protocol in list(getattr(left_target, "_observers", [])):
            if not isinstance(protocol, EntanglementGenerationA):
                continue
            if getattr(getattr(protocol, "rule", None), "reservation", None) != reservation:
                continue
            if protocol not in self.owner.protocols:
                continue
            acp = getattr(self.owner, "adaptive_continuous", None)
            if acp is not None:
                acp.counters["cache_protocol_boundary_fallbacks"] += 1
                acp.lifecycle_events.append({
                    "event": "cache_protocol_boundary_fallback",
                    "time_ps": self.owner.timeline.now(),
                    "reservation": reservation.identity,
                    "left": self.owner.name,
                    "right": right_name,
                    "pair": msg.pair,
                    "answer": bool(msg.answer),
                })
            self.owner.timeline.schedule(Event(
                self.owner.timeline.now(),
                Process(protocol, "start", []),
                self.owner.timeline.schedule_counter,
            ))
            return

    def _cache_edge_key(self, reservation: Reservation, left: str, right: str) -> tuple[int, tuple[str, str]]:
        return id(reservation), tuple(sorted((left, right)))

    def _cache_edge_satisfied(self, reservation: Reservation, left: str, right: str) -> bool:
        return self._cache_edge_key(reservation, left, right) in self.cache_satisfied_edges

    def _cache_edge_pending(self, reservation: Reservation, left: str, right: str) -> bool:
        return self._cache_edge_key(reservation, left, right) in self.cache_pending_edges

    def _mark_cache_edge_satisfied(self, reservation: Reservation, left: str, right: str) -> None:
        self.cache_satisfied_edges.add(self._cache_edge_key(reservation, left, right))

    def _clear_cache_edge_pending(self, reservation: Reservation, left: str, right: str) -> None:
        self.cache_pending_edges.discard(self._cache_edge_key(reservation, left, right))

    def _retire_application_generation_protocols(self, left_node, left_target, right_node, right_target, reservation) -> int:
        retired = 0
        retired += left_node.resource_manager._retire_generation_protocol_for_memory(left_target, reservation)
        retired += right_node.resource_manager._retire_generation_protocol_for_memory(right_target, reservation)
        if retired:
            left_node.adaptive_continuous.counters["cache_protocol_boundary_adoptions"] += 1
        return retired

    def _retire_generation_protocol_for_memory(self, memory, reservation) -> int:
        retired = 0
        for protocol in list(memory._observers):
            if not isinstance(protocol, EntanglementGenerationA):
                continue
            if getattr(getattr(protocol, "rule", None), "reservation", None) != reservation:
                continue
            for event in list(getattr(protocol, "scheduled_events", [])):
                if event.time >= self.owner.timeline.now():
                    self.owner.timeline.remove_event(event)
            if protocol.rule and protocol in protocol.rule.protocols:
                protocol.rule.protocols.remove(protocol)
            for collection in (self.owner.protocols, self.waiting_protocols, self.pending_protocols):
                if protocol in collection:
                    collection.remove(protocol)
            memory.detach(protocol)
            retired += 1
        if retired:
            memory.attach(memory.memory_array)
        return retired

    def _activate_adopted_application_memory(self, memory) -> None:
        memo_info = self.memory_manager.get_info_by_memory(memory)
        for rule in self.rule_manager:
            memories_info = rule.is_valid(memo_info)
            if len(memories_info) > 0:
                rule.do(memories_info)
                for info in memories_info:
                    info.to_occupied()
                return

    def _candidate_valid(self, pair, left_node, right_node, reservation) -> bool:
        left_memory = left_node.timeline.get_entity_by_name(pair[0][1])
        right_memory = right_node.timeline.get_entity_by_name(pair[1][1])
        left_info = left_node.resource_manager.memory_manager.get_info_by_memory(left_memory)
        right_info = right_node.resource_manager.memory_manager.get_info_by_memory(right_memory)
        if left_info.state not in (MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED):
            return False
        if right_info.state not in (MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED):
            return False
        if left_info.remote_node != right_node.name or right_info.remote_node != left_node.name:
            return False
        return min(left_info.fidelity, right_info.fidelity) >= reservation.fidelity

    def _target_cache_adoptable(self, node, memory, reservation) -> bool:
        info = node.resource_manager.memory_manager.get_info_by_memory(memory)
        if info.state == MemoryInfo.RAW:
            return True
        if info.state != MemoryInfo.OCCUPIED:
            return False
        for protocol in getattr(memory, "_observers", []):
            if not isinstance(protocol, EntanglementGenerationA):
                continue
            if getattr(getattr(protocol, "rule", None), "reservation", None) == reservation:
                return True
        return False

    def _mark_adopted(self, node, memory, remote_node, remote_memory, reservation, metadata) -> None:
        memory.entangled_memory["node_id"] = remote_node
        memory.entangled_memory["memo_id"] = remote_memory
        memory.qdc_generation_source = "background"
        memory.qdc_generation_time_ps = metadata.get("generation_time_ps", node.timeline.now())
        memory.qdc_elementary_sources = [{
            "source": "background",
            "link": tuple(sorted((node.name, remote_node))),
            "generation_time_ps": memory.qdc_generation_time_ps,
        }]
        memory.qdc_application_reservation = str(reservation)
        memory.qdc_application_reservation_object = reservation
        info = node.resource_manager.memory_manager.get_info_by_memory(memory)
        info.to_entangled()

    def update(self, protocol, memory, state: str) -> None:
        if (
            state == MemoryInfo.ENTANGLED
            and isinstance(protocol, EntanglementGenerationA)
            and protocol.primary
        ):
            record_generation_success(
                self.owner,
                protocol.remote_node_name,
                protocol.middle,
            )
        if isinstance(protocol, EntanglementSwappingA) and state == MemoryInfo.RAW:
            self._propagate_swapping_provenance(protocol)
        if state == MemoryInfo.RAW:
            self._release_adopted_cache_edge(memory)
        self.memory_manager.update(memory, state)
        if isinstance(protocol, BBPSSWProtocol) and state == MemoryInfo.RAW:
            self._handle_purification_update(protocol, memory, state)
        if state == MemoryInfo.RAW:
            self._clear_qdc_attrs(memory)
        else:
            self._mark_generation_source(protocol, memory, state)

        if protocol:
            memory.detach(protocol)
            memory.attach(memory.memory_array)
            if protocol.rule and protocol in protocol.rule.protocols:
                protocol.rule.protocols.remove(protocol)

        if protocol in self.owner.protocols:
            self.owner.protocols.remove(protocol)
        if protocol in self.waiting_protocols:
            self.waiting_protocols.remove(protocol)
        if protocol in self.pending_protocols:
            self.pending_protocols.remove(protocol)

        memo_info = self.memory_manager.get_info_by_memory(memory)
        for rule in self.rule_manager:
            memories_info = rule.is_valid(memo_info)
            if len(memories_info) > 0:
                rule.do(memories_info)
                for info in memories_info:
                    info.to_occupied()
                record_memory_occupancy(self.owner)
                return

        self.owner.get_idle_memory(memo_info)
        record_memory_occupancy(self.owner)

    def _mark_generation_source(self, protocol, memory, state: str) -> None:
        if isinstance(protocol, BBPSSWProtocol):
            self._handle_purification_update(protocol, memory, state)
            return
        if protocol is None or not isinstance(protocol, EntanglementGenerationA):
            return
        if state != MemoryInfo.ENTANGLED:
            return
        reservation = getattr(getattr(protocol, "rule", None), "reservation", None)
        if isinstance(reservation, AdaptiveReservation):
            acp = self.owner.adaptive_continuous
            pair = ((self.owner.name, memory.name), (memory.entangled_memory["node_id"], memory.entangled_memory["memo_id"]))
            memory.qdc_generation_source = "background"
            memory.qdc_generation_time_ps = self.owner.timeline.now()
            acp.add_generated_entanglement_pair(pair, reservation)
            acp.counters["background_generation_successes"] += 1
            self._maybe_start_background_purification(protocol, pair)
            return
        memory.qdc_generation_source = "application"
        memory.qdc_generation_time_ps = self.owner.timeline.now()
        memory.qdc_elementary_sources = [{
            "source": "application",
            "link": tuple(sorted((self.owner.name, memory.entangled_memory["node_id"]))),
            "generation_time_ps": memory.qdc_generation_time_ps,
        }]

    def _release_adopted_cache_edge(self, memory) -> None:
        reservation = getattr(memory, "qdc_application_reservation_object", None)
        remote_name = memory.entangled_memory.get("node_id")
        if reservation is None or remote_name is None:
            return
        self._clear_cache_edge_pending(reservation, self.owner.name, remote_name)
        self.cache_satisfied_edges.discard(
            self._cache_edge_key(reservation, self.owner.name, remote_name)
        )
        remote_node = self.owner.timeline.get_entity_by_name(remote_name)
        remote_manager = getattr(remote_node, "resource_manager", None)
        if isinstance(remote_manager, ACPResourceManager):
            remote_manager._clear_cache_edge_pending(reservation, self.owner.name, remote_name)
            remote_manager.cache_satisfied_edges.discard(
                remote_manager._cache_edge_key(reservation, self.owner.name, remote_name)
            )

    def _propagate_swapping_provenance(self, protocol: EntanglementSwappingA) -> None:
        if getattr(protocol, "qdc_provenance_propagated", False):
            return
        protocol.qdc_provenance_propagated = True
        sources = []
        for memory in (protocol.left_memo, protocol.right_memo):
            memory_sources = list(getattr(memory, "qdc_elementary_sources", ()))
            if memory_sources:
                sources.extend(dict(source) for source in memory_sources)
            else:
                sources.append({
                    "source": getattr(memory, "qdc_generation_source", "application"),
                    "link": tuple(sorted((self.owner.name, memory.entangled_memory["node_id"]))),
                })
        background_count = sum(1 for source in sources if source.get("source") == "background")
        if background_count == len(sources):
            generation_source = "background"
        elif background_count:
            generation_source = "mixed"
        else:
            generation_source = "application"
        for memory_name in (protocol.left_remote_memo, protocol.right_remote_memo):
            remote_memory = self.owner.timeline.get_entity_by_name(memory_name)
            if remote_memory is None:
                continue
            remote_memory.qdc_elementary_sources = [dict(source) for source in sources]
            remote_memory.qdc_generation_source = generation_source

    def _maybe_start_background_purification(self, protocol, pair: tuple) -> None:
        acp = self.owner.adaptive_continuous
        if not acp.purify or not getattr(protocol, "primary", False):
            return
        if not self._background_pair_valid(self.owner, pair):
            acp.remove_entanglement_pair(pair, reason="stale")
            acp.counters["purification_candidates_stale"] += 1
            return
        pair2 = acp.find_purification_partner(pair)
        remote_node = self.owner.timeline.get_entity_by_name(pair[1][0])
        remote_acp = getattr(remote_node, "adaptive_continuous", None) if remote_node is not None else None
        if remote_acp is None:
            return
        while pair2 is not None and not self._background_pair_valid(self.owner, pair2):
            acp.remove_entanglement_pair(pair2, reason="stale")
            acp.counters["purification_candidates_stale"] += 1
            pair2 = acp.find_purification_partner(pair)
        if pair2 is None:
            return
        reverse_pair = (pair[1], pair[0])
        reverse_pair2 = (pair2[1], pair2[0])
        if (
            reverse_pair not in remote_acp.generated_entanglement_pairs
            or reverse_pair2 not in remote_acp.generated_entanglement_pairs
            or not self._background_pair_valid(remote_node, reverse_pair)
            or not self._background_pair_valid(remote_node, reverse_pair2)
        ):
            return
        predicted_fidelity = self._predicted_purification_fidelity(pair, pair2)
        current_fidelity = acp.get_fidelity(pair)
        if predicted_fidelity <= current_fidelity:
            acp.counters["purification_skipped_non_improving"] += 1
            acp.lifecycle_events.append({
                "event": "purification_skipped_non_improving",
                "time_ps": self.owner.timeline.now(),
                "kept_pair": pair,
                "measured_pair": pair2,
                "current_fidelity": current_fidelity,
                "predicted_fidelity": predicted_fidelity,
            })
            return

        acp.remove_entanglement_pair(pair, reason="purification_input")
        acp.remove_entanglement_pair(pair2, reason="purification_input")
        remote_acp.remove_entanglement_pair(reverse_pair, reason="purification_input")
        remote_acp.remove_entanglement_pair(reverse_pair2, reason="purification_input")

        local_protocol = self._create_purification_protocol(self.owner, pair, pair2)
        remote_protocol = self._create_purification_protocol(remote_node, reverse_pair, reverse_pair2)
        local_protocol.set_others(remote_protocol.name, remote_node.name, [reverse_pair[0][1], reverse_pair2[0][1]])
        remote_protocol.set_others(local_protocol.name, self.owner.name, [pair[0][1], pair2[0][1]])
        local_protocol.rule = getattr(protocol, "rule", None)
        remote_protocol.rule = getattr(protocol, "rule", None)
        self.owner.protocols.append(local_protocol)
        remote_node.protocols.append(remote_protocol)
        acp.counters["purification_attempts"] += 1
        remote_acp.counters["purification_attempts"] += 1
        event = {
            "event": "purification_started",
            "time_ps": self.owner.timeline.now(),
            "kept_pair": pair,
            "measured_pair": pair2,
            "remote": remote_node.name,
        }
        acp.lifecycle_events.append(event)
        remote_acp.lifecycle_events.append(dict(event, node=remote_node.name))

        start_time = self.owner.timeline.now() + int(self.owner.cchannels[remote_node.name].delay)
        self.owner.timeline.schedule(Event(
            start_time,
            Process(self, "start_background_purification", [local_protocol, remote_protocol]),
            self.owner.timeline.schedule_counter,
        ))

    def _predicted_purification_fidelity(self, pair: tuple, pair2: tuple) -> float:
        """Return the official BDS BBPSSW output fidelity without mutating state."""
        kept_memory = self.owner.timeline.get_entity_by_name(pair[0][1])
        measured_memory = self.owner.timeline.get_entity_by_name(pair2[0][1])
        remote_node = self.owner.timeline.get_entity_by_name(pair[1][0])
        if kept_memory is None or measured_memory is None or remote_node is None:
            return 0.0
        preview = ACPBackgroundPurification(
            self.owner,
            "ACP.BBPSSW.preview",
            kept_memory,
            measured_memory,
        )
        preview.set_others(
            "ACP.BBPSSW.preview",
            remote_node.name,
            [pair[1][1], pair2[1][1]],
        )
        try:
            _, output_bds = preview.purification_res()
        except Exception:
            return 0.0
        return float(output_bds[0])

    def start_background_purification(self, local_protocol: BBPSSWProtocol, remote_protocol: BBPSSWProtocol) -> None:
        remote_node = self.owner.timeline.get_entity_by_name(local_protocol.remote_node_name)
        remote_acp = getattr(remote_node, "adaptive_continuous", None) if remote_node is not None else None
        if (
            remote_node is None
            or not self._purification_protocol_valid(local_protocol)
            or not remote_node.resource_manager._purification_protocol_valid(remote_protocol)
        ):
            self._abort_purification_protocol(local_protocol)
            if remote_node is not None:
                remote_node.resource_manager._abort_purification_protocol(remote_protocol)
            self.owner.adaptive_continuous.counters["purification_aborts"] += 1
            if remote_acp is not None:
                remote_acp.counters["purification_aborts"] += 1
            return
        remote_protocol.start()
        local_protocol.start()

    def _purification_protocol_valid(self, protocol: BBPSSWProtocol) -> bool:
        remote_nodes = set()
        memory_names = set()
        for memory in protocol.memories:
            if memory.name in memory_names:
                return False
            memory_names.add(memory.name)
            if memory.entangled_memory["node_id"] is None:
                return False
            info = self.memory_manager.get_info_by_memory(memory)
            if info.state != MemoryInfo.OCCUPIED:
                return False
            try:
                self.owner.timeline.quantum_manager.get(memory.qstate_key)
            except Exception:
                return False
            remote_nodes.add(memory.entangled_memory["node_id"])
        return len(remote_nodes) == 1

    def _abort_purification_protocol(self, protocol: BBPSSWProtocol) -> None:
        for memory in list(protocol.memories):
            info = self.memory_manager.get_info_by_memory(memory)
            if info.state != MemoryInfo.RAW:
                self.update(protocol, memory, MemoryInfo.RAW)
        if protocol in self.owner.protocols:
            self.owner.protocols.remove(protocol)

    def _background_pair_valid(self, node, pair: tuple) -> bool:
        memory = node.timeline.get_entity_by_name(pair[0][1])
        if memory is None or memory.entangled_memory["node_id"] != pair[1][0] or memory.entangled_memory["memo_id"] != pair[1][1]:
            return False
        info = node.resource_manager.memory_manager.get_info_by_memory(memory)
        return info.state in (MemoryInfo.ENTANGLED, MemoryInfo.PURIFIED)

    def _create_purification_protocol(self, node, pair: tuple, pair2: tuple):
        kept_memory = node.timeline.get_entity_by_name(pair[0][1])
        measured_memory = node.timeline.get_entity_by_name(pair2[0][1])
        protocol = ACPBackgroundPurification(node, f"ACP.BBPSSW.{kept_memory.name}.{measured_memory.name}", kept_memory, measured_memory)
        for memory in (kept_memory, measured_memory):
            memory.detach(memory.memory_array)
            memory.attach(protocol)
            node.resource_manager.memory_manager.get_info_by_memory(memory).to_occupied()
        return protocol

    def _handle_purification_update(self, protocol: BBPSSWProtocol, memory, state: str) -> None:
        acp = getattr(self.owner, "adaptive_continuous", None)
        if acp is None:
            return
        if state == MemoryInfo.RAW:
            acp.counters["adaptive_memory_slots_recycled_purification"] += 1
            if memory is protocol.kept_memo:
                acp.counters["purification_failures"] += 1
                acp.lifecycle_events.append({
                    "event": "purification_failed",
                    "time_ps": self.owner.timeline.now(),
                    "memory": memory.name,
                    "remote": protocol.remote_node_name,
                })
            return
        if state == MemoryInfo.PURIFIED and memory is protocol.kept_memo:
            pair = ((self.owner.name, memory.name), (memory.entangled_memory["node_id"], memory.entangled_memory["memo_id"]))
            memory.qdc_generation_source = "background_purified"
            memory.qdc_generation_time_ps = self.owner.timeline.now()
            acp.add_generated_entanglement_pair(pair, getattr(protocol, "rule", None))
            acp.counters["purification_successes"] += 1
            acp.lifecycle_events.append({
                "event": "purification_succeeded",
                "time_ps": self.owner.timeline.now(),
                "pair": pair,
                "remote": protocol.remote_node_name,
                "fidelity": self.memory_manager.get_info_by_memory(memory).fidelity,
            })

    def _clear_qdc_attrs(self, memory) -> None:
        for name in (
            "qdc_generation_source",
            "qdc_generation_time_ps",
            "qdc_elementary_sources",
            "qdc_application_reservation",
            "qdc_application_reservation_object",
        ):
            if hasattr(memory, name):
                delattr(memory, name)


class ACPRSVPProtocol(RSVPProtocol):
    def create_rules_adaptive(self, path: list[str], reservation: AdaptiveReservation) -> list[Rule]:
        memory_indices = [card.memory_index for card in self.timecards if reservation in card.reservations]
        index = path.index(self.owner.name)
        rules = []
        if index > 0:
            rules.extend(self._adaptive_generation_rules(
                path[index - 1],
                path,
                index,
                memory_indices[:reservation.memory_size],
                reservation,
                requester=False,
            ))
        if index < len(path) - 1:
            rules.extend(self._adaptive_generation_rules(
                path[index + 1],
                path,
                index,
                memory_indices[:reservation.memory_size],
                reservation,
                requester=True,
            ))
        for rule in rules:
            rule.set_reservation(reservation)
        return rules

    def _adaptive_generation_rules(
        self,
        neighbor,
        path,
        index,
        memory_indices,
        reservation,
        *,
        requester,
    ):
        lanes = middle_nodes(self.owner, neighbor)
        rules = []
        for lane_index, middle in enumerate(lanes):
            parallel = len(lanes) > 1
            action_args = {"mid": middle, "path": path, "index": index}
            if requester:
                action_args.update({
                    "name": self.owner.name,
                    "reservation": reservation,
                    "parallel_lane": parallel,
                })
                action = eg_rule_action_request_adaptive
            else:
                action = eg_rule_action_await_adaptive
            condition_args = {"memory_indices": memory_indices}
            if parallel:
                condition_args.update({
                    "lane_index": lane_index,
                    "lane_count": len(lanes),
                })
            rules.append(Rule(
                20 if not requester else 10,
                action,
                parallel_eg_rule_condition if parallel else eg_rule_condition,
                action_args,
                condition_args,
            ))
        return rules

    def load_rules_adaptive(self, rules: list[Rule], reservation: AdaptiveReservation) -> None:
        self.accepted_reservations.append(reservation)
        acp = self.owner.adaptive_continuous
        for card in self.timecards:
            if reservation in card.reservations:
                memory = self.memo_arr[card.memory_index]
                acp.adaptive_memory_names.add(memory.name)
                self.owner.timeline.schedule(Event(reservation.end_time, Process(self.owner.resource_manager, "update", [None, memory, MemoryInfo.RAW]), self.owner.timeline.schedule_counter))

                self.owner.timeline.schedule(Event(reservation.end_time, Process(acp, "adaptive_memory_used_minus_one", [memory]), self.owner.timeline.schedule_counter))
        for rule in rules:
            self.owner.timeline.schedule(Event(reservation.start_time, Process(self.owner.resource_manager, "load", [rule]), self.owner.timeline.schedule_counter))
            self.owner.timeline.schedule(Event(reservation.end_time, Process(self.owner.resource_manager, "expire", [rule]), self.owner.timeline.schedule_counter))


class ACPQuantumRouter(QuantumRouter):
    def __init__(self, name, tl, memo_size=50, seed=None, component_templates=None, gate_fid=1, meas_fid=1):
        component_templates = component_templates or {}
        super().__init__(name, tl, memo_size, seed, component_templates, gate_fid, meas_fid)
        self.resource_manager = ACPResourceManager(self, self.memo_arr_name)
        ensure_parallel_metrics(self)
        old_rsvp = self.network_manager.rsvp
        rsvp = ACPRSVPProtocol(self, f"{self.name}.RSVP", self.memo_arr_name)
        rsvp.timecards = self.network_manager.timecards
        rsvp.lower_protocols.append(self.network_manager.forward)
        rsvp.upper_protocols.append(self.network_manager)
        self.network_manager.forward.upper_protocols = [
            rsvp if protocol is old_rsvp else protocol
            for protocol in self.network_manager.forward.upper_protocols
        ]
        self.network_manager.protocol_stack[-1] = rsvp

        self.adaptive_continuous = AdaptiveContinuousProtocol(
            self,
            adaptive_max_memory=int(component_templates.get("adaptive_max_memory", 0)),
            period_ps=int(component_templates.get("acp_period_ps", 100_000_000_000)),
            strategy=component_templates.get("acp_strategy", "freshest"),
            delta=float(component_templates.get("acp_delta", 0.05)),
            update_prob=bool(component_templates.get("acp_update_prob", True)),
            background_enabled=bool(component_templates.get("acp_background_enabled", True)),
            purify=bool(component_templates.get("acp_purify", False)),
            cache_endpoint_processing_delay_ps=int(component_templates.get("acp_cache_endpoint_processing_delay_ps", 0)),
            execution_profile=component_templates.get("acp_execution_profile", "asynchronous"),
        )

    def init(self):
        super().init()
        self.adaptive_continuous.init()

    def receive_message(self, src: str, msg) -> None:
        if msg.receiver == "adaptive_continuous":
            self.adaptive_continuous.received_message(src, msg)
            return
        super().receive_message(src, msg)


class ACPRouterNetTopo(RouterNetTopo):
    def __init__(self, config_source: str | dict, acp_options: dict[str, Any] | None = None):
        self.acp_options = acp_options or {}
        super().__init__(config_source)

    def record_served_path(self, path: tuple[str, ...], timestamp: int) -> None:
        """Forward successful application-path feedback to each ACP node."""
        for node_name in path:
            node = self.tl.get_entity_by_name(node_name)
            node.adaptive_continuous.record_served_path(list(path), timestamp)

    def _add_nodes(self, config: dict):
        for node in config[Topo.ALL_NODE]:
            seed = node[Topo.SEED]
            node_type = node[Topo.TYPE]
            name = node[Topo.NAME]
            template_name = node.get(Topo.TEMPLATE, None)
            template = dict(self.templates.get(template_name, {}))
            template.update(self.acp_options)
            if node_type == self.BSM_NODE:
                node_obj = BSMNode(name, self.tl, self.bsm_to_router_map[name], component_templates=template)
            elif node_type == self.QUANTUM_ROUTER:
                node_obj = ACPQuantumRouter(
                    name,
                    self.tl,
                    node.get(self.MEMO_ARRAY_SIZE, 0),
                    seed,
                    template,
                    node.get("gate_fidelity", 1),
                    node.get("measurement_fidelity", 1),
                )
            else:
                raise ValueError(f"Unknown type of node '{node_type}'")
            node_obj.set_seed(seed)
            self.nodes[node_type].append(node_obj)

    def _add_bsm_node_to_router(self) -> None:
        super()._add_bsm_node_to_router()
        configure_parallel_middle_nodes(self)
