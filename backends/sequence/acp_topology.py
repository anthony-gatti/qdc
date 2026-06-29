"""ACP-aware SeQUeNCe topology/node integration."""

from __future__ import annotations

from typing import Any

from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.network_management.rsvp import RSVPProtocol
from sequence.network_management.reservation import Reservation
from sequence.resource_management.action_condition_set import (
    eg_match_func,
    eg_rule_action_await,
    eg_rule_action_request,
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


def eg_rule_action_await_adaptive(memories_info: list[MemoryInfo], args: Arguments):
    return eg_rule_action_await(memories_info, args)


def eg_rule_action_request_adaptive(memories_info: list[MemoryInfo], args: Arguments):
    return eg_rule_action_request(memories_info, args)


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


class ACPResourceManager(ResourceManager):
    """Resource manager extension for ACP inventory and cache adoption."""

    def __init__(self, owner: QuantumRouter, memory_array_name: str):
        super().__init__(owner, memory_array_name)
        self.memory_manager = ACPMemoryManager(owner.components[memory_array_name])
        self.memory_manager.set_resource_manager(self)
        self.cache_satisfied_reservations: set[int] = set()

    def generate_load_rules(self, path: list[str], reservation: Reservation, timecards: list, memory_array_name: str):
        if isinstance(reservation, AdaptiveReservation):
            return super().generate_load_rules(path, reservation, timecards, memory_array_name)

        activation_time = reservation.start_time
        if getattr(self.owner, "adaptive_continuous", None) is not None:
            self.owner.timeline.schedule(Event(
                activation_time,
                Process(self, "initiate_cached_pairs_for_reservation", [path, reservation, timecards, memory_array_name]),
                -10,
            ))
            activation_time += self._cache_coordination_delay(path)
        self._generate_no_purification_rules(path, reservation, timecards, memory_array_name, activation_time)

    def _cache_coordination_delay(self, path: list[str]) -> int:
        delay = 0
        for left, right in zip(path, path[1:]):
            if left == self.owner.name and right in self.owner.cchannels:
                delay = max(delay, int(self.owner.cchannels[right].delay))
            elif right == self.owner.name and left in self.owner.cchannels:
                delay = max(delay, int(self.owner.cchannels[left].delay))
        if delay <= 0:
            return 0
        return 2 * delay + 2 * self._cache_endpoint_processing_delay()

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
            rules.append(Rule(
                10,
                eg_rule_action_await,
                eg_rule_condition,
                {"mid": self.owner.map_to_middle_node[path[index - 1]], "path": path, "index": index},
                {"memory_indices": memory_indices[:reservation.memory_size]},
            ))
        if index < len(path) - 1:
            selected = memory_indices[:reservation.memory_size] if index == 0 else memory_indices[reservation.memory_size:]
            rules.append(Rule(
                10,
                eg_rule_action_request,
                eg_rule_condition,
                {
                    "mid": self.owner.map_to_middle_node[path[index + 1]],
                    "path": path,
                    "index": index,
                    "name": self.owner.name,
                    "reservation": reservation,
                },
                {"memory_indices": selected},
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

    def load_application_rule(self, rule: Rule, reservation: Reservation) -> None:
        if id(reservation) in self.cache_satisfied_reservations:
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
        if not self._target_raw(self.owner, left_target):
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
            ordered = sorted(candidates, key=left_acp.get_fidelity, reverse=True)

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
            answer = self._target_raw(self.owner, right_target)
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
            return 0
        left_acp = left_node.adaptive_continuous
        right_acp = getattr(right_node, "adaptive_continuous", None)
        right_pair = (pair[1], pair[0])
        if pair not in left_acp.generated_entanglement_pairs:
            return 0
        if right_acp is None or right_pair not in right_acp.generated_entanglement_pairs:
            return 0
        if not self._candidate_valid(pair, left_node, right_node, reservation):
            return 0
        left_bg = left_node.timeline.get_entity_by_name(pair[0][1])
        right_bg = right_node.timeline.get_entity_by_name(pair[1][1])
        left_target = left_node.components[left_node.memo_arr_name][msg.left_target_index]
        right_target = right_node.components[right_node.memo_arr_name][msg.right_target_index]
        if not self._target_raw(left_node, left_target) or not self._target_raw(right_node, right_target):
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
        left_acp.lifecycle_events.append({
            "event": "cache_ownership_transferred",
            "time_ps": left_node.timeline.now(),
            "reservation": reservation.identity,
            "pair": pair,
            "app_pair": ((left_node.name, left_target.name), (right_node.name, right_target.name)),
        })
        left_node.get_idle_memory(left_node.resource_manager.memory_manager.get_info_by_memory(left_target))
        right_node.get_idle_memory(right_node.resource_manager.memory_manager.get_info_by_memory(right_target))
        left_node.resource_manager.update(None, left_bg, MemoryInfo.RAW)
        right_node.resource_manager.update(None, right_bg, MemoryInfo.RAW)
        left_acp.remove_entanglement_pair(pair, reason="application")
        left_acp.adaptive_memory_used_minus_one(left_bg)
        left_acp.counters["cache_hits"] += 1
        left_acp.counters["background_pairs_reused"] += 1
        if right_acp is not None:
            right_acp.remove_entanglement_pair(right_pair, reason="application")
            right_acp.adaptive_memory_used_minus_one(right_bg)
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

    def _target_raw(self, node, memory) -> bool:
        return node.resource_manager.memory_manager.get_info_by_memory(memory).state == MemoryInfo.RAW

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
        info = node.resource_manager.memory_manager.get_info_by_memory(memory)
        info.to_entangled()

    def update(self, protocol, memory, state: str) -> None:
        self.memory_manager.update(memory, state)
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
                return

        self.owner.get_idle_memory(memo_info)

    def _mark_generation_source(self, protocol, memory, state: str) -> None:
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
            return
        memory.qdc_generation_source = "application"
        memory.qdc_generation_time_ps = self.owner.timeline.now()
        memory.qdc_elementary_sources = [{
            "source": "application",
            "link": tuple(sorted((self.owner.name, memory.entangled_memory["node_id"]))),
            "generation_time_ps": memory.qdc_generation_time_ps,
        }]

    def _clear_qdc_attrs(self, memory) -> None:
        for name in ("qdc_generation_source", "qdc_generation_time_ps", "qdc_elementary_sources", "qdc_application_reservation"):
            if hasattr(memory, name):
                delattr(memory, name)


class ACPRSVPProtocol(RSVPProtocol):
    def create_rules_adaptive(self, path: list[str], reservation: AdaptiveReservation) -> list[Rule]:
        memory_indices = [card.memory_index for card in self.timecards if reservation in card.reservations]
        index = path.index(self.owner.name)
        rules = []
        if index > 0:
            rules.append(Rule(
                20,
                eg_rule_action_await_adaptive,
                eg_rule_condition,
                {"mid": self.owner.map_to_middle_node[path[index - 1]], "path": path, "index": index},
                {"memory_indices": memory_indices[:reservation.memory_size]},
            ))
        if index < len(path) - 1:
            rules.append(Rule(
                10,
                eg_rule_action_request_adaptive,
                eg_rule_condition,
                {
                    "mid": self.owner.map_to_middle_node[path[index + 1]],
                    "path": path,
                    "index": index,
                    "name": self.owner.name,
                    "reservation": reservation,
                },
                {"memory_indices": memory_indices[:reservation.memory_size]},
            ))
        for rule in rules:
            rule.set_reservation(reservation)
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
            cache_endpoint_processing_delay_ps=int(component_templates.get("acp_cache_endpoint_processing_delay_ps", 0)),
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
