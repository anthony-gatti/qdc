"""Minimal paper-faithful Adaptive Continuous Protocol for SeQUeNCe v1.0.0."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from enum import Enum, auto
from itertools import accumulate
from typing import Optional

from sequence.constants import EPSILON
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.message import Message
from sequence.network_management.reservation import Reservation
from sequence.protocol import Protocol


class ACPMsgType(Enum):
    REQUEST = auto()
    RESPOND = auto()
    PATH_FEEDBACK = auto()


class ACPMessage(Message):
    def __init__(self, msg_type: ACPMsgType, reservation: Reservation | None = None, **kwargs):
        super().__init__(msg_type, receiver="adaptive_continuous")
        self.reservation = reservation
        self.answer = kwargs.get("answer")
        self.path = kwargs.get("path")
        self.timestamp = kwargs.get("timestamp")


class AdaptiveReservation(Reservation):
    """Reservation marker for ACP-owned neighbor-link background memory."""

    def __str__(self) -> str:
        return (
            f"|ACP; initiator={self.initiator}; responder={self.responder}; "
            f"path={self.path}; start_time={self.start_time:,}; end_time={self.end_time:,}; "
            f"memory_size={self.memory_size}; fidelity={self.fidelity}|"
        )


class AdaptiveContinuousProtocol(Protocol):
    """Distributed ACP state machine attached to one quantum router."""

    def __init__(
        self,
        owner,
        adaptive_max_memory: int,
        period_ps: int,
        strategy: str = "freshest",
        delta: float = 0.05,
        update_prob: bool = True,
        background_enabled: bool = True,
    ):
        super().__init__(owner, "adaptive_continuous")
        self.adaptive_max_memory = adaptive_max_memory
        self.period_ps = period_ps
        self.strategy = strategy
        self.delta = delta
        self.update_prob = update_prob
        self.background_enabled = background_enabled
        self.has_empty_neighbor = True
        self.probability_table: dict[Optional[str], float] = {}
        self.adaptive_memory_used = 0
        self.adaptive_memory_names: set[str] = set()
        self.generated_entanglement_pairs: set[tuple] = set()
        self.generated_pair_metadata: dict[tuple, dict] = {}
        self.path_feedback: list[tuple[int, list[str]]] = []
        self.counters = Counter()
        self.lifecycle_events: list[dict] = []

    def init(self) -> None:
        self.init_probability_table()
        self._schedule_start(0)

    def init_probability_table(self) -> None:
        neighbors = sorted(
            dst for dst, next_hop in self.owner.network_manager.get_forwarding_table().items()
            if dst == next_hop
        )
        keys: list[Optional[str]] = list(neighbors)
        if self.has_empty_neighbor:
            keys.append(None)
        if not keys:
            self.probability_table = {None: 1.0}
            return
        self.probability_table = {key: 1 / len(keys) for key in keys}
        assert abs(sum(self.probability_table.values()) - 1) < EPSILON

    def start(self) -> None:
        self.counters["start_invocations"] += 1
        if not self.background_enabled or self.adaptive_max_memory <= 0:
            return
        if self.adaptive_memory_used >= self.adaptive_max_memory:
            self.counters["start_blocked_memory_cap"] += 1
            self._schedule_start(self.period_ps // 1000)
            return
        neighbor = self.select_neighbor()
        if neighbor is None:
            self.counters["start_selected_none"] += 1
            self._schedule_start(self.period_ps // 100)
            return

        self.adaptive_memory_used += 1
        self._record_memory_high_watermark()
        self.counters["ac_request_sent"] += 1
        cc_delay = int(self.owner.cchannels[neighbor].delay)
        start_time = self.owner.timeline.now() + 2 * cc_delay
        end_time = start_time + self.period_ps
        reservation = AdaptiveReservation(self.owner.name, neighbor, start_time, end_time, 1, 0.5)
        if self.owner.network_manager.rsvp.schedule(reservation):
            self.owner.send_message(neighbor, ACPMessage(ACPMsgType.REQUEST, reservation))
        else:
            self.adaptive_memory_used -= 1
            self.counters["local_schedule_failed"] += 1
            self._remove_from_timecards(reservation)
            self._schedule_start(self.period_ps // 1000)

    def select_neighbor(self) -> Optional[str]:
        keys = []
        probs = []
        for key, prob in sorted(self.probability_table.items(), key=lambda item: "" if item[0] is None else item[0]):
            keys.append(key)
            probs.append(prob)
        index = bisect_left(list(accumulate(probs)), self.owner.get_generator().random())
        return keys[min(index, len(keys) - 1)]

    def received_message(self, src: str, msg: ACPMessage) -> None:
        if msg.msg_type is ACPMsgType.REQUEST:
            self._handle_request(src, msg)
        elif msg.msg_type is ACPMsgType.RESPOND:
            self._handle_response(src, msg)
        elif msg.msg_type is ACPMsgType.PATH_FEEDBACK and msg.path:
            self.record_served_path(msg.path, msg.timestamp or self.owner.timeline.now())

    def _handle_request(self, src: str, msg: ACPMessage) -> None:
        self.counters["ac_request_received"] += 1
        reservation = msg.reservation
        if reservation is None or self.adaptive_memory_used >= self.adaptive_max_memory:
            self.owner.send_message(src, ACPMessage(ACPMsgType.RESPOND, reservation, answer=False))
            return
        if self.owner.network_manager.rsvp.schedule(reservation):
            self.adaptive_memory_used += 1
            self._record_memory_high_watermark()
            path = [src, self.owner.name]
            reservation.set_path(path)
            rules = self.owner.network_manager.rsvp.create_rules_adaptive(path, reservation)
            self.owner.network_manager.rsvp.load_rules_adaptive(rules, reservation)
            self.owner.send_message(src, ACPMessage(ACPMsgType.RESPOND, reservation, answer=True, path=path))
        else:
            self.owner.send_message(src, ACPMessage(ACPMsgType.RESPOND, reservation, answer=False))

    def _handle_response(self, src: str, msg: ACPMessage) -> None:
        self.counters["ac_respond_received"] += 1
        reservation = msg.reservation
        if reservation is None:
            self._schedule_start(self.period_ps // 1000)
            return
        if not msg.answer:
            self.adaptive_memory_used = max(0, self.adaptive_memory_used - 1)
            self._remove_from_timecards(reservation)
            self.counters["remote_schedule_failed"] += 1
        else:
            reservation.set_path(msg.path)
            rules = self.owner.network_manager.rsvp.create_rules_adaptive(msg.path, reservation)
            self.owner.network_manager.rsvp.load_rules_adaptive(rules, reservation)
        self._schedule_start(3 * max(1, self.period_ps // 1000))

    def _remove_from_timecards(self, reservation: Reservation) -> None:
        for card in self.owner.network_manager.get_timecards():
            card.remove(reservation)

    def _schedule_start(self, delay: int) -> None:
        if not self.background_enabled or self.adaptive_max_memory <= 0:
            return
        delay = max(0, delay)
        random_delay = int(self.owner.get_generator().uniform(0, delay)) if delay else 0
        self.owner.timeline.schedule(Event(self.owner.timeline.now() + random_delay, Process(self, "start", [])))
        self.counters["start_events_scheduled"] += 1

    def add_generated_entanglement_pair(self, pair: tuple, reservation: Reservation | None = None) -> None:
        if pair in self.generated_entanglement_pairs:
            return
        now = self.owner.timeline.now()
        self.generated_entanglement_pairs.add(pair)
        metadata = {
            "generation_time_ps": now,
            "adaptive_reservation": str(reservation or ""),
            "link": tuple(sorted((pair[0][0], pair[1][0]))),
        }
        self.generated_pair_metadata[pair] = metadata
        self.generated_pair_metadata[(pair[1], pair[0])] = metadata
        self.counters["background_endpoint_records"] += 1
        self.lifecycle_events.append({"event": "background_pair_available", "time_ps": now, "pair": pair})

    def remove_entanglement_pair(self, pair: tuple, reason: str = "other") -> None:
        reverse = (pair[1], pair[0])
        removed = None
        if pair in self.generated_entanglement_pairs:
            removed = pair
        elif reverse in self.generated_entanglement_pairs:
            removed = reverse
        if removed is None:
            return
        self.generated_entanglement_pairs.remove(removed)
        self.generated_pair_metadata.pop(removed, None)
        self.generated_pair_metadata.pop((removed[1], removed[0]), None)
        self.counters[f"background_pairs_removed_{reason}"] += 1
        self.lifecycle_events.append({"event": "background_pair_removed", "time_ps": self.owner.timeline.now(), "pair": removed, "reason": reason})

    def match_generated_entanglement_pair(self, this_node: str, remote_node: str) -> tuple | None:
        candidates = [
            pair for pair in self.generated_entanglement_pairs
            if pair[0][0] == this_node and pair[1][0] == remote_node
        ]
        self.counters["cache_checks"] += 1
        if not candidates:
            self.counters["cache_misses"] += 1
            return None
        self.counters["cache_hits"] += 1
        if self.strategy == "random":
            index = int(self.owner.get_generator().integers(0, len(candidates)))
            return sorted(candidates)[index]
        return max(candidates, key=self.get_fidelity)

    def get_fidelity(self, pair: tuple) -> float:
        memory = self.owner.timeline.get_entity_by_name(pair[0][1])
        try:
            memory.bds_decohere()
            return memory.get_bds_fidelity()
        except Exception:
            return getattr(memory, "fidelity", 0.0)

    def adaptive_memory_used_minus_one(self, memory) -> None:
        if memory.name not in self.adaptive_memory_names:
            return
        self.adaptive_memory_names.remove(memory.name)
        self.adaptive_memory_used = max(0, self.adaptive_memory_used - 1)
        for pair in list(self.generated_entanglement_pairs):
            if pair[0][1] == memory.name or pair[1][1] == memory.name:
                self.remove_entanglement_pair(pair, reason="expired")
        self.counters["adaptive_memory_released"] += 1

    def _record_memory_high_watermark(self) -> None:
        self.counters["adaptive_memory_high_watermark"] = max(
            self.counters["adaptive_memory_high_watermark"],
            self.adaptive_memory_used,
        )

    def record_served_path(self, path: list[str], timestamp: int | None = None) -> None:
        timestamp = self.owner.timeline.now() if timestamp is None else timestamp
        self.path_feedback.append((timestamp, list(path)))
        if not self.update_prob:
            return
        this = self.owner.name
        if this not in path:
            return
        index = path.index(this)
        neighbors = set()
        if index > 0:
            neighbors.add(path[index - 1])
        if index < len(path) - 1:
            neighbors.add(path[index + 1])
        updated = False
        for neighbor in list(self.probability_table):
            if neighbor is not None and neighbor in neighbors:
                self.probability_table[neighbor] += self.delta
                updated = True
        if not updated and None in self.probability_table:
            self.probability_table[None] += self.delta
        total = sum(self.probability_table.values())
        for neighbor in list(self.probability_table):
            self.probability_table[neighbor] /= total
        self.counters["probability_updates"] += 1

    def send_path_feedback(self, node: str, path: list[str], timestamp: int) -> None:
        self.owner.send_message(node, ACPMessage(ACPMsgType.PATH_FEEDBACK, path=path, timestamp=timestamp))
