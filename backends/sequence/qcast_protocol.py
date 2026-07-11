"""Classical control messages and reservation markers for Q-CAST slots."""

from __future__ import annotations

from enum import Enum, auto

from sequence.message import Message
from sequence.network_management.reservation import Reservation
from sequence.protocol import Protocol


class QCASTMessageType(Enum):
    PLAN = auto()
    LINK_STATE = auto()


class QCASTMessage(Message):
    def __init__(self, msg_type: QCASTMessageType, slot_id: int, payload=None):
        super().__init__(msg_type, receiver="qcast_control")
        self.slot_id = slot_id
        self.payload = payload


class QCASTControlProtocol(Protocol):
    """Per-router endpoint for modeled Q-CAST control-plane traffic."""

    def __init__(self, owner):
        super().__init__(owner, "qcast_control")
        self.scheduler = None
        self.plan_by_slot = {}
        self.link_state_by_slot = {}

    def configure(self, scheduler) -> None:
        self.scheduler = scheduler

    def received_message(self, src: str, msg: QCASTMessage) -> None:
        if msg.msg_type is QCASTMessageType.PLAN:
            self.plan_by_slot[msg.slot_id] = msg.payload
        elif msg.msg_type is QCASTMessageType.LINK_STATE:
            self.link_state_by_slot.setdefault(msg.slot_id, {})[src] = msg.payload
        if self.scheduler is not None:
            self.scheduler.received_control_message(self.owner.name, src, msg)


class QCASTReservation(Reservation):
    """Identity-based reservation used for one assigned elementary lane."""

    def __init__(self, *args, slot_id: int, lane_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_id = slot_id
        self.lane_id = lane_id

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QCASTReservation)
            and self.slot_id == other.slot_id
            and self.lane_id == other.lane_id
        )

    def __hash__(self) -> int:
        return hash((self.slot_id, self.lane_id))
