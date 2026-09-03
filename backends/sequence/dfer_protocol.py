"""Local classical control endpoint and reservation marker for DFER."""

from __future__ import annotations

from enum import Enum, auto

from sequence.message import Message
from sequence.network_management.reservation import Reservation
from sequence.protocol import Protocol


class DFERMessageType(Enum):
    STATE_QUERY = auto()
    STATE_RESPONSE = auto()


class DFERMessage(Message):
    def __init__(
        self,
        msg_type: DFERMessageType,
        reservation_id: int,
        epoch: int,
        payload=None,
    ):
        super().__init__(msg_type, receiver="dfer_control")
        self.reservation_id = reservation_id
        self.epoch = epoch
        self.payload = payload


class DFERControlProtocol(Protocol):
    """Per-router endpoint for DFER's one-hop asynchronous control traffic."""

    def __init__(self, owner):
        super().__init__(owner, "dfer_control")
        self.scheduler = None

    def configure(self, scheduler) -> None:
        self.scheduler = scheduler

    def begin_hop(self, reservation_id: int) -> None:
        if self.scheduler is not None:
            self.scheduler.begin_hop(self.owner.name, reservation_id)

    def record_elementary_success(
        self,
        operation_id: str,
        time_ps: int,
        fidelity: float,
    ) -> None:
        if self.scheduler is not None:
            self.scheduler.record_elementary_success(
                operation_id,
                time_ps,
                fidelity,
            )

    def received_message(self, src: str, msg: DFERMessage) -> bool:
        if self.scheduler is not None:
            self.scheduler.received_control_message(self.owner.name, src, msg)
        return True


class DFERReservation(Reservation):
    """Identity-based reservation for one DFER elementary-link operation."""

    def __init__(self, *args, operation_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_id = operation_id

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DFERReservation)
            and self.operation_id == other.operation_id
        )

    def __hash__(self) -> int:
        return hash(self.operation_id)
