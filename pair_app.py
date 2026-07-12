"""Application and result collection for single-pair paper workloads."""

from collections import defaultdict

from sequence.app.request_app import RequestApp
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.constants import MILLISECOND

from results import BackendResult, RequestResult


class PairRequestApp(RequestApp):
    """Record one-pair request latency while using the normal RSVP path."""

    def __init__(self, node, served_path_observer=None):
        super().__init__(node)
        self._served_path_observer = served_path_observer
        self.time_to_serve = {}
        self.entanglement_fidelities = defaultdict(list)
        self.reservation_outcomes = {}
        self._path_feedback_recorded = set()

    def start(self, responder, start_t, end_t, memo_size, fidelity,
              entanglement_number=1, identity=0):
        self.node.reserve_net_resource(
            responder, start_t, end_t, memo_size, fidelity,
            entanglement_number, identity,
        )

    def get_reservation_result(self, reservation, result):
        self.reservation_outcomes[reservation.identity] = bool(result)
        super().get_reservation_result(reservation, result)

    def get_memory(self, info):
        if info.state != MemoryInfo.ENTANGLED:
            return
        reservation = self.memo_to_reservation.get(info.index)
        if reservation is None or info.fidelity < reservation.fidelity:
            return

        if info.remote_node == reservation.responder:
            self.entanglement_fidelities[reservation].append(info.fidelity)
            self.time_to_serve[reservation] = (
                self.node.timeline.now() - reservation.start_time
            )
            self._record_successful_path(reservation)

        self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
        if len(self.entanglement_fidelities.get(reservation, ())) >= reservation.entanglement_number:
            self.node.resource_manager.expire_rules_by_reservation(reservation)

    def _record_successful_path(self, reservation) -> None:
        if id(reservation) in self._path_feedback_recorded:
            return
        self._path_feedback_recorded.add(id(reservation))
        path = getattr(reservation, "path", [])
        if not path or self._served_path_observer is None:
            return
        self._served_path_observer(tuple(path), self.node.timeline.now())


def collect_pair_results(name_to_app, requests, backend_name, seed):
    """Return one result row per offered request, including failures."""
    completed = {}
    rejected = set()
    for app in name_to_app.values():
        outcomes = getattr(app, "reservation_outcomes", {})
        rejected.update(identity for identity, accepted in outcomes.items() if not accepted)
        for reservation, tts in app.time_to_serve.items():
            fidelities = app.entanglement_fidelities.get(reservation, [])
            completed[reservation.identity] = (tts, fidelities[0] if fidelities else None)

    rows = []
    for identity, src, dst, start, _end, _memory, _fidelity, _pairs in requests:
        success = identity in completed
        tts, fidelity = completed.get(identity, (None, None))
        rows.append(RequestResult(
            request_id=identity, src=src, dst=dst, start_time_ps=start,
            time_to_serve_ms=tts / MILLISECOND if tts is not None else None,
            fidelity=fidelity, success=success,
            failure_reason="" if success else ("reservation_rejected" if identity in rejected else "deadline"),
        ))
    return BackendResult(
        backend_name=backend_name, seed=seed, num_nodes=len(name_to_app),
        request_results=rows,
    )
