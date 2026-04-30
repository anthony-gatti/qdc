"""
QPQ (Quantum Private Query) application for Sequence.

Manages multi-round QPQ queries on top of Sequence's entanglement
infrastructure. Each query consists of 2 rounds, each requiring
2*log(N)+1 Bell pairs delivered within a deadline.

"""

import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from sequence.app.request_app import RequestApp
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.constants import SECOND
import sequence.utils.log as log

# ACP imports for path caching
ACP_DIR = os.path.join(os.path.dirname(__file__), '..', 'acp')
if ACP_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(ACP_DIR))

from reservation import ReservationAdaptive

if TYPE_CHECKING:
    from node import QuantumRouterAdaptive
    from sequence.network_management.reservation import Reservation


# ---------- Data structures ----------

@dataclass
class QPQRound:
    """Tracks state for one round of a QPQ query."""
    round_num: int                      # 1 or 2
    pairs_needed: int                   # 2*log(N)+1
    pairs_delivered: int = 0
    fidelities: List[float] = field(default_factory=list)
    start_time_ps: int = 0              # when round was issued
    end_time_ps: int = 0                # when last pair was delivered
    deadline_met: bool = False
    reservation: Optional[object] = None  # the Sequence Reservation object


@dataclass
class QPQQuery:
    """Tracks state for a complete QPQ query (2 rounds)."""
    query_id: int
    src: str
    dst: str
    database_size_log: int
    fidelity_threshold: float
    round_deadline_ps: int              # max time to deliver all pairs in one round
    reservation_end_ps: int             # overall time window end
    rounds: Dict[int, QPQRound] = field(default_factory=dict)
    success: bool = False
    failure_reason: str = ""
    total_time_ps: int = 0              # from round 1 start to round 2 end


@dataclass
class QPQResult:
    """Final result for a QPQ query, used by the results collector."""
    query_id: int
    src: str
    dst: str
    database_size_log: int
    pairs_per_round: int
    success: bool
    failure_reason: str = ""
    total_time_ms: float = 0.0
    round1_time_ms: float = 0.0
    round2_time_ms: float = 0.0
    round1_pairs: int = 0
    round2_pairs: int = 0
    round1_avg_fidelity: float = 0.0
    round2_avg_fidelity: float = 0.0
    round1_deadline_met: bool = False
    round2_deadline_met: bool = False


# ---------- Application ----------

MILLISECOND = int(1e9)


class QPQApp(RequestApp):
    """QPQ application managing multi-round queries.

    Attach to every router in the network. Only initiator-side instances
    manage query state; responder-side instances just handle pair delivery
    and ACP cache updates.

    Usage:
        app = QPQApp(router_node)
        app.submit_query(query_id=0, responder="router_7",
                         start_time=2*SECOND, end_time=60*SECOND,
                         database_size_log=20, fidelity=0.7,
                         round_deadline_ps=5*SECOND)
    """

    def __init__(self, node: "QuantumRouterAdaptive"):
        super().__init__(node)

        # Query tracking (initiator side)
        self.queries: Dict[int, QPQQuery] = {}
        self._reservation_to_query: Dict[object, tuple] = {}  # reservation -> (query_id, round_num)
        self._pending_rounds: Dict[int, tuple] = {}  # start_time -> (query_id, round_num)

        # Pair tracking (both sides)
        self.entanglement_timestamps = defaultdict(list)
        self.entanglement_fidelities = defaultdict(list)

    def submit_query(
        self,
        query_id: int,
        responder: str,
        start_time: int,
        end_time: int,
        database_size_log: int,
        fidelity: float,
        round_deadline_ps: int,
    ) -> None:
        """Submit a QPQ query. Initiates round 1.

        Args:
            query_id: unique identifier for this query.
            responder: name of QDC server node.
            start_time: simulation time (ps) to begin round 1.
            end_time: simulation time (ps) by which everything must complete.
            database_size_log: n = log2(N), determines pairs per round.
            fidelity: minimum acceptable fidelity per pair.
            round_deadline_ps: max time (ps) to deliver all pairs in one round.
        """
        pairs = 2 * database_size_log + 1

        query = QPQQuery(
            query_id=query_id,
            src=self.node.name,
            dst=responder,
            database_size_log=database_size_log,
            fidelity_threshold=fidelity,
            round_deadline_ps=round_deadline_ps,
            reservation_end_ps=end_time,
        )
        self.queries[query_id] = query

        self._start_round(query, round_num=1, start_time=start_time, end_time=end_time)

    def _start_round(
        self,
        query: QPQQuery,
        round_num: int,
        start_time: int,
        end_time: int,
    ) -> None:
        """Issue a reservation for one round of a QPQ query."""
        pairs = 2 * query.database_size_log + 1

        rnd = QPQRound(
            round_num=round_num,
            pairs_needed=pairs,
            start_time_ps=start_time,
        )
        query.rounds[round_num] = rnd

        # Track pending round so we can map reservation -> query later
        self._pending_rounds[start_time] = (query.query_id, round_num)

        log.logger.info(
            f"{self.node.name} QPQ query {query.query_id} round {round_num}: "
            f"requesting {pairs} pairs from {query.dst}"
        )

        self.node.reserve_net_resource(
            query.dst,
            start_time,
            end_time,
            1,                          # memo_size
            query.fidelity_threshold,
            pairs,                      # entanglement_number
            query.query_id,             # id
        )

    def get_memory(self, info: "MemoryInfo") -> None:
        """Callback when a memory becomes entangled.

        Handles both initiator and responder sides.
        On the initiator side, tracks pair delivery and manages round transitions.
        """
        if info.state != "ENTANGLED":
            return

        if info.index not in self.memo_to_reservation:
            return

        reservation = self.memo_to_reservation[info.index]

        # Map reservation to query if we haven't yet
        self._try_map_reservation(reservation)

        if info.remote_node == reservation.responder:
            # We are the initiator
            self._handle_initiator_pair(info, reservation)
        elif info.remote_node == reservation.initiator:
            # We are the responder (QDC side)
            self._handle_responder_pair(info, reservation)

    def _try_map_reservation(self, reservation) -> None:
        """Try to map a reservation to a (query_id, round_num) pair."""
        if reservation in self._reservation_to_query:
            return

        # Match by start_time
        start_time = reservation.start_time
        if start_time in self._pending_rounds:
            query_id, round_num = self._pending_rounds.pop(start_time)
            self._reservation_to_query[reservation] = (query_id, round_num)

            # Store reservation reference on the round
            if query_id in self.queries:
                query = self.queries[query_id]
                if round_num in query.rounds:
                    query.rounds[round_num].reservation = reservation

    def _handle_initiator_pair(self, info: "MemoryInfo", reservation) -> None:
        """Handle a delivered pair on the initiator (client) side."""
        if info.fidelity < reservation.fidelity:
            log.logger.info(
                f"{self.node.name}: pair fidelity {info.fidelity:.4f} "
                f"below threshold {reservation.fidelity}"
            )
            return

        # Track delivery
        self.entanglement_timestamps[reservation].append(self.node.timeline.now())
        self.entanglement_fidelities[reservation].append(info.fidelity)

        # Free memory for next pair
        self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)

        # ACP path caching
        self._cache_entangled_path(reservation)
        self._send_entangled_path(reservation)

        # Check round completion
        pairs_delivered = len(self.entanglement_timestamps[reservation])

        if pairs_delivered == reservation.entanglement_number:
            self._on_round_complete(reservation)

    def _handle_responder_pair(self, info: "MemoryInfo", reservation) -> None:
        """Handle a delivered pair on the responder (QDC) side."""
        if info.fidelity >= reservation.fidelity:
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
            self._cache_entangled_path(reservation)
        else:
            log.logger.info(
                f"{self.node.name}: responder pair fidelity {info.fidelity:.4f} "
                f"below threshold {reservation.fidelity}"
            )

    def _on_round_complete(self, reservation) -> None:
        """Called when all pairs for a round have been delivered."""
        # Expire rules for this round's reservation
        self.node.resource_manager.expire_rules_by_reservation(reservation)
        self._send_expire_rules_message(reservation)

        # Find query and round
        if reservation not in self._reservation_to_query:
            log.logger.warning(f"{self.node.name}: completed reservation not mapped to query")
            return

        query_id, round_num = self._reservation_to_query[reservation]
        query = self.queries[query_id]
        rnd = query.rounds[round_num]

        # Record round results
        rnd.end_time_ps = self.node.timeline.now()
        rnd.pairs_delivered = len(self.entanglement_timestamps[reservation])
        rnd.fidelities = list(self.entanglement_fidelities[reservation])

        # Check deadline
        round_duration = rnd.end_time_ps - rnd.start_time_ps
        rnd.deadline_met = (round_duration <= query.round_deadline_ps)

        round_time_ms = round_duration / MILLISECOND
        avg_fid = sum(rnd.fidelities) / len(rnd.fidelities) if rnd.fidelities else 0

        log.logger.info(
            f"{self.node.name} QPQ query {query_id} round {round_num} complete: "
            f"{rnd.pairs_delivered} pairs in {round_time_ms:.2f} ms, "
            f"avg fidelity {avg_fid:.4f}, deadline {'met' if rnd.deadline_met else 'MISSED'}"
        )

        if round_num == 1:
            if rnd.deadline_met:
                # Start round 2 with buffer for RSVP propagation
                # The reservation request must arrive at all hops
                # before start_time. Use 1 second as a safe buffer.
                now = self.node.timeline.now()
                # Estimate buffer from path length: each hop needs ~2 classical
                # channel delays for RSVP request + response propagation
                if hasattr(rnd.reservation, 'path') and rnd.reservation.path:
                    path_len = len(rnd.reservation.path)
                else:
                    path_len = 5  # conservative default
                rsvp_buffer = path_len * 50 * MILLISECOND + 100 * MILLISECOND
                round2_start = now + rsvp_buffer
                if round2_start >= query.reservation_end_ps:
                    query.success = False
                    query.failure_reason = "no_time_for_round2"
                    query.total_time_ps = now - query.rounds[1].start_time_ps
                    log.logger.info(
                        f"{self.node.name} QPQ query {query.query_id} FAILED: "
                        f"no time remaining for round 2"
                    )
                    return
                self._start_round(query, round_num=2,
                                  start_time=round2_start,
                                  end_time=query.reservation_end_ps)
            else:
                query.success = False
                query.failure_reason = "round1_deadline_missed"
                query.total_time_ps = rnd.end_time_ps - rnd.start_time_ps
                log.logger.info(
                    f"{self.node.name} QPQ query {query_id} FAILED: round 1 deadline missed"
                )

        elif round_num == 2:
            if rnd.deadline_met:
                query.success = True
                query.total_time_ps = rnd.end_time_ps - query.rounds[1].start_time_ps
                log.logger.info(
                    f"{self.node.name} QPQ query {query_id} SUCCEEDED in "
                    f"{query.total_time_ps / MILLISECOND:.2f} ms"
                )
            else:
                query.success = False
                query.failure_reason = "round2_deadline_missed"
                query.total_time_ps = rnd.end_time_ps - query.rounds[1].start_time_ps
                log.logger.info(
                    f"{self.node.name} QPQ query {query_id} FAILED: round 2 deadline missed"
                )

    # ---------- ACP integration ----------

    def _cache_entangled_path(self, reservation) -> None:
        """Feed entangled path to ACP probability table."""
        if hasattr(reservation, 'path') and reservation.path:
            timestamp = self.node.timeline.now()
            cache = self.node.adaptive_continuous.cache
            cache.append((timestamp, reservation.path))

    def _send_entangled_path(self, reservation) -> None:
        """Send entangled path to intermediate nodes for ACP cache."""
        if not hasattr(reservation, 'path') or not reservation.path:
            return
        path = reservation.path
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                node = path[i]
                time = self.node.timeline.now()
                self.node.adaptive_continuous.send_entanglement_path(
                    node, time, reservation
                )

    def _send_expire_rules_message(self, reservation) -> None:
        """Send expire rules to intermediate nodes."""
        if not hasattr(reservation, 'path') or not reservation.path:
            return
        path = reservation.path
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                node = path[i]
                self.node.adaptive_continuous.send_expire_rules_message(
                    node, reservation
                )

    # ---------- Results collection ----------

    def get_results(self) -> List[QPQResult]:
        """Collect results for all completed queries on this node."""
        results = []
        for query_id, query in self.queries.items():
            r = QPQResult(
                query_id=query_id,
                src=query.src,
                dst=query.dst,
                database_size_log=query.database_size_log,
                pairs_per_round=2 * query.database_size_log + 1,
                success=query.success,
                failure_reason=query.failure_reason,
            )

            if query.total_time_ps > 0:
                r.total_time_ms = query.total_time_ps / MILLISECOND

            if 1 in query.rounds:
                rnd1 = query.rounds[1]
                r.round1_pairs = rnd1.pairs_delivered
                r.round1_deadline_met = rnd1.deadline_met
                if rnd1.end_time_ps > 0 and rnd1.start_time_ps > 0:
                    r.round1_time_ms = (rnd1.end_time_ps - rnd1.start_time_ps) / MILLISECOND
                if rnd1.fidelities:
                    r.round1_avg_fidelity = sum(rnd1.fidelities) / len(rnd1.fidelities)

            if 2 in query.rounds:
                rnd2 = query.rounds[2]
                r.round2_pairs = rnd2.pairs_delivered
                r.round2_deadline_met = rnd2.deadline_met
                if rnd2.end_time_ps > 0 and rnd2.start_time_ps > 0:
                    r.round2_time_ms = (rnd2.end_time_ps - rnd2.start_time_ps) / MILLISECOND
                if rnd2.fidelities:
                    r.round2_avg_fidelity = sum(rnd2.fidelities) / len(rnd2.fidelities)

            results.append(r)

        return results