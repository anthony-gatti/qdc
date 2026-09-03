"""Unmodified Q-CAST execution with Q-GUARD final purification only."""

from __future__ import annotations

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.resource_management.memory_manager import MemoryInfo

from backends.sequence.qcast_distributed import QCASTDistributedDemandScheduler
from backends.sequence.qcast_scheduler import (
    QCASTDemandScheduler,
    QCASTLaneEdge,
    _Slot,
)
from backends.sequence.qguard_scheduler import QGUARDDemandScheduler, _PairRef
from workloads.base import PairDelivery


class QCASTE2EDemandScheduler(QGUARDDemandScheduler):
    """Append only Q-GUARD's final purifier to paper-distributed Q-CAST.

    Q-CAST's Phase 1--4 methods are called explicitly below so this variant
    cannot accidentally use Q-GUARD's fidelity-bearing link state, equal-split
    per-hop targets, EXG ranking, or elementary-link purification workflows.
    """

    def _broadcast_link_state(self, slot: _Slot) -> int:
        return QCASTDemandScheduler._broadcast_link_state(self, slot)

    def start_distributed_major_path(
        self,
        slot_id: int,
        major_path_id: str,
    ) -> None:
        QCASTDistributedDemandScheduler.start_distributed_major_path(
            self,
            slot_id,
            major_path_id,
        )

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
        qualification_time = max(
            self._major_path_check_ps[key]
            for key in expected
        ) + 1
        self._finish_scheduled_slots.add(slot.slot_id)
        self.timeline.schedule(Event(
            qualification_time,
            Process(self, "begin_end_to_end_qualification", [slot.slot_id]),
        ))

    def check_delivery(
        self,
        slot_id: int,
        delivery_id: str,
        reservation_id: int,
        nodes: tuple[str, ...],
        assignments: tuple[QCASTLaneEdge, ...],
        used_recovery: bool,
    ) -> None:
        """Retain Q-CAST's completed pair instead of rejecting on fidelity."""
        slot = self.slot
        context = self.contexts.get(reservation_id)
        if (
            slot is None
            or slot.slot_id != slot_id
            or context is None
            or context.terminal
        ):
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
        self._slot_pairs[slot_id][reservation_id].append(_PairRef(
            pair_id=delivery_id,
            left=nodes[0],
            right=nodes[-1],
            left_memory_index=assignments[0].memory_index(nodes[0]),
            right_memory_index=assignments[-1].memory_index(nodes[-1]),
            carrier=None,
            path=nodes,
            provenance=assignments,
            used_recovery=used_recovery,
            p4_start_ps=slot.delivery_p4_start_ps.get(delivery_id),
        ))
        self.counters["qcast_e2e_end_to_end_pairs_assembled"] += 1

    def begin_end_to_end_qualification(self, slot_id: int) -> None:
        slot = self.slot
        if (
            slot is None
            or slot.slot_id != slot_id
            or slot_id in self._slot_qualification_started
        ):
            return
        self._slot_qualification_started.add(slot_id)
        demand_ids = {path.demand_id for path in slot.plan.major_paths}
        reservation_ids = {
            context.demand.reservation_id
            for context in self.contexts.values()
            if not context.terminal and context.demand.demand_id in demand_ids
        }
        self._slot_qualification_pending[slot_id] = set(reservation_ids)
        self.counters["qcast_e2e_final_qualification_phases"] += 1
        if not reservation_ids:
            self._schedule_slot_finish(slot_id)
            return
        for reservation_id in sorted(reservation_ids):
            self._continue_end_to_end_qualification(slot_id, reservation_id)

    def _deliver_qualified_pair(
        self,
        context,
        pair: _PairRef,
        fidelity: float,
    ) -> None:
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
                tuple(sorted((left, right)))
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
        self.counters["qcast_e2e_pairs_qualified"] += 1
        self.events.append({
            "event": "pair_delivered",
            "time_ps": self.timeline.now(),
            "demand_id": context.demand.demand_id,
            "fidelity": fidelity,
            "path": list(pair.path),
            "qcast_role": role,
            "qcast_e2e_qualified": True,
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

    def diagnostics(self) -> dict:
        diagnostics = QCASTDistributedDemandScheduler.diagnostics(self)
        diagnostics["qcast_e2e"] = {
            "variant": "qcast_phase1_through_phase4_plus_qguard_final_e2e",
            "max_purification_rounds": self.algorithm.max_purification_rounds,
            "purification_events": self.purification_events,
            "uses_official_bell_diagonal_bbpssw": True,
            "final_end_to_end_purification": True,
            "per_hop_fidelity_planning": False,
            "link_state_includes_realized_fidelity": False,
            "qcast_distributed_decisions": self.distributed_decisions,
        }
        return diagnostics
