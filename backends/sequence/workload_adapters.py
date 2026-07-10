"""Workload adapters for the common SeQUeNCe runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.router_net_topo import RouterNetTopo

from backends.sequence.demand_service import SequenceDemandService
from pair_app import PairRequestApp, collect_pair_results
from results import BackendResult
from workloads.qpq import QPQTransaction


class SequenceWorkloadAdapter(ABC):
    def __init__(
        self,
        network_topology,
        workload,
        algorithm_name: str,
        served_path_observer: Callable[[tuple[str, ...], int], None] | None = None,
    ):
        self.network_topology = network_topology
        self.workload = workload
        self.algorithm_name = algorithm_name
        self.served_path_observer = served_path_observer

    @abstractmethod
    def schedule(self) -> None:
        """Attach applications and submit initial workload stages."""

    @abstractmethod
    def finalize(self) -> None:
        """Finalize unfinished application state after the timeline ends."""

    @abstractmethod
    def collect(self) -> BackendResult:
        """Return standardized workload results."""

    def diagnostics(self) -> dict:
        return {}


class SinglePairSequenceAdapter(SequenceWorkloadAdapter):
    def __init__(self, network_topology, workload, algorithm_name: str, served_path_observer=None):
        super().__init__(network_topology, workload, algorithm_name, served_path_observer)
        self.apps = {
            router.name: PairRequestApp(router, served_path_observer)
            for router in network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        }
        self.requests = workload.requests()

    def schedule(self) -> None:
        for request in self.requests:
            identity, src, dst, start, end, memory, fidelity, pairs = request
            self.apps[src].start(dst, start, end, memory, fidelity, pairs, identity)

    def finalize(self) -> None:
        return

    def collect(self) -> BackendResult:
        return collect_pair_results(
            self.apps,
            self.requests,
            self.algorithm_name,
            self.workload.seed,
        )


class QPQSequenceAdapter(SequenceWorkloadAdapter):
    def __init__(self, network_topology, workload, algorithm_name: str, served_path_observer=None):
        super().__init__(network_topology, workload, algorithm_name, served_path_observer)
        self.services = {
            router.name: SequenceDemandService(router, served_path_observer)
            for router in network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        }
        self.transactions = [
            QPQTransaction(spec, self.services[spec.source].submit)
            for spec in workload.queries()
        ]

    def schedule(self) -> None:
        timeline = self.network_topology.get_timeline()
        for transaction in self.transactions:
            timeline.schedule(Event(
                transaction.spec.start_time_ps,
                Process(transaction, "start", []),
                0,
            ))

    def finalize(self) -> None:
        now = self.network_topology.get_timeline().now()
        for service in self.services.values():
            service.finalize(now)
        for transaction in self.transactions:
            transaction.finalize(now)

    def collect(self) -> BackendResult:
        return BackendResult(
            backend_name=self.algorithm_name,
            seed=self.workload.seed,
            num_nodes=len(self.services),
            request_results=[
                transaction.to_request_result()
                for transaction in sorted(self.transactions, key=lambda item: item.spec.query_id)
            ],
        )

    def diagnostics(self) -> dict:
        counters = Counter()
        counters_by_node = {}
        events = []
        for node_name, service in self.services.items():
            counters.update(service.counters)
            counters_by_node[node_name] = dict(service.counters)
            events.extend(service.events)
        return {
            "counters": dict(counters),
            "counters_by_node": counters_by_node,
            "events": sorted(events, key=lambda event: event.get("time_ps", 0)),
            "transactions": {
                str(transaction.spec.query_id): {
                    "success": transaction.success,
                    "failure_reason": transaction.failure_reason,
                    "rounds_started": sorted(transaction.rounds),
                }
                for transaction in self.transactions
            },
        }


_ADAPTERS: dict[str, type[SequenceWorkloadAdapter]] = {
    "single_pair": SinglePairSequenceAdapter,
    "qpq": QPQSequenceAdapter,
}


def create_sequence_workload_adapter(
    adapter_name: str,
    network_topology,
    workload,
    algorithm_name: str,
    served_path_observer: Callable[[tuple[str, ...], int], None] | None = None,
) -> SequenceWorkloadAdapter:
    try:
        adapter_type = _ADAPTERS[adapter_name]
    except KeyError as exc:
        raise ValueError(f"No SeQUeNCe adapter registered for workload {adapter_name!r}") from exc
    return adapter_type(
        network_topology,
        workload,
        algorithm_name,
        served_path_observer,
    )
