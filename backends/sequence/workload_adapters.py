"""Workload adapters for the common SeQUeNCe runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.router_net_topo import RouterNetTopo

from backends.sequence.demand_service import SequenceDemandService
from backends.sequence.qcast_scheduler import QCASTDemandScheduler
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
        algorithm=None,
    ):
        self.network_topology = network_topology
        self.workload = workload
        self.algorithm_name = algorithm_name
        self.served_path_observer = served_path_observer
        self.algorithm = algorithm

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
    def __init__(self, network_topology, workload, algorithm_name: str, served_path_observer=None, algorithm=None):
        super().__init__(network_topology, workload, algorithm_name, served_path_observer, algorithm)
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
    def __init__(self, network_topology, workload, algorithm_name: str, served_path_observer=None, algorithm=None):
        super().__init__(network_topology, workload, algorithm_name, served_path_observer, algorithm)
        routers = network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        if algorithm_name == "qcast":
            self.scheduler = QCASTDemandScheduler(
                network_topology,
                algorithm,
                workload.controller_node,
            )
            self.services = None
            self.transactions = [
                QPQTransaction(spec, self.scheduler.submit)
                for spec in workload.queries()
            ]
        else:
            self.scheduler = None
            self.services = {
                router.name: SequenceDemandService(router, served_path_observer)
                for router in routers
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
        if self.scheduler is not None:
            self.scheduler.finalize(now)
        else:
            for service in self.services.values():
                service.finalize(now)
        for transaction in self.transactions:
            transaction.finalize(now)

    def collect(self) -> BackendResult:
        return BackendResult(
            backend_name=self.algorithm_name,
            seed=self.workload.seed,
            num_nodes=len(self.network_topology.get_nodes_by_type(
                RouterNetTopo.QUANTUM_ROUTER
            )),
            request_results=[
                transaction.to_request_result()
                for transaction in sorted(self.transactions, key=lambda item: item.spec.query_id)
            ],
        )

    def diagnostics(self) -> dict:
        if self.scheduler is not None:
            diagnostics = self.scheduler.diagnostics()
            diagnostics["transactions"] = self._transaction_diagnostics()
            return diagnostics
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
            "transactions": self._transaction_diagnostics(),
        }

    def _transaction_diagnostics(self) -> dict:
        return {
            str(transaction.spec.query_id): {
                "success": transaction.success,
                "failure_reason": transaction.failure_reason,
                "rounds_started": sorted(transaction.rounds),
            }
            for transaction in self.transactions
        }


class ConcurrentPairSequenceAdapter(SequenceWorkloadAdapter):
    """Use the shared demand contract with Q-CAST or ordinary RSVP."""

    def __init__(self, network_topology, workload, algorithm_name: str, served_path_observer=None, algorithm=None):
        super().__init__(
            network_topology,
            workload,
            algorithm_name,
            served_path_observer,
            algorithm,
        )
        from workloads.concurrent_pairs import ConcurrentPairTransaction

        self.transactions = [
            ConcurrentPairTransaction(spec)
            for spec in workload.requests()
        ]
        routers = network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        if algorithm_name == "qcast":
            self.scheduler = QCASTDemandScheduler(
                network_topology,
                algorithm,
                workload.controller_node,
            )
            self.services = None
        else:
            self.scheduler = None
            self.services = {
                router.name: SequenceDemandService(router, served_path_observer)
                for router in routers
            }

    def schedule(self) -> None:
        timeline = self.network_topology.get_timeline()
        for transaction in self.transactions:
            submitter = (
                self.scheduler.submit
                if self.scheduler is not None
                else self.services[transaction.spec.source].submit
            )
            timeline.schedule(Event(
                transaction.spec.start_time_ps,
                Process(submitter, "__call__", [transaction.spec.demand(), transaction]),
                0,
            ))

    def finalize(self) -> None:
        now = self.network_topology.get_timeline().now()
        if self.scheduler is not None:
            self.scheduler.finalize(now)
        else:
            for service in self.services.values():
                service.finalize(now)
        for transaction in self.transactions:
            transaction.finalize(now)

    def collect(self) -> BackendResult:
        return BackendResult(
            backend_name=self.algorithm_name,
            seed=self.workload.seed,
            num_nodes=len(self.network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)),
            request_results=[
                transaction.to_request_result()
                for transaction in sorted(self.transactions, key=lambda item: item.spec.request_id)
            ],
        )

    def diagnostics(self) -> dict:
        if self.scheduler is not None:
            return self.scheduler.diagnostics()
        counters = Counter()
        for service in self.services.values():
            counters.update(service.counters)
        return {"counters": dict(counters)}


_ADAPTERS: dict[str, type[SequenceWorkloadAdapter]] = {
    "single_pair": SinglePairSequenceAdapter,
    "qpq": QPQSequenceAdapter,
    "concurrent_pairs": ConcurrentPairSequenceAdapter,
}


def create_sequence_workload_adapter(
    adapter_name: str,
    network_topology,
    workload,
    algorithm_name: str,
    served_path_observer: Callable[[tuple[str, ...], int], None] | None = None,
    algorithm=None,
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
        algorithm,
    )
