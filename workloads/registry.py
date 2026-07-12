"""Configuration-driven application workload registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from workloads.base import Workload
from workloads.concurrent_pairs import ConcurrentPairWorkload
from workloads.qpq import QPQWorkload
from workloads.single_pair import SinglePairPaperWorkload


WorkloadFactory = Callable[[Mapping[str, Any], int], Workload]
_WORKLOADS: dict[str, WorkloadFactory] = {}


def register_workload(name: str):
    def decorator(factory: WorkloadFactory) -> WorkloadFactory:
        if name in _WORKLOADS:
            raise ValueError(f"Workload {name!r} is already registered")
        _WORKLOADS[name] = factory
        return factory
    return decorator


@register_workload("qpq")
def _qpq_factory(config: Mapping[str, Any], seed: int) -> Workload:
    return QPQWorkload.from_config(config, seed=seed)


@register_workload("single_pair")
def _single_pair_factory(config: Mapping[str, Any], seed: int) -> Workload:
    workload = config.get("workload", {})
    hardware = config.get("hardware", {})
    return SinglePairPaperWorkload(
        num_requests=int(workload.get("request_count", workload.get("requests", 100))),
        link_distance_m=float(workload.get("link_distance_m", 10_000.0)),
        memories_per_node=int(hardware.get("memories_per_node", 10)),
        request_rate_hz=float(workload.get("request_rate_hz", 10.0)),
        request_window_s=float(workload.get("request_window_s", 0.08)),
        fidelity_threshold=float(workload.get("fidelity_threshold", 0.5)),
        link_parallelism=int(hardware.get(
            "link_parallelism",
            hardware.get("bsm_lanes_per_link", 1),
        )),
        seed=seed,
    )


@register_workload("concurrent_pairs")
def _concurrent_pairs_factory(config: Mapping[str, Any], seed: int) -> Workload:
    return ConcurrentPairWorkload.from_config(config, seed)


def create_workload(name: str, config: Mapping[str, Any], seed: int) -> Workload:
    try:
        factory = _WORKLOADS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_WORKLOADS))
        raise ValueError(f"Unknown workload {name!r}; available: {choices}") from exc
    return factory(config, seed)


def workload_names() -> tuple[str, ...]:
    return tuple(sorted(_WORKLOADS))
