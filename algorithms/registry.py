"""Configuration-driven routing algorithm registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from algorithms.acp import AdaptiveContinuous
from algorithms.base import RoutingAlgorithm
from algorithms.dfer import DFER
from algorithms.odo import ShortestPathOnDemand
from algorithms.qcast import QCAST, QCAST_CONTROL_PAPER_DISTRIBUTED
from algorithms.qcast_e2e import QCASTE2E
from algorithms.qguard import QGUARD


AlgorithmFactory = Callable[[Mapping[str, Any]], RoutingAlgorithm]
_ALGORITHMS: dict[str, AlgorithmFactory] = {}


def register_algorithm(name: str):
    def decorator(factory: AlgorithmFactory) -> AlgorithmFactory:
        if name in _ALGORITHMS:
            raise ValueError(f"Algorithm {name!r} is already registered")
        _ALGORITHMS[name] = factory
        return factory
    return decorator


def _acp_options(config: Mapping[str, Any]) -> dict[str, Any]:
    period_ps = config.get("period_ps")
    if period_ps is None:
        period_ps = int(float(config.get("control_period_s", 0.1)) * 10**12)
    return {
        "adaptive_max_memory": int(config.get(
            "adaptive_max_memory",
            config.get("max_background_memories_per_node", 5),
        )),
        "cache_strategy": str(config.get("cache_strategy", config.get("selection", "freshest"))),
        "period_ps": int(period_ps),
        "delta": float(config.get("delta", 0.05)),
        "background_enabled": bool(config.get("background_enabled", True)),
        "purify": bool(config.get("purify", config.get("purification", False))),
        "execution_profile": str(config.get("execution_profile", "asynchronous")),
    }


@register_algorithm("odo")
def _odo_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    del config
    return ShortestPathOnDemand()


@register_algorithm("qcast")
def _qcast_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    generation_window_ps = config.get("generation_window_ps")
    if generation_window_ps is None:
        generation_window_ps = int(float(config.get("generation_window_s", 0.005)) * 10**12)
    processing_delay_ps = config.get("control_processing_delay_ps")
    if processing_delay_ps is None:
        processing_delay_ps = int(float(config.get("control_processing_delay_s", 0.0001)) * 10**12)
    return QCAST(
        edge_width=int(config.get("edge_width", 3)),
        generation_window_ps=int(generation_window_ps),
        control_processing_delay_ps=int(processing_delay_ps),
        swap_success_probability=float(config.get("swap_success_probability", 0.9)),
        link_state_hops=int(config.get("link_state_hops", 3)),
        recovery_paths_per_segment=int(config.get("recovery_paths_per_segment", 1)),
        max_recovery_paths_per_major=int(config.get("max_recovery_paths_per_major", 12)),
        max_major_paths=int(config.get("max_major_paths", 200)),
        max_hops=int(config.get("max_hops", 8)),
    )


@register_algorithm("qcast_distributed")
def _qcast_distributed_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    algorithm = _qcast_factory(config)
    return QCAST(
        edge_width=algorithm.edge_width,
        generation_window_ps=algorithm.generation_window_ps,
        control_processing_delay_ps=algorithm.control_processing_delay_ps,
        swap_success_probability=algorithm.swap_success_probability,
        link_state_hops=algorithm.link_state_hops,
        recovery_paths_per_segment=algorithm.recovery_paths_per_segment,
        max_recovery_paths_per_major=algorithm.max_recovery_paths_per_major,
        max_major_paths=algorithm.max_major_paths,
        max_hops=algorithm.max_hops,
        control_mode=QCAST_CONTROL_PAPER_DISTRIBUTED,
    )


@register_algorithm("qguard")
def _qguard_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    algorithm = _qcast_factory(config)
    return QGUARD(
        edge_width=algorithm.edge_width,
        generation_window_ps=algorithm.generation_window_ps,
        control_processing_delay_ps=algorithm.control_processing_delay_ps,
        swap_success_probability=algorithm.swap_success_probability,
        link_state_hops=algorithm.link_state_hops,
        recovery_paths_per_segment=algorithm.recovery_paths_per_segment,
        max_recovery_paths_per_major=algorithm.max_recovery_paths_per_major,
        max_major_paths=algorithm.max_major_paths,
        max_hops=algorithm.max_hops,
        max_purification_rounds=int(config.get("max_purification_rounds", 20)),
    )


@register_algorithm("qcast_e2e")
def _qcast_e2e_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    algorithm = _qcast_factory(config)
    return QCASTE2E(
        edge_width=algorithm.edge_width,
        generation_window_ps=algorithm.generation_window_ps,
        control_processing_delay_ps=algorithm.control_processing_delay_ps,
        swap_success_probability=algorithm.swap_success_probability,
        link_state_hops=algorithm.link_state_hops,
        recovery_paths_per_segment=algorithm.recovery_paths_per_segment,
        max_recovery_paths_per_major=algorithm.max_recovery_paths_per_major,
        max_major_paths=algorithm.max_major_paths,
        max_hops=algorithm.max_hops,
        max_purification_rounds=int(config.get("max_purification_rounds", 20)),
    )


@register_algorithm("dfer")
def _dfer_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    processing_delay_ps = config.get("control_processing_delay_ps")
    if processing_delay_ps is None:
        processing_delay_ps = int(
            float(config.get("control_processing_delay_s", 0.0001)) * 10**12
        )
    retry_delay_ps = config.get("resource_retry_delay_ps")
    if retry_delay_ps is None:
        retry_delay_ps = int(
            float(config.get("resource_retry_delay_s", 0.0001)) * 10**12
        )
    return DFER(
        cutoff_fidelity=float(config.get("cutoff_fidelity", 0.7)),
        max_purification_rounds=int(config.get("max_purification_rounds", 20)),
        control_processing_delay_ps=int(processing_delay_ps),
        resource_retry_delay_ps=int(retry_delay_ps),
        swap_success_probability=float(
            config.get("swap_success_probability", 0.9)
        ),
        max_hops=int(config.get("max_hops", 64)),
    )


@register_algorithm("acp")
@register_algorithm("acp_freshest")
def _acp_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    return AdaptiveContinuous(**_acp_options(config))


@register_algorithm("acp_random")
def _acp_random_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    return AdaptiveContinuous(**(_acp_options(config) | {"cache_strategy": "random"}))


@register_algorithm("acp_purify")
def _acp_purify_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    return AdaptiveContinuous(**(
        _acp_options(config)
        | {"purify": True, "algorithm_name": "acp_purify"}
    ))


@register_algorithm("ucp")
def _ucp_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    return AdaptiveContinuous(**(
        _acp_options(config)
        | {"update_prob": False, "algorithm_name": "ucp"}
    ))


@register_algorithm("ucp_purify")
def _ucp_purify_factory(config: Mapping[str, Any]) -> RoutingAlgorithm:
    return AdaptiveContinuous(**(
        _acp_options(config)
        | {"update_prob": False, "purify": True, "algorithm_name": "ucp_purify"}
    ))


def create_algorithm(name: str, config: Mapping[str, Any] | None = None) -> RoutingAlgorithm:
    try:
        factory = _ALGORITHMS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(f"Unknown algorithm {name!r}; available: {choices}") from exc
    return factory(config or {})


def algorithm_names() -> tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))
