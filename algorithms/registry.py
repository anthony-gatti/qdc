"""Configuration-driven routing algorithm registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from algorithms.acp import AdaptiveContinuous
from algorithms.base import RoutingAlgorithm
from algorithms.odo import ShortestPathOnDemand


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
