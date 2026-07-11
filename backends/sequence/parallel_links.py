"""Shared physical link-parallelism support for SeQUeNCe backends."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.network_management.reservation import Reservation
from sequence.resource_management.action_condition_set import (
    eg_rule_action_await,
    eg_rule_condition,
    ep_rule_action_await,
    ep_rule_action_request,
    ep_rule_condition_await,
    ep_rule_condition_request,
    es_rule_action_A,
    es_rule_action_B,
    es_rule_condition_A,
    es_rule_condition_B,
    es_rule_condition_B_end,
)
from sequence.resource_management.resource_manager import ResourceManager
from sequence.resource_management.rule_manager import Rule
from sequence.topology.router_net_topo import RouterNetTopo


def expand_parallel_links(config: dict, link_parallelism: int) -> dict:
    """Expand each router link into independent midpoint BSM/channel lanes.

    A parallelism of one intentionally preserves the supplied topology byte for
    byte so established ODO and ACP simulations retain their current behavior.
    """
    if link_parallelism <= 0:
        raise ValueError("Link parallelism must be positive")
    if link_parallelism == 1:
        return deepcopy(config)

    expanded = deepcopy(config)
    configured = expanded.get("link_parallelism")
    if configured is not None:
        if configured != link_parallelism:
            raise ValueError("Topology was expanded with a different link parallelism")
        return expanded

    bsm_nodes = {
        node["name"]: node
        for node in expanded["nodes"]
        if node["type"] == RouterNetTopo.BSM_NODE
    }
    if not bsm_nodes:
        raise ValueError("Parallel links require midpoint BSM nodes")
    router_seeds = {
        node["name"]: int(node.get("seed", 0))
        for node in expanded["nodes"]
        if node["type"] == RouterNetTopo.QUANTUM_ROUTER
    }
    endpoints = {name: [] for name in bsm_nodes}
    for channel in expanded.get("qchannels", []):
        if channel["destination"] in endpoints:
            endpoints[channel["destination"]].append(channel["source"])

    def lane_name(name: str, lane: int) -> str:
        return f"{name}.lane_{lane}"

    nodes = [
        node for node in expanded["nodes"]
        if node["type"] != RouterNetTopo.BSM_NODE
    ]
    for name, node in sorted(bsm_nodes.items()):
        link_endpoints = sorted(endpoints[name])
        if len(link_endpoints) != 2:
            raise ValueError(f"BSM {name!r} must connect exactly two routers")
        endpoint_seed = (
            router_seeds[link_endpoints[0]] * 1_000_033
            + router_seeds[link_endpoints[1]] * 1_000_037
        )
        for lane in range(link_parallelism):
            duplicate = deepcopy(node)
            duplicate["name"] = lane_name(name, lane)
            duplicate["seed"] = (
                int(node.get("seed", 0)) * 1_000_003 + endpoint_seed + lane
            ) % (2**31)
            nodes.append(duplicate)

    qchannels = []
    for channel in expanded.get("qchannels", []):
        destination = channel["destination"]
        if destination not in bsm_nodes:
            qchannels.append(channel)
            continue
        for lane in range(link_parallelism):
            duplicate = deepcopy(channel)
            duplicate["destination"] = lane_name(destination, lane)
            qchannels.append(duplicate)

    cchannels = []
    for channel in expanded.get("cchannels", []):
        source = channel["source"]
        destination = channel["destination"]
        bsm_name = source if source in bsm_nodes else (
            destination if destination in bsm_nodes else None
        )
        if bsm_name is None:
            cchannels.append(channel)
            continue
        for lane in range(link_parallelism):
            duplicate = deepcopy(channel)
            if source == bsm_name:
                duplicate["source"] = lane_name(bsm_name, lane)
            if destination == bsm_name:
                duplicate["destination"] = lane_name(bsm_name, lane)
            cchannels.append(duplicate)

    expanded["nodes"] = nodes
    expanded["qchannels"] = qchannels
    expanded["cchannels"] = cchannels
    expanded["link_parallelism"] = link_parallelism
    return expanded


def configure_parallel_middle_nodes(topology) -> None:
    """Retain every midpoint BSM lane alongside SeQUeNCe's legacy map."""
    lanes: dict[tuple[str, str], list[str]] = {}
    for middle, endpoints in topology.bsm_to_router_map.items():
        if len(endpoints) != 2:
            raise ValueError(f"BSM {middle!r} does not have two endpoints")
        left, right = endpoints
        lanes.setdefault((left, right), []).append(middle)
        lanes.setdefault((right, left), []).append(middle)
    for router in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        router.parallel_middle_nodes = {
            neighbor: tuple(sorted(middles))
            for (owner, neighbor), middles in lanes.items()
            if owner == router.name
        }
        router.link_parallelism = max(
            (len(middles) for middles in router.parallel_middle_nodes.values()),
            default=1,
        )
        ensure_parallel_metrics(router)


def middle_nodes(owner, neighbor: str) -> tuple[str, ...]:
    lanes = getattr(owner, "parallel_middle_nodes", {}).get(neighbor)
    if lanes:
        return tuple(lanes)
    return (owner.map_to_middle_node[neighbor],)


def ensure_parallel_metrics(owner) -> dict:
    metrics = getattr(owner, "parallel_link_metrics", None)
    if metrics is None:
        metrics = {
            "generation_attempts": Counter(),
            "generation_successes": Counter(),
            "memory_high_watermark": 0,
        }
        owner.parallel_link_metrics = metrics
    return metrics


def _metric_key(owner_name: str, remote_name: str, middle: str) -> tuple[str, str, str]:
    left, right = sorted((owner_name, remote_name))
    return left, right, middle


def record_generation_attempt(owner, remote_name: str, middle: str) -> None:
    metrics = ensure_parallel_metrics(owner)
    metrics["generation_attempts"][_metric_key(owner.name, remote_name, middle)] += 1


def record_generation_success(owner, remote_name: str, middle: str) -> None:
    metrics = ensure_parallel_metrics(owner)
    metrics["generation_successes"][_metric_key(owner.name, remote_name, middle)] += 1


def record_memory_occupancy(owner) -> None:
    metrics = ensure_parallel_metrics(owner)
    occupied = sum(
        info.state != "RAW"
        for info in owner.resource_manager.memory_manager
    )
    metrics["memory_high_watermark"] = max(
        metrics["memory_high_watermark"], occupied
    )


def parallel_eg_rule_condition(memory_info, manager, args):
    if not eg_rule_condition(memory_info, manager, args):
        return []
    lane_offset = args["memory_indices"].index(memory_info.index)
    return [memory_info] if lane_offset % args["lane_count"] == args["lane_index"] else []


def parallel_eg_rule_action_request(memories_info, args):
    memory = memories_info[0].memory
    mid = args["mid"]
    path = args["path"]
    index = args["index"]
    owner = memory.memory_array.owner
    remote = path[index + 1]
    protocol = EntanglementGenerationA.create(
        owner=None,
        name=f"EGA.{memory.name}",
        middle=mid,
        other=remote,
        memory=memory,
    )
    record_generation_attempt(owner, remote, mid)
    req_args = {
        "name": args["name"],
        "reservation": args["reservation"],
        "mid": mid,
    }
    return protocol, [remote], [parallel_eg_match_func], [req_args]


def parallel_eg_match_func(protocols, args):
    for protocol in protocols:
        if (
            isinstance(protocol, EntanglementGenerationA)
            and protocol.remote_node_name == args["name"]
            and protocol.middle == args["mid"]
            and protocol.rule is not None
            and protocol.rule.get_reservation() == args["reservation"]
        ):
            return protocol
    return None


class ParallelResourceManager(ResourceManager):
    """Native ResourceManager with lane-specific generation rules."""

    def generate_load_rules(
        self,
        path: list[str],
        reservation: Reservation,
        timecards: list,
        memory_array_name: str,
    ):
        memory_indices = [
            card.memory_index for card in timecards if reservation in card.reservations
        ]
        index = path.index(self.owner.name)
        rules = []
        if index > 0:
            rules.extend(self._generation_rules(
                path[index - 1],
                path,
                index,
                memory_indices[:reservation.memory_size],
                reservation,
                requester=False,
            ))
        if index < len(path) - 1:
            selected = (
                memory_indices[:reservation.memory_size]
                if index == 0
                else memory_indices[reservation.memory_size:]
            )
            rules.extend(self._generation_rules(
                path[index + 1],
                path,
                index,
                selected,
                reservation,
                requester=True,
            ))

        if index > 0:
            rules.append(Rule(10, ep_rule_action_request, ep_rule_condition_request, {}, {
                "memory_indices": memory_indices[:reservation.memory_size],
                "reservation": reservation,
                "purification_mode": reservation.purification_mode,
            }))
        if index < len(path) - 1:
            selected = (
                memory_indices if index == 0
                else memory_indices[reservation.memory_size:]
            )
            rules.append(Rule(10, ep_rule_action_await, ep_rule_condition_await, {}, {
                "memory_indices": selected,
                "fidelity": reservation.fidelity,
                "purification_mode": reservation.purification_mode,
            }))

        if index == 0:
            rules.append(Rule(10, es_rule_action_B, es_rule_condition_B_end, {}, {
                "memory_indices": memory_indices,
                "target_remote": path[-1],
                "fidelity": reservation.fidelity,
            }))
        elif index == len(path) - 1:
            rules.append(Rule(10, es_rule_action_B, es_rule_condition_B_end, {}, {
                "memory_indices": memory_indices,
                "target_remote": path[0],
                "fidelity": reservation.fidelity,
            }))
        else:
            reduced_path = path[:]
            while reduced_path.index(self.owner.name) % 2 == 0:
                reduced_path = [
                    node for offset, node in enumerate(reduced_path)
                    if offset % 2 == 0 or offset == len(reduced_path) - 1
                ]
            reduced_index = reduced_path.index(self.owner.name)
            condition_args = {
                "memory_indices": memory_indices,
                "left": reduced_path[reduced_index - 1],
                "right": reduced_path[reduced_index + 1],
                "fidelity": reservation.fidelity,
            }
            rules.append(Rule(10, es_rule_action_A, es_rule_condition_A, {
                "swapping_success_prob": self.owner.swapping_success_prob,
                "swapping_degradation": self.owner.swapping_degradation,
            }, condition_args))
            rules.append(Rule(10, es_rule_action_B, es_rule_condition_B, {}, condition_args))

        for rule in rules:
            rule.set_reservation(reservation)
            self.owner.timeline.schedule(Event(
                reservation.start_time,
                Process(self.owner.resource_manager, "load", [rule]),
                self.owner.timeline.schedule_counter,
            ))
            self.owner.timeline.schedule(Event(
                reservation.end_time,
                Process(self.owner.resource_manager, "expire", [rule]),
                self.owner.timeline.schedule_counter,
            ))
        for card in timecards:
            if reservation in card.reservations:
                self.owner.timeline.schedule(Event(
                    reservation.end_time,
                    Process(self.owner.resource_manager, "update", [
                        None,
                        self.owner.components[memory_array_name][card.memory_index],
                        "RAW",
                    ]),
                    self.owner.timeline.schedule_counter,
                ))

    def load(self, rule):
        result = super().load(rule)
        record_memory_occupancy(self.owner)
        return result

    def update(self, protocol, memory, state: str):
        if (
            state == "ENTANGLED"
            and isinstance(protocol, EntanglementGenerationA)
            and protocol.primary
        ):
            record_generation_success(
                self.owner,
                protocol.remote_node_name,
                protocol.middle,
            )
        super().update(protocol, memory, state)
        record_memory_occupancy(self.owner)

    def _generation_rules(
        self,
        neighbor: str,
        path: list[str],
        index: int,
        memory_indices: list[int],
        reservation: Reservation,
        *,
        requester: bool,
    ) -> list[Rule]:
        lanes = middle_nodes(self.owner, neighbor)
        rules = []
        for lane_index, middle in enumerate(lanes):
            condition_args = {
                "memory_indices": memory_indices,
            }
            parallel = len(lanes) > 1
            if parallel:
                condition_args.update({
                    "lane_index": lane_index,
                    "lane_count": len(lanes),
                })
            action_args = {"mid": middle, "path": path, "index": index}
            if requester:
                action_args.update({
                    "name": self.owner.name,
                    "reservation": reservation,
                })
                action = parallel_eg_rule_action_request
            else:
                action = eg_rule_action_await
            rules.append(Rule(
                10,
                action,
                parallel_eg_rule_condition if parallel else eg_rule_condition,
                action_args,
                condition_args,
            ))
        return rules


def collect_parallel_link_diagnostics(network_topology) -> dict:
    links: dict[tuple[str, str], dict] = {}
    for middle, endpoints in network_topology.bsm_to_router_map.items():
        if len(endpoints) != 2:
            continue
        left, right = sorted(endpoints)
        entry = links.setdefault((left, right), {
            "channels": {},
            "parallelism": 0,
        })
        entry["channels"].setdefault(middle, {
            "generation_attempts": 0,
            "generation_successes": 0,
        })
        entry["parallelism"] = len(entry["channels"])
    memory_occupancy = {}
    for router in network_topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        metrics = ensure_parallel_metrics(router)
        memory_occupancy[router.name] = {
            "high_watermark": metrics["memory_high_watermark"],
            "final": dict(Counter(
                info.state for info in router.resource_manager.memory_manager
            )),
        }
        for metric_name in ("generation_attempts", "generation_successes"):
            for (left, right, middle), value in metrics[metric_name].items():
                entry = links.setdefault((left, right), {
                    "channels": {},
                    "parallelism": 0,
                })
                channel = entry["channels"].setdefault(middle, {
                    "generation_attempts": 0,
                    "generation_successes": 0,
                })
                channel[metric_name] += value
                entry["parallelism"] = len(entry["channels"])
    return {
        "links": {
            f"{left}|{right}": value
            for (left, right), value in sorted(links.items())
        },
        "memory_occupancy": memory_occupancy,
    }
