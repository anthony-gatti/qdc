"""SeQUeNCe router factory for Q-CAST control and physical execution."""

from __future__ import annotations

from copy import deepcopy

from sequence.topology.node import BSMNode, QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.topology import Topology as Topo

from backends.sequence.qcast_protocol import QCASTControlProtocol


def expand_qcast_parallel_links(config: dict, edge_width: int) -> dict:
    """Give each Q-CAST edge-width unit an independent BSM/channel lane."""
    if edge_width <= 0:
        raise ValueError("Q-CAST edge width must be positive")
    expanded = deepcopy(config)
    if expanded.get("qcast_parallel_edge_width") is not None:
        if expanded["qcast_parallel_edge_width"] != edge_width:
            raise ValueError("Q-CAST topology was expanded with a different edge width")
        return expanded

    bsm_nodes = {
        node["name"]: node
        for node in expanded["nodes"]
        if node["type"] == RouterNetTopo.BSM_NODE
    }
    if not bsm_nodes:
        raise ValueError("Q-CAST topology requires midpoint BSM nodes")
    router_seeds = {
        node["name"]: int(node.get("seed", 0))
        for node in expanded["nodes"]
        if node["type"] == RouterNetTopo.QUANTUM_ROUTER
    }
    bsm_endpoints = {name: [] for name in bsm_nodes}
    for channel in expanded.get("qchannels", []):
        if channel["destination"] in bsm_endpoints:
            bsm_endpoints[channel["destination"]].append(channel["source"])

    def lane_name(name: str, lane: int) -> str:
        return f"{name}.qcast_lane_{lane}"

    nodes = [
        node
        for node in expanded["nodes"]
        if node["type"] != RouterNetTopo.BSM_NODE
    ]
    for name, node in sorted(bsm_nodes.items()):
        endpoints = sorted(bsm_endpoints[name])
        if len(endpoints) != 2:
            raise ValueError(f"Q-CAST BSM {name!r} must connect two routers")
        endpoint_seed = (
            router_seeds[endpoints[0]] * 1_000_033
            + router_seeds[endpoints[1]] * 1_000_037
        )
        for lane in range(edge_width):
            duplicate = deepcopy(node)
            duplicate["name"] = lane_name(name, lane)
            duplicate["seed"] = (
                int(node.get("seed", 0)) * 1_000_003
                + endpoint_seed
                + lane
            ) % (2**31)
            nodes.append(duplicate)

    qchannels = []
    for channel in expanded.get("qchannels", []):
        destination = channel["destination"]
        if destination not in bsm_nodes:
            qchannels.append(channel)
            continue
        for lane in range(edge_width):
            duplicate = deepcopy(channel)
            duplicate["destination"] = lane_name(destination, lane)
            qchannels.append(duplicate)

    cchannels = []
    for channel in expanded.get("cchannels", []):
        source = channel["source"]
        destination = channel["destination"]
        bsm_name = (
            source
            if source in bsm_nodes
            else destination if destination in bsm_nodes else None
        )
        if bsm_name is None:
            cchannels.append(channel)
            continue
        for lane in range(edge_width):
            duplicate = deepcopy(channel)
            if source == bsm_name:
                duplicate["source"] = lane_name(bsm_name, lane)
            if destination == bsm_name:
                duplicate["destination"] = lane_name(bsm_name, lane)
            cchannels.append(duplicate)

    expanded["nodes"] = nodes
    expanded["qchannels"] = qchannels
    expanded["cchannels"] = cchannels
    expanded["qcast_parallel_edge_width"] = edge_width
    return expanded


class QCASTQuantumRouter(QuantumRouter):
    def __init__(self, name, tl, memo_size=50, seed=None, component_templates=None, gate_fid=1, meas_fid=1):
        super().__init__(
            name,
            tl,
            memo_size,
            seed,
            component_templates or {},
            gate_fid,
            meas_fid,
        )
        self.qcast_control = QCASTControlProtocol(self)
        self.protocols.append(self.qcast_control)


class QCASTRouterNetTopo(RouterNetTopo):
    """Pristine SeQUeNCe topology using ACP-free Q-CAST-aware routers."""

    def _add_nodes(self, config: dict) -> None:
        for node in config[Topo.ALL_NODE]:
            seed = node[Topo.SEED]
            node_type = node[Topo.TYPE]
            name = node[Topo.NAME]
            template_name = node.get(Topo.TEMPLATE, None)
            template = self.templates.get(template_name, {})

            if node_type == self.BSM_NODE:
                node_obj = BSMNode(
                    name,
                    self.tl,
                    self.bsm_to_router_map[name],
                    component_templates=template,
                )
            elif node_type == self.QUANTUM_ROUTER:
                node_obj = QCASTQuantumRouter(
                    name,
                    self.tl,
                    node.get(self.MEMO_ARRAY_SIZE, 0),
                    component_templates=template,
                    gate_fid=float(node.get("gate_fidelity", 1.0)),
                    meas_fid=float(node.get("measurement_fidelity", 1.0)),
                )
            else:
                raise ValueError(f"Unknown type of node {node_type!r}")

            node_obj.set_seed(seed)
            self.nodes[node_type].append(node_obj)
