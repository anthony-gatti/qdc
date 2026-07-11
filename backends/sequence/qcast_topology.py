"""SeQUeNCe router factory for Q-CAST control and physical execution."""

from __future__ import annotations

from sequence.topology.node import BSMNode, QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.topology import Topology as Topo

from backends.sequence.qcast_protocol import QCASTControlProtocol
from backends.sequence.parallel_links import (
    configure_parallel_middle_nodes,
    ensure_parallel_metrics,
    expand_parallel_links,
)


def expand_qcast_parallel_links(config: dict, edge_width: int) -> dict:
    """Compatibility alias retained for callers from the initial port."""
    return expand_parallel_links(config, edge_width)


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
        ensure_parallel_metrics(self)
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

    def _add_bsm_node_to_router(self) -> None:
        super()._add_bsm_node_to_router()
        configure_parallel_middle_nodes(self)
