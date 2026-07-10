"""Ordinary SeQUeNCe routers with QDC physical router parameters applied."""

from __future__ import annotations

from sequence.topology.node import BSMNode, QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.topology import Topology as Topo


class ConfiguredRouterNetTopo(RouterNetTopo):
    """Upstream router topology with explicit gate and measurement fidelity.

    SeQUeNCe's JSON loader does not forward these QDC node fields to
    ``QuantumRouter``. This adapter preserves the upstream router, RSVP, and
    resource-manager implementation while applying the configured hardware.
    """

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
                node_obj = QuantumRouter(
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
