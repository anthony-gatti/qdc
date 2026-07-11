"""Ordinary SeQUeNCe routers with QDC physical router parameters applied."""

from __future__ import annotations

from sequence.topology.node import BSMNode, QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.topology import Topology as Topo

from backends.sequence.parallel_links import (
    ParallelResourceManager,
    configure_parallel_middle_nodes,
    ensure_parallel_metrics,
)


class ParallelQuantumRouter(QuantumRouter):
    """Upstream router with lane-aware resource generation rules."""

    def __init__(
        self,
        name,
        tl,
        memo_size=50,
        seed=None,
        component_templates=None,
        gate_fid=1,
        meas_fid=1,
    ):
        super().__init__(
            name,
            tl,
            memo_size,
            seed,
            component_templates or {},
            gate_fid,
            meas_fid,
        )
        self.resource_manager = ParallelResourceManager(self, self.memo_arr_name)
        ensure_parallel_metrics(self)


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
                node_obj = ParallelQuantumRouter(
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
