"""SeQUeNCe router factory for DFER local control and physical execution."""

from __future__ import annotations

from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.topology.node import BSMNode
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.topology import Topology as Topo

from backends.sequence.dfer_protocol import DFERControlProtocol
from backends.sequence.qcast_topology import QCASTQuantumRouter, QCASTResourceManager
from backends.sequence.parallel_links import configure_parallel_middle_nodes


class DFERResourceManager(QCASTResourceManager):
    """Observe DFER elementary success without changing SeQUeNCe protocols."""

    def update(self, protocol, memory, state: str) -> None:
        super().update(protocol, memory, state)
        if (
            state == MemoryInfo.ENTANGLED
            and isinstance(protocol, EntanglementGenerationA)
            and protocol.primary
            and hasattr(memory, "dfer_operation_id")
        ):
            self.owner.dfer_control.record_elementary_success(
                memory.dfer_operation_id,
                self.owner.timeline.now(),
                memory.fidelity,
            )


class DFERQuantumRouter(QCASTQuantumRouter):
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
            component_templates,
            gate_fid,
            meas_fid,
        )
        self.resource_manager = DFERResourceManager(self, self.memo_arr_name)
        self.dfer_control = DFERControlProtocol(self)
        self.protocols.append(self.dfer_control)


class DFERRouterNetTopo(RouterNetTopo):
    """Pristine SeQUeNCe topology populated with DFER-aware routers."""

    def _add_nodes(self, config: dict) -> None:
        for node in config[Topo.ALL_NODE]:
            node_type = node[Topo.TYPE]
            name = node[Topo.NAME]
            template = self.templates.get(node.get(Topo.TEMPLATE), {})
            if node_type == self.BSM_NODE:
                node_obj = BSMNode(
                    name,
                    self.tl,
                    self.bsm_to_router_map[name],
                    component_templates=template,
                )
            elif node_type == self.QUANTUM_ROUTER:
                node_obj = DFERQuantumRouter(
                    name,
                    self.tl,
                    node.get(self.MEMO_ARRAY_SIZE, 0),
                    component_templates=template,
                    gate_fid=float(node.get("gate_fidelity", 1.0)),
                    meas_fid=float(node.get("measurement_fidelity", 1.0)),
                )
            else:
                raise ValueError(f"Unknown type of node {node_type!r}")
            node_obj.set_seed(node[Topo.SEED])
            self.nodes[node_type].append(node_obj)

    def _add_bsm_node_to_router(self) -> None:
        super()._add_bsm_node_to_router()
        configure_parallel_middle_nodes(self)
