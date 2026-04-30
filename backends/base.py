"""
Abstract base class for QPQ evaluation backends.

Each backend builds a SeQUeNCe simulation, runs it with a given topology
and request queue, and returns standardized results.
"""

from abc import ABC, abstractmethod
from results import BackendResult


class BackendBase(ABC):
    """Interface that every routing algorithm backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend, e.g. 'odo', 'acp'."""
        ...

    @property
    def adaptive_max_memory(self) -> int:
        """ACP memory budget to set in the topology JSON template.
        Override in subclasses that use ACP infrastructure.
        """
        return 0

    @abstractmethod
    def run(
        self,
        topo_json_path: str,
        request_queue: list,
        config: dict,
    ) -> BackendResult:
        """Build a SeQUeNCe simulation, execute it, and return results.

        Args:
            topo_json_path: path to SeQUeNCe-compatible JSON topology config.
            request_queue: list of request tuples, each containing:
                (id, src_name, dst_name, start_time, end_time,
                 memo_size, fidelity, entanglement_number)
                Same format as used by ACP's demo.py / RequestAppTimeToServe.
            config: full experiment config dict (from YAML), for any
                additional parameters the backend needs.

        Returns:
            BackendResult with per-request timing and fidelity data.
        """
        ...