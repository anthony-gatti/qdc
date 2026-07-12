# Pre-plugin QPQ implementation

These files preserve the original current-SeQUeNCe ODO/QPQ baseline for result
comparison. They are historical reference material and are not imported by the
supported runners.

The old `QPQApp` combined application state, SeQUeNCe reservation handling,
ACP-specific hooks, result instrumentation, and a fixed round-2 setup buffer.
The supported implementation separates those responsibilities across:

- `workloads/qpq.py`
- `backends/sequence/demand_service.py`
- `backends/sequence/workload_adapters.py`
- `results.py`

Imports in this directory intentionally retain their historical names and are
not expected to run in place.
