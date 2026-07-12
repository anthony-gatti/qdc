# Workload plugins

Workloads describe application transactions without importing routing
algorithms or SeQUeNCe. They emit `EntanglementDemand` objects and consume the
delivery callbacks defined in `workloads/base.py`. Simulator-specific adapters
live under `backends/sequence/`.

## QPQ model

`QPQWorkload` generates client-to-QDC query specifications. Each
`QPQTransaction` has two strictly sequential rounds. For `n = log2(N)`, every
round requires `2n + 1` qualifying end-to-end Bell pairs, so a successful query
uses `4n + 2` pairs. Round 2 is submitted only after all round-1 pairs arrive.

The model includes application arrival, RSVP setup, pair generation, cache
reuse, swapping, fidelity rejection, round deadlines, and transaction
deadlines in network time-to-serve. It intentionally does not simulate qRAM
execution, teleportation gates, query-register decoherence, or QPQ security
measurements. Those are future application-model extensions.

`SequenceDemandService` is the reusable translation from workload demands to
native SeQUeNCe reservations. Algorithm-specific behavior remains on the other
side of that boundary. In particular, ACP path feedback enters through
`ACPRouterNetTopo.record_served_path`; the QPQ state machine does not know ACP
exists.

When ACP background purification is enabled, cached pairs use SeQUeNCe's
Bell-diagonal BBPSSW model only when that model predicts a fidelity increase.
With the default 0.99 local gate and measurement fidelities, purifying fresh
0.99 link pairs is correctly skipped because the modeled local operations would
reduce their fidelity.

## Adding a workload

1. Implement a simulator-neutral `Workload` and transaction state machine.
2. Register it in `workloads/registry.py`.
3. Add a SeQUeNCe adapter in `backends/sequence/workload_adapters.py`.
4. Return the shared `BackendResult` / `RequestResult` schema and finalize every
   offered transaction exactly once.
