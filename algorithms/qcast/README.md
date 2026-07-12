# Q-CAST implementation

This package implements Q-CAST without purification. It is split into a
SeQUeNCe-independent planner and a SeQUeNCe execution adapter.

## Slot flow

1. Requests originate at the workload's QDC node and enter one slot batch.
2. `planner.py` runs greedy EDA using the paper's expected-throughput metric.
   Major paths reserve node memories and physical edge lanes without
   contention. Recovery paths are then reserved from residual resources for
   major-path segments up to `link_state_hops`.
3. The QDC sends every router an explicit local plan message. The message
   carries lane, midpoint BSM, neighbor, role, and memory-index assignments.
   Sender processing and classical propagation delays are modeled.
4. Each assigned lane runs upstream SeQUeNCe v1.0.0 single-heralded elementary
   generation for `generation_window_ps`. Failed attempts retry through normal
   resource-manager rules until the window closes.
5. Routers exchange explicit link-state messages with nodes up to
   `link_state_hops` away. P4 waits for the worst modeled processing and
   propagation delay.
6. Successful major lanes are used directly. Failed major edges may be
   replaced only by complete, successful, non-overlapping recovery lanes that
   were reserved for the affected segments.
7. Multi-hop paths use upstream `EntanglementSwappingA/B`. Delivery is recorded
   only after the endpoint memories confirm the expected remote memory owners
   and the fidelity threshold is met.
8. Slot resources are returned to `RAW`; unfinished demands enter another slot
   until completion or deadline.

## SeQUeNCe boundary

`backends/sequence/qcast_topology.py` supplies an ACP-free router subclass with
a Q-CAST control protocol. Physical parallelism is a shared workload hardware
parameter, `hardware.link_parallelism`; the common SeQUeNCe adapter expands all
algorithms' router links into that many independent midpoint BSM/channel lanes.
Q-CAST's `edge_width` is only a path-scheduling cap over the shared lanes. It
does not create hardware.

`backends/sequence/qcast_scheduler.py` owns slot timing, physical lane
allocation, control messages, rule installation, recovery selection, and swap
sequencing. The workload sees only the shared demand/callback contract.

`experiments/qcast_control_timing.md` records timing instrumentation for the
current centralized baseline and the regression scenario for the distributed
P3/P4 port.

## Deliberate differences and current limits

- Physical generation, loss, detector behavior, memory decoherence, classical
  propagation, processing delay, and swapping are event-driven SeQUeNCe
  behavior. The reference Kotlin simulator treats phases more abstractly.
- The analytical per-window link probability follows native v1.0.0
  single-heralded semantics and is an estimate; diagnostics report all inputs
  needed to compare it with observed lane yield.
- A central scheduler deterministically orchestrates P4 after all modeled
  k-hop reports arrive. Recovery choices are constrained to the same reserved
  segment information, but this is not a node-by-node implementation of the
  paper's distributed XOR procedure.
- Purification is intentionally deferred.
- Both `concurrent_pairs` and QPQ use the shared demand/callback contract. For
  QPQ, the QDC router controls Q-CAST slots while each query retains its client
  and QDC endpoints. Round two is submitted only after the exact round-one pair
  quota is delivered; round and transaction deadlines remain workload-owned.
- ODO, ACP, and Q-CAST share the same `hardware.link_parallelism` topology.
  Native RSVP still has different scheduling semantics from Q-CAST slots, so
  comparisons measure algorithms under equal physical lanes rather than an
  identical control plane.
