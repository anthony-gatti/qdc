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
a Q-CAST control protocol. It also expands each router link into `edge_width`
independent midpoint BSM and optical-channel lanes. This is necessary because
the paper's EXT calculation assumes independent parallel channels. The
expansion is used only by Q-CAST and does not modify upstream SeQUeNCe or the
ODO/ACP topologies.

`backends/sequence/qcast_scheduler.py` owns slot timing, physical lane
allocation, control messages, rule installation, recovery selection, and swap
sequencing. The workload sees only the shared demand/callback contract.

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
- Purification and QPQ integration are intentionally deferred. The initial
  workload is `concurrent_pairs`, which supports concurrent QDC-originated Bell
  pair demands and also runs through ODO for baseline comparison.
