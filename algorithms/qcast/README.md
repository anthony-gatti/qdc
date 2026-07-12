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
   `link_state_hops` away. In `qcast_distributed`, each selected major path
   enters P4 as soon as the routers on that path and its reserved recovery
   paths have completed their local P3 exchanges. Unrelated routers cannot
   delay that decision. The retained `qcast` profile waits at the historical
   network-wide barrier.
6. Successful major lanes are used directly. The distributed profile applies
   the paper's deterministic XOR over the major lane and successful,
   contention-free recovery loops. It prefers shorter recovery paths and
   allows each physical recovery lane to support at most one delivered pair.
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

`backends/sequence/qcast_scheduler.py` owns common slot timing, physical lane
allocation, control messages, rule installation, and swap sequencing.
`backends/sequence/qcast_distributed.py` is the paper-local P3/P4 policy. It
constructs each recovery decision only from state actually available in the
selected path's router-local views, filtered to that major path and its
pre-reserved recovery lanes. The coordinator schedules events on the common
SeQUeNCe timeline but has no network-wide P4 release condition. The workload
sees only the shared demand/callback contract.

Use `qcast_distributed` for paper-local experiments. `qcast` is retained as a
centralized comparison profile so previous results remain reproducible.

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
- Q-CAST P2 still uses globally consistent topology and source-destination
  inputs, as specified by the paper. The QDC router disseminates that
  deterministic plan with modeled classical delay; this occurs before pair
  generation and is not a dynamic global-link-state dependency.
- The distributed scheduler is a simulation coordinator, not a separate
  process per physical router. Its P3 message receipt, decision inputs, and P4
  release times are nevertheless router-local and auditable in diagnostics.
- Swaps run through SeQUeNCe's physical swapping protocols and are sequenced
  along the selected final path. The paper and Kotlin reference abstract P4
  swapping more coarsely and do not supply gate-level timing.
- Purification is intentionally deferred.
- Both `concurrent_pairs` and QPQ use the shared demand/callback contract. For
  QPQ, the QDC router controls Q-CAST slots while each query retains its client
  and QDC endpoints. Round two is submitted only after the exact round-one pair
  quota is delivered; round and transaction deadlines remain workload-owned.
- ODO, ACP, and Q-CAST share the same `hardware.link_parallelism` topology.
  Native RSVP still has different scheduling semantics from Q-CAST slots, so
  comparisons measure algorithms under equal physical lanes rather than an
  identical control plane.
