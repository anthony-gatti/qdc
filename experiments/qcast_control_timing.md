# Q-CAST Control-Timing Baseline

This records the current centralized Q-CAST control baseline before a
paper-style distributed P3/P4 implementation. It uses pristine SeQUeNCe v1.0.0
and records exact elementary-pair success times, local P3 readiness, the global
P4 barrier, swap/delivery time, and pair age.

## Instrumentation

Q-CAST diagnostics now include `workload_diagnostics.timing`:

- `slots`: plan receipts, elementary-success time per lane, local P3 readiness
  per router, global P4 release, and P2/P3 wait components.
- `deliveries`: the oldest elementary-pair age at P2 close, global P4, and
  application delivery, plus the global-barrier excess beyond the selected
  path's local P3 readiness.
- `summary`: aggregate timing values in picoseconds.

The centralized implementation retains its existing behavior. These diagnostics
are a baseline, not a distributed approximation.

## Five-Node QPQ Mesh

The matched three-seed QPQ pilots use 20 km links, 5 s coherence, 0.1 ms
per-message processing, and `link_state_hops: 3`. Each local P3 view spans the
entire five-router mesh, so the centralized global barrier is ready only 1 ps
after each delivered path's local P3 view.

| Lanes/link | Delivered pairs | Slots | Mean P2 residual age | P3 control wait | Global-barrier excess |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 90 | 148 | 2.941 ms | 0.300 ms | 0.000001 ms |
| 3 | 90 | 55 | 2.900 ms | 0.300 ms | 0.000001 ms |

The residual P2 age is the time between an elementary pair's actual success and
the end of the fixed generation window. It is not a centralized-control cost.
This topology therefore offers no evidence for or against the benefit of local
P4 decisions.

Outputs:

- `/tmp/qcast_timing_mesh_p1`
- `/tmp/qcast_timing_mesh_p3`

## Heterogeneous-Control Stress Case

The regression test constructs an eight-router linear network with a one-hop
request from router 0 to router 1. A remote, unused router 6--7 edge is either
1 km or 1,000 km. The target path has 2 ms memory coherence, a 1 ms generation
window, and `link_state_hops: 1`.

| Remote edge | Request outcome | Target path local P3 ready | Global P4 | Extra global wait | Oldest pair age at P4 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 km | success | 11.240 ms | 11.240 ms | 0.000001 ms | 0.905 ms |
| 1,000 km | failure before P4 | 16.235 ms | 21.230 ms | 4.995 ms | 5.900 ms |

The long remote edge is not part of the request's quantum path. It delays the
current implementation because the centralized scheduler waits for every
router's P3 state. The local switch nodes have enough information about 5 ms
earlier. With the configured 2 ms coherence, the requested pair expires before
the centralized P4 barrier opens.

Reproduce the central-baseline assertion with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc \
/home/amg671/.conda/envs/qdc/bin/python -m unittest \
  tests.test_qcast.QCASTSequenceIntegrationTest.test_central_p4_barrier_can_outwait_a_local_path -v
```

## Implication

This gives a concrete paper-style distributed-Q-CAST target: after P2, routers
on the selected major/recovery paths should act when their own required `k`-hop
P3 state is ready, rather than at the network-wide maximum. The eventual port
must preserve the same physical channels, classical propagation, processing
delays, and SeQUeNCe swapping, then demonstrate that the 1,000 km remote edge
does not delay the router 0--1 decision.
