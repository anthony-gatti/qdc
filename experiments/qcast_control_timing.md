# Q-CAST Control-Timing Locality Validation

This records the centralized Q-CAST control baseline and the corrected
paper-local P3/P4 profile. It uses pristine SeQUeNCe v1.0.0 and records exact
elementary-pair success times, local P3 readiness, P4 release, swap/delivery
time, and pair age.

## Instrumentation

Q-CAST diagnostics now include `workload_diagnostics.timing`:

- `slots`: plan receipts, elementary-success time per lane, local P3 readiness
  per router, global P4 release, and P2/P3 wait components.
- `deliveries`: the oldest elementary-pair age at P2 close, global P4, and
  application delivery, plus the global-barrier excess beyond the selected
  path's local P3 readiness.
- `summary`: aggregate timing values in picoseconds.

The centralized implementation retains its existing behavior under `qcast`.
The paper-local implementation is selected as `qcast_distributed`.

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

| Profile | Remote edge | Request outcome | Target path local P3 ready | P4 start | Extra wait | Oldest pair age at P4 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| centralized | 1 km | success | 11.240 ms | 11.240 ms | 0.000001 ms | 0.905 ms |
| centralized | 1,000 km | failure before P4 | 16.235 ms | 21.230 ms | 4.995 ms | 5.900 ms |
| paper distributed | 1,000 km | success, F=0.99 | 16.235 ms | 16.235 ms | 0.000001 ms | 0.905 ms |

The long remote edge is not part of the request's quantum path. It delays the
centralized profile because that scheduler waits for every router's P3 state.
The local switch nodes have enough information `4.995 ms` earlier. With the
configured 2 ms coherence, the requested pair expires before the centralized
P4 barrier opens. The distributed profile starts P4 one simulation tick after
the selected path is locally ready, completes in `6.235 ms`, and is unaffected
by the remote report.

Reproduce both assertions with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc \
/home/amg671/.conda/envs/qdc/bin/python -m unittest \
  tests.test_qcast.QCASTSequenceIntegrationTest.test_central_p4_barrier_can_outwait_a_local_path \
  tests.test_qcast.QCASTSequenceIntegrationTest.test_paper_distributed_p4_ignores_unrelated_remote_link -v
```

## Matched Small-Mesh Check

On the existing five-router, 20 km QPQ mesh with `link_state_hops: 3`, the local
view spans the whole topology. Centralized and distributed profiles therefore
match exactly across seeds 0--2: 9/9 queries succeed, with per-seed mean TTS of
`114.000`, `243.367`, and `188.933 ms` and identical fidelity. This is the
expected reduction to centralized timing when the configured local range is
effectively network-wide.

The distributed profile also passes a forced-major-link integration test using
a physically generated recovery lane, exact two-round QPQ pair counts, memory
capacity checks, and final-RAW cleanup. Its diagnostics list every decision's
scope, visible lane states, witnesses, XOR selections, and path-specific P4
start time.
