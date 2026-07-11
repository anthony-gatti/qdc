# QPQ Shared-Link-Parallelism Pilot

This report supersedes the earlier Q-CAST-only width-three comparison. Physical
link capacity is now common workload hardware:

```yaml
hardware:
  link_parallelism: 1  # or 3
```

Every `link_parallelism: 3` router link has three independent midpoint
BSM/quantum-channel lanes for ODO, ACP, and Q-CAST. Q-CAST `edge_width` only
caps how many shared physical lanes its planner schedules; it adds no hardware.

## Reproduction

The pilots use QDC branch `clean-acp-rebuild`, pristine SeQUeNCe v1.0.0,
Bell-diagonal states, single-heralded generation, 20 km links,
`link_fidelity: 0.99`, `memory_efficiency: 0.5`, 20 memories/router,
gate/measurement fidelity 0.99, swapping probability 0.9, no purification,
and three deterministic seeds (0, 1, 2). ACP uses asynchronous freshest reuse
with a four-memory cache cap.

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc /home/amg671/.conda/envs/qdc/bin/python -u \
  experiments/run.py --config config/qpq_qcast_matched.yaml \
  --output /tmp/qdc_shared_parallel_mesh_p1_final

MPLCONFIGDIR=/tmp/matplotlib-qdc /home/amg671/.conda/envs/qdc/bin/python -u \
  experiments/run.py --config config/qpq_qcast_matched_parallel3.yaml \
  --output /tmp/qdc_shared_parallel_mesh_p3_final

MPLCONFIGDIR=/tmp/matplotlib-qdc /home/amg671/.conda/envs/qdc/bin/python -u \
  experiments/run.py --config config/qpq_qcast_direct_parallel3.yaml \
  --output /tmp/qdc_shared_parallel_direct_p3_final
```

The mesh is a five-router hub-spoke topology centered on router 2 with three
extra mesh edges. Three clients each make one QPQ query, requiring five pairs
per round and ten per transaction. The direct topology has two routers and no
recovery path.

## Equal-Hardware Mesh Results

TTS and fidelity are over successful queries only.

| Lanes/link | Algorithm | Success | Mean TTS ms | Median ms | P95 ms | Mean fidelity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | ODO | 9/9 | 116.614 | 115.204 | 191.406 | 0.981272 |
| 1 | ACP freshest m4 | 9/9 | 61.235 | 59.701 | 108.703 | 0.980342 |
| 1 | Q-CAST w1 | 9/9 | 182.100 | 162.500 | 393.700 | 0.977580 |
| 3 | ODO | 9/9 | 38.101 | 35.301 | 79.602 | 0.981383 |
| 3 | ACP freshest m4 | 6/9 | 36.868 | 40.001 | 50.001 | 0.990000 |
| 3 | Q-CAST w3 | 9/9 | 61.911 | 45.300 | 147.100 | 0.977637 |

The three ACP failures are `round2_reservation_rejected`, not deadline,
fidelity, cache-cap, or ownership failures. With three lanes, native RSVP
reserves up to three application memories per active link so it can use the
shared hardware. This concurrent, 20-memory regime exposes admission control;
it is not a result adjusted to make algorithms match.

## Direct Topology

| Algorithm | Success | Mean TTS ms | Mean fidelity | Q-CAST role |
| --- | ---: | ---: | ---: | --- |
| ODO | 3/3 | 33.334 | 0.990000 | n/a |
| ACP freshest m4 | 3/3 | 11.567 | 0.990000 | n/a |
| Q-CAST w3 | 3/3 | 45.000 | 0.990000 | 18 major, 0 recovery |

Q-CAST therefore schedules the same direct link but is not latency-identical to
ODO: its explicit plan and link-state control phases are still modeled.

## Diagnostics And Invariants

Each `diagnostics.json` now includes:

- `parallel_links.links`: physical channels, attempts, and successful
  elementary generations by lane.
- `parallel_links.memory_occupancy`: high-watermark and final state by router.
- `workload_diagnostics.scheduled_lane_usage_by_link`: Q-CAST scheduled use of
  each physical BSM lane.
- `workload_diagnostics.counters`: selected paths and explicit major/recovery
  end-to-end delivery counts.

Across mesh seeds, Q-CAST selected 402 major and 113 recovery paths at one lane
and delivered 88 major plus 2 recovery-supported pairs. At three lanes it
selected 143 major and 44 recovery paths and delivered 87 major plus 3 recovery
supported pairs. Fewer paths at three lanes means fewer scheduling slots, not
less physical hardware.

All pilot memories end `RAW`. ACP's adaptive-cache high-watermark is at most 4
per router and its reservation accounting is consistent. Integration tests cover
shared three-channel visibility for all algorithms, Q-CAST memory allocation,
and final RAW state.

## Interpretation

The prior Q-CAST width-three improvement was not an algorithm-only gain: three
physical lanes were available only to Q-CAST. With shared hardware, ODO improves
from 116.614 ms to 38.101 ms and Q-CAST from 182.100 ms to 61.911 ms. ACP's
successful queries are also faster, but its admission behavior needs a focused
capacity/concurrency study.

Q-CAST uses genuine mesh recovery, but only 3 of 90 delivered three-lane pairs
use recovery. These pilots do not establish an algorithmic multipath advantage
over ODO under equal hardware. They remove the hardware confound and make
Q-CAST's remaining control overhead visible.

## Remaining Limitations

- Q-CAST uses centrally orchestrated but explicitly delayed classical control,
  not a literal distributed XOR implementation.
- RSVP and Q-CAST slots have different admission semantics despite equal lanes.
- This is a three-seed pilot, not a full statistical comparison.
