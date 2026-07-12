# Algorithm Regimes

This study maps conditions under which the current ODO, ACP, and paper-local
Q-CAST implementations are preferable. It is a matched, finite pilot, not a
parameter sweep or a claim that any algorithm dominates generally.

All runs use pristine SeQUeNCe v1.0.0, Bell-diagonal states, native
single-heralded generation, `link_parallelism: 1`, and no purification.
Q-CAST refers to `qcast_distributed` throughout.

## ACP: Repeated QPQ Traffic With Warm Cache

Thirty seeds of the existing five-router hub-spoke QPQ mesh were run with
three 20 km client-to-QDC transactions per seed. The topology has three extra
mesh links; each transaction requires ten end-to-end pairs. Hardware has 20
memories per node and 0.5 memory efficiency. ACP has a four-memory adaptive
cache cap.

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc \
/home/amg671/.conda/envs/qdc/bin/python -u experiments/run.py \
  --config config/qpq_qcast_matched.yaml \
  --output /tmp/qdc_algorithm_regimes_hub_p1_final \
  --algorithms odo acp_freshest qcast_distributed \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
```

| Algorithm | Success | Mean TTS | Median TTS | P95 TTS | Mean fidelity |
| --- | ---: | ---: | ---: | ---: | ---: |
| ODO | 90/90 | 126.346 ms | 120.303 ms | 204.086 ms | 0.980084 |
| ACP freshest, cap 4 | 89/90 | 66.787 ms | 58.702 ms | 137.564 ms | 0.980253 |
| Q-CAST distributed | 90/90 | 212.627 ms | 166.750 ms | 472.095 ms | 0.976730 |

ACP is the clear latency winner for repeated QPQ traffic. Its diagnostics record
612 normalized physical background-pair reuses over the study. The one ACP
failure is a round-two deadline, so the result is a latency tradeoff rather
than an unconditional success-rate advantage.

## ODO: Direct Pair With Tight Coherence Budget

This regime has one direct 10 km request, 0.5 ms memory coherence, 0.5 memory
efficiency, and a 50--150 ms request window. ACP keeps its four-memory cache
cap; Q-CAST uses a 1 ms generation window so P2/P3/P4 fit the coherence budget.

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc \
/home/amg671/.conda/envs/qdc/bin/python -u experiments/run_algorithm_regimes.py \
  --case direct_short_coherence \
  --output /tmp/qdc_algorithm_regimes_direct_short_coherence_final \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
```

| Algorithm | Success | Mean TTS | Median TTS | P95 TTS | Mean fidelity |
| --- | ---: | ---: | ---: | ---: | ---: |
| ODO | 30/30 | 0.700 ms | 0.700 ms | 0.700 ms | 0.990000 |
| ACP freshest, cap 4 | 30/30 | 1.008 ms | 1.250 ms | 1.560 ms | 0.990000 |
| Q-CAST distributed | 30/30 | 31.200 ms | 31.200 ms | 31.200 ms | 0.990000 |

ODO wins because a direct fresh pair is inexpensive and has no cache-reuse
request/response coordination. ACP did reuse 26 normalized physical cached
pairs, but that does not outweigh its coordination and replenishment path in
this direct, low-latency workload. Q-CAST pays repeated slot-control overhead
that is not recovered when there is no path diversity.

## Q-CAST: Symmetric Multipath With Small Cache Budget

The reusable six-router ring added in this change provides two equal three-hop
routes from router 0 to router 3. One request runs from 20--160 ms on 20 km
links with 0.5 memory efficiency, 12 memories per node, and 5 s coherence.
ACP is deliberately limited to one background memory per node: this is the
small-cache regime, not a claim about ACP with a larger cache.

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc \
/home/amg671/.conda/envs/qdc/bin/python -u experiments/run_algorithm_regimes.py \
  --case ring_single_opposite \
  --output /tmp/qdc_algorithm_regimes_ring_single_final \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
```

| Algorithm | Success | Mean TTS | Median TTS | P95 TTS | Mean fidelity |
| --- | ---: | ---: | ---: | ---: | ---: |
| ODO | 30/30 | 34.281 ms | 26.701 ms | 70.352 ms | 0.904896 |
| ACP freshest, cap 1 | 30/30 | 27.534 ms | 26.701 ms | 55.972 ms | 0.900068 |
| Q-CAST distributed | 30/30 | 25.080 ms | 20.900 ms | 41.800 ms | 0.904356 |

Q-CAST is the latency winner here. Its diagnostics show 72 selected major paths
and 30 major-path deliveries: the two real ring arcs are reserved concurrently
when available, reducing the tail without adding physical channels. No recovery
path or injected link failure was needed.

## Important Negative Result: Single-Source Fan-Out

The same runner's `ring_fanout` case sends five simultaneous requests from one
QDC around a ten-router ring. The QDC still has only two adjacent physical
links. Across ten seeds, ACP completed 50/50 requests, ODO 46/50, and Q-CAST
0/50 under the configured 250 ms deadline. The result is expected from the
current Q-CAST slot model: exclusive allocation can serve at most the two
source-adjacent links in a generation slot, while each multi-hop path requires
all its elementary links to succeed together in that slot.

This is a real limitation to retain in future comparisons, not a reason to tune
the implementation toward a favorable result.

## Interpretation

- Use ACP for repeated, locality-stable QPQ traffic when enough cache memory is
  available and latency is the primary objective.
- Use ODO for direct or sparse traffic, particularly when cache coordination is
  comparable to fresh-pair establishment and deterministic latency matters.
- Use Q-CAST when requests are single-pair, the graph offers comparable
  contention-free paths, and the ACP cache budget is too small to cover the
  relevant links. It is not currently a good fit for high fan-out, multi-pair
  QPQ bursts from one QDC.

The ring is now a supported common topology value, `topology.type: ring`, for
both `concurrent_pairs` and QPQ workloads. The QPQ-specific ring contention
configuration is in `config/qpq_ring_contention.yaml`.
