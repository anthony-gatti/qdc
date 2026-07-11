# QPQ algorithm comparison

These are diagnostic comparisons, not paper-scale performance claims. All runs
use current upstream SeQUeNCe behavior, Bell-diagonal states, single-heralded
generation, physical loss, memory decoherence, swapping, and modeled classical
delays. Purification is disabled.

## Matched one-BSM-width results

Q-CAST uses `edge_width: 1` in these rows, so every algorithm has one midpoint
BSM per router link. This is not perfect capacity equivalence: native ODO/ACP
can time-multiplex several memory protocols through that BSM, while Q-CAST
width one reserves one paper-style lane per slot. Reported TTS includes only
successful queries. The direct and memory-constrained regimes use three seeds;
the concurrent-mesh regime uses five.

| Regime | Algorithm | Success | Mean TTS (ms) |
| --- | --- | ---: | ---: |
| Direct, cache-friendly, 3 pairs/round | ODO | 3/3 | 17.00 |
| Direct, cache-friendly, 3 pairs/round | ACP | 3/3 | 3.32 |
| Direct, cache-friendly, 3 pairs/round | Q-CAST w1 | 3/3 | 40.63 |
| Concurrent mesh, 3 clients, 5 pairs/round | ODO | 15/15 | 109.04 |
| Concurrent mesh, 3 clients, 5 pairs/round | ACP | 15/15 | 55.79 |
| Concurrent mesh, 3 clients, 5 pairs/round | Q-CAST w1 | 15/15 | 156.57 |
| Memory constrained, 2 clients, 5 pairs/round | ODO | 6/6 | 150.89 |
| Memory constrained, 2 clients, 5 pairs/round | ACP | 2/6 | 64.90 |
| Memory constrained, 2 clients, 5 pairs/round | Q-CAST w1 | 6/6 | 247.48 |

ACP is strongest when its cache has time and memory to produce useful pairs.
ODO is strongest under the tested six-memory constraint: ACP's three-memory cap
leaves too little application capacity, while Q-CAST pays slot-control latency.
No tested width-one regime makes Q-CAST fastest.

## Paper-style parallel Q-CAST links

Width three gives Q-CAST three independent midpoint BSM/channel lanes per
router link. ODO and ACP currently use one midpoint BSM, so this is a Q-CAST
scaling result rather than a hardware-matched algorithm comparison.

| Workload | ODO | ACP | Q-CAST w1 | Q-CAST w3 |
| --- | ---: | ---: | ---: | ---: |
| 3 clients, 5 pairs/round, mean TTS (ms) | 109.04 | 55.79 | 156.57 | 56.36 |
| 3 clients, 11 pairs/round, mean TTS (ms) | 265.70 | 136.53 | not run | 139.33 |

Width three reduces Q-CAST slot count and makes it competitive with ACP. It
does not consistently beat ACP in these small samples. Recovery contributes to
some delivered pairs but is not the main source of the improvement; parallel
major-path lane throughput is.

## Window sensitivity

For the width-three, 11-pair workload over three seeds:

| Generation window | Success | Mean TTS (ms) |
| --- | ---: | ---: |
| 1 ms | 8/9 | 347.70 |
| 2 ms | 9/9 | 252.60 |
| 5 ms | 9/9 | 139.33 |
| 10 ms | 9/9 | 140.76 |

Short windows spend too much time in plan and link-state coordination and may
not complete enough physical attempts. Longer windows reduce control cycles but
delay delivery until the generation phase closes. Five milliseconds is a
reasonable point for this hardware/workload, not a universal optimum.

## Reproduction

The checked-in matched configuration can be run with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-qdc \
/home/amg671/.conda/envs/qdc/bin/python -u experiments/run.py \
  --config config/qpq_qcast_matched.yaml \
  --output qpq_qcast_matched
```

The next fairness improvement is a shared multi-channel physical topology that
ODO, ACP, and Q-CAST can all use. Until then, width-three results must not be
described as an algorithm-only speedup.
