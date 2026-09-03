# Q-GUARD Evaluation

This is a matched evaluation of the base equal-split Q-GUARD implementation
against paper-local Q-CAST. It uses pristine SeQUeNCe v1.0.0, the common
Bell-diagonal/single-heralded runtime, and shared physical
`link_parallelism`. Q-GUARD's `edge_width` is set to that shared lane count;
it does not create hardware unavailable to any comparison algorithm.

The results below were produced on QDC commit `276482e` and SeQUeNCe commit
`ffd7c837`. They are finite deterministic-seed experiments, not claims of
universal dominance.

## Three-hop QPQ Threshold Sweep

One fixed QPQ transaction runs from `router_3` to the QDC at `router_0` on a
four-router 1 km linear topology. It requires six pairs across two rounds.
Hardware uses initial link fidelity 0.9, perfect gate/measurement efficiency,
16 memories per node, 5 s coherence, and four physical lanes per link.
The QPQ round deadline is 0.5 s.

| Fidelity threshold | Q-CAST | Q-GUARD | Q-GUARD mean TTS | Q-GUARD mean fidelity |
| --- | ---: | ---: | ---: | ---: |
| 0.70 | 10/10 | 10/10 | 10.490 ms | 0.734295 |
| 0.75 | 0/10 | 10/10 | 25.619 ms | 0.796575 |
| 0.80 | 0/10 | 10/10 | 85.352 ms | 0.834826 |
| 0.85 | 0/10 | 0/10 | -- | -- |

At 0.70 Q-GUARD invoked no purification and matched Q-CAST to simulation
precision. At 0.75 it executed 282 purification attempts (31 failures), all
on hops. At 0.80 it executed 1,005 attempts (123 failures), including 75
end-to-end purification attempts. Every run preserved locality, stayed within
the 16-memory cap, and returned all memories to `RAW`.

ODO and ACP without purification were also run at 0.70 and 0.80. At 0.80 both
missed the first round in the pilot, as did Q-CAST; Q-GUARD completed the
transaction. At 0.70, ODO completed 7/10 with 0.434 ms mean TTS, Q-CAST and
Q-GUARD completed 10/10, and ACP had no warm-cache opportunity and missed
round two in all ten runs. This is a one-request cold-cache control, not a
comparison against ACP's known repeated-traffic regime.

The 0.85 result is an important limit. Q-GUARD never delivered an invalid
pair, but it exhausted the request deadline attempting purification. This
follows the paper's specified design: EXG estimates costs with symmetric
BBPSSW rounds, while Phase 5 greedily improves the highest-fidelity realized
pair with another available pair. Four raw pairs can be analytically
width-feasible but still fail to reach the target under that asymmetric,
stochastic execution. This is a paper-level planning approximation, not a
SeQUeNCe inconsistency.

## Shared-Capacity Check

The same 0.80 QPQ workload was run with Q-GUARD under different shared lane
counts. The scheduler width exactly matched physical `link_parallelism`.

| Physical lanes/link | Success | Mean TTS | Interpretation |
| --- | ---: | ---: | --- |
| 1 | 0/5 | -- | No simultaneous purification inputs. |
| 2 | 0/5 | -- | Hop outputs remain below the equal-split target. |
| 3 | 5/5 | 78.052 ms | Final end-to-end purification can qualify outputs. |
| 4 | 10/10 | 85.352 ms | Full per-hop plan is available. |

The gain therefore depends on shared physical capacity rather than a hidden
Q-GUARD-only channel expansion.

## Path-Diverse Ring

This case has two equal three-hop arcs between `router_0` and `router_3` in a
six-router ring. It uses 10 km links, initial fidelity 0.9, four shared lanes
per link, a 0.8 threshold, and a single-pair request.

| Algorithm | Success | Mean TTS | P95 TTS | Mean fidelity |
| --- | ---: | ---: | ---: | ---: |
| Q-CAST | 0/20 | -- | -- | -- |
| Q-GUARD | 20/20 | 8.073 ms | 12.723 ms | 0.833916 |

Q-GUARD used both ring arcs: across the study it made 52 output selections on
each arc. It used no recovery path in these healthy-link runs, so this result
demonstrates concurrent candidate-path scheduling plus purification, not a
recovery-path advantage. In a five-seed all-algorithm check, ODO, ACP without
purification, and Q-CAST each completed 0/5, while Q-GUARD completed 5/5.

## Controls And Negative Result

For the non-binding six-router ring case (0.99 initial fidelity, 0.7
threshold, one shared lane), Q-CAST and Q-GUARD were exactly equivalent over
30 seeds: 30/30 success, 98.900 ms mean TTS, 0.914645 mean fidelity, and zero
purification attempts.

For five simultaneous requests from one QDC in a ten-router ring with one
shared lane, both Q-CAST and Q-GUARD completed 0/50 requests over ten seeds.
Q-GUARD does not cure Q-CAST's exclusive slot-allocation/source-adjacent-link
bottleneck.

## Conclusion

Q-GUARD has a clear operating regime: multihop requests whose raw paths miss
the fidelity threshold but have enough shared concurrent pair resources for
physical purification. It is not a universal latency improvement; it is
equivalent to Q-CAST when no purification is needed, incurs substantial
purification latency at binding thresholds, and reaches the same physical
limit under insufficient width or high fan-out contention.

The reusable runner is `experiments/run_qguard_evaluation.py`. Its JSON
reports include per-seed results, purification/recovery counts, final-memory
checks, memory-cap checks, and Q-GUARD locality checks.
