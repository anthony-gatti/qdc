# Bottleneck20 ACP divergence diagnosis

Date: 2026-07-09

## Reconstructed reference

- ACP source: archived commit `f533c8b` (2024-11-28).
- SeQUeNCe source: commit `5826c443`, the v0.6.5-era simulator used by the
  archived ACP code.
- A detached run of seed 0 reproduced every archived request TTS and fidelity
  exactly. Its mean ACP-purify TTS was 5.071431597 ms.
- Current runtime: pristine SeQUeNCe v1.0.0, commit `ffd7c837`.

No physical parameter was calibrated during this diagnosis.

## Root causes

1. The clean port decremented `adaptive_memory_used` when cache reuse or
   purification recycled a qubit, although the ACP reservation, timecard, and
   generation rule remained active. This created uncounted reservations and
   starved the bottleneck link. ACP slots are now released once, at reservation
   expiry. **Classification: correctness bug in the clean implementation.**
2. The archived implementation did not use the paper's per-served-path update
   literally. It updated every 100 ms and rewarded phantom `None` at nodes with
   no relevant path. This suppresses traffic from unused leaves.
   **Classification: archived-code behavior that is ambiguous in the paper.**
3. The archived implementation aligned every ACP reservation expiry to a global
   100 ms epoch. Simultaneous free slots let `router_9` and `router_10` accept
   one another. Independent asynchronous lifetimes rarely free both endpoints
   together under leaf pressure. **Classification: archived-code artifact that
   conflicts with the paper's asynchronous description.**
4. The archived bottleneck workload contains 110 requests, switches after
   request 55, and resets the traffic RNG at the phase boundary. The prior
   harness used 100 requests, switched after 50, and continued the RNG stream.
   **Classification: configuration mismatch.**
5. Cache preflight previously delayed every fresh miss. Preflight and normal
   protocol pairing now run concurrently; only an edge with a pending cache
   confirmation waits. **Classification: correctness bug in the clean cache
   integration.**

Items 2 and 3 are exposed as `paper_legacy`. The normal ACP default remains
`asynchronous`; the synchronized behavior is not presented as the realistic
interpretation of the paper's asynchronous model.

## Full 20-seed result

Each row contains 2,200 successful requests. Purification is enabled for UCP
and ACP. Values in `Archived` come from the checked-in raw paper runs.

| Algorithm | Runtime | Mean TTS (ms) | Median | p95 | p99 | Mean fidelity |
|---|---|---:|---:|---:|---:|---:|
| ODO | Archived | 12.4675 | 10.8602 | 26.9105 | 34.1165 | 0.846967 |
| ODO | v1.0.0 | 12.4084 | 10.7102 | 27.0605 | 35.6106 | 0.846863 |
| UCP | Archived | 10.2176 | 8.6101 | 24.5180 | 33.5107 | 0.837290 |
| UCP | v1.0.0 | 10.3643 | 8.6101 | 25.1104 | 34.8606 | 0.837148 |
| ACP | Archived | 5.2929 | 2.6100 | 16.8603 | 27.5105 | 0.851732 |
| ACP | v1.0.0 | 5.0543 | 2.5350 | 16.1103 | 25.7105 | 0.851162 |

ACP phase means are 5.3214 ms before and 4.7871 ms after the traffic change;
the archived values are 5.7800 ms and 4.8058 ms. The current ACP result is
4.5% faster overall, a small current-SeQUeNCe/stochastic difference rather than
the previous functional divergence.

A seed-0 run with the realistic default `asynchronous` profile completed all
110 requests at 10.7770 ms mean TTS. Its accounting invariants passed, but it
generated fewer useful central-link pairs because independently expiring slots
at the two bottleneck endpoints rarely became available together. This result
shows that the remaining paper/profile difference is caused by the archived
synchronization policy, not by cache reuse, purification, or a memory leak.

Current ACP cached-link counts per request were `{0: 24, 1: 350, 2: 825,
3: 1001}`. It reused 5,003 physical cached link pairs, including 1,857 on the
central bottleneck link. All node high-water marks were at or below five, and
all end-of-run counters matched live ACP reservations.

## Validation commands

```bash
/home/amg671/.conda/envs/qdc/bin/python -m unittest test_paper_workload.py

/home/amg671/.conda/envs/qdc/bin/python experiments/run_single_pair_paper.py \
  --output /tmp/qdc_single_pair_final --requests 100 \
  --algorithms odo acp_freshest acp_random acp_purify

/home/amg671/.conda/envs/qdc/bin/python experiments/run_paper_scenario.py \
  --scenario bottleneck20 --output OUTPUT --seeds 20 \
  --algorithms odo ucp_purify acp_purify \
  --acp-execution-profile paper_legacy
```

The ODO QPQ pilot passes. The ACP QPQ pilot remains intentionally unsupported
by `CleanAlgorithmBackend`; QPQ integration is the next separate milestone.

## Output locations

- Full paper-profile results: `/tmp/qdc_bottleneck20_final_5seed` and
  `/tmp/qdc_bottleneck20_final_seed5_19` (with ODO seeds 0-4 in
  `/tmp/qdc_bottleneck20_odo_5seed`).
- Final two-node run: `/tmp/qdc_single_pair_final`.
- Final asynchronous seed-0 diagnostic:
  `/tmp/qdc_bottleneck20_async_accounting_fixed_seed0`.
- Reconstructed archived seed-0 run: `/tmp/qdc_original_seed0`.
