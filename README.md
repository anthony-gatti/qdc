# QDC Quantum-Routing Evaluation Framework

QDC is an application-aware quantum-routing evaluation framework built on
SeQUeNCe v1.0.0. It evaluates routing algorithms under realistic workloads,
classical communication delays, memory limits, loss, decoherence, and swapping
rather than treating every entanglement request as an isolated pair.

The current application workload is Quantum Private Query (QPQ) for a Quantum
Data Center (QDC). The architecture is intended to accommodate additional
algorithms and application state machines without changing the common runtime
or result schema.

## QPQ Model

A QPQ query is a two-round transaction between a client router and the QDC.
Each round requires `2 * log2(N) + 1` Bell pairs for a database of size `N`, so
each query requires `4 * log2(N) + 2` pairs total. A query succeeds only when
both rounds deliver their exact pair quotas before their deadlines and above
the configured fidelity threshold.

The framework measures the entanglement-service layer: query success, time to
serve (TTS), pair-arrival time, fidelity, failures, memory occupancy, and
algorithm-specific diagnostics. qRAM execution and QPQ cheat-detection
operations occur after pair delivery and are outside the simulator scope.

## Supported Algorithms

All supported algorithms use the same SeQUeNCe physical topology and common
`BackendResult` / `RequestResult` schema.

| Algorithm | Configuration name | Current role |
| --- | --- | --- |
| Shortest Path On-Demand | `odo` | Native current-SeQUeNCe on-demand baseline with no pregeneration. |
| Adaptive Continuous Protocol | `acp_freshest`, `acp_random` | Continuous neighboring-pair generation, cache reuse, adaptive path feedback, and an optional purification profile. |
| Uniform Continuous Protocol | `ucp` | ACP without adaptive probability updates; retained for paper comparisons. |
| Q-CAST, centralized control | `qcast` | Historical centrally released P3/P4 control profile for reproducibility. |
| Q-CAST, paper-local control | `qcast_distributed` | Recommended Q-CAST profile. It uses explicit `k`-hop link-state messages and path-local P4 release with XOR recovery. |
| Q-GUARD | `qguard` | Fidelity-aware, paper-local Q-CAST extension with equal-split purification planning, EXG recovery ranking, physical BBPSSW, and strict final qualification. |
| DFER | `dfer` | Asynchronous hop-by-hop DLFR/DFPS routing with local state exchange, pumping purification, sequential swapping, and strict fidelity qualification. |

Q-CAST currently implements the no-purification algorithm. Its P2 plan is
deterministic from globally consistent topology and demand inputs, while its
dynamic P3/P4 link-state and swapping decisions observe the paper's locality
constraint. See [algorithms/qcast/README.md](algorithms/qcast/README.md) for
the exact SeQUeNCe boundary and known modeling limits.

Q-GUARD implements the paper's base equal-split variant. It extends the
paper-local Q-CAST control path without adding a global link-state round, then
executes purification through SeQUeNCe's official Bell-diagonal BBPSSW
protocol. See [algorithms/qguard/README.md](algorithms/qguard/README.md) for
the algorithm/runtime boundary and deliberate realistic-model differences.
The matched validation regimes and conclusions are recorded in
[experiments/qguard_evaluation.md](experiments/qguard_evaluation.md).

DFER is independent of the Q-CAST slot architecture. Each current
entanglement endpoint queries only closer adjacent routers, computes a
remaining-fidelity requirement, selects the feasible neighbor with greatest
expected EDR, and physically generates, pumps, and swaps before advancing.
See [algorithms/dfer/README.md](algorithms/dfer/README.md) for the paper's
equation ambiguities and their explicit implementation.

## Workloads And Topologies

Three workload plugins are available:

- `qpq`: two-round QPQ transactions with exact pair accounting.
- `single_pair`: the ACP paper-style single-pair validation workload.
- `concurrent_pairs`: controlled batches of independent pair requests, useful
  for algorithm and topology diagnostics.

Topology configuration supports `hub_spoke`, `linear`, and `ring` router
graphs. Every router pair has modeled classical channels. The shared hardware
parameter `hardware.link_parallelism` creates the same number of physical
link/BSM lanes for ODO, ACP, and Q-CAST; Q-CAST `edge_width` is a scheduling
limit, not extra hardware.

## Current Evidence

The current 30-seed regime study identifies real, configuration-dependent
advantages rather than a universal winner:

- ACP is fastest for repeated QPQ traffic with a warm adaptive cache.
- ODO is fastest for direct low-coherence traffic where fresh generation is
  cheaper than cache coordination.
- Paper-local Q-CAST is fastest for a single-pair request over two symmetric
  ring routes when ACP has a small cache budget.
- Q-CAST is not currently competitive for a high fan-out, multi-pair burst
  from one QDC under its exclusive slot-allocation model.

Exact configurations, commands, results, and limitations are documented in
[experiments/algorithm_regimes.md](experiments/algorithm_regimes.md).

## Repository Layout

```text
qdc/
|- algorithms/              Routing algorithm plugins and registry
|  |- acp/                  ACP and UCP configuration
|  |- odo/                  Shortest-path on-demand baseline
|  |- qcast/                SeQUeNCe-independent Q-CAST planner
|  |- qguard/               Q-GUARD configuration and fidelity-planning math
|  `- dfer/                 DFER DLFR, pumping, and DFPS math
|- workloads/               QPQ, single-pair, and concurrent-pair state machines
|- backends/sequence/       SeQUeNCe runtime, adapters, protocols, and schedulers
|- experiments/             Generic, paper, Q-CAST, and regime-study runners
|- config/                  YAML workload and algorithm configurations
|- tests/                   Unit and SeQUeNCe integration coverage
|- topology.py              Hub-spoke, linear, and ring topology generators
|- results.py               Common result schema and CSV writer
`- sweep2d.py               ODO/ACP characterization sweep runner
```

Historical ACP diagnostics and pre-plugin material are retained as reference
only. Supported commands do not import the archived ACP implementation.

## Setup

The supported Python environment is `/home/amg671/.conda/envs/qdc/bin/python`.
The sibling `SeQUeNCe` checkout is kept pristine at v1.0.0 commit `ffd7c837`.

```text
qdc_project/
|- qdc/
`- SeQUeNCe/
```

```bash
QDC_PYTHON=/home/amg671/.conda/envs/qdc/bin/python
"$QDC_PYTHON" --version
"$QDC_PYTHON" -m pip install -e /home/amg671/qdc_project/SeQUeNCe
"$QDC_PYTHON" -c 'import sequence; print(sequence.__file__)'
```

The import should resolve under `/home/amg671/qdc_project/SeQUeNCe`.

ACP execution profiles are explicit:

- `asynchronous` is the default. It updates probabilities after each served
  path and gives each background reservation its own lifetime.
- `paper_legacy` reproduces the archived 100 ms windowed update schedule,
  idle-node reward of the phantom `None` choice, and period-aligned expiry.
  Use it only when reproducing archived paper experiments.

## Running Experiments

Run the matched QPQ ODO/ACP pilot:

```bash
/home/amg671/.conda/envs/qdc/bin/python experiments/run.py \
  --config config/qpq.yaml \
  --output /tmp/qdc_qpq
```

Run ODO, ACP, and paper-local Q-CAST under equal shared hardware:

```bash
/home/amg671/.conda/envs/qdc/bin/python experiments/run.py \
  --config config/qpq_qcast_matched.yaml \
  --output /tmp/qdc_qcast_matched \
  --algorithms odo acp_freshest qcast_distributed
```

Run the reusable topology/algorithm regime pilots:

```bash
/home/amg671/.conda/envs/qdc/bin/python experiments/run_algorithm_regimes.py \
  --case ring_single_opposite \
  --output /tmp/qdc_ring_regime \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Run ACP paper-style validation:

```bash
/home/amg671/.conda/envs/qdc/bin/python experiments/run_single_pair_paper.py \
  --output /tmp/qdc_single_pair \
  --algorithms odo acp_freshest acp_random acp_purify

/home/amg671/.conda/envs/qdc/bin/python experiments/run_paper_scenario.py \
  --scenario bottleneck20 \
  --output /tmp/qdc_bottleneck20 \
  --seeds 20 \
  --algorithms odo ucp_purify acp_purify \
  --acp-execution-profile paper_legacy
```

Run the full test suite:

```bash
/home/amg671/.conda/envs/qdc/bin/python -m pytest -q
```

`sweep2d.py` remains the ODO/ACP characterization runner. Use
`experiments/run.py` for configuration-driven comparisons that include
Q-CAST, Q-GUARD, and DFER. A focused Q-GUARD purification check is available
as:

```bash
/home/amg671/.conda/envs/qdc/bin/python experiments/run.py \
  --config config/qguard_validation.yaml \
  --output /tmp/qdc_qguard_validation
```

Run the deterministic DFER DLFR, pumping, and sequential-swapping check:

```bash
/home/amg671/.conda/envs/qdc/bin/python experiments/run.py \
  --config config/dfer_validation.yaml \
  --output /tmp/qdc_dfer_validation
```

Run the reusable Q-GUARD threshold/path-diversity study:

```bash
/home/amg671/.conda/envs/qdc/bin/python \
  experiments/run_qguard_evaluation.py \
  --case ring_fidelity_bound \
  --threshold 0.8 \
  --link-parallelism 4 \
  --output /tmp/qdc_qguard_ring_fidelity
```

## Next Steps

- Benchmark Q-GUARD against ODO, ACP, and paper-local Q-CAST in fidelity-bound
  QPQ regimes with matched physical parallelism.
- Extend QPQ evaluation to controlled ring and larger path-diverse topologies,
  then evaluate concurrent client traffic rather than only well-spaced queries.
- Add realistic topology imports and hardware sensitivity studies.
- Make fidelity binding through longer links, lower initial fidelity, or deeper
  paths before comparing purification-aware algorithms.
- Add workload state machines for additional QDC applications and eventually
  heterogeneous application mixes.

## Modeling Notes

The framework intentionally models physical generation, attenuation, detector
efficiency, memory decoherence, classical propagation, endpoint processing,
and SeQUeNCe swapping. It does not yet model decoherence of the application
query registers while a QPQ transaction waits for its pairs, or the downstream
QPQ security calculation from delivered-pair fidelity.
