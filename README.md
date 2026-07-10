# QPQ Evaluation Framework

A simulation framework for evaluating entanglement routing algorithms under Quantum Private Query (QPQ) workloads on Quantum Data Center (QDC) architectures. Built on [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe).

## Project goal

Implement the QPQ protocol on a QDC and simulate realistic client workloads to evaluate how different entanglement routing algorithms perform — primarily, the range through which they can provide quality service for a sufficient fraction of their requests. The framework is meant to support a broad set of algorithms (on-demand, ACP, Q-CAST, Q-GUARD, EFiRAP, etc.) so they can be compared head-to-head under the same workload assumptions.

Long term, the framework should generalize beyond QPQ to other QDC applications (multi-party private quantum communication, distributed sensing, blind quantum computation), evaluating algorithms under heterogeneous workloads where multiple application types share the network simultaneously.

## How QPQ works

Quantum Private Queries let a client (Alice) retrieve item *j* from a server's (Bob's) database of size *N* without revealing *j*, while also detecting if Bob tries to peek at her query. The protocol exchanges entangled query registers between Alice and the QDC.

In this framework, each QPQ query is modeled as a **2-round entanglement distribution** between a client router and the QDC hub:

- **Round 1**: Alice sends a `log N`-qubit query register to Bob (consuming `log N` Bell pairs via teleportation), Bob runs his qRAM and returns a `log N + 1`-qubit response register (consuming another `log N + 1` pairs). Total round 1: `2(log N) + 1 = 2n+1` Bell pairs.
- **Round 2**: Same exchange repeats with Alice's second query register (the superposition state used for cheat detection).

Total per query: `2(2n+1) = 4n+2` Bell pairs delivered between client and QDC. A query succeeds if all pairs are delivered before the round deadlines and meet the fidelity threshold.

The framework measures **network-level cost only**: time-to-serve, fidelity, success rate. Downstream protocol details (Bob's qRAM execution, Alice's cheat-detection measurement) are out of scope — they're treated as instantaneous after pair delivery.

## What's been done so far

Validated QPQ/ODO baseline:

- **Topology generator**: central-hop and linear topologies with parametric depth, link distance, and density.
- **QPQ application**: SeQUeNCe `Application` subclass implementing the 2-round protocol structure with per-round deadlines and pair-arrival timestamping.
- **Workload generator**: produces QPQ query specs with configurable client count, queries per client, inter-query timing, and database size.
- **Backend abstraction**: the QPQ ODO backend plugs into the sweep runner through `BackendBase`.
- **Sweep infrastructure**: 2D parameter sweeps over `(distance × seed)` and `(database_size × seed)` with consistent seed counts, producing a unified per-query CSV.
- **Plotting**: six characterization charts (success-rate heatmap, TTS by distance, fidelity by hops, db-size scaling, failure decomposition, pair-arrival timeline).

Supported behavior on SeQUeNCe v1.0.0:

- **QPQ ODO** — validated on-demand shortest path with no pregeneration.
- **Single-pair ODO, UCP, and ACP** — clean algorithm/workload plugins used by the paper-validation harness.
- **ACP background purification** — Bell-diagonal BBPSSW for cached elementary pairs in the single-pair runtime.

ACP has not yet been integrated with the QPQ workload. The QPQ ACP sweep backend
therefore fails explicitly instead of silently falling back to ODO behavior.

## Repository layout

```
qdc/
├── algorithms/              # ODO and ACP algorithm configurations
├── workloads/               # Workload plugins and paper-scenario data
├── backends/
│   ├── sequence/            # SeQUeNCe runtime and ACP-aware adapter
│   └── odo_backend.py        # Validated QPQ ODO backend
├── experiments/             # Reproducible paper-validation runners
├── tests/                   # Supported runtime and workload tests
├── legacy/                  # Non-runnable pre-rebuild reference material
├── config/                  # YAML experiment configurations
├── common.py                # Time-unit constants, QPQ pair-count formulas
├── qpq_app.py               # SeQUeNCe Application implementing 2-round QPQ
├── results.py               # Result dataclasses + CSV I/O
├── sweep2d.py               # Experiment orchestration: 2D sweep + db-size sweep
├── plot.py                  # Generate all 6 charts from sweep CSVs
├── topology.py              # Hub-spoke and linear topology generation
└── workload.py              # QPQ query spec generation
```

The remaining root-level QPQ modules are the validated pre-plugin ODO baseline.
They are active, not deprecated. Their migration into the workload and SeQUeNCe
adapter packages should happen as a dedicated QPQ integration change, rather
than alongside ACP compatibility work.

## Setup

The supported environment is Python 3.12.13 with pristine SeQUeNCe v1.0.0 at
commit `ffd7c837`. ACP integration code is contained in `backends/sequence`;
there is no imported historical ACP package or patched SeQUeNCe checkout.

```
qdc_project/
├── qdc/                     # this repository
└── SeQUeNCe/                # pristine v1.0.0 checkout
```

```bash
QDC_PYTHON=/home/amg671/.conda/envs/qdc/bin/python
"$QDC_PYTHON" --version
"$QDC_PYTHON" -m pip install -e /home/amg671/qdc_project/SeQUeNCe
"$QDC_PYTHON" -c 'import sequence; print(sequence.__file__)'
```

The import must resolve under `/home/amg671/qdc_project/SeQUeNCe`.

Normal application reservations use SeQUeNCe v1.0.0's official reservation,
single-heralded generation, Bell-diagonal purification, swapping, and
notification architecture.

ACP execution profiles are explicit:

- `asynchronous` is the default algorithm profile. It updates probabilities
  after each served path and gives each background reservation its own lifetime.
- `paper_legacy` reproduces the archived experiment code's 100 ms windowed
  probability updates, idle-node reward of the phantom `None` choice, and
  period-aligned reservation expiry. The alignment is retained only for paper
  reproduction because the paper itself describes ACP as asynchronous.

## Running experiments

Paper single-pair validation:

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

Targeted tests:

```bash
/home/amg671/.conda/envs/qdc/bin/python -m unittest discover -s tests
```

QPQ ODO pilot:

```bash
python sweep2d.py --config config/default.yaml --output pilot_output --pilot
python plot.py --sweep-dir pilot_output
```

Full QPQ ODO sweep (~17 hours, run in tmux):

```bash
python sweep2d.py --config config/default.yaml --output sweep2d_final
python plot.py --sweep-dir sweep2d_final --output-dir sweep2d_final/figures
```

The default sweep is ODO only until ACP is integrated with QPQ. ACP paper
validation is run through the single-pair entrypoints above.

CLI flags:
- `--num-nodes N` — topology size (default 25, gives hop depth 1–7)
- `--seeds K` — seeds per cell (default 15)
- `--skip-primary` / `--skip-dbsize` — run only one sweep
- `--demand-diagnostics` — write per-reservation application demand JSON

Detailed diagnostics are stored below the output directory and do not expand
the primary schema-v2 CSV.

## Next steps

**Algorithm implementations**:

- **Q-CAST**: proactive multi-path entanglement distribution.
- **Q-GUARD**: fidelity-aware extension of Q-CAST with purification planning.
- **EFiRAP**: entanglement fidelity-aware routing with purification.
- **LP-based optimal baseline** for small topologies (gives an upper bound on what any heuristic could achieve).

Each new algorithm should plug in via `BackendBase` and produce results in the same CSV format, so existing sweep and plotting infrastructure works unchanged.

**Evaluation extensions**:

- **Realistic topologies**: rerun characterization on Topology Zoo backbones (SURFnet, GÉANT, Colt, etc.) instead of synthetic. Different generated topologies also likely produce different protocol orderings.
- **Push the fidelity boundary**: current sweep keeps fidelity well above the 0.7 threshold at all tested operating points, so fidelity-aware algorithms have nothing to differentiate on. Need longer distances (60+ km), lower initial fidelity, or deeper hops to find where fidelity becomes binding.
- **Concurrent client load**: current workload has well-spaced queries (6s period, single round at a time per client). Need to characterize what happens under heavy concurrent load where hub memory contention becomes the bottleneck.
- **Hardware parameter sensitivity**: sweep link generation rate, swap success probability, memory coherence time independently to understand which physical parameters most strongly determine the frontier.

**Modeling improvements**:

- **Register-coherence modeling**: the simulation tracks Bell pair fidelity but doesn't model decoherence of Alice's and Bob's local query registers while they wait for sequential pairs. Could matter at long distances or large database sizes.
- **Security metric integration**: connect per-pair fidelity to the QPQ information bound `I_B ≤ c·ε^(1/4)·log₂N` so the framework can directly report cheat-detection probability as a function of network conditions.

## Future directions

Beyond QPQ, the same QDC architecture supports other applications described in the QDC paper (Liu, Hann, Jiang 2023):

- **Multi-party private quantum communication** combines QPQ with quantum secret sharing across multiple non-cooperating QDCs.
- **Distributed sensing with data compression** uses QRAM to compress quantum data before transmission, reducing entanglement cost for sensor networks.
- **Blind quantum computation** outsources computations to QDCs without revealing what is computed.

Each application has its own workload characteristics (different pair counts per request, different fidelity tolerances, different concurrency patterns). The most interesting evaluation question is how routing algorithms hold up under **heterogeneous workloads** where multiple application types share the same QDC simultaneously. This requires generalizing the workload generator and adding application-specific backends, but the rest of the framework should carry over.
