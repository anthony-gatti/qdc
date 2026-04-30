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

Framework and infrastructure:

- **Topology generator**: hub-spoke and linear topologies with parametric depth, link distance, and density.
- **QPQ application**: SeQUeNCe `Application` subclass implementing the 2-round protocol structure with per-round deadlines and pair-arrival timestamping.
- **Workload generator**: produces QPQ query specs with configurable client count, queries per client, inter-query timing, and database size.
- **Backend abstraction**: each routing algorithm implements `BackendBase` and plugs into the same experiment runner.
- **Sweep infrastructure**: 2D parameter sweeps over `(distance × seed)` and `(database_size × seed)` with consistent seed counts, producing a unified per-query CSV.
- **Plotting**: six characterization charts (success-rate heatmap, TTS by distance, fidelity by hops, db-size scaling, failure decomposition, pair-arrival timeline).

Algorithms implemented:

- **ODO** (on-demand shortest path) — baseline
- **ACP** (continuous pre-generation) — via the ACP fork of SeQUeNCe

A first round of evaluation has been run for ODO and ACP across 4 link distances (10/20/30/40 km), 7 hop counts (1–7), 4 database sizes (n = 5, 10, 15, 20), with 15 seeds per configuration. See `sweep2d_final/figures/` for results.

## Repository layout

```
qdc_eval/
├── backends/                # Backend interface + algorithm implementations
├── config/                  # YAML experiment configurations
├── sweep2d_final/           # Output of main 2D sweep (CSVs + figures)
├── common.py                # Time-unit constants, QPQ pair-count formulas
├── qpq_app.py               # SeQUeNCe Application implementing 2-round QPQ
├── results.py               # Result dataclasses + CSV I/O
├── sweep2d.py               # Experiment orchestration: 2D sweep + db-size sweep
├── plot_all.py              # Generate all 6 charts from sweep CSVs
├── topology.py              # Hub-spoke and linear topology generation
└── workload.py              # QPQ query spec generation
```

## Setup

This framework depends on two upstream repositories that are not included here. Expected directory layout:

```
qdc/
├── qdc_eval/                # this repo
├── SeQUeNCe/                # https://github.com/sequence-toolbox/SeQUeNCe
└── acp/                     # ACP fork of SeQUeNCe
```

`backends/acp_backend.py` adds `../acp/` to `sys.path` at import.

```bash
python -m venv qdc_env
source qdc_env/bin/activate

cd ../SeQUeNCe && pip install -e .
pip install numpy networkx matplotlib pyyaml
```

Python 3.11 used during development.

## Running experiments

Quick run (~30 min):

```bash
python sweep2d.py --config config/default.yaml --output pilot_output --pilot
python plot.py --sweep-dir pilot_output
```

Full sweep (~17 hours, run in tmux):

```bash
python sweep2d.py --config config/default.yaml --output sweep2d_final
python plot.py --sweep-dir sweep2d_final --output-dir sweep2d_final/figures
```

Default sweep is 4 distances × 15 seeds × 2 backends + a database-size sub-sweep at 20 km.

CLI flags:
- `--num-nodes N` — topology size (default 25, gives hop depth 1–7)
- `--seeds K` — seeds per cell (default 15)
- `--skip-primary` / `--skip-dbsize` — run only one sweep

## Next steps

**Algorithm implementations**:

- **Q-CAST**: proactive multi-path entanglement distribution.
- **Q-GUARD**: fidelity-aware extension of Q-CAST with purification planning.
- **EFiRAP**: entanglement fidelity-aware routing with purification.
- **LP-based optimal baseline** for small topologies (gives an upper bound on what any heuristic could achieve).

Each new algorithm should plug in via `BackendBase` and produce results in the same CSV format, so existing sweep and plotting infrastructure works unchanged.

**Evaluation extensions**:

- **Realistic topologies**: rerun characterization on Topology Zoo backbones (SURFnet, GÉANT, Colt, etc.) instead of synthetic hub-spoke. Tree-structured topologies and meshes likely produce different protocol orderings.
- **Push the fidelity boundary**: current sweep keeps fidelity well above the 0.7 threshold at all tested operating points, so fidelity-aware algorithms have nothing to differentiate on. Need longer distances (60+ km) or deeper hops to find where fidelity becomes binding.
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