# Q-GUARD implementation

This package implements the base equal-split Q-GUARD algorithm from
[Fidelity-Guaranteed Entanglement Routing with Distributed Purification
Planning](https://arxiv.org/abs/2605.00246). It does not implement the paper's
Q-GUARD-WS or Q-GUARD-FP variants.

## Algorithm flow

1. Q-GUARD reuses the validated paper-local Q-CAST Phase 1 path discovery and
   Phase 2 EDA major/recovery path reservations. `edge_width` remains a
   scheduling bound over the shared physical `hardware.link_parallelism`.
2. Reserved lanes run upstream SeQUeNCe v1.0.0 single-heralded elementary-pair
   generation during the configured generation window.
3. Phase 3 link-state messages carry each reserved lane's success and realized
   fidelity in the existing local exchange. Q-GUARD adds no global control
   round.
4. Each major path begins Phase 4 as soon as its `link_state_hops`-local state
   is available. It computes the paper's Werner-parameter equal-split target.
   A detour receives the fidelity budget of the major-path span it replaces.
5. Symmetric BBPSSW planning maps each hop target to a purification depth and
   raw-pair cost. Recovery combinations that repair failed major links are
   ranked by the paper's EXG expression. Width determines analytical
   feasibility; realized pair availability contributes the `A_min` penalty.
6. Selected hops execute physical BBPSSW using SeQUeNCe's Bell-diagonal
   `BBPSSWProtocol`, including its explicit endpoint messages and classical
   propagation delay. Protocol start also includes the configured sender-side
   endpoint processing delay. The highest-fidelity available pairs are
   combined first.
7. Surviving hop pairs use the existing upstream SeQUeNCe swapping chain.
   End-to-end outputs below the application threshold may be purified again;
   only a physically present pair at or above the threshold is delivered.
8. The common workload callback records delivery, while slot cleanup returns
   every reserved memory to `RAW`.

## Integration boundary

`algorithms/qguard/__init__.py` contains configuration and pure paper math:
Werner conversion, equal/detour fidelity splitting, ideal BBPSSW planning, and
EXG. It intentionally has no SeQUeNCe resource-manager logic.

`backends/sequence/qguard_scheduler.py` is the SeQUeNCe adapter. It subclasses
the paper-local Q-CAST scheduler so Phases 1-3, explicit control messages,
physical lane reservations, and swap execution retain their validated
behavior. The subclass owns Q-GUARD Phase 4/5 state, physical purification,
strict final qualification, counters, and audit records. QPQ and
`concurrent_pairs` continue to use the common demand/callback contract and
result schema.

No SeQUeNCe source file is changed by this implementation.

## Realistic-model differences

- The paper evaluates an abstract slotted network without memory decoherence
  or operation time. This runtime uses realized SeQUeNCe fidelity, continuous
  simulation time, memory decoherence, optical loss, BSM behavior, control
  propagation, and purification messages.
- The paper's symmetric ideal BBPSSW recurrence is used to plan raw-pair cost.
  Runtime purification uses the actual fidelities held by SeQUeNCe and can
  fail stochastically. It also follows the paper's greedy asymmetric Phase 5
  rule, which repeatedly improves the highest-fidelity pair using another
  available pair. Consequently, the symmetric `2^r` cost estimate is an EXG
  planning heuristic rather than a guarantee that a width-feasible hop will
  reach its target. An underfilled route receives lower EXG but can still be
  attempted with its physically available pairs; strict final qualification
  prevents any below-threshold result from being delivered.
- When no width-feasible per-hop plan exists on an intact major path, Q-GUARD
  may swap an unqualified output and use the paper's final end-to-end
  purification stage. Recovery detours must remain width-feasible.
- SeQUeNCe marks a successful BBPSSW output `PURIFIED`; the adapter normalizes
  that resource-manager label to `ENTANGLED` after protocol completion because
  the upstream swapping protocol consumes entangled memories. The physical
  Bell-diagonal state is not rewritten.
- The scheduler is a simulation coordinator, but every dynamic decision is
  limited to messages visible within the selected major/recovery path scope.
  Diagnostics expose decision lane IDs, witnesses, and locality invariants.

Use configuration name `qguard`. `max_purification_rounds` defaults to `20`.
