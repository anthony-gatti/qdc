# Q-CAST-E2E

Q-CAST-E2E is an experimental ablation used to separate Q-GUARD's final
purification benefit from its per-hop planning and scheduling decisions.

It runs paper-distributed Q-CAST unchanged through:

1. major/recovery path selection;
2. reservation and memory allocation;
3. elementary generation and Boolean link-state exchange;
4. Q-CAST XOR recovery selection and entanglement swapping.

At Q-CAST's normal delivery check, an intact end-to-end pair is retained rather
than immediately rejected when it misses the request threshold. The retained
pool is then passed to the same Bell-diagonal SeQUeNCe BBPSSW implementation and
the same final end-to-end qualification loop used by Q-GUARD.

This variant does **not** exchange realized fidelity during Q-CAST link-state
collection, calculate equal-split per-hop targets, purify elementary links, or
rank recovery paths using Q-GUARD's EXG score. Consequently, Q-CAST-E2E versus
Q-CAST isolates final purification, while Q-GUARD versus Q-CAST-E2E isolates
Q-GUARD's per-hop fidelity planning, elementary purification, and EXG recovery
ranking as a group.

The scheduler emits `qcast_e2e` diagnostics. Purification counters retain their
`qguard_` prefix because the physical routine is deliberately shared rather
than copied.
