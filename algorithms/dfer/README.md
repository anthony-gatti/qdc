# DFER

This plugin implements **Distributed Entanglement Routing Scheme With Fidelity
Guarantee in Quantum Networks** (Tan et al., IEEE TNSE, 2026).

## Algorithm/runtime boundary

`algorithms/dfer` contains the simulator-neutral DLFR fidelity calculation,
pumping plan, DFPS EDR score, and configuration.  The SeQUeNCe adapter in
`backends/sequence/dfer_scheduler.py` implements local request/response state
exchange, elementary generation, pumping purification, and hop-by-hop swapping.

DFER does not use Q-CAST slots or Q-GUARD route planning.  It shares only the
common physical link-parallelism facilities and upstream SeQUeNCe protocols.

## Paper interpretations

The paper uses `l_curr` inconsistently in Algorithm 1 and Eq. 16 prints a
negative exponent for the remaining-link requirement.  The implementation
uses the surrounding prose's physically consistent definition:

```text
omega_current * omega_link ** remaining_hops >= omega_threshold
```

Eq. 22 also omits the `1/4` term and conflicts with the paper's earlier Werner
swapping equation.  Runtime swapping therefore uses upstream SeQUeNCe's
Bell-diagonal protocol.  DFER pumping consumes one fresh elementary pair per
round, as specified by the paper, rather than Q-GUARD's binary purification
tree.

DFPS considers only adjacent routers with a shorter topological distance to
the destination and a feasible DLFR plan.  It selects the greatest expected
distribution rate using physical generation rate, pumping success/cost, and
remaining swap success.

## Realistic-model additions

Neighbor state queries and responses are explicit classical messages.  Every
send includes configured endpoint processing delay and SeQUeNCe channel
propagation.  Generation, BBPSSW purification, decoherence, and swapping use
the configured physical hardware and current Bell-diagonal state.
