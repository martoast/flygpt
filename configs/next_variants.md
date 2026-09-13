# Deliberate follow-ups after baseline A

Status: **planned only**. Do not modify the active continuation. Baseline A's
architecture and optimizer are frozen in `target_v1.json` and its checkpoint.

A stalled validation curve identifies failure of a configuration under its
budget, not an intrinsic MaleCNS capacity limit. A downward curve can support
learning/slow learning. Evidence for approaching the teacher additionally
requires a shrinking CE gap, shrinking forward KL, improving teacher agreement,
and a recurrent-edge ablation penalty. A few noisy checkpoints cannot establish
an asymptotic convergence rate.

| Arm | Only intended change relative to A | Rationale |
|---|---|---|
| A | Existing two-tick, leak-0.65, fixed-population model | Preserve a reference trajectory |
| B | More internal ticks per byte | Test propagation depth |
| C | Leak / neuron time constants | Test temporal retention |
| D | Input/output placement, still disjoint | Test access to graph pathways |
| E | Weight-based normalization / stabilization | Test optimization and dynamics |

Choose and commit exact values, initialization transformations, and budgets
before starting any arm. Reuse identical teacher, split, evaluation windows,
optimizer family, seeds, and data order. Report both token and wall-clock
budgets: extra ticks change compute. For topology claims, run the same arms on
real and degree-preserving controls, with at least five exploratory seeds and
10+ confirmatory seeds. Do not compare a tuned fly arm against an untuned null.

Variant E must preserve the causal topology. For example, normalization based
on fixed degrees or allowed incoming weight parameters preserves forward edge
support. **Global hidden-state LayerNorm would couple otherwise unconnected
neurons and is not an acceptable silent normalization change.** Any such
architecture change would need a different scientific claim and explicit
bypass audit.

No wetware claim follows from any of these digital rate-network experiments.
