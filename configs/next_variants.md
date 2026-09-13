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

Interpretation details to preserve when those experiments are designed:

- More ticks at the same per-tick leak also changes per-byte decay. Report that
  confound and consider a separately labelled time-matched control; do not
  silently rescale leak in the B arm and call it a one-factor experiment.
- Different input/output placement can change population degrees and shortest
  access paths. Record those distributions and keep population sizes fixed.
- A capacity claim needs more than a plateau: demonstrate adequate optimization
  on an easier positive control, check gradients/stability and training-versus-
  validation errors, and compare deliberate architecture changes under matched
  budgets. Even then, conclusions apply to a specified parameterization/task,
  not to the computational limits of a living fly brain.

Separately from architectural variants B–E, a matched **CE-only continuation**
is required before attributing any gain specifically to distillation rather
than additional supervised training. This objective control must use A's
initial checkpoint, additional bytes, sampling sequence, optimizer, and
architecture. Improving KL/teacher agreement establishes closer behavior on
measured inputs, not by itself a causal advantage of KL training over CE alone.
