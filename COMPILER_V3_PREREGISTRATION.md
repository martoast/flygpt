# Compiler v3 — six-condition engineering pilot

Question: can internal information from a fixed pretrained TinyGPT teacher improve acquisition of its substitution function in full MaleCNS beyond teacher-hard answers and matched shuffled-representation controls?

This is one seed (300), six methods on the same M4 Mac mini. It is a method-screening pilot, not a replicated scientific finding. A positive signal requires a separately frozen multi-seed replication. This choice limits the initial compute commitment; no one-seed significance claim will be made.

## Conditions and fixed budget

| Condition | Training objective |
|---|---|
| A_hard | CE against teacher-generated exact answers |
| B_soft | Pure teacher-to-student KL, temperature 2, multiplied by T²; no hard CE loss term |
| C_hidden | Hard CE + normalized hidden alignment through learned linear projection |
| D_hidden_shuffled | Same as C, with fixed deranged teacher-example correspondence |
| E_relational | Hard CE + off-diagonal cosine-geometry alignment through learned linear projection |
| F_relational_shuffled | Same as E, with fixed deranged teacher-example correspondence |

The alignment coefficient is 1. Projections map the disjoint output-neuron population to the teacher hidden width (64); C–F use the same projection initialization rule. They are auxiliary training heads discarded at inference, never decoder bypasses. The original full MaleCNS topology, rate dynamics, I/O populations, optimizer, learning rate, clipping and initialization rules remain those of the reference specification. No recurrent edge is added.

Each condition receives batch eight × 512 updates = 4,096 example presentations. Entire optimizer updates, not individual examples, are the stopping unit. Method order is shuffled once using Python Random(33000 + seed), before any method runs. No hyperparameter search is performed. The previous batch-one or batch-two results are historical context, not matched baselines for this cohort. B is explicitly pure KL and is not the old mixed CE/KL condition.

All six student training paths consume only the frozen teacher-generated target file. No original ground-truth training answers enter their losses. Teacher-hard generation is replayed from prompts only, checked against the frozen target artifact, timed and never repaired from ground truth. The teacher was already qualified on this task; new test performance cannot select or gate this screen.

## Fixed shuffled correspondence and numerical qualification

Partition the 2,048 training examples into fixed groups of eight using generator seed 125000 + model seed. For each group create one frozen single-cycle permutation using seed 175000 + model seed. Every example receives a different example's teacher state, always from the same donor whenever that group recurs. No per-update reshuffling occurs. The cycle also avoids preserving unordered example pairs. Every method uses the same groups and the same group-sampling schedule (seed 75000 + model seed), sampled with replacement across updates.

Cache the teacher's final normalized-layer hidden states and logits on teacher-generated sequences. For every unique group used in the complete frozen schedule, audit the response-position cosine matrices:

`|| S_T - P S_T P^T ||_F > 1e-6`

The Frobenius norm is taken over all response positions and example-pair entries together. Require the criterion in both float64 audit arithmetic and float32 training arithmetic. Also require a nonzero normalized hidden-target difference above the same epsilon. Record all group norms and the coverage of every scheduled batch. Preserve each group's full feature distribution, scale and dimensions through permutation. If any audit fails, stop before student training; do not resample groups, modify permutations or relax epsilon to pass.

## Data and test lock

Keep the original training and 256-example validation data. Choose 128 new test IDs by the fixed SHA256 salt in the configuration, excluding every original train/validation/qualification/test allocation AND all 128 topology-confirmation test IDs. No original or confirmation final test is reused. Record hashes and a mechanical leakage audit without displaying new test cases.

All six specs, schedules, cache hashes, audits, source files, endpoints and stopping rules must be frozen and published before full-budget student training. A single-update full-graph memory probe with a disposable model seed is allowed solely to confirm batch-eight feasibility; it is not a candidate checkpoint and cannot select hyperparameters by performance. If it fails for memory, stop and document the failure rather than silently reduce the batch. Final test evaluation is locked until every six-condition checkpoint for this pilot finishes. No final-test-driven changes are permitted.

## Endpoints and timing

Primary: exact-answer accuracy difference to A at the fixed budget. For representation-specific evidence, additionally report C−D and E−F; a candidate needs both an advantage over A and over its matched shuffled control. These are descriptive single-seed contrasts, not proof of a general transfer effect.

Efficiency: first observed accuracy ≥85% on all 256 validation cases at updates 32, 64, 128, 256, 384 and 512. Report the bracket between observation checkpoints, examples presented, acquisition seconds and end-to-end wall seconds. If the threshold is never reached, report right-censoring beyond 512 updates. Do not infer exact crossing times between observations or extend a run to reach threshold. Later drops are visible in the full curve.

Acquisition cost includes full one-time teacher answer generation (all methods), teacher checkpoint setup, full representation/logit extraction and caching (B–F), student setup, feature retrieval, projection/alignment and forward/backward/optimizer work. Teacher pretraining is excluded because the teacher is given. Explicitly report end-to-end wall time including validation and checkpoint I/O as well. Cached artifacts are reused operationally, but the primary standalone acquisition accounting charges their full method-appropriate preparation cost; amortization is secondary and must be labeled.

Secondary: normalized validation accuracy learning-curve area over the observed checkpoint range, response CE, teacher KL/agreement, final generation exactness, hidden norms/saturation, gradient norms and zero-edge ablation. An accuracy ratio to A is optional and undefined when A is zero. Report all six conditions and failures, not only a selected winner.

## Preservation and separation

Use a separate compiler branch/worktree and `results/compiler_v3/`. Preserve cache, all checkpoint/optimizer/sampling state, group correspondence, timing components, provenance and evaluations on the mini-attached Seagate. Push code and small artifacts through the working SSH/Git relay when direct GitHub authentication is unavailable. Do not modify or pool with the completed topology confirmation. Topology mechanism work is a separate study.
