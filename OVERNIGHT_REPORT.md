# G2c overnight report

Updated: 2026-09-13T09:09:08.920075+00:00. All results are computational; no living tissue was used.

This is adaptive exploratory research. The primary outcome is exact autoregressive complete-answer accuracy, including the end marker, on unseen input instances. Response CE is in nats per task symbol, not nats per byte.

The first task applies a fixed one-to-one substitution to six symbols from a four-symbol alphabet. Training has 2,048 distinct inputs; validation 256; each qualification and final-test partition has 128. Instance IDs are allocated without replacement. This tests unseen instances at a fixed length, not length generalization or natural-language ability.

**Run status: in progress.** Pending results must not be interpreted as failures or successes.

## Did we establish a teacher that genuinely generalizes?

| Task / candidate | Validation exact | Locked qualification exact | Gate |
|---|---:|---:|---|
| substitute_6/teacher_0 | 100.0% | 100.0% | PASS |

A pass establishes ≥95% observed exact accuracy on this finite held-out sample. It does not prove correctness on every possible input. Thresholds were fixed before qualification.

Baselines for **substitute_6**, on validation: GRU 100.0% exact; ngram1 0.0%, ngram3 0.0%, ngram6 0.0%. Teacher parameters: 102,656; GRU: 102,861.

## Matched locked-test results

No matched cohort has completed final testing yet. The final test remains locked until every condition reaches the same budget.

## Answers to the critical questions

- **Did MaleCNS itself generalize?** Pending full-graph held-out exact-answer evaluation.
- **Did distillation improve over CE-only?** Pending matched results.
- **Did recurrent-edge ablation destroy capability?** Pending intact-versus-zero-edge exact-answer evaluation.
- **Did biological topology differ from rewiring?** Pending matched results.
- **Which hypothesis is supported/falsified?** A qualified teacher removes the teacher-capability bottleneck for the passed task; no student-transfer conclusion yet.
- **Single highest-value next experiment:** Finish the frozen full-MaleCNS KD / CE-only / rewired KD cohort and its locked exact-answer test.

## Learning curves and provenance

Every saved progress file contains per-update loss, gradient norm, response symbols seen, hidden-state RMS/saturation, validation exact accuracy/CE/KL at fixed checkpoints and cumulative wall-clock time. Final evaluation JSON retains per-instance predictions for paired analysis.


Protocol: `configs/g2c_overnight_v1.json`. Qualification receipts, frozen cohort specs, code/input hashes, teacher/graph hashes, optimizer and sampling RNG are preserved. Checkpoints are copied and hash-verified on `/Volumes/Seagate/FlyGPT Backups/G2c-overnight/`; raw graphs/checkpoints stay out of Git. Code and small results are committed/pushed at stage boundaries.

G1 remains a finite-function encoding result, not novel-composition generalization. G2/G2b negative results remain unchanged. G2c does not support claims about living brains, natural language, or a global minimum substrate capacity.
