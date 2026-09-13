# G2c overnight report

Updated: 2026-09-13T12:08:40.735121+00:00. All results are computational; no living tissue was used.

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

| Task / substrate | Seed | Updates | KD exact | CE-only exact | Rewired KD exact | Transfer (pp) | Topology (pp) | KD zero-edge exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| substitute / full | 0 | 1024 | 63.3% | 83.6% | 22.7% | -20.31 | +40.62 | 0.0% |
| substitute / full | 0 | 512 | 11.7% | 25.8% | 10.9% | -14.06 | +0.78 | 0.0% |
| substitute / induced_16384_edges_1p0 | 0 | 512 | 0.0% | 0.8% | 0.0% | -0.78 | +0.00 | 0.0% |
| substitute / induced_4096_edges_1p0 | 0 | 512 | 0.0% | 0.0% | 0.0% | +0.00 | +0.00 | 0.0% |

Each comparison has matched training examples/order, updates, optimizer, initialization rule, I/O populations and architecture. KD necessarily adds teacher forward-pass overhead; wall-clock times are recorded rather than claimed identical. “Full” uses all 166,700 MaleCNS neurons and 25,582,938 directed edges; induced subgraphs are explicitly labeled.

| Task / substrate / seed / updates | Condition | Response CE | Teacher KL | Exact teacher agreement | Zero-edge CE |
|---|---|---:|---:|---:|---:|
| substitute/full/0/1024 | real_kd | 0.2118 | 0.2118 | 63.3% | 1.9006 |
| substitute/full/0/1024 | real_ce | 0.0592 | 0.0592 | 83.6% | 1.9251 |
| substitute/full/0/1024 | rewired_kd | 0.6896 | 0.6896 | 22.7% | 1.7567 |
| substitute/full/0/512 | real_kd | 0.6359 | 0.6359 | 11.7% | 1.9073 |
| substitute/full/0/512 | real_ce | 0.4131 | 0.4131 | 25.8% | 1.9157 |
| substitute/full/0/512 | rewired_kd | 0.5727 | 0.5727 | 10.9% | 1.7163 |
| substitute/induced_16384_edges_1p0/0/512 | real_kd | 1.1284 | 1.1284 | 0.0% | 1.9098 |
| substitute/induced_16384_edges_1p0/0/512 | real_ce | 0.8047 | 0.8047 | 0.8% | 1.9660 |
| substitute/induced_16384_edges_1p0/0/512 | rewired_kd | 1.2007 | 1.2007 | 0.0% | 1.9275 |
| substitute/induced_4096_edges_1p0/0/512 | real_kd | 1.2899 | 1.2899 | 0.0% | 1.6834 |
| substitute/induced_4096_edges_1p0/0/512 | real_ce | 1.2444 | 1.2443 | 0.0% | 1.7938 |
| substitute/induced_4096_edges_1p0/0/512 | rewired_kd | 1.2531 | 1.2531 | 0.0% | 1.6873 |

## Answers to the critical questions

- **Did MaleCNS itself generalize?** At 1024 updates, seed 0, distilled exact accuracy is 63.3% and CE-only is 83.6% on 128 unseen instances. Partial accuracy is not mastery.
- **Did distillation improve over CE-only?** Observed paired difference: -20.31 percentage points. This comparison does not support a distillation advantage.
- **Did recurrent-edge ablation destroy capability?** Distilled exact accuracy changes from 63.3% to 0.0%; response CE changes from 0.2118 to 1.9006. Without strong intact capability, this cannot establish destruction of a mastered function.
- **Did biological topology differ from rewiring?** Observed difference: +40.62 percentage points. A single seed is insufficient evidence of a robust topology advantage.
- **Which hypothesis is supported/falsified?** The current fixed training configuration has not shown a KD generalization advantage. This does not falsify representability or all possible compilation procedures.
- **Single highest-value next experiment:** Use matched training/validation probes and the preplanned budget extension to distinguish basic learnability from KD-specific failure.

## Learning curves and provenance

Every saved progress file contains per-update loss, gradient norm, response symbols seen, hidden-state RMS/saturation, validation exact accuracy/CE/KL at fixed checkpoints and cumulative wall-clock time. Final evaluation JSON retains per-instance predictions for paired analysis.

- `results/g2c_overnight/substitute_6/full/seed_0/real_ce`: 1024 updates, 7,168 response symbols, 40.9 min; latest validation exact 81.2%, CE 0.0879.
- `results/g2c_overnight/substitute_6/full/seed_0/real_kd`: 1024 updates, 7,168 response symbols, 40.9 min; latest validation exact 62.5%, CE 0.3055.
- `results/g2c_overnight/substitute_6/full/seed_0/rewired_kd`: 1024 updates, 7,168 response symbols, 42.1 min; latest validation exact 28.1%, CE 0.6057.
- `results/g2c_overnight/substitute_6/induced_16384_edges_1p0/seed_0/real_ce`: 512 updates, 3,584 response symbols, 0.3 min; latest validation exact 0.0%, CE 0.7730.
- `results/g2c_overnight/substitute_6/induced_16384_edges_1p0/seed_0/real_kd`: 512 updates, 3,584 response symbols, 0.3 min; latest validation exact 0.0%, CE 1.1185.
- `results/g2c_overnight/substitute_6/induced_16384_edges_1p0/seed_0/rewired_kd`: 512 updates, 3,584 response symbols, 0.3 min; latest validation exact 0.0%, CE 1.1994.
- `results/g2c_overnight/substitute_6/induced_4096_edges_1p0/seed_0/real_ce`: 512 updates, 3,584 response symbols, 0.1 min; latest validation exact 0.0%, CE 1.2436.
- `results/g2c_overnight/substitute_6/induced_4096_edges_1p0/seed_0/real_kd`: 512 updates, 3,584 response symbols, 0.1 min; latest validation exact 0.0%, CE 1.2917.
- `results/g2c_overnight/substitute_6/induced_4096_edges_1p0/seed_0/rewired_kd`: 512 updates, 3,584 response symbols, 0.1 min; latest validation exact 0.0%, CE 1.2488.

Protocol: `configs/g2c_overnight_v1.json`. Qualification receipts, frozen cohort specs, code/input hashes, teacher/graph hashes, optimizer and sampling RNG are preserved. Checkpoints are copied and hash-verified on `/Volumes/Seagate/FlyGPT Backups/G2c-overnight/`; raw graphs/checkpoints stay out of Git. Code and small results are committed/pushed at stage boundaries.

G1 remains a finite-function encoding result, not novel-composition generalization. G2/G2b negative results remain unchanged. G2c does not support claims about living brains, natural language, or a global minimum substrate capacity.
