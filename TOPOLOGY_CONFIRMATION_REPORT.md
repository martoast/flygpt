# Topology confirmation v2 — five fresh pairs

Primary criterion passed: **True**.

| Seed | MaleCNS | Rewired | Gap (pp) |
|---|---:|---:|---:|
| 100 | 88.28% | 55.47% | +32.81 |
| 101 | 81.25% | 54.69% | +26.56 |
| 102 | 91.41% | 26.56% | +64.84 |
| 103 | 89.06% | 64.06% | +25.00 |
| 104 | 79.69% | 60.16% | +19.53 |

Mean gap +33.75 pp; 95% paired t interval [+11.38, +56.12] pp; primary p=0.013815; sign-flip sensitivity p=0.0625.

All ten checkpoints completed before test evaluation. Input and checkpoint audit passed. Full per-case CE/KL, agreement and recurrent ablations are retained in each paired result; training curves remain in progress files.
The preregistered primary criterion passed: all five paired gaps were positive, with mean MaleCNS accuracy 85.94% versus rewired 52.19%. Recurrent-edge removal reduced exact accuracy to 0% in all ten checkpoints. This supports a topology-dependent advantage for this fixed-corpus, fixed-budget substitution experiment; it does not identify a structural mechanism or establish distillation benefit.

The paired t test assumes approximately independent, normally distributed seed differences. With only five pairs, its interval and p-value require caution. The exact two-sided sign-flip sensitivity test gave p=0.0625 and does not meet 0.05; this is also its minimum possible value with five nonzero paired differences. The sensitivity test was prespecified as a supplement, not an alternative success criterion. Fresh test inputs were disjoint from all earlier allocated splits; the training and validation corpus was intentionally reused.

Hardware amendment: seeds 100–101 trained on M1; 102–104 on M4. Both conditions within each pair used the same machine. Reported effect is across these two hardware blocks; no hardware-specific significance claim. All final evaluation ran on the primary M1 after every checkpoint completed.
