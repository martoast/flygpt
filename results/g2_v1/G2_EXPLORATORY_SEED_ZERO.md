# G2 exploratory seed-zero result

User-directed truncation after teacher weakness and real seed-zero outcomes; not the original five-seed stopping plan. G2 test data remain locked. All three axes are preserved in the original per-run records.

The table uses the fixed eight relation validation sentences and final 512-update checkpoints. Relation focus and span KL are post hoc diagnostics.

| Student | Relation span CE | Relation span KL |
|---|---:|---:|
| real_ce | 2.1752 | 1.9614 |
| real_kd | 2.3972 | 2.1399 |
| rewired_kd | 2.6177 | 2.3519 |

Teacher relation span CE: 1.1768. N-gram span CE: 1.5863.

CE-only minus KD differences (positive means KD improves):
- real, seed 0: CE -0.2220; KL -0.1786; both improve: False.

This single-seed validation comparison cannot establish replicated generalization or a biological topology advantage. It documents the distinction between ordinary sequence prediction and the specific held-out behavior sought for transfer.
