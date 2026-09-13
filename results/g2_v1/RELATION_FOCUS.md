# G2 relation-span comparison at 512

Post hoc focus on the fixed eight relation validation sentences. This does not replace the three-axis protocol or create a new confirmatory endpoint. Missing controls remain pending.

| Model | Seed | Span CE | Span KL to teacher |
|---|---:|---:|---:|
| Teacher | 0 | 1.1768 | 0 |
| N-gram order 4 | — | 1.5863 | not measured |
| real_ce | 0 | 2.1752 | 1.9614 |
| real_kd | 0 | 2.3972 | 2.1399 |
| rewired_kd | 0 | 2.6177 | 2.3519 |

Positive CE-only minus KD differences in both span CE and span KL would support beneficial distillation on these validation positions. Replication and untouched evaluation are needed for a generalization claim. Teacher advantage over the n-gram alone does not establish teacher superiority over retrieval or a simple RNN.
