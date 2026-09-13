# G2: post hoc teacher KL on held-out spans

Read-only validation diagnostic added after observing the real seed-zero 256 result. Training, stopping, and test access are unchanged. Matching checkpoints: 256 and 512. Lower KL means greater teacher similarity, not necessarily better generalization.

| Condition | Seed | Step | Holdout | Student span CE | Teacher span CE | Span KL (T=1) | Span agreement |
|---|---:|---:|---|---:|---:|---:|---:|
| real_kd | 0 | 256 | validation_attribute | 2.5576 | 2.1003 | 2.4720 | 0.360 |
| real_kd | 0 | 256 | validation_relation | 2.9916 | 1.1768 | 2.7470 | 0.231 |
| real_kd | 0 | 256 | validation_combined | 2.7686 | 2.0162 | 2.7505 | 0.253 |
