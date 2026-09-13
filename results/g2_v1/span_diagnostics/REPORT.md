# G2: post hoc teacher KL on held-out spans

Read-only validation diagnostic added after observing the real seed-zero 256 result. Training, stopping, and test access are unchanged. Matching checkpoints: 256 and 512. Lower KL means greater teacher similarity, not necessarily better generalization.

| Condition | Seed | Step | Holdout | Student span CE | Teacher span CE | Span KL (T=1) | Span agreement |
|---|---:|---:|---|---:|---:|---:|---:|
| real_ce | 0 | 256 | validation_attribute | 2.5077 | 2.1003 | 2.3314 | 0.360 |
| real_ce | 0 | 256 | validation_relation | 2.8575 | 1.1768 | 2.5497 | 0.231 |
| real_ce | 0 | 256 | validation_combined | 2.6680 | 2.0162 | 2.5898 | 0.242 |
| real_ce | 0 | 512 | validation_attribute | 1.8107 | 2.1003 | 1.6513 | 0.260 |
| real_ce | 0 | 512 | validation_relation | 2.1752 | 1.1768 | 1.9614 | 0.212 |
| real_ce | 0 | 512 | validation_combined | 2.0723 | 2.0162 | 2.0750 | 0.232 |
| real_kd | 0 | 256 | validation_attribute | 2.5576 | 2.1003 | 2.4720 | 0.360 |
| real_kd | 0 | 256 | validation_relation | 2.9916 | 1.1768 | 2.7470 | 0.231 |
| real_kd | 0 | 256 | validation_combined | 2.7686 | 2.0162 | 2.7505 | 0.253 |
| real_kd | 0 | 512 | validation_attribute | 1.9051 | 2.1003 | 1.8721 | 0.260 |
| real_kd | 0 | 512 | validation_relation | 2.3972 | 1.1768 | 2.1399 | 0.212 |
| real_kd | 0 | 512 | validation_combined | 2.2463 | 2.0162 | 2.2677 | 0.232 |
| rewired_kd | 0 | 256 | validation_attribute | 2.6543 | 2.1003 | 2.5399 | 0.140 |
| rewired_kd | 0 | 256 | validation_relation | 3.3068 | 1.1768 | 3.3039 | 0.154 |
| rewired_kd | 0 | 256 | validation_combined | 2.9471 | 2.0162 | 3.1349 | 0.141 |
| rewired_kd | 0 | 512 | validation_attribute | 1.9119 | 2.1003 | 1.8584 | 0.360 |
| rewired_kd | 0 | 512 | validation_relation | 2.6177 | 1.1768 | 2.3519 | 0.212 |
| rewired_kd | 0 | 512 | validation_combined | 2.3535 | 2.0162 | 2.3822 | 0.253 |
