# FlyGPT measured results

Only completed JSON records are summarized. Missing runs are not negative results.

## language

| Condition | Completed seeds | Mean validation CE |
|---|---:|---:|
| real | 1 | 3.1754 |
| gru | 1 | 1.7881 |

A_bio uses performance = -CE. Positive values favor real topology.
No completed paired real/rewired comparison yet.

## distill

| Condition | Completed seeds | Mean validation CE |
|---|---:|---:|

A_bio uses performance = -CE. Positive values favor real topology.
No completed paired real/rewired comparison yet.

## Teacher-gap continuation

| Additional steps | Validation CE | Gap to teacher |
|---:|---:|---:|
| 0 | 3.1754 | 2.8504 |
| 64 | 2.8628 | 2.5379 |
| 128 | 2.5221 | 2.1971 |

Complete: False. Single-seed feasibility only; no topology-superiority conclusion.

