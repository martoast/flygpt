# Compiler v3: six-condition pilot

One-seed engineering pilot, not a replicated finding; no topology comparison in this cohort

All six models trained before opening the fresh compiler test. No statistical significance or replication claim is made.

Seed 300

| Method | Exact | Gap to hard (pp) | Response CE | KL | Zero-edge exact | Acquisition min | End-to-end min | First ≥85% validation update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_hard | 93.75% | +0.00 | 0.03380 | 0.03379 | 0.00% | 87.2 | 113.3 | 384 |
| B_soft | 100.00% | +6.25 | 0.00227 | 0.00229 | 0.00% | 87.2 | 113.2 | 384 |
| C_hidden | 100.00% | +6.25 | 0.00526 | 0.00525 | 0.00% | 115.4 | 153.4 | 384 |
| D_hidden_shuffled | 96.09% | +2.34 | 0.02093 | 0.02093 | 0.00% | 116.0 | 154.2 | 384 |
| E_relational | 100.00% | +6.25 | 0.00390 | 0.00389 | 0.00% | 87.2 | 113.3 | 384 |
| F_relational_shuffled | 95.31% | +1.56 | 0.02485 | 0.02484 | 0.00% | 87.2 | 113.3 | 384 |

A representation-specific pilot signal requires beating both A and its matched shuffled control; even then independent seeds are needed. Full learning curves and threshold intervals are preserved in comparison.json.
Acquisition time charges full standalone teacher-answer generation and, where used, extraction/cache cost. End-to-end time additionally includes validation and checkpoint I/O. Teacher pretraining is excluded. Final-test evaluation overhead is recorded separately in evaluation manifests/logs and is not a training-efficiency advantage.

Hardware amendment: C/D trained on M1; A/B/E/F on M4. Cross-machine raw times do not establish compiler efficiency. Cached teacher preparation was measured on M4 for all methods. C/D versus A is additionally hardware-confounded; real/shuffled pairs share hardware. See COMPILER_V3_HARDWARE_AMENDMENT.md.
