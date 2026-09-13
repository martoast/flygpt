# Compiler benchmark report

Updated 2026-09-13T20:04:33.567137+00:00. Computational full MaleCNS experiments; no living tissue.

The substitution dataset, teacher, 128-instance primary test (`test_extension`), recurrent architecture and optimization budgets are fixed. This test was already inspected: the tournament is exploratory, not fresh confirmatory evaluation. Exact complete-answer accuracy is primary.

**Status:** running / queued. Empty result cells are pending, not failures.

Teacher-only generation audit: 2048/2048 generated training answers equal the original answers. Consequently hard-target CE has exactly the same objective as supervised CE. A separate run checks implementation equivalence; this does not establish improved sample efficiency or direct parameter translation.

## Paired CE-only replication

| Seed | MaleCNS CE exact | Rewired CE exact | Difference (pp) | MaleCNS zero-edge exact |
|---|---:|---:|---:|---:|
| 0 | 83.6% | 71.1% | +12.50 | 0.0% |
| 1 | 85.9% | 44.5% | +41.41 | 0.0% |
| 2 | 94.5% | 56.2% | +38.28 | 0.0% |
| 3 | 91.4% | 42.2% | +49.22 | 0.0% |

Seed zero reuses the original real-CE checkpoint; it is not counted as a new independent replication. Rewired seed zero is newly trained. Seeds 1–4 are fresh paired replications.

all available seeds: n=4, mean MaleCNS CE 88.9%, mean paired topology difference +35.35 pp. Seed SD: 5.00 pp (accuracy), 15.91 pp (difference). Paired gaps range from +12.50 to +49.22 pp; 4/4 are positive.

fresh seeds 1–4: n=3, mean MaleCNS CE 90.6%, mean paired topology difference +42.97 pp. Seed SD: 4.35 pp (accuracy), 5.63 pp (difference). Paired gaps range from +38.28 to +49.22 pp; 3/3 are positive.

Seed-zero topology difference is +12.50 pp under CE versus +40.62 pp under KD; their difference is -28.12 pp. This is a descriptive topology-by-objective comparison, not a replicated interaction estimate.

## Compiler outcomes

| Method / seed | Exact | CE reference exact | Difference (pp) | Response CE | Teacher KL | Zero-edge exact |
|---|---:|---:|---:|---:|---:|---:|



Batch-one objective selection: pending; full-validation exact accuracy, then CE, then name determines selection before candidate test evaluation.

C5 uses a separate batch-two cohort with ground-truth CE, hard-teacher CE and T=2 KD controls, each 512 updates / 1,024 examples. It cannot be compared to batch-one results as if optimization were identical.

Next decisions follow [DECISION_TREE.md](DECISION_TREE.md): complete every CE pair, then the frozen teacher-only compiler tournament. New graph families and harder functions are not running.

## Execution and provenance

- `results/compiler_v1/paired_ce/seed_0/rewired_ce`: 1024 updates; 7168 response symbols; 41.7 min; validation exact 78.1%.
- `results/compiler_v1/paired_ce/seed_1/real_ce`: 1024 updates; 7168 response symbols; 40.7 min; validation exact 81.2%.
- `results/compiler_v1/paired_ce/seed_1/rewired_ce`: 1024 updates; 7168 response symbols; 41.7 min; validation exact 28.1%.
- `results/compiler_v1/paired_ce/seed_2/real_ce`: 1024 updates; 7168 response symbols; 41.3 min; validation exact 96.9%.
- `results/compiler_v1/paired_ce/seed_2/rewired_ce`: 1024 updates; 7168 response symbols; 42.9 min; validation exact 50.0%.
- `results/compiler_v1/paired_ce/seed_3/real_ce`: 1024 updates; 7168 response symbols; 40.6 min; validation exact 78.1%.
- `results/compiler_v1/paired_ce/seed_3/rewired_ce`: 1024 updates; 7168 response symbols; 41.6 min; validation exact 59.4%.

Every checkpoint is preserved with hashes. Archives go to Seagate when available and to bounded local staging while disconnected; staged files are migrated and verified on reconnect. Final checkpoint references may be symlinks to conserve internal storage. Losses, hidden-state norms/saturation, gradients, optimizer state, RNG, input hashes and per-instance evaluation outputs are preserved. Source/protocol: `results/compiler_v1/frozen_plan.json` and `configs/compiler_v1.json`.

The unresolved questions are robustness across independent seeds, topology effects under CE, and whether teacher-derived objectives match or improve supervised training. “Matches” is a descriptive two-percentage-point target, not a formal noninferiority conclusion. Positive screening results alone are not robust transfer evidence.
