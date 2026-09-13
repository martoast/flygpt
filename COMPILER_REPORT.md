# Compiler benchmark report

Updated 2026-09-13T13:56:30.039541+00:00. Computational full MaleCNS experiments; no living tissue.

The substitution dataset, teacher, 128-instance primary test (`test_extension`), recurrent architecture and optimization budgets are fixed. This test was already inspected: the tournament is exploratory, not fresh confirmatory evaluation. Exact complete-answer accuracy is primary.

**Status:** running / queued. Empty result cells are pending, not failures.

Teacher-only generation audit: 2048/2048 generated training answers equal the original answers. Consequently hard-target CE has exactly the same objective as supervised CE. A separate run checks implementation equivalence; this does not establish improved sample efficiency or direct parameter translation.

## Paired CE-only replication

| Seed | MaleCNS CE exact | Rewired CE exact | Difference (pp) | MaleCNS zero-edge exact |
|---|---:|---:|---:|---:|
| pending | — | — | — | — |

Seed zero reuses the original real-CE checkpoint; it is not counted as a new independent replication. Rewired seed zero is newly trained. Seeds 1–4 are fresh paired replications.

## Compiler outcomes

| Method / seed | Exact | CE reference exact | Difference (pp) | Response CE | Teacher KL | Zero-edge exact |
|---|---:|---:|---:|---:|---:|---:|



Batch-one objective selection: pending; full-validation exact accuracy, then CE, then name determines selection before candidate test evaluation.

C5 uses a separate batch-two cohort with ground-truth CE, hard-teacher CE and T=2 KD controls, each 512 updates / 1,024 examples. It cannot be compared to batch-one results as if optimization were identical.

## Execution and provenance


All checkpoints are hash-verified on Seagate. Final local checkpoints may be symlinks to those archives to conserve internal storage. Losses, hidden-state norms/saturation, gradients, optimizer state, RNG, input hashes and per-instance evaluation outputs are preserved. Source/protocol: `results/compiler_v1/frozen_plan.json` and `configs/compiler_v1.json`.

The unresolved questions are robustness across independent seeds, topology effects under CE, and whether teacher-derived objectives match or improve supervised training. “Matches” is a descriptive two-percentage-point target, not a formal noninferiority conclusion. Positive screening results alone are not robust transfer evidence.
