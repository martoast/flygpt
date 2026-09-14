# Compiler acquisition before saturation — Omarchy study

This supersedes the design draft for this new study only. User authorized using Omarchy alongside the unchanged M4 replication. Registered before any full-budget Omarchy run or evaluation of its new test. Prior pilot validation motivated the 256-update endpoint; no current locked M4 test has been used. This is a separate Linux cohort, never pooled with the M4 or pilot results.

## Frozen cohort

Ten new paired seeds 500–509, six conditions each: A teacher-hard; B pure temperature-2 soft KL; C hard + hidden alignment; D hard + fixed-deranged hidden alignment; E hard + relational alignment; F hard + fixed-deranged relational alignment. Preserve the v3 teacher, training/validation corpus, graph, I/O placement, recurrence, losses, coefficients, AdamW settings, learning rate and four-thread CPU/SciPy backend. All 60 models run sequentially on the same i7-8700K Linux machine. No GPU experiment or hardware comparison is part of this cohort.

Every model stops after **256 updates with batch eight**, totaling 2,048 example presentations. Save checkpoints and evaluate the unchanged 256-case validation set every 32 updates. Save the 128-update prefix, without running a separate model. Run seeds in increasing order; within each, use the unchanged Random(33000 + seed) method ordering rule. No architecture, loss or budget tuning, early success stopping or extensions. The desired 50–80% teacher-hard regime is not an exclusion criterion. Preserve ceiling, floor and failure outcomes.

Use the unchanged fixed groups, deterministic derangements and frozen sampling schedule per seed. Audit all scheduled groups in float32 and float64 above 1e-6. Failed audits stop the entire cohort, with no replacement seed or reshuffling. A disposable full-graph one-update probe with seed 90003 and one validation case verifies CPU/memory feasibility before full-budget training. It cannot select settings from accuracy. Python/package versions, graph/teacher/data hashes and probe hardware/timing are recorded and frozen. The Linux adapter only converts ru_maxrss from KiB to bytes and uses the existing local checkpoint archive; it does not change learning.

## Fresh data and test lock

Exclude every original G2c allocation, topology confirmation test, v3 pilot test and the M4 confirmation's 512 test inputs: 3,712 excluded IDs of 4,096. Select 256 of the remaining 384 with the fixed SHA256 salt in config, leaving 128 unused. Training and validation remain byte-identical. No model evaluates the new test until all 60 final checkpoints are complete and verified. Preserve and publish the plan, sources, specs, numerical audits and input hashes before full-budget training. The teacher is fixed and is not selected or requalified using this test.

After cohort completion, evaluate both 256-update primary checkpoints and saved 128-update secondary checkpoints on the same new test. The 128 results cannot choose a primary budget or rescue a failed 256 endpoint. No 512 arm is added; previous-study 512 scores are contextual, not a matched efficiency comparison.

## Analysis

Primary at 256: accuracy benefit (method minus control) and response-CE benefit (control minus method) for B−A, C−D, E−F, C−A and E−A. CE includes response EOS, unchanged from v3. The model seed is the unit, n=10; no test-case pseudoreplication. Report all pairs, means, SDs, unadjusted two-sided 95% t intervals, paired-t p-values and exhaustive two-sided sign-flip sensitivity p-values. Holm-correct the ten paired-t tests jointly at .05. Require positive mean and adjusted p<.05 for an endpoint-specific finding. A representation benefit over answers alone requires both hard-answer and shuffled comparisons on the same endpoint. Report degeneracy/zero variance explicitly. Normality assumptions remain a limitation; no universal task/architecture claim follows from this fixed teacher and corpus.

Secondary: 128-update exact accuracy and CE, validation area over updates 32–256, first observed validation crossings of 50% and 75% with intervals/right-censoring, teacher KL/agreement, gradients/state norms and numerical stability. No secondary significance fishing or promotion to primary. Distinguish unique examples from repeated presentations. A fixed-budget advantage alone does not quantify time saved; coincident threshold observations do not establish faster acquisition.

Record standalone teacher-answer generation, representation extraction where used, setup and student compute; report validation, checkpoint and test-evaluation costs. All timing comparisons are within this Linux cohort. Correct information can improve CE without establishing new exact-answer capability, calibration or unique representation necessity. Evaluate zero-edge ablation for all final checkpoints (and the stored prefix evaluator also reports it); surviving capability would qualify the causal claim.

## Preservation and stopping

Keep all checkpoint/optimizer/RNG states on Omarchy's disk. A MacBook relay copies them to mini-attached Seagate with SHA-256 verification; originals remain until verified, and local storage has a safety reserve. Connectivity failures delay remote backup, not silently discard artifacts. Push code/small results to the assigned Git branch through the MacBook relay, without installing GitHub credentials on Omarchy. Preserve all results, including negative results. Linux sleep inhibition is best-effort; record its status and keep the controller independent of SSH lifetime. No unregistered training proceeds after this cohort.
