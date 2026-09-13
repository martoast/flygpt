# Topology confirmation v2 — registered before new training or test evaluation

The exploratory study found five positive paired gaps (mean +35.16 percentage points). It motivates this independent-seed confirmation; it is not included in the primary confirmatory estimate.

## Frozen comparison

Five fresh model seeds 100–104 compare full MaleCNS CE against directed degree-preserving rewired CE. Rewiring seeds are 877–881 respectively, using the unchanged generator with ten successful swaps per edge and existing self-loops preserved. No graph is selected by performance or spectral properties.

Keep the original six-symbol substitution task, training and validation files byte-identical, model architecture, I/O assignment, optimizer, initialization rules, sampling rule (75000 + model seed), batch one, and 1,024-update budget. Use the final checkpoint, regardless of validation. The original teacher remains a diagnostic reference; its score cannot gate or select these CE runs.

## New locked test

Exclude all IDs allocated to ANY split in the original dataset manifest, including both qualification splits and all three test splits. From the remaining 1,152 domain IDs choose 128 by SHA256 ordering with the salt in the configuration. Generate answers with the unchanged substitution function. Record the file hash and exclusion audit without displaying cases. The test file is excluded from Git and copied to external backup; hashing and mechanical leakage audits are permitted. No model is evaluated on it until all ten training checkpoints have completed and every frozen input has been verified. All model and analysis choices are committed and pushed before training.

This locks model evaluation, not cryptographic access: the local user/agent can access the file. Prior training/validation data are intentionally reused. Claims are conditional on that corpus and fixed sequence length.

## Primary endpoint and decision

For each seed compute the difference in greedy exact-answer accuracy (including end marker) on the same 128 test cases. The experimental replication unit is the paired seed, not the pooled case count.

Primary analysis: two-sided paired Student-t test, alpha 0.05, with mean, sample SD and 95% confidence interval. Confirmation requires a positive mean and p < 0.05. Report all five differences. The normal-difference assumption with n=5 is a limitation. An exact two-sided paired sign-flip test over 2^5 assignments is a prespecified sensitivity analysis and cannot replace a failed primary criterion. Report the fraction of positive gaps and accuracy variance descriptively. Whether the lower interval exceeds +10 pp is a secondary practical-effect assessment.

Secondary endpoints: response CE, teacher-to-student KL, exact teacher agreement, zero-edge accuracy/CE, and normalized learning-curve areas at shared checkpoints. No multiplicity-adjusted mechanism claims are made. Recurrent-edge ablation resets state and regenerates answers. Publish all outcomes, including negative ones.

## Stopping, faults, and future work

Run all five pairs unchanged; no early success/futility stopping, tuning, additional seeds, or budget increases based on observations. Missing pairs prevent a complete-study claim. Preserve failed runs without replacing seeds. Execution faults can resume identical state/configuration with a recorded incident; scientific changes require a new study.

After confirmation, prioritize crossed graph × initialization experiments before structural mechanisms. Reciprocity, spectral properties, motifs, modules and hubs require separately audited controls; they are not independent knobs or a guaranteed nested hierarchy. Compiler v2 representation-transfer design can proceed separately, with no competing full-graph training during this confirmation.

The existing hard-label run is allowed to finish, then the old compiler tournament is postponed. The handoff does not modify its frozen training code. Checkpoints remain backed up by the existing verified storage adapter.

## Budget amendment before confirmation begins

The user explicitly requested five fresh pairs (ten models), replacing the ten-pair v1 plan before any confirmatory model trained or new test was evaluated. The superseded protocol and handoff receipt are retained. Task, architecture, optimizer, per-model budget, test assignment salt and primary analysis remain unchanged. With five pairs, uncertainty is wider and the exact two-sided sign-flip sensitivity test has minimum p=0.0625; it cannot reach 0.05 even with five positive gaps. The prespecified paired t test remains primary, and its small-sample assumption must be reported. Four or five wins alone do not override the registered primary criterion.
