# Independent ten-pair topology replication

New study authorized after the completed five-pair confirmation and its +33.75 pp mean effect were examined. Preserve that study unchanged; do not append seeds to its stopping rule or pool it into this primary analysis. This is replication of learning on a fixed topology, not a compiler-method test or a mechanism experiment.

## Design

Ten model seeds 600–609, paired with ten new directed degree-preserving graph seeds 1900–1909. For each model seed, train full MaleCNS and its assigned rewired graph with the original CE-only G2c architecture and settings: batch one, 1,024 updates, lr .001, original AdamW/clipping, fixed population seed 2026, disjoint I/O, leak .65, two internal ticks, degree normalization, four-thread CPU/SciPy. Preserve the reference teacher for diagnostic KL only; CE training uses ground-truth task labels, not distillation. Same training data and sampling seed 75000+model seed within every pair. All 20 models and all evaluation run on the M1 MacBook.

Use the unchanged graph-controls implementation: exactly ten successful directed swaps per edge parameter, identical individual in/out degrees, edge count and self-loops preserved. Audit every graph and hash it before training. This procedure is not claimed to sample uniformly from all degree-matched graphs. No replacement of a bad-performing graph, seed or run. Degree preservation does not preserve reciprocity, motifs, spectrum or modules. Graph realization and model initialization remain coupled across pairs; this study does not identify the mechanism or independently estimate graph-versus-initialization effects.

Run seeds in increasing order, with real then rewired within each pair, as in the original confirmation. This fixed order is a limitation for wall-clock interpretation, not a basis for an efficiency claim. Save original checkpoints 64,128,256,512,768,1024 and the same first 32 validation cases. Freeze all specs, graph audits, source/config/data/teacher/graph hashes and publish to GitHub before training. Execution failures may resume the identical optimizer/RNG state; numerical failures remain failures and missing pairs prevent a complete-cohort claim. No accuracy-driven extensions or tuning.

## Fresh locked test

Exclude all original G2c train/validation/qualification/test IDs, the earlier topology test, compiler pilot test, M4 compiler confirmation's reserved 512 IDs and Omarchy efficiency study's reserved 256 IDs. That excludes 3,968 of the 4,096 inputs. Allocate the remaining **128** to this study with deterministic order and recorded hash. No unseen inputs remain in this finite task afterward; future fresh-domain studies must use a separately validated larger domain. Training and validation stay unchanged. The test is closed until all 20 final checkpoints are complete and their hashes verify. Do not inspect interim final-test results to change protocol choices.

## Analysis

Primary: per-seed final exact-answer accuracy difference MaleCNS minus rewired. Ten paired seeds, not 128 independent experimental replicates per model. Two-sided paired Student-t test at alpha .05; support requires a positive mean and p<.05. Report all pairs, sample SD, unadjusted 95% paired-t interval, exact exhaustive two-sided sign-flip sensitivity p, number of positive pairs, and separate condition means/SDs. The exact test is sensitivity, not a second route to declaring success. No pooling with exploratory seeds 0–4 or confirmation seeds 100–104. Normality/independence assumptions and the shared corpus/teacher/task limit interpretation.

Secondary descriptive outcomes: CE/KL/agreement, original validation learning curves and their trapezoidal area, per-condition variance, whether the lower mean-gap interval exceeds 10 pp, and zero-edge accuracy/CE for every final model. No secondary significance fishing, no claim of asymptotic capacity or of evolution selecting this structure for substitution. A positive result supports this topology's fixed-budget learning/generalization advantage for this task and optimizer.

## Storage and execution

The MacBook prepares the graphs and runs this study independently of the unchanged M4 and Omarchy queues. Keep the latest checkpoint locally for resume/evaluation. A background backup worker hashes each checkpoint, copies it to the Seagate on the mini, verifies the destination, and only then may reclaim older local checkpoint copies. Latest/final checkpoint copies remain local. Preserve all versions externally. If connectivity or storage prevents safe progress, pause rather than discard unverified artifacts. Snapshot code, graph files/audits, data and small results off-machine and push the assigned Git branch at milestones. No automatic experiment follows completion.
