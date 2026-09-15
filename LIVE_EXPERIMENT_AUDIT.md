# Live experimental integrity audit — 2026-09-15 UTC

No blocking defect was found in the checks below. Continue the frozen runs unchanged. This is an operational and implementation audit, not a declaration of compiler success or an interim hypothesis test.

## Snapshot

| Study | Finished training | Saved models inspected | Planned models |
|---|---:|---:|---:|
| M1 topology replication | 16 | 17 | 20 |
| M4 compiler confirmation | 9 | 10 | 30 |
| Omarchy acquisition efficiency | 4 | 5 | 60 |

Snapshots were taken at 05:40–05:41 UTC. Training continued during the audit, so these counts are not synchronized live status.

## Checks and evidence

- All 285 frozen input entries matched their recorded SHA-256 hashes (80 topology, 81 confirmation, 124 efficiency; entries include shared inputs). All 32 inspected jobs had contiguous saved traces, finite recorded scalar metrics, and a latest archived checkpoint matching its manifest and progress hash.
- Recorded losses matched the intended decomposition: CE for topology/teacher-hard, pure recorded KL for soft KD, and CE plus weighted alignment for representation methods. Compiler group IDs matched the frozen schedule and example counts matched step × batch size. This checks recorded arithmetic; regression tests additionally exercise objective behavior.
- Dataset hashes and unique instance IDs passed within-study train/validation/test separation. Final-test model evaluation was NOT performed. Reading IDs and hashes for leakage auditing does not supply test performance for model selection. Existing data-generation tests also passed.
- All 15 frozen compiler shuffle audits passed. The minimum recorded float32 relational target difference was approximately 3.34, far above epsilon 1e-6. Tests verify fixed derangements, correspondence-sensitive relational targets, and reject degenerate controls.
- Independently reloaded all ten topology rewires. Each retained 166,700 neurons, 25,582,938 directed edges, all per-neuron in/out degrees, and the original 101 self-loops. This extra pass checked counts/degrees; uniqueness and successful swap counts remain covered by frozen graph-generation audits and regression tests.
- 50 targeted regression tests passed on the actual three execution environments: 19 M1, 17 M4, 14 Linux. Coverage includes sparse forward/backward correctness, disjoint populations, prompt consumption, no decoder bypass, exact resume behavior, compiler auxiliary-head isolation, and final-test locks. Transformer nested-tensor warnings were performance-path warnings, not test failures.
- Source review: teacher attention is causal; generation feeds its own predictions; decoder reads the output population; auxiliary projection is used for training loss, not as an inference bypass.
- Maximum recorded hidden-state saturation across inspected jobs was 1.752%; no widespread saturation or nonfinite dynamics was observed. Pre-clipping gradient spikes alone are not divergence evidence.

## Full-connectome validation-only causal diagnostic

Loaded the completed M4 seed-400 teacher-hard checkpoint and evaluated the first eight validation cases, with no performance-based case selection. Exact accuracy was 62.5% intact versus 0% with recurrent edges zeroed. Response CE increased from 0.12771 to 1.95116. Checkpoint hash and case IDs are in `results/live_audit_20260915/validation_ablation.json`.

This small diagnostic confirms that this saved full-size checkpoint executes and depends on recurrent edges on these cases. It is not the final ablation across all models and is not an accuracy estimate for the entire validation set.

## What the interim results do and do not say

All eight completed topology pairs favor MaleCNS on the existing 32-case validation subset. That is encouraging but is not the locked final-test replication result.

Compiler results are mixed. In completed M4 seed 400, both hidden and hidden-shuffled reach 100% validation accuracy; exact accuracy therefore cannot establish a representation-specific benefit there. In the first Linux seed at 256 updates, soft KD is 33.20%, teacher-hard 62.50%, and hidden alignment 72.27% validation accuracy. That is not a completed six-condition comparison or replicated inference. It shows why negative results and all matched controls must be retained. No parameters, stopping rules, or method choices were changed after inspecting these values.

## Storage and timing

At inspection, M1 had about 13 GiB free internally; M4 about 14 GiB; Seagate about 759 GiB. Existing transfer receipts recorded 101 topology checkpoint backups and 191 Linux artifact backups, with recent verified transfers. These are transfer-time hash receipts, not an exhaustive rehash of every historical backup. M4 checkpoint storage is on Seagate; topology cleanup removes older local archives only after verified transfer.

Limited Mac internal headroom remains worth monitoring. The Linux worker also still requires the machine to stay awake; its earlier noninteractive sleep-inhibition request was denied. Neither issue currently showed a failed training job in this audit.

The audit briefly shared CPU and I/O with training, including approximately 26 seconds of single-threaded M4 full-model validation inference. Any precise wall-clock efficiency interpretation must account for this external contention; updates/examples comparisons remain the cleaner measures. No running job or frozen training file was altered.

## Reproduction and limits

Raw snapshots, checkpoint/degree/ablation evidence, and test counts are in `results/live_audit_20260915/`. Run its `audit.py` from each study worktree with arguments STUDY_RESULTS_ROOT OUTPUT_JSON to repeat frozen-input, trace, latest-checkpoint, split, and shuffle checks. Use the test paths in `regression_tests.json` with `python -m pytest -q`.

This audit does not certify every historical checkpoint, independently regenerate all teacher features, or rule out every possible scientific confound. It does establish that the checked artifacts and critical implementation paths are consistent with the frozen protocol. Preserve all current conditions through their registered stopping points, then unlock and analyze each cohort as planned.
