# Compiler v3 hardware allocation amendment

User-requested on 2026-09-13 (local time), after A began and before any final-test evaluation. This supersedes only the original single-M4 allocation and serial scheduling. Frozen source, specs, teacher, data, shuffle audit, optimizer, seed, batch eight, 512 updates, checkpoints and final-test lock remain unchanged.

- M4: A_hard, E_relational, F_relational_shuffled, B_soft, retaining relative frozen order.
- M1 MacBook: D_hidden_shuffled, C_hidden, retaining relative frozen order.
- No duplicate model training. Preserve the already-running M4 A process.
- A disposable one-update M1 memory probe must pass before its assigned training. It cannot select hyperparameters or change the batch.
- Both machines use CPU, four PyTorch/OMP/Numba threads and the existing pinned package versions. Save per-run hardware provenance.
- Each real/shuffled pair stays on one machine. C/D versus A is a cross-machine descriptive pilot contrast; hardware is an additional limitation. Raw acquisition/wall times across machines cannot establish a method efficiency advantage. Report update/example thresholds and machine-specific timings, including the M4 origin of cached teacher-preparation timings. Any positive candidate needs a hardware-balanced replication.
- The mini remains the sole final-test coordinator. It must receive hash-verified M1 checkpoint archives, progress and completion receipts for both assigned jobs before opening the unchanged final test. Preserve original M1 manifests and archive locations; separate import receipts identify Seagate copies.
- Back up M1 checkpoints over SSH to the Seagate physically attached to the mini. Never remove unverified local originals. All six outcomes remain mandatory; a failed worker must stop rather than be silently substituted.

The new orchestration wrapper is outside the original frozen training implementation. Its hash and this amendment's hash are recorded in a separate allocation receipt and published before starting full-budget M1 training. The original frozen plan and publication receipt remain intact.
