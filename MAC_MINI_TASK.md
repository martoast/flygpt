# M4 Mac mini assignment: benchmark first, then allocate full-graph work

Read this file first after cloning `https://github.com/martoast/flygpt`.

## Ownership and purpose

The primary Mac is running the five-pair topology confirmation. Because this second machine is an M4, first benchmark the unchanged full-graph implementation. Do not independently start, modify, or evaluate confirmation: the primary agent must assign complete remaining pairs and record hardware ownership before any move, preventing duplicate work and mixed-machine pairs. The compiler screen below is a prepared fallback assignment, not the immediate default.

Create branch `experiment/mac-mini-compiler-v2`. Keep results in `results/macmini_compiler_v2/` and the report in `MAC_MINI_REPORT.md`. Do not modify the frozen engine, dataset, teacher, graph, or other study outputs. Do not merge your branch into main yourself. Push your branch and give the primary agent its name, commit and report summary for verification and merge.

## Inputs that cloning does not supply

The real graph and teacher checkpoint are outside Git. Obtain `flygpt-mac-mini-inputs.tar.gz` and its `.sha256` file from the primary machine. They are prepared in `artifacts/mac_mini_handoff/` and copied to `/Volumes/Seagate/FlyGPT Backups/Mac mini handoff/`. AirDrop, local-network copy, or an external drive can move this roughly 80 MB package. Do not disconnect Seagate from the primary machine during a checkpoint copy.

The package contains only the existing compiler task inputs; it does NOT contain the new confirmation test. Verify its SHA256, then extract it into the repository root. The runner verifies each extracted input against the committed dependency manifest before training. Do not regenerate or substitute missing inputs.

## Setup

Use Python 3.12 and at least 16 GiB of available disk space. Create `.venv` in the repository. Install `requirements-mac-mini.txt`, which records the primary machine's relevant versions. If exact versions cannot be installed or the graph cannot fit in memory, document the blocker and stop before full-graph training; do not silently change versions or architecture. Record chip, memory, OS, Python and package versions. CPU/SciPy with four threads is intentional; no MPS/CUDA substitutions within the cohort.

```sh
git clone https://github.com/martoast/flygpt.git
cd flygpt
git switch -c experiment/mac-mini-compiler-v2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac-mini.txt
```

After receiving the package, verify/extract it from its containing directory:

```sh
shasum -a 256 -c flygpt-mac-mini-inputs.tar.gz.sha256
tar -xzf flygpt-mac-mini-inputs.tar.gz -C /absolute/path/to/flygpt
```

Then, from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_g2c.py tests/test_compiler.py
.venv/bin/python -m scripts.mac_mini_compiler --preflight
.venv/bin/python -m scripts.mac_mini_compiler --benchmark
```

The benchmark performs 64 batch-one CE updates with the unchanged full MaleCNS architecture and reports time through validation/checkpointing. It uses a disposable seed, never opens a final test, and must never be reused as a confirmatory checkpoint. Push `results/macmini_compiler_v2/benchmark.json` to the assigned branch and report the branch to the primary agent. Do not infer speed from the chip name or compare this short run directly to a full 1,024-update runtime without accounting for overhead.

The primary agent can then transfer whole confirmation pairs under a documented hardware amendment. Until that allocation exists, do not duplicate its runs. To run the fallback compiler assignment ONLY once allocated to it, use `.venv/bin/python -m scripts.mac_mini_compiler` without `--benchmark`.

Run longer controllers persistently (for example in a terminal session with caffeinate, or detached with stdout/stderr captured). The runner also starts caffeinate. Check its status and failure files; do not assume a silent process completed. If interrupted, rerun the same command to resume verified checkpoints. No duplicate controller is allowed.

## Prepared fallback experiment — run only after assignment

Five conditions, all on this same machine and real MaleCNS graph:

1. Ground-truth CE.
2. Teacher-hard CE.
3. Teacher-hard CE plus soft KD, alpha 0.5, temperature 2.
4. Teacher-hard CE plus normalized hidden alignment, weight 1.
5. Teacher-hard CE plus relational geometry alignment, weight 1.

All use model seed 200, sampling seed 75200, batch two, 512 updates (1,024 training examples drawn with replacement), the existing architecture and optimizer, and identical checkpoint schedules. This is a matched batch-two cohort. Do not compare it directly to the primary Mac's batch-one CE numbers as though that were a controlled method effect.

The original teacher generated identical training answers to the ground truth; hard-label CE is therefore an implementation sanity control, not evidence of a stronger transfer claim. Hidden projection is auxiliary during training only; relational transfer compares cross-example geometry. Neither may add an inference bypass.

Freeze all five specifications before any run. Complete all conditions before final evaluation on the OLD `test_extension` split. This is an exploratory, previously inspected test, not the new confirmation test. Never use final results to tune or add methods. Full validation, final exact accuracy, response CE, KL, teacher agreement and zero-edge ablations must be retained for every condition. Report differences to this cohort's hard-label baseline, including negative results. Stop after the five-condition report; replication or new methods require a separately frozen follow-up.

## Artifacts and handback

Commit and push code-independent small results on your own branch after each stage. Do not commit `.pt`, `.npz`, archives, or bulk files. The runner retains checkpoints locally when Seagate is absent, respecting a 3 GiB reserve; local files are not an off-machine backup. Preserve ALL intermediate archives.

At completion the runner creates `artifacts/mac_mini_return/flygpt-mac-mini-final-checkpoints.tar.gz`, its checksum, and a tracked manifest of the five final checkpoints. Transfer that archive back to the primary machine outside Git. Include `MAC_MINI_REPORT.md`, branch/commit, hardware/runtime, test status, and any failures. The primary agent verifies artifact hashes and checks small result paths before merging; it must not pool this compiler screen with the topology confirmation.

Authorization: the user requested this additional computer work on its own branch. No further approval is needed for reversible setup, these fixed runs, checkpoint backups, and pushing that branch. Do not send messages to other people, deploy anything, or launch unrelated experiments.
