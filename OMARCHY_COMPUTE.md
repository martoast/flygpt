# Omarchy research worker

Reachable over Tailscale SSH as `alex@omarchy.tail61505e.ts.net` (100.97.166.59). On the MacBook, host keys are pinned from authenticated tailnet metadata in `~/.ssh/flygpt_omarchy_known_hosts`; no private key or GitHub credential was copied to Linux.

Hardware: Intel i7-8700K, six cores/twelve threads, 16 GB RAM, GTX 1070 8 GB, approximately 418 GB initially free on the system disk. The registered efficiency study uses **CPU/SciPy with four threads**. The GPU is not part of this cohort.

Workspace: `/home/alex/flygpt_efficiency`, branch `experiment/compiler-efficiency`. Isolated Python 3.12.12 environment with torch 2.14.0+cpu, numpy 2.5.3, scipy 1.18.1, numba 0.67.0, pytest 9.1.1. Exact metadata is frozen in `results/compiler_efficiency/environment.json`. Graph and teacher input hashes match the original MaleCNS project artifacts.

Study: seeds 500–509, all six methods, 256 updates each; saved 128-update prefix secondary. See `COMPILER_EFFICIENCY_PREREGISTRATION.md`. The active M4 replication is a separate unchanged study. Full Linux graph/memory probe passed, and all ten numerical shuffle audits passed before full-budget training.

Controller: `.venv/bin/python -m scripts.run_compiler_efficiency`, detached from SSH. Status, failures and controller log live in `results/compiler_efficiency/`. The controller holds a file lock; do not launch a second process over the same jobs. Resume uses saved optimizer/RNG/checkpoint provenance checks and never changes the stopping budget.

Linux denied noninteractive `systemd-inhibit` authorization. Keep this desktop awake and connected; no lock-screen or system sleep policy was silently disabled. The inhibition status is recorded separately.

Checkpoints remain on Linux under `artifacts/compiler_pending/results/compiler_efficiency/`. Local disk reserve checks prevent continuing when archive space is exhausted. The MacBook runs `python -m scripts.relay_omarchy`: it streams checkpoint and metadata backups to `/Volumes/Seagate/FlyGPT Backups/G2c-overnight/` on the mini, verifies SHA-256 before publishing each backup, and retains Linux originals. MacBook relay receipts/logs live in `results/compiler_efficiency_relay/`. Backups catch up when the MacBook and mini are reachable; training does not depend on that network path.

Linux commits to its own branch and pushes to `/home/alex/flygpt-outbox.git`. The MacBook relay fetches the branch and pushes to GitHub with its existing authentication. No automatic merge to main. The final-test lock covers all 60 models, and every outcome is retained.
