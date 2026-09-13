# Active two-Mac confirmation allocation

The M4 Mac mini is reachable from the primary Mac through Tailscale and the dedicated SSH key:

```sh
ssh -i ~/.ssh/flygpt_mac_mini_ed25519 -o IdentitiesOnly=yes alex@alexs-mac-mini
```

Mini workspace: `/Users/alex/flygpt`. Its branch is `experiment/mac-mini-compiler-v2` (created for the benchmark, now also hosting its assigned confirmation work). The original M1 workspace remains `/Users/alex/Desktop/flygpt_v01`.

The independent 64-update benchmark measured 135.46 seconds through validation on M4 versus roughly 181 seconds in prior M1 runs. This is an approximate 1.34x throughput ratio including validation overhead, not a guaranteed full-run speedup. The mini passed all twelve existing G2c/compiler tests with the pinned package versions. It has 16 GiB RAM. Seagate is now attached to the mini.

The recorded hardware amendment assigns complete pairs:

| Machine | Confirmation seeds |
|---|---|
| M1 MacBook | 100, 101 |
| M4 Mac mini | 102, 103, 104 |

No seed, task, architecture, initialization rule, optimizer, update budget or test changed. Both conditions within each pair train on the same machine. The aggregate analysis spans two hardware blocks and must report that limitation. No final evaluation occurs until all ten checkpoints finish; evaluation stays on M1. Hardware assignment was based on a separate timing benchmark before new final-test evaluation.

The original `run_topology_confirmation_v2.py` and training engines remain unchanged. `scripts/distributed_confirmation.py` adapts scheduling and imports M4 final checkpoints with SHA256 verification. It uses the unchanged original runner for the all-checkpoints test gate and analysis. M1/mini directory differences in frozen input paths are mapped explicitly and verified by content.

M4 checkpoints go directly to Seagate. M1 checkpoints remain in local staging while `scripts/mirror_m1_checkpoints.py` copies them to the mini-attached Seagate, verifies hashes and records receipts in `results/topology_confirmation_v2/remote_backup.json`. Keep local originals. A running mirror is not proof every copy has completed; inspect the receipts.

The remote SSH session could not read the mini's GitHub credentials during setup. Its Git push URL therefore points to `/Users/alex/flygpt-outbox.git`; the primary Mac can fetch the branch over authenticated SSH and push/merge from its own working GitHub connection. Do not move private keys or access tokens between machines. The benchmark branch was already fetched, inspected, merged and pushed by this route.

Monitor M4: `results/topology_confirmation_v2/remote_status.json`, `remote_controller.log`, and `remote_failure.json` if present. Monitor M1: `distributed_controller.log`, original status/progress files, and backup receipts. The original scheduler PID in old status files is historical during handoff; verify the active OS processes. Do not start the original single-machine controller alongside the distributed coordinator or duplicate any assigned run.
