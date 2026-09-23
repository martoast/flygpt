# CP0 — baseline harness recovered

2026-09-23. **Gate passed.** No model was trained; this checkpoint only verifies
the benchmark, the recorded V5 model and a new inference bridge.

## Inputs

- Sim: `fire-drone` at `b945e541`, plus one uncommitted change to
  `sim/client/src/vision/goal-probe.ts` (sha256 `c20a347a…`) adding the
  `actorBridgeUrl` option. Dev server: `npm run dev`.
- Model: V5 epoch20 `runs/training/goal-navigation-v5/epoch-020/policy.onnx`,
  sha256 `8702c263…04241`, matching the recorded freeze.
- The five V5 training sources in `runs/navigation` (a symlink to Seagate) all
  match their recorded sha256 (24,615 rows):
  `goal-teacher-velocity-v3`, `goal-student-velocity-v3`,
  `goal-tree-teacher-velocity-v3`, `goal-tree-student-v3-velocity-v3` and
  `goal-tree-student-v4-velocity-v3`.

## 1. Browser replay of the recorded 48 flights

Same six probe invocations as `docs/goal-navigation-v5-results.json`.
Outputs are in `sim/runs/navigation/cp0-replay/`.

- **48/48 pass.** 47/48 match every recorded metric exactly.
- heldOut-6103 still passes but arrives at 13.77 s against the recorded 13.85 s.
  It reproduces identically on two reruns, so it is deterministic drift from
  commits made since Sep 17. There are 58 commits touching `client/src/vision`,
  and they were not bisected.
- **Today's replay is the reference for all later comparisons, not the Sep 17
  record.**

## 2. Inference bridge

`scripts/flight_bridge.py` (this repo, sha256 `b638961f…`) serves an actor over
local HTTP:
- `POST /reset` at the start of each flight, so recurrent state never crosses
  flights
- `POST /act` once per inner physics step, with the same 83-float observation
  the browser actor receives
- it rejects malformed observations and acts from a different flight

The sim never reads wall-clock time and awaits every observe and act call, so
bridge latency cannot change outcomes. Latency is about 0.4 ms per call.

**Open-loop parity.** `sim/scripts/check-actor-bridge-parity.mjs` fed 2,000
recorded observations through browser WASM onnxruntime and through the bridge
running CPU onnxruntime 1.22.1 with the same ONNX file. The largest action
difference was **3.3e-6**, where actions range over ±1. This is rounding
difference between the two ONNX runtimes, not a transport error.

**Closed loop through the bridge** (`sim/runs/navigation/cp0-bridge/`):
- **48/48 pass.** The report records the bridge's model hash.
- Most arrival times move by 0.2 s or less, and 0/48 are bit-identical to the
  browser run.
- Training-1110, an open route with no obstacle, goes from 8.30 s to 12.33 s.
  From t≈2 s the drone swerves up to 6.5 m sideways and returns, with angular
  acceleration of 28.5 against 4.6. It is deterministic across two reruns.
- The rendering and training rows show no bridge fault. V5 has a decision
  point here where about 1e-6 of action noise switches between flying straight
  and a detour.

## Consequences for CP1–CP4

1. **Every arm, including the MLP baseline, is evaluated through the same
   bridge backend.** Browser and bridge outcomes are not interchangeable at the
   trajectory level.
2. **Pass/fail is the only primary endpoint.** Arrival time, path length,
   smoothness and clearance change with the inference backend alone, so they
   are secondary and descriptive only.
3. **The connectome actor must be bitwise deterministic on CPU.** The SciPy
   CSR propagation used in FlyGPT is deterministic. Threaded scatter-add is not
   allowed at evaluation.
4. The bridge currently serves ONNX only. CP3 adds a backend that composes
   V5's `VelocityResidual` controller (`sim/training/goal_controller.py`) with
   a connectome module in place of its 83→192→192→3 `net`. The authored
   stabilizer stays byte-identical across arms.

## Reproduction

```sh
# fire-drone/sim, with `npm run dev` running
M=/@fs$PWD/runs/training/goal-navigation-v5/epoch-020/policy.onnx
node scripts/goal-navigation-probe.mjs "{\"mode\":\"student\",\"split\":\"heldOut\",\"course\":\"trees\",\"modelUrl\":\"$M\"}" out.json
# flygpt_v01, using the sim venv (it has onnxruntime)
~/Documents/alex/fire-drone/sim/.venv/bin/python scripts/flight_bridge.py --backend onnx --model <policy.onnx> --port 8799
# then add "actorBridgeUrl":"http://127.0.0.1:8799" to the probe options
```

## Output hashes (sha256)

| File | Replay | Bridge |
|---|---|---|
| tree-training | `e573a71e…` | `11ec765b…` |
| block-training | `5939cddc…` | `78ddfbbb…` |
| tree-validation | `9dadcaba…` | `e00ed725…` |
| block-validation | `7fa8b88b…` | `79f15ba4…` |
| tree-heldout | `be7b4885…` | `0b1d1baf…` |
| block-heldout | `911777d6…` | `c46aa9af…` |
