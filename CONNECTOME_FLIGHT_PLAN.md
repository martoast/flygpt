# Connectome flight — checkpoint plan

Status: plan, 2026-09-22. Not a preregistration. Each checkpoint below freezes
its own protocol before its training starts. FlyGPT's substitution studies are
paused. The Omarchy efficiency cohort stays trained but not evaluated, and its
test stays locked.

## Question

Can a recurrent network constrained to the full MaleCNS connectome fly the fire-drone
simulator? And when inputs and outputs enter and leave through anatomically correct
neurons, does the real wiring fly better than degree-matched rewiring?

This replaces the arbitrary `(x+1) mod 4` task. Visually guided flight is the task
this graph evolved for, and the drone simulator already provides closed-loop grading,
a privileged teacher, DAgger data and frozen seed discipline.

## Starting facts (measured 2026-09-22)

**Drone side** (`~/Documents/alex/fire-drone/sim`):
- The benchmark is goal-navigation V5 epoch20 (`goal-navigation-flat-v1.onnx`). It passes
  48/48 flat block and low-tree flights. Blinding obstacles in RGB only drops it to 8/16,
  so vision is doing real work.
- The actor takes 83 inputs: 64 learned depth values from 128×72 RGB, the GPS goal,
  velocity and gravity estimates, body rates and the previous action. It runs every inner
  physics step (100 Hz), and camera frames arrive at 20 Hz.
- V5 is a *structured* actor. The learned part outputs a horizontal velocity correction;
  an authored controller does the stabilization.
- The 24,615-row V5 training set and teacher captures sit behind `runs/navigation`, which
  is a symlink to `/Volumes/Seagate/fire-drone-sim-reference/local-artifacts/macbook-offload-20260919/navigation`.
  **Seagate is currently mounted on neither Mac.**
- All 48 V5 seeds are consumed development data, so they are not unseen.

**Connectome side** (`data/processed/malecns.npz`, stored as A[pre, post]):
- **Photoreceptors:** there are 3,377 R1–R6 cells, which is partial and asymmetric
  coverage (root side R 3,746 / L 2,345 across all `ol_sensory`). None have hex
  coordinates.
- **Hex-mapped columns:** 23,720 columnar neurons carry `assignedOlHex1/2`, about 1,770
  columns each for L1, L2, L5, Mi1, Tm1 and T1. L1 and L2 are the direct photoreceptor
  targets, and 96% of R1–R6 synapse onto L1.
- **Motor neurons:** wing motor (`wm`) 67, haltere motor (`hm`) 16.
- **Flight-neuropil descending neurons:** upper tectulum (`ut`) 276, haltere tectulum
  (`ht`) 26, intermediate tectulum (`it`) 18, wing tectulum (`wt`) 2. There are 1,314
  descending neurons in total.
- **Hops from R1–R6:** descending neurons are 2–4 hops away (median 3), and wing and
  haltere motor neurons 3–4 (median 4).
- **Ticks per step:** FlyGPT's two ticks per symbol cannot carry an eye signal to motor
  neurons. The tick count per control step must come from these distances.
- **Speed:** the FlyGPT CPU engine took about 2.4 s per batch-one update on roughly
  13-symbol sequences.

## Claim boundaries (all checkpoints)

- This is a rate model with trainable signed weights on fixed anatomical topology. It is
  not a spiking model and uses no transmitter signs. It is not wetware, and it is not
  evidence of how real flies fly.
- Stages A–B keep V5's GPS goal, ideal state estimates and authored stabilizer. The
  connectome replaces only the learned steering. Do not call that GPS-free flight or
  end-to-end flight control.
- **Zero-edge ablation is uninformative by construction.** Input and output populations
  are disjoint, so removing the recurrent edges disconnects them. Report it, but treat it
  as a wiring check. Informative ablations are targeted lesions (CP5) and camera blinding.
- Hardware stays inside a pair or seed block. There is no pooling across studies, all
  outcomes are published including failures, and nothing is tuned against a locked test.

---

## CP0 — Recover the baseline harness (no training)

1. Mount Seagate. Verify the `runs/navigation` offload against its recorded hashes.
2. Replay packaged V5 on all 48 recorded seeds. The outcomes and metrics must match
   `goal-navigation-v5-results.json`.
3. **Inference bridge.** Browser ONNX cannot run a 166,700-neuron sparse recurrent net.
   - The sim sends each observation over the existing wire boundary to a Python actor
     process and applies the returned action.
   - The sim is fixed-step, so wall-clock latency does not change sim outcomes.
   - Parity gate: V5 served through the bridge reproduces its browser outcomes on all
     48 seeds, action-for-action within float tolerance.

**Gate:** data verified, replay identical, bridge parity passes. If the data cannot be
recovered, regenerate teacher captures from training seeds only and record that change.

**Result (2026-09-23): passed.** See `results/connectome_flight/cp0/CP0_REPORT.md`.
- All V5 sources match their hashes. Replay: 48/48 pass, 47/48 bit-identical.
- Bridge: 3.3e-6 open-loop action parity, 48/48 closed-loop passes.
- Closed-loop trajectories shift with ~1e-6 backend noise, up to a 6.5 m swerve on
  training-1110. So: every arm is evaluated through the same bridge backend, pass/fail
  is the only primary endpoint, and the connectome actor must be bitwise deterministic
  on CPU.

## CP1 — Wiring map and observation contract (no training)

Freeze `connectome-pilot-v1`, a JSON contract plus `io_map.json` with hashes.

- **Visual input.** Map the 8×8 depth grid (Stage A) retinotopically onto L1 and L2
  columns by hex coordinate, both eyes, within the camera's 140° field of view. Record
  the azimuth/elevation-to-hex rule. Photoreceptors are not the input layer, because
  their coverage is incomplete.
- **Non-visual inputs.**
  - Body rates and gravity have biological analogues in haltere, neck and antennal
    mechanosensory neurons. The contract names the chosen `vnc_sensory` / `cb_sensory`
    types.
  - The GPS goal and previous action have no biological analogue. They enter a declared
    population chosen by a fixed rule, labeled non-biological.
- **Readout.** The primary readout is a linear readout from flight-neuropil descending
  neurons (`ut`, `ht`, `it`, `wt`: 322 neurons). Wing and haltere motor neurons (83) are
  the alternate. Pick one before training, not after.
- **Random-I/O arm.** Same channel counts, with FlyGPT's population-seed rule over the
  whole graph and disjoint populations.
- **Ticks per control step.** Set from graph distance: at least the maximum
  input→readout hop count across all input channels, currently 4. Performance never
  chooses it.
- **Reachability audit.** Every input channel reaches the readout within the declared
  ticks in the real graph *and* in every control graph.

**Gate:** contract frozen, audits pass, and the random and anatomical maps are both
hashed.

## CP2 — Throughput and control-rate decision (disposable probes only)

1. Benchmark forward and forward+backward time per tick at the planned batch size on the
   M1, M4, Omarchy CPU (4 threads) and Omarchy GTX 1070 with CUDA sparse ops. Record
   memory use.
2. Decide the connectome's update rate.
   - At 100 Hz with 4 ticks per step, a 30 s flight is 12,000 ticks. At about 30 ms per
     tick that is about 6 min per flight, or about 5 h per 48-flight evaluation per model.
   - The alternative is 20 Hz, running on camera frames with the action held across the
     five inner steps.
   - If 20 Hz is chosen, **retrain and re-baseline the MLP at 20 Hz**, so every arm runs
     under the same interface.
3. Project the full wall time of CP4 from the measured numbers. If it exceeds the declared
   budget, shrink the seed count or schedule. Do not switch to a subgraph: a smaller
   graph changes capacity and is a different study.

**Gate:** rate and hardware allocation frozen, CP4 wall-time projection recorded.

## CP3 — Learnability pilot (one seed, development only)

1. **Software check.** Run the full training and flight loop on a small synthetic graph.
   This validates plumbing only and is not connectome evidence.
2. **Full-graph pilot.** One seed each of MaleCNS-anatomical and the MLP, trained on V5
   training rows with V5's loss (4·nav-velocity + 0.2·action + 0.05·rollout-velocity +
   4·clearance + 0.1·arrival-speed + 0.001·control-change). Use DAgger on training seeds
   only.
3. **Report:**
   - open-loop action error on held-out *rows*
   - closed-loop passes on the 32 training seeds
   - hidden-state saturation and gradient norms

**Gate:** the connectome actor passes some closed-loop training flights and its loss
falls. If it passes zero, diagnose on this same task: check the readout, ticks and
normalization, and record every change. Do not open evaluation seeds and do not add
edges.

## CP4 — Primary matched study (preregistered separately before training)

**Arms,** all with identical data, DAgger rounds, optimizer, budget and seeds:

| Arm | Role |
|---|---|
| MaleCNS, anatomical I/O | Treatment |
| Degree-preserving rewired, anatomical I/O | **Primary control** |
| Reciprocity-preserving rewire, anatomical I/O | Secondary. Tests the FlyGPT reciprocity lead |
| MaleCNS, random I/O | Secondary. Does anatomical placement matter? |
| V5-architecture MLP | Engineering reference, not a topology control |

**Rules:**
- Initialization matches degree normalization. Log each graph's initial spectral radius,
  because FlyGPT's rewired accuracy tracked it with r=0.97. If compute allows, add a
  spectral-radius-matched rewire.
- **Fresh test.** Generate new block and low-tree seeds with a frozen list, disjoint from
  every V5 seed. They stay locked until every CP4 model is trained and hash-verified.
  Development uses only V5 training and validation seeds.
- **Primary endpoint.** Per model seed, the closed-loop pass rate on fresh test flights
  under V5's unchanged grading: arrive within 3 m, ≤2 m/s held 1 s, ≤30 s, zero contacts,
  2–6 m band on the tree course. The paired difference is MaleCNS minus degree-rewired.
- **Statistics.** Paired t with a sign-flip sensitivity test, as in FlyGPT. Use at least
  5 seeds, and 10 if CP2's budget allows.
- **Secondary endpoints:**
  - collisions, minimum clearance, time to goal and path efficiency
  - RGB camera-blinding ablation for every arm
  - learning-curve area across DAgger rounds

**Gate:** results published whatever their sign.

## CP5 — Lesions: which circuits does it use? (after CP4)

On trained MaleCNS-anatomical models, apply lesions without retraining:

- motion vision: T4/T5
- looming detection: LPLC2 and LC types
- the giant fiber escape pathway: DNp01
- **controls:** a size-matched random lesion and a size-matched lesion inside the same
  superclass for each of the above

Compare collision and pass changes. A pathway-specific deficit beyond its matched
controls is the result that would say something about the biology. Fix the lesion list
before evaluation.

## CP6 — The eye drives the brain (raw pixels)

Remove the learned depth CNN and feed luminance on the hex lattice directly into the
L1/L2 columns. The connectome then does all the visual processing. Compare against a
CNN+MLP baseline given the same pixels. This needs its own teacher/DAgger data at
camera resolution and a fresh locked test. **Headline claim if it passes:** the fly
connectome, receiving camera input through its own visual system, steers a simulated
drone around obstacles.

## CP7 — Harder flight (later, separately frozen)

In order:
1. rolling terrain (`goal-ranch-hard-v1`, still unsolved by V5)
2. direct thrust/body-rate output with no authored stabilizer
3. noisy onboard estimation instead of ideal state
4. the GPS-free patrol contract
5. fleet fire response

Each step gets its own test and protocol.

---

## Where things live

- Connectome engine, I/O maps and studies: this repo, on a new branch
  `experiment/connectome-flight`.
- Sim, teacher, grading and the observation bridge: `fire-drone/sim`, with a separate
  scenario contract.
- The fleet stays on its authored autopilot until a connectome pilot passes the fleet's
  own acceptance.
