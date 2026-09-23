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

## CP0b — Realistic flight stack: `drone-env-v2` (before any training)

Decided 2026-09-23: the pilot, whether human, MLP or connectome, sits exactly where a
real pilot's hands sit. v1 is idealized. Body rates are set directly with a 40 ms lag,
there are no motors or inertia, and throttle and turning authority are independent.
Manual flying is a velocity-command point mass. v1 stays as it is, and every earlier
result remains valid only on v1.

**Command path**, the same for every pilot:

1. **Pilot → radio sticks.** Four RC channels in Mode 2:
   - left stick: throttle and yaw
   - right stick: roll and pitch
   - CRSF/ELRS range and resolution, radio-link latency

   This is the **sim-to-real boundary**. On the real drone a companion computer sends
   the same channels over CRSF or MSP.
2. **Betaflight replica flight controller** (4.5.x, cited from source).
   - **ANGLE mode:** stick → angle setpoint → level gain → rate setpoint. Yaw is rate
     control through the rates curve.
   - **Rate PID:** Betaflight scaling, filtered simulated gyro, I-term handling and TPA.
   - **Attitude estimate:** Betaflight's own, computed from the simulated IMU.
   - **Mixer:** Quad X with airmode, DShot idle and output limits.
3. **Motors:** four motors with asymmetric spin-up/down, thrust = kf·ω², yaw torque =
   kq·ω², and battery-voltage sag.
4. **Rigid body:** full inertia tensor, gyroscopic terms, body and rotor drag, wind.
5. **Sensors:** gyro and accelerometer noise and bias. Vibration is added later.

**Airframe.** The learning-loop testbed from `fire-drone/docs/hardware/scout-reference-build.md`:
5-inch, F722 running Betaflight, T-Motor F60 Pro IV 1950 KV on HQ Ethix S5 props, 6S
1200 mAh, about 650–750 g. Each parameter is tagged *measured*, *published* or
*estimate*, with a domain-randomization range. Estimates are replaced by bench
measurements (thrust stand, bifilar pendulum, Blackbox logs) once the airframe exists.

**Rates.** Physics and the flight-controller loop run at a declared rate of at least
1 kHz. That is a simplification of Betaflight's 4–8 kHz loop and must be shown stable.
The pilot runs at 100 Hz and the camera at 20 Hz.

**Gates:**
1. **Physics sanity:**
   - hover throttle matches the predicted value
   - free fall and energy checks pass
   - motor spin directions give the correct yaw torque sign
   - both saturation directions work: with airmode, low throttle still gives attitude
     authority and full throttle limits it
2. **Flight-controller replica:** angle and rate step responses are stable at the
   default Betaflight tune, with no oscillation. Every constant is traceable to the
   Betaflight source. A real Blackbox comparison follows once hardware exists.
3. **Parity:** the TypeScript and Python implementations agree within the v1
   sim-to-sim tolerance, and the dynamics are deterministic.
4. **Flyable by a human:** manual flying goes through the *same* stack, gamepad in
   Mode 2, angle mode by default. Alex's own check that it feels like a real quad.
   Assisted GPS modes are outer loops that output sticks, and are labeled as such.
5. **Re-baseline:** an authored outer loop (goal/velocity → angle-mode sticks) flies
   the goal-nav cases. The V5-style MLP is retrained through sticks on v2, and its
   48-flight result through the bridge becomes the new CP0 reference.

**Scope:**
- Migrating the live fleet and UI is a separate, later step.
- v2 changes the connectome's readout to the four stick channels, or to a velocity
  correction fed through the authored velocity→stick loop in Stage A. Decide in CP1.
- The network runs only at the pilot rate, so CP2's connectome compute is unaffected.

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
- **Readout.** A linear readout from **all 1,314 descending neurons**, the brain's
  entire output to the body, 2–4 hops from the eye. A subclass filter was rejected:
  the flight-neuropil set (`ut`/`ht`/`it`/`wt`, 322) keeps DNg02, the wing-amplitude
  neurons, but drops DNp01 (giant fiber, `lt`), DNa02 and DNb01. Wing and haltere motor
  neurons (83) are the alternate. Pick one before training, not after.
- **Two weight models, both declared in the contract:**
  - **Trained:** every anatomical edge has a trainable signed weight (the FlyGPT
    setup). This tests the wiring as an architecture.
  - **Frozen biological:** weight = synapse count × transmitter sign, fixed. Only the
    input encoder gains and the readout are trained. This is the approach FlyDrones and
    haltere take, and it is what makes "the fly's own circuitry does the flying" a
    testable claim.
  - **Signs** come from `body-neurotransmitters-male-cns-v1.0.feather` (`consensus_nt`;
    166,522 of 166,700 neurons covered):
    - acetylcholine is excitatory; GABA, glutamate and histamine are inhibitory
    - "unclear" (2,999) falls back to the cell-type prediction
    - neuromodulators (dopamine, octopamine, serotonin: 541 neurons) get zero fast
      weight; including them is a declared sensitivity variant
    - one global gain scales weights so the linearized dynamics start stable, set from
      the spectral radius and never from flight performance
  - **Rewired controls** in the frozen arm carry each edge's biological weight and sign
    onto its new endpoint.
- **Random-I/O arm.** Same channel counts, with FlyGPT's population-seed rule over the
  whole graph and disjoint populations.
- **Ticks per control step.** Set from graph distance: at least the maximum
  input→readout hop count across all input channels, currently 4. Performance never
  chooses it.
- **Reachability audit.** Every input channel reaches the readout within the declared
  ticks in the real graph *and* in every control graph.

**Gate:** contract frozen, audits pass, and the random and anatomical maps are both
hashed.

**Result (2026-09-23): passed.** See `results/connectome_flight/cp1/CP1_REPORT.md` and
`connectome-pilot-v1.json`.
- **Eye map:** a regular hex lattice, with orientation from lamina positions and
  declared anchors at Δφ = 5°.
- **Depth:** 59 of 64 cells mapped. The 5 uncovered upper-front cells are dropped for
  every arm.
- **Readout:** all 1,314 descending neurons.
- **Ticks per control step: 4**, the hops needed to reach half the readout in the real
  graph and 10 rewired controls.
- **Stage A output:** a 3-dimensional velocity correction into the authored velocity
  loop.

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
2. **Frozen-weights pilot first.** It is the cheapest arm to train: the
   166,700-neuron graph only runs forward, and with a readout-only variant there is no
   backpropagation through it. Report:
   - whether the frozen dynamics are stable and non-saturating on flight inputs
   - whether descending-neuron activity varies with visual obstacles at all, compared
     with a camera-blinded run, before any readout is fitted
   - closed-loop passes on training seeds after fitting the readout
3. **Full-graph pilot.** One seed each of MaleCNS-anatomical and the MLP, trained on V5
   training rows with V5's loss (4·nav-velocity + 0.2·action + 0.05·rollout-velocity +
   4·clearance + 0.1·arrival-speed + 0.001·control-change). Use DAgger on training seeds
   only.
4. **Report:**
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
| MaleCNS, anatomical I/O, trained weights | Treatment (architecture question) |
| Degree-preserving rewired, anatomical I/O, trained weights | **Primary control** for the row above |
| MaleCNS, anatomical I/O, **frozen biological weights** | Treatment (native circuitry question) |
| Degree-preserving rewired, frozen weights carried with their edges | **Primary control** for the row above |
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
  2–6 m band on the tree course. The paired difference is MaleCNS minus degree-rewired,
  computed separately for the trained and frozen weight models. That makes two primary
  contrasts, Holm-corrected.
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

## Prior work (checked 2026-09-23)

- **[FlyDrones](https://github.com/pietroagazzi/FlyDrones)** (the project behind @c10ned's
  posts): the full MaleCNS graph runs as a frozen Shiu-2024 LIF network with
  transmitter-signed weights. Computed optic flow is injected into T4/T5 and computed
  looming into LPLC2/LC4. Six hand-mapped DNs drive the sticks: DNg02 L+R gives
  throttle, R−L gives yaw, and DNp01 gives escape. A safety governor sits outside the
  brain. There is no quantitative evaluation and there are no controls, and the
  browser demo uses an 850-neuron stand-in.
- **[haltere](https://github.com/skulitom/haltere):** a 30,000-neuron MaleCNS rate
  subgraph with fixed topology. Encoders and motor readouts are trained by imitation
  and RL, and it flies Liftoff through its stick → PID → mixer stack. It has an MLP
  baseline but no rewired controls. Its README reports 0/5 first-exposure courses and
  "brain 0/2 finishes; PD 1/2".
- **This study adds:** matched rewired and lesion controls, locked fresh test flights,
  pass/fail grading, a realistic Betaflight flight stack, and a full-graph comparison of
  trained against frozen weights. Any engineered visual features injected mid-pathway,
  like FlyDrones' T4/T5 and LPLC2 inputs, are labeled engineered input.

## Where things live

- Connectome engine, I/O maps and studies: this repo, on a new branch
  `experiment/connectome-flight`.
- Sim, teacher, grading and the observation bridge: `fire-drone/sim`, with a separate
  scenario contract.
- The fleet stays on its authored autopilot until a connectome pilot passes the fleet's
  own acceptance.
