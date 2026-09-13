# FlyGPT status — 2026-09-12

## Latest measured checkpoint: 512 continuation updates

**Validation CE: 0.851437 nats/byte**, versus frozen teacher **0.324945**.
The trajectory is **3.175374 → 2.862806 → 2.522078 → 1.871875 → 0.851437**
at 0/64/128/256/512 additional updates. This supports continued learning on
this finite grammar task, not an established asymptote or a capacity limit.

- Teacher gap: 0.526492 nats/byte.
- Forward teacher KL (T=1): 0.574970 nats/byte.
- Teacher argmax agreement: 73.05%; target-byte accuracy: 69.14%.
- Zero-edge ablation CE: 3.107468, a +2.256031 penalty relative to intact.
- Last gradient L2 norm before clipping: 1.5157 (clipping threshold 1.0).
- Hidden-state mean L2 norm: 81.34; mean RMS: 0.1992.
- Fraction of sampled hidden values with abs(h)>0.95: 1.1744%.
- Total student training bytes seen: 16,896, including the 512-byte CE pilot.
- Cumulative training-process wall time: 2,943 seconds (about 49 minutes).

The biological topology, populations, ticks, leak, and optimizer settings were
unchanged during continuation. The model is still outside the predefined
near-teacher threshold of 0.425. The same baseline has resumed toward 1,024
updates with optimizer and data RNG state preserved. Conditional continuation
can reach 2,048; the test split remains untouched until those decisions finish.
No architectural variant has been started.

The local 512-step checkpoint is
`results/malecns_v1/target/snapshots/real_0_step_0512.pt`.
`results/fly_real.pt` remains the original supervised pilot for provenance;
use the snapshot path to query this newer model. All 15 control graphs (five
seeds each of rewired/configuration/ER) have been constructed, but the matched
training comparison remains incomplete. No biological-topology advantage or
causal benefit of distillation over matched CE-only training is established.

**Task boundary:** all validation sentences appear somewhere in the finite
training corpus. This tests learning its predictive distribution, not novel
composition, general language understanding, or wetware programming. Full
metrics and plot: `results/malecns_v1/target/TRAJECTORY.md` and `trajectory.png`.

The sections below retain the earlier pilot results and context.

## Measured real-data progress

The official MaleCNS v1.0 minconf-0.5 segment table and curated annotations have
been downloaded, hashed, and processed locally. The node rule is **non-null
annotation superclass**, retaining every selected node, including isolates:

- 166,700 annotated neurons.
- 25,582,938 directed neuron-pair edges.
- 124,177,617 anatomical synaptic contacts between selected neurons.
- Giant strongly connected component: 165,314 neurons.

The raw segment table contains 151,856,684 rows and 311,833,243 contacts;
126,273,746 rows have at least one endpoint outside the annotated neuron
universe. Those excluded segments must not be mislabeled as extra neurons.
See `results/malecns_v1/graph_provenance.json` and `graph_statistics.json` for
source hashes, distributions, sampled path/clustering estimates, SCCs,
structural spectral estimates, and annotation coverage.

Hardware: Apple M1, 8 GB RAM. Full-graph CPU CSR propagation with custom exact
first-order autograd is feasible. CPU and MPS scatter were benchmarked and
were slower. Full training/evaluation is substantially more expensive than
an isolated recurrent tick. No full dense adjacency is allocated.

## First language checkpoint: trained, but unsuccessful

`results/fly_real.pt` is a local 101 MB checkpoint trained on the complete
annotated MaleCNS graph, with fixed topology, trainable signed weights,
disjoint input/output populations and two recurrent ticks per byte.
The first supervised pilot used only 64 updates / 512 training bytes.

On the same 256 validation bytes from a **synthetic grammar corpus**:

| Model / intervention | CE, nats per byte |
|---|---:|
| Frozen three-layer TinyGPT teacher | 0.324945 |
| Untrained MaleCNS model | 5.537795 |
| Trained MaleCNS pilot | 3.175374 |
| MaleCNS pilot, all recurrent edges zeroed | 3.081525 |
| Smoothed training-corpus unigram baseline | 2.889558 |

The pilot is **not close to the teacher**. It also fails to beat the unigram
baseline, and removing recurrence slightly improves its loss. Common-suffix
prompts change intact logits, but this signal is not yet useful for prediction.
Autoregressive querying runs and yields largely incoherent text. This does
**not** meet the non-trivial held-out language / necessary recurrence success
criterion. The arbitrary question "Why is the sky blue?" is outside the
training corpus and has no reason to receive a meaningful answer.

The first 24-update mixed-task memory screen is also inconclusive. Only one
seed of real and GRU memory pilots has completed at this status update; do not
present it as a five-seed result. Completed records live under
`results/malecns_v1/memory/` and `language/`.

## Active teacher-gap continuation

The user prioritized approaching the teacher's loss without changing topology.
`configs/target_v1.json` fixes 512 additional updates with 32-byte contexts,
frozen-teacher CE/KL distillation, learning rate 0.001, alpha 0.5 and temperature
2. "Close" is defined in advance as <=0.425 nats/byte. Architecture/topology are
unchanged. Checkpoints and validation results are saved at 64/128/256/512 steps.
The run's existence is not evidence of success: consult
`results/malecns_v1/target/real_0.json` when available for completed checkpoints.

The five-seed matched screen is implemented and preregistered but **not yet
complete**. Degree-preserving rewires request ten successful swaps per edge;
construction is expensive on this machine. The screen ladder can resume from
completed result files. It has been paused to prioritize the teacher-gap run;
control construction continues separately. No A_bio result is available yet.

## What is not demonstrated

No useful natural-language MaleCNS model, topology advantage, successful
teacher compilation, biological sign/time-constant constraints, spiking
implementation, or wetware programming has been established. Later biological
realism and programmability claims are gated on stronger computational results.

## Historical synthetic milestone

The older 32-node synthetic query-path experiment remains a software causal
check: common-suffix prompts yielded different answers intact and collapsed
under zero-edge ablation. It is a **synthetic graph**, not a MaleCNS finding.
The query implementation's repeated-last-prompt-byte bug and hard-coded hosted
paths have now been fixed. Automated tests check exact sparse gradients,
disjoint populations, no bypass, topology-bound checkpoint loading, teacher
causality, directed nulls, and preprocessing aggregation.

### Continuation trajectory update

The first two saved continuation checkpoints reduce validation CE from
3.175374 (start) to 2.862806 (64 additional updates) to 2.522078 (128).
That supports an initial learning trend, **still far from the teacher**.
It does not establish convergence or a capacity limit. All requested causal,
KL, agreement, gradient, state and timing diagnostics are recorded separately
in `results/malecns_v1/target/TRAJECTORY.md` and `trajectory.json` as each
checkpoint is inspected. The active architecture and optimizer remain fixed.

The fixed interface audit found 1,025 input and 1,025 output neurons with zero
overlap, 938 direct input-to-output anatomical edges, and four output neurons
unreachable from any input. Those nodes remain in baseline A. This audit is
saved for the later placement variant; it is not a reason to alter the active
run after observing losses.

At 256 additional updates, validation CE is **1.871875**, teacher KL is
**1.601164**, and zero-edge CE is **2.956929**. This is continued improvement
and a positive recurrent-edge ablation penalty on the fixed validation windows;
it remains far from the teacher's 0.324945 CE.

Corpus limitation: the grammar has only 384 possible sentences; the 2,000-line
training corpus contains 379 unique sentences, and all 100 validation sentences
occur verbatim somewhere in it. Split byte positions are distinct, as specified
before training, but this task cannot establish novel-composition generalization.
This audit does not change baseline A or unblind the final test split.

### Preservation and primary control queue

The 512-update milestone has an independently copied local archive at
`artifacts/malecns_seed0_step0512/`. Its tracked manifest is
`results/malecns_v1/target/milestone_0512_archive.json`. Checkpoint, teacher,
graph, body-ID mapping, pilot, corpus and metric copies were hash-verified.
All AdamW parameter states are at update 512; replaying the data sampler
reproduces the saved RNG state exactly. Exact validation input/target bytes
are archived. This is a same-disk copy, not an off-machine backup; large
artifacts remain excluded from Git. The historical dirty-worktree flag is
retained rather than treating the commit alone as complete source provenance.

`python -m scripts.matched_rewired` queues seed-zero degree-preserving rewiring
behind the existing baseline controller. It runs the same 64-update CE pilot,
then the 512/1024/2048 continuation stages actually completed by baseline A,
using identical data, optimizer settings, sampling seeds and architecture.
Control performance does not determine its budget. Each completed stage gets
the same KL, agreement, zero-edge, gradient and hidden-state diagnostics.
Runtime is measured, not claimed equal. This first pair cannot establish a
topology advantage; five paired exploratory seeds remain outstanding.
