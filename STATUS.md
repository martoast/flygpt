# FlyGPT status — 2026-09-12

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
