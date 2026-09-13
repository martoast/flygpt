# FlyGPT preregistration v0.1

## Primary question
Can a recurrent neural network whose allowed recurrent edges are fixed by the MaleCNS connectome learn autoregressive sequence/language tasks, and does real biological topology differ from topology-matched controls?

## Primary hypotheses
H1: Real MaleCNS topology achieves lower held-out loss than an untrained/readout-only baseline.
H2: Under identical neuron count, edge count, parameterization, optimizer, data order, and compute budget, real topology differs from a directed degree-preserving rewired control.
H3: A connectome-constrained student can reduce KL divergence to a frozen tiny-transformer teacher during distillation.

## Primary controls
1. Directed degree-preserving edge-swap graph.
2. Configuration-model graph preserving in/out-degree distributions.
3. Erdos-Renyi graph matched on N and E.
4. Conventional GRU with matched trainable parameter budget where feasible.

## Primary endpoints
- Synthetic delayed-recall accuracy.
- Validation cross entropy / bits per byte on byte-level language modeling.
- Distillation efficiency E=(L0-Lstudent)/(L0-Lteacher), reported with confidence intervals.
- Effective memory horizon on delayed-recall probes.

## Statistical protocol
- Minimum 5 random seeds per condition for screening; 10+ for confirmatory comparisons.
- Hyperparameter search budget identical across topology conditions.
- Report all attempted confirmatory configurations, not only successes.
- Bootstrap 95% CI across seeds; paired comparisons when data order/initialization can be paired.
- Hold a final untouched test set until architecture/hyperparameters are frozen.

## Falsification criteria
The topology-specific hypothesis is unsupported if the real graph fails to outperform degree-preserving rewired controls consistently after matched tuning and compute, or if any apparent advantage disappears under initialization/degree matching.

## Staged biological constraints
A0 unrestricted signed effective weights on allowed edges.
A1 initialize magnitude from synapse count.
A2 impose neurotransmitter-derived sign constraints.
A3 restrict weight drift around anatomy-informed initialization.
A4 neuron-type-specific dynamics.
A5 spiking/LIF dynamics with biologically plausible delays/time constants.

## Interpretation boundary
Success demonstrates functional realizability in a connectome-constrained *model*. It does not demonstrate that the learned state can currently be written into living tissue, nor that anatomical synapse count equals a freely programmable biological weight.

## 2026-09-12 hardware-limited screen v1 (before training)

The machine is an 8 GB Apple M1. The complete annotated-neuron graph has
166,700 nodes and 25,582,938 directed pairs. The initial fixed budgets are in
`configs/screen_v1.json`: five seeds, all four graph conditions plus GRU,
24 mixed-task memory updates and 64 byte-LM updates. This is a deliberately
small feasibility screen (512 language-training bytes per seed), **not a
sufficient language-capability or topology-advantage experiment**. Every
attempt and failure must be reported. No fly-only tuning is allowed.

Neuron inclusion: all unique annotation body IDs with non-null `superclass`,
including isolated nodes. No edge threshold beyond one aggregated contact.
Raw segments outside that universe are excluded and counted in provenance.

Use random, fixed, disjoint input/output populations (population seed 2026),
trainable signed weights initialized N(0, 0.9² / destination indegree),
leak 0.65 and two ticks per byte. Anatomy-derived self-edges are allowed.
No extra recurrent edges are allowed in the real condition. The neuron-local
leak is part of the specified dynamics, not an added anatomical edge.

The directed rewire fixes existing self-loops and requests 10 successful
swaps per total edge. It reports acceptance and overlap; this does not prove
Markov-chain mixing. Configuration controls are directed stub-matched
multigraphs (self-loops and parallel edges allowed, independent edge weights).
ER uses the same N, E and self-loop count. GRU is approximately parameter-matched.
Equal tokens/steps do not imply equal wall-clock or FLOPs; report runtime.

Memory uses a mixture of delayed bit/symbol, length-three copy, two-item
associative recall, three-state grammar, and parity. Each task's readout is
restricted to its known answer alphabet. C(delay) is reported as accuracy and
chance-adjusted accuracy, not as a classical linear-reservoir capacity estimate.
The eight-target-per-delay evaluation is too small for strong conclusions.
The byte corpus is a synthetic compositional grammar with independently drawn
splits, not a natural-language understanding benchmark. Report untrained,
unigram and zero-edge losses, and paired seed-wise comparisons. Distillation
uses the same initial seed and token budget as supervised training. The
teacher has a separately reported, larger training budget.

Ordering amendment before the first language result: the user explicitly
prioritized closing the gap to the grammar teacher (0.324945 nats/byte on the
fixed validation windows). After the initial real-graph memory screen, run the
language conditions before the remaining memory conditions. Budgets and seeds
are unchanged. A single successful checkpoint is feasibility evidence only;
biological-topology claims still require the matched multi-seed comparisons.

Target continuation (specified before the first language validation result):
`configs/target_v1.json` fixes a 512-update continuation from the 64-update CE
pilot, with optimizer reset, 32-byte contexts, learning rate 0.001, alpha 0.5,
and temperature 2. Evaluation is at 64/128/256/512 updates on the same windows.
"Close" is operationalized as <=0.425 nats/byte, about 0.10 above the teacher.
The graph, populations, leak, and ticks remain unchanged. This is a feasibility
run, not a biological-topology comparison. Apply the same extension to controls;
do not report topology superiority until matched multi-seed runs complete.

Read-only diagnostic amendment requested during baseline A: at every saved
checkpoint, record forward teacher KL (T=1 and training-scaled T=2), teacher
argmax agreement, zero-edge CE, pre-clipping training gradient norms,
hidden-state L2/RMS and fraction abs(h)>0.95 at byte boundaries, cumulative
training bytes, and elapsed training/diagnostic time. A companion process
reads immutable checkpoint snapshots; it does not change the running model.

After the fixed 512-update final checkpoint, evaluate the previously untouched
synthetic-grammar test split exactly once with non-overlapping 32-byte target
windows and the same frozen teacher. Use the final checkpoint, not a test-selected
checkpoint. Paired window bootstrap intervals describe text-window variability,
not variability across training seeds or proof of a biological topology effect.
