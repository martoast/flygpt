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
