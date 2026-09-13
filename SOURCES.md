# Sources and interpretation (checked 2026-09-12)

## Dataset identity

- [Official MaleCNS download portal](https://male-cns.janelia.org/download/):
  versioned segment-pair Feather and curated annotations, CC-BY. The connection
  table includes all segments; it is not itself a list of neurons. Our complete
  annotated-neuron graph uses non-null annotation `superclass` and retains isolates.
- [MaleCNS project page](https://male-cns.janelia.org/): identifies the v1.0 release
  and links the [MaleCNS Cell paper](https://www.cell.com/cell/fulltext/S0092-8674%2826%2900942-6)
  and [preprint](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2).
  Paper full text was inaccessible from this research session; graph facts here
  are grounded in the official files and their recorded hashes.
- [Berg et al., Distributed control circuits across a brain-and-cord connectome](https://www.nature.com/articles/s41586-026-10735-w)
  concerns **BANC**, not the source of this MaleCNS graph. The previous source list
  placed it beside MaleCNS without making that distinction. Do not conflate datasets.

## Fixed-connectome optimization and controls

- [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3),
  [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/),
  [official FlyVis implementation](https://github.com/TuragaLab/flyvis):
  task-optimized, connectome-constrained fly visual-system models predict neural
  responses. This motivates differentiable fixed-topology models; it does not
  establish autoregressive language transfer or whole-MaleCNS superiority.
- [Dhiman, Topological Sensitivity in Connectome-Constrained Neural Networks, 2026 preprint](https://arxiv.org/abs/2604.04033):
  a FlyVis-based control study reports that initialization and degree matching
  remove apparent advantages. Treat this as a methodological warning and
  test topology with from-scratch, matched initialization and strong nulls.
- [Network statistics of the whole-brain connectome of Drosophila, Nature 2024](https://www.nature.com/articles/s41586-024-07968-y):
  precedent for graph statistics and topology analysis; this is a different
  connectome and cannot supply our MaleCNS measurements.

## Transfer and computational implementation

- [Hinton, Vinyals and Dean, Distilling the Knowledge in a Neural Network, 2015](https://arxiv.org/abs/1503.02531):
  teacher probability distributions and temperature-scaled distillation.
  Our objective is CE plus forward KL from frozen teacher to student.
- [BrainTrace, Model-agnostic linear-memory online learning in spiking neural networks, 2026](https://www.nature.com/articles/s41467-026-68453-w),
  [open text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12913608/):
  online learning and memory reduction for SNNs. Our current implementation
  instead uses exact first-order BPTT through a rate network with sparse CPU
  matrix products; it does not claim to implement BrainTrace.
- [Zhu et al., SpikeGPT, 2023](https://arxiv.org/abs/2302.13939):
  generative spiking language models in an engineered architecture. Spiking
  activations alone do not make topology connectome-derived or weights biologically writable.
- [Neuromorphic Simulation of Drosophila Melanogaster Brain Connectome on Loihi 2, 2025](https://arxiv.org/abs/2508.16792):
  reports FlyWire-scale neuromorphic simulation, not MaleCNS language training
  or physical implantation of a learned model in a fly.
- [Optimizing Memory in Reservoir Computers](https://arxiv.org/abs/2201.01605):
  memory/dynamics precedent. Our task accuracy versus delay is not the classical
  sum of linear reconstruction capacities and is labelled accordingly.

## Physical programmability boundary

- [Matsuzaki et al., Structural basis of long-term potentiation in single dendritic spines, 2004](https://www.nature.com/articles/nature02617)
  and [Photoactivatable CaMKII induces synaptic plasticity in single synapses, 2021](https://www.nature.com/articles/s41467-021-21025-6)
  demonstrate localized plasticity manipulations in experimental preparations.
  They do not demonstrate arbitrary setting of millions of weights in a living
  fly. No experiment in this repository uses living tissue.

Weight-noise or quantization tolerance would constrain the precision of a
*digital parameterization*, conditional on its task and interfaces. It would
not alone establish a synaptic programmer, correct biological dynamics,
read/write access, or transfer to a living animal.
