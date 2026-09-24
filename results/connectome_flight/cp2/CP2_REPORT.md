# CP2 — throughput and the frozen-weights network (M1, in progress)

2026-09-23. This covers `benchmark_m1.json` (`scripts/connectome_flight/cp2_benchmark.py`)
and `frozen_gain_scan.txt` (`scripts/connectome_flight/frozen_gain_scan.py`). These are
disposable measurements: nothing here is selected by flight performance.

## Throughput (Apple M1, 4 threads, SciPy CSR plus the FlyGPT Numba edge gradient)

The graph is the full MaleCNS: 166,700 neurons and 25,582,938 synapse pairs.

| Batch | Forward per tick | Forward + backward per tick |
|---|---|---|
| 1 | 25 ms | 70 ms |
| 8 | 287 ms (36 ms per sample) | 649 ms |
| 32 | 299 ms (9.4 ms per sample) | 1.33 s (42 ms per sample) |

At 4 ticks per control step (the CP1 contract), a 30 s flight costs:

| Control rate | Ticks per flight | Forward only (frozen arm, inference) | Forward + backward, batch 1 |
|---|---|---|---|
| 100 Hz | 12,000 | about 5.1 min | about 14 min |
| 20 Hz | 2,400 | about 1.0 min | about 2.8 min |

**Implication:** 20 Hz control, driven by camera frames, is affordable for the frozen
arm and for evaluation. The trained-weights arm is CPU-expensive. Batching
(9 ms per sample at batch 32) and the Omarchy GTX 1070 are still to be benchmarked
before the rate and hardware decision is frozen.

## The frozen biological network does not transmit as contracted

**Contract rule (CP1):** weight = synapse count × presynaptic transmitter sign, with one
global gain setting the spectral radius to 0.9.
- Signs: 103,720 excitatory neurons, 59,262 inhibitory, and 3,718 with zero fast
  weight (neuromodulators or missing). 2,999 "unclear" neurons were resolved by their
  cell type.
- The raw spectral radius is 3,777.6, dominated by hub neurons, so the gain comes out
  at 2.4e-4.
- **Drive check:** random visual input into the anatomical L1/L2 channels for 2 s gives
  a mean descending-neuron |activity| of about 6e-6, and 0 of 1,314 DNs above 1e-3.
  The rule makes the network silent at the readout.

**Diagnostic alternative:** weights normalised by each neuron's total incoming synapse
count, so each input is the fraction of that neuron's synapses, with sign kept.
- The spectral radius at gain 1 is already 0.915.
- The gain scan (`frozen_gain_scan.txt`) found **no gain that meets all three
  propagation conditions**: reaching at least 50% of DNs, under 1% saturation, and
  activity decaying after the input stops.
  - At gain 1.25 or below, the signal fades before the readout (1–3% of DNs reached).
  - At gain 2.0 or above, it reaches 82–98% of DNs, but 56–83% of the activity
    remains 1 s after the input is removed. The network becomes self-sustaining once
    gain × radius exceeds 1.

**Interpretation:** a threshold-free rate model (leaky tanh) on the fly's own weights
either attenuates the 3–4-hop eye-to-DN pathway, or it reverberates globally.
Biological selectivity depends on firing thresholds, which is why the published
whole-brain fly model (Shiu et al. 2024) is spiking (LIF). This is a pre-flight finding
and requires a documented amendment to the CP1 weight-model rule before the frozen arm
can be piloted.
