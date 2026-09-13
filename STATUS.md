# FlyGPT Lab v0.3 — Status

## What has been demonstrated in this runtime

The end-to-end query path has been validated on a small synthetic sparse recurrent graph with **disjoint input and output neuron populations** and **three recurrent micro-steps per input symbol**.

Validation task:
- Prompt 1: `Q:a?\nA:` → target `x`
- Prompt 2: `Q:b?\nA:` → target `y`
- Both prompts share the same final `A:` suffix, so the answer cannot be determined from the current symbol alone.
- Network: 32 recurrent nodes, 320 directed edges, 9-character vocabulary.
- Seed: 123.

Result after training:
- Intact recurrent graph: `a -> x`, `b -> y`
- All recurrent edge weights set to zero: both prompts collapse to `:`.

This is a causal ablation showing that prompt-specific information traverses the recurrent graph. It is **not yet a MaleCNS result** because the official binary graph could not be downloaded into this hosted runtime.

Raw result: `results/query_path_validation.json`.

## Architecture correction from v0.2

v0.2 allowed the input projection and output readout to touch the same global hidden state. In an ablation, zeroing recurrent edges did not destroy the toy answer, revealing a shortcut. v0.3 fixes this by:

1. Injecting symbols only into a dedicated input-neuron population.
2. Reading logits only from a disjoint output-neuron population.
3. Giving the recurrent graph multiple internal propagation ticks per symbol.

This makes recurrent connectivity causally necessary for prompt-dependent responses.

## Next run on the real connectome

1. Run `python download_malecns.py` on a machine with normal outbound access.
2. Convert the official Feather edge table with `python -m src.prepare_malecns ...`.
3. Run temporal-memory and query-path tests on the real MaleCNS graph plus matched controls.
4. Train byte/character language models.
5. Train a TinyGPT teacher and distill into the MaleCNS-constrained student.
6. Run edge, region, sign, quantization, and weight-noise ablations.

## Scientific boundary

A successful digital MaleCNS language model would show that a computation can be instantiated on a biologically derived topology. It would **not** demonstrate that an intact living fly brain has been programmed. Physical synaptic-state writing remains a separate unsolved wetware problem.
