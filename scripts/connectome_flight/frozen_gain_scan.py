"""CP2 exploratory: frozen weights normalised by each neuron's total input synapses, gain scan
against the propagation rule (DN reach >= 50%, saturation < 1%, activity decays after input off).
Diagnostic only; selects nothing by flight performance.

Usage: python -m scripts.connectome_flight.frozen_gain_scan
"""
import json, numpy as np, scipy.sparse as sp
from scripts.connectome_flight.cp2_benchmark import load_graph, presynaptic_signs, LEAK, TICKS, IO_MAP
W = load_graph(); n = W.shape[0]; sign, _ = presynaptic_signs(n)
total_in = np.asarray(W.sum(axis=1)).ravel()
S = sp.diags(1 / np.maximum(total_in, 1)).dot(W).tocsr().multiply(sign[np.newaxis, :]).tocsr().astype(np.float32)
io = json.load(open(IO_MAP))['anatomical']; readout = np.array(io['readout'])
depth = [np.array(io['channels'][f'depth[{i}]'], dtype=int) for i in range(64)]
for gain in np.arange(1.0, 3.01, 0.25):
    rng = np.random.default_rng(0); h = np.zeros(n, np.float32)
    for step in range(100):
        drive = np.zeros(n, np.float32)
        for g in depth:
            if len(g): drive[g] = rng.uniform(0, 1)
        for _ in range(TICKS): h = LEAK * h + (1 - LEAK) * np.tanh(gain * (S @ h) + drive)
    active = (np.abs(h[readout]) > 1e-3).mean(); sat = (np.abs(h) > 0.95).mean(); on = np.abs(h).mean()
    for step in range(100):  # input off for 1 s
        for _ in range(TICKS): h = LEAK * h + (1 - LEAK) * np.tanh(gain * (S @ h))
    residual = np.abs(h).mean() / max(on, 1e-12)
    ok = active >= 0.5 and sat < 0.01 and residual < 0.01
    print(f'gain {gain:.2f}: DN active {active:.3f}  saturated {sat:.4f}  residual after 1 s off {residual:.2e}  {"PASS" if ok else ""}')
