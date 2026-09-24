"""CP2: connectome throughput on this machine, and the frozen biological network.

1. Tick cost of the full MaleCNS recurrent step (CSR, rows = destinations), forward
   only and forward + backward (the FlyGPT CSRMultiply kernels), at several batch
   sizes. Disposable measurements: nothing here selects a setting by performance.
2. The frozen-weights network of connectome-pilot-v1: weight = synapse count x
   transmitter sign (consensus_nt), one global gain setting the spectral radius of
   the scaled recurrent matrix to 0.9 (contract rule, fixed before any flight).
3. A drive check: inject random input into the anatomical L1/L2 depth channels and
   confirm the frozen dynamics stay bounded and reach the descending-neuron readout.

Usage: python -m scripts.connectome_flight.cp2_benchmark --out results/connectome_flight/cp2/benchmark_m1.json
"""
import argparse
import json
import platform
import resource
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from src.provenance import sha256
from src.sparse_ops import edge_gradient

GRAPH = 'data/processed/malecns.npz'
BODY_IDS = 'data/processed/malecns_body_ids.npy'
NT = 'data/raw/body-neurotransmitters-male-cns-v1.0.feather'
IO_MAP = 'results/connectome_flight/cp1/io_map.json'
CONTRACT = 'results/connectome_flight/cp1/connectome-pilot-v1.json'
SIGN = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0, 'histamine': -1.0}
TARGET_RADIUS = 0.9
LEAK = 0.65
TICKS = 4


def load_graph():
    """CSR with rows = destinations (post), columns = sources (pre), synapse counts."""
    z = np.load(GRAPH)
    A = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))  # A[pre, post]
    return A.T.tocsr().astype(np.float32)


def presynaptic_signs(n):
    ids = np.load(BODY_IDS)
    nt = pd.read_feather(NT).set_index('body').reindex(ids)
    consensus = nt.consensus_nt.astype(object)
    unclear = consensus.eq('unclear')
    consensus[unclear] = nt.celltype_predicted_nt[unclear]
    sign = consensus.map(SIGN).astype(float).fillna(0.0).to_numpy()  # modulators/missing -> 0
    counts = {
        'excitatory': int((sign > 0).sum()),
        'inhibitory': int((sign < 0).sum()),
        'zeroFastWeight': int((sign == 0).sum()),
        'unclearResolvedByCellType': int(unclear.sum()),
    }
    return sign.astype(np.float32), counts


def timed(fn, repeats=3):
    fn()  # warm-up (Numba compile, caches)
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return float(np.median(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--batches', default='1,8,32')
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    W = load_graph()
    n = W.shape[0]
    report = {
        'machine': {'platform': platform.platform(), 'processor': platform.processor(), 'python': platform.python_version()},
        'inputs': {p: sha256(p) for p in (GRAPH, BODY_IDS, NT, IO_MAP, CONTRACT)},
        'nNeurons': n,
        'nEdges': int(W.nnz),
    }

    # 1. Tick timing on unit random weights (cost does not depend on values).
    Wt = W.T.tocsr()
    timing = {}
    for batch in [int(b) for b in args.batches.split(',')]:
        H = rng.standard_normal((batch, n)).astype(np.float32)
        G = rng.standard_normal((batch, n)).astype(np.float32)
        forward = timed(lambda: W @ H.T)
        backward = timed(lambda: (Wt @ G.T, edge_gradient(H, G, W.indices, W.indptr)))
        timing[str(batch)] = {'forwardSeconds': forward, 'backwardSeconds': backward, 'forwardBackwardSeconds': forward + backward}
    report['tickTiming'] = timing
    one = timing['1']
    flight_seconds = 30
    report['projection'] = {
        f'{hz}Hz': {
            'ticksPerFlight': flight_seconds * hz * TICKS,
            'forwardOnlySecondsPerFlight': flight_seconds * hz * TICKS * one['forwardSeconds'],
            'forwardBackwardSecondsPerFlightBatch1': flight_seconds * hz * TICKS * one['forwardBackwardSeconds'],
        }
        for hz in (100, 20)
    }

    # 2. Frozen biological network.
    sign, sign_counts = presynaptic_signs(n)
    frozen = W.multiply(sign[np.newaxis, :]).tocsr()  # column j = presynaptic j
    frozen.eliminate_zeros()
    start = time.perf_counter()
    radius_raw = float(abs(eigs(frozen.astype(np.float64), k=1, which='LM', return_eigenvectors=False, maxiter=5000, tol=1e-6)[0]))
    gain = TARGET_RADIUS / radius_raw
    report['frozen'] = {
        'signCounts': sign_counts,
        'edgesWithFastWeight': int(frozen.nnz),
        'rawSpectralRadius': radius_raw,
        'globalGain': gain,
        'targetSpectralRadius': TARGET_RADIUS,
        'eigenSeconds': time.perf_counter() - start,
    }
    frozen = (frozen * gain).astype(np.float32)

    # 3. Drive check: 64 depth channels at plausible amplitude, frozen dynamics.
    io = json.load(open(IO_MAP))['anatomical']
    readout = np.array(io['readout'])
    depth = [np.array(io['channels'][f'depth[{i}]'], dtype=int) for i in range(64)]
    h = np.zeros(n, np.float32)
    trace = []
    for step in range(200):  # 2 s at 100 Hz, 4 ticks each
        drive = np.zeros(n, np.float32)
        for group in depth:
            if len(group):
                drive[group] = rng.uniform(0, 1)
        for _ in range(TICKS):
            h = LEAK * h + (1 - LEAK) * np.tanh(frozen @ h + drive)
        if step % 20 == 19:
            trace.append({
                'step': step + 1,
                'meanAbsAll': float(np.abs(h).mean()),
                'meanAbsReadout': float(np.abs(h[readout]).mean()),
                'fractionSaturated': float((np.abs(h) > 0.95).mean()),
                'readoutActive': float((np.abs(h[readout]) > 1e-3).mean()),
            })
    report['frozenDriveCheck'] = {
        'note': 'random uniform(0,1) per depth cell per step into anatomical L1/L2 channels; no input gains or readout trained',
        'trace': trace,
        'bounded': bool(np.isfinite(h).all() and np.abs(h).max() <= 1.0),
    }
    report['peakRssBytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ('tickTiming', 'projection', 'frozen')}, indent=1))
    print(json.dumps(report['frozenDriveCheck']['trace'][::3], indent=1))


if __name__ == '__main__':
    main()
