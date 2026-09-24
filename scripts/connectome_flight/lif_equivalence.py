"""Check the parallel LIF engine against the single-core dense reference and time both.

Same seed, same stimulus sequence (random depth drive at 20 Hz windows, then input
off). Reports per-window total/DN spike counts for both engines, the fraction of
neurons with identical spike counts, and wall time per 50 ms of brain time.

Usage: python -m scripts.connectome_flight.lif_equivalence --out results/connectome_flight/cp2/lif_equivalence.json
"""
import argparse
import json
import time

import numpy as np

from scripts.connectome_flight.cp2_benchmark import IO_MAP
from scripts.connectome_flight.lif import load_brain


def drive(engine, windows_on, windows_off, stim, cell_of, readout):
    brain = load_brain(stim, engine=engine)
    rng = np.random.default_rng(0)
    brain.run(0.1, np.zeros(len(stim)))  # compile outside timing
    brain.reset()
    total = np.zeros(brain.n, np.int64)
    per_window, wall = [], []
    for w in range(windows_on + windows_off):
        rates = 150.0 * rng.uniform(0, 1, 64)[cell_of] if w < windows_on else np.zeros(len(stim))
        start = time.perf_counter()
        counts = brain.run(50, rates)
        wall.append(time.perf_counter() - start)
        total += counts
        per_window.append((int(counts.sum()), int(counts[readout].sum())))
    return total, per_window, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--windows-on', type=int, default=20)
    parser.add_argument('--windows-off', type=int, default=10)
    args = parser.parse_args()
    io = json.load(open(IO_MAP))['anatomical']
    readout = np.array(io['readout'])
    cells = [np.array(io['channels'][f'depth[{i}]'], dtype=np.int64) for i in range(64)]
    stim = np.concatenate([c for c in cells if len(c)])
    cell_of = np.concatenate([np.full(len(c), i) for i, c in enumerate(cells) if len(c)])
    results = {}
    for engine in ('parallel', 'dense'):
        total, per_window, wall = drive(engine, args.windows_on, args.windows_off, stim, cell_of, readout)
        results[engine] = {'total': total, 'perWindow': per_window, 'wall': wall}
    a, d = results['parallel'], results['dense']
    report = {
        'identicalSpikeCountFraction': float((a['total'] == d['total']).mean()),
        'totalSpikes': {'parallel': int(a['total'].sum()), 'dense': int(d['total'].sum())},
        'dnSpikes': {'parallel': int(a['total'][readout].sum()), 'dense': int(d['total'][readout].sum())},
        'maxAbsNeuronCountDifference': int(np.abs(a['total'] - d['total']).max()),
        'wallSecondsPer50ms': {
            e: {'withInputMean': float(np.mean(results[e]['wall'][:args.windows_on])),
                'inputOffMean': float(np.mean(results[e]['wall'][args.windows_on:]))}
            for e in results
        },
        'perWindow': {e: results[e]['perWindow'] for e in results},
    }
    on = report['wallSecondsPer50ms']['parallel']['withInputMean']
    report['realTimeFactorParallelWithInput'] = 0.05 / on
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != 'perWindow'}, indent=1))


if __name__ == '__main__':
    main()
