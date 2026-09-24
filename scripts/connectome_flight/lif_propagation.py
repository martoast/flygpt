"""CP2: does vision reach the descending neurons in the frozen LIF brain, selectively
and without runaway? Same question the rate model failed (frozen_gain_scan).

Drive: every L1/L2 neuron of the 59 mapped depth cells gets Poisson input at
r_poi_max * x, with x drawn uniformly per cell every 50 ms (a 20 Hz control
step); then input is switched off. Diagnostic only; no flight, nothing selected.

Usage: python -m scripts.connectome_flight.lif_propagation --out results/connectome_flight/cp2/lif_propagation.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from scripts.connectome_flight.cp2_benchmark import BODY_IDS, IO_MAP
from scripts.connectome_flight.lif import PARAMS, calibrated_params, load_brain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds-on', type=float, default=1.0)
    parser.add_argument('--seconds-off', type=float, default=0.5)
    parser.add_argument('--published-wsyn', action='store_true', help='uncalibrated FlyWire w_syn')
    args = parser.parse_args()
    io = json.load(open(IO_MAP))['anatomical']
    readout = np.array(io['readout'])
    cells = [np.array(io['channels'][f'depth[{i}]'], dtype=np.int64) for i in range(64)]
    stim = np.concatenate([c for c in cells if len(c)])
    cell_of = np.concatenate([np.full(len(c), i) for i, c in enumerate(cells) if len(c)])
    params = PARAMS if args.published_wsyn else calibrated_params()
    brain = load_brain(stim, params=params)
    ids = np.load(BODY_IDS)
    ann = pd.read_feather('data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId').reindex(ids)
    superclass = ann.superclass.fillna('unknown').to_numpy()
    rng = np.random.default_rng(0)

    windows = []
    total = np.zeros(brain.n, np.int64)
    start = time.perf_counter()
    for w in range(int(args.seconds_on * 20)):
        x = rng.uniform(0, 1, 64)
        counts = brain.run(50, PARAMS['r_poi_max'] * x[cell_of])
        total += counts
        windows.append({'window': w, 'input': True, 'spikes': int(counts.sum()),
                        'dnSpikes': int(counts[readout].sum()), 'dnActive': int((counts[readout] > 0).sum())})
    on_seconds = time.perf_counter() - start
    for w in range(int(args.seconds_off * 20)):
        counts = brain.run(50, np.zeros(len(stim)))
        windows.append({'window': len(windows), 'input': False, 'spikes': int(counts.sum()),
                        'dnSpikes': int(counts[readout].sum()), 'dnActive': int((counts[readout] > 0).sum())})
    active = total > 0
    by_class = pd.Series(superclass[active]).value_counts().head(12).to_dict()
    rate_hz = total / args.seconds_on
    report = {
        'params': params,
        'stimulatedNeurons': int(len(stim)),
        'secondsPer50msWindow': on_seconds / (args.seconds_on * 20),
        'neuronsThatSpiked': int(active.sum()),
        'spikingBySuperclass': by_class,
        'readout': {
            'dnThatSpiked': int(active[readout].sum()),
            'of': int(len(readout)),
            'meanRateHz': float(rate_hz[readout].mean()),
            'maxRateHz': float(rate_hz[readout].max()),
        },
        'nonStimulatedMeanRateHz': float(np.delete(rate_hz, stim).mean()),
        'windows': windows,
        'spikesPer50msAfterInputOff': [w['spikes'] for w in windows if not w['input']],
    }
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ('windows', 'params')}, indent=1))


if __name__ == '__main__':
    main()
