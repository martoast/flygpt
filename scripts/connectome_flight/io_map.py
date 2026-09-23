"""CP1: build the connectome pilot's input/readout map and audit reachability.

Maps each of the goal pilot's 83 observation channels (fire-drone
`controller.ts:visionObservation`) onto MaleCNS neurons, and fixes the readout
population. It produces two maps with identical channel sizes:

  anatomical  depth cells -> L1/L2 neurons of the columns viewing that direction
              (eye_map.json); body rates -> haltere afferents; velocity and gravity ->
              Johnston's-organ wind/gravity neurons; goal and previous action ->
              declared random central-brain neurons (non-biological).
              Readout: all descending neurons.
  random      each channel and the readout redrawn uniformly from the whole graph,
              same sizes, disjoint (FlyGPT population-seed rule, seed 2026).

Ticks per control step = the largest, over channels and over the real graph and every
degree-rewired control, of the hops a channel needs to reach half of the readout
neurons. Depth cells no eye column views are dropped for every arm, MLP included.
Performance never enters.

Usage: python -m scripts.connectome_flight.io_map --eye-map results/connectome_flight/cp1/eye_map.json \
           --out results/connectome_flight/cp1/io_map.json
"""
import argparse
import glob
import hashlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.graphs import load_npz

ANNOTATIONS = 'data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather'
GRAPH = 'data/processed/malecns.npz'
BODY_IDS = 'data/processed/malecns_body_ids.npy'
REWIRED = sorted(glob.glob('data/processed/controls/rewired_*.npz'))
POPULATION_SEED = 2026
NONBIOLOGICAL_PER_CHANNEL = 20

# fire-drone scenarios/patrol-depth-interface-v1.json (detector-expert-v1 convention):
# row-major, column = i % 8; positive yaw is body-left; increasing row is up.
DEPTH_YAW = [-61.25, -43.75, -26.25, -8.75, 8.75, 26.25, 43.75, 61.25]
DEPTH_ELEVATION = [-39.375, -28.125, -16.875, -5.625, 5.625, 16.875, 28.125, 39.375]
HALF_YAW, HALF_ELEVATION = 8.75, 5.625

# Observation layout, 83 channels (controller.ts visionObservation).
CHANNELS = (
    [('goal', i) for i in range(3)]
    + [('velocity', i) for i in range(3)]
    + [('gravity', i) for i in range(3)]
    + [('rates', i) for i in range(3)]
    + [('previous_action', i) for i in range(4)]
    + [('goal_confident', i) for i in range(3)]
    + [('depth', i) for i in range(64)]
)
assert len(CHANNELS) == 83


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_graph(path):
    """Binary directed adjacency, rows = presynaptic (src/graphs.py formats)."""
    n, src, dst, _ = load_npz(path)
    return sp.csr_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n))


def split_disjoint(indices, parts, rng):
    shuffled = rng.permutation(indices)
    return [np.sort(chunk) for chunk in np.array_split(shuffled, parts)]


def anatomical_map(a, position, eye_map, rng):
    side = a.somaSide.fillna(a.rootSide)
    index = lambda frame: np.array([position[b] for b in frame.bodyId if b in position])
    groups = {}
    # Depth cells: L1/L2 neurons whose column looks into the cell's angular bin.
    lamina = a[a.type.isin(['L1', 'L2']) & a.assignedOlHex1.notna()]
    direction = {}
    for eye, columns in eye_map['eyes'].items():
        for c in columns:
            direction[(eye, c['hex'][0], c['hex'][1])] = (c['azimuthDeg'], c['elevationDeg'])
    keys = list(zip(side.loc[lamina.index], lamina.assignedOlHex1.astype(int), lamina.assignedOlHex2.astype(int)))
    view = np.array([direction.get(k, (np.nan, np.nan)) for k in keys])
    lamina_index = np.array([position.get(b, -1) for b in lamina.bodyId])
    for i in range(64):
        yaw, elevation = DEPTH_YAW[i % 8], DEPTH_ELEVATION[i // 8]
        # Half-open bins so a column on a shared edge belongs to exactly one cell.
        inside = ((view[:, 0] >= yaw - HALF_YAW) & (view[:, 0] < yaw + HALF_YAW)
                  & (view[:, 1] >= elevation - HALF_ELEVATION) & (view[:, 1] < elevation + HALF_ELEVATION))
        groups[('depth', i)] = np.sort(lamina_index[inside & (lamina_index >= 0)])
    # Mechanosensory analogues (declared encodings, not identified codes).
    rates_pool = index(a[a.subclass.eq('haltere')])
    for i, part in enumerate(split_disjoint(rates_pool, 3, rng)):
        groups[('rates', i)] = part
    jo_pool = index(a[a.subclass.eq('wind_gravity')])
    for i, part in enumerate(split_disjoint(jo_pool, 6, rng)):
        groups[('velocity' if i < 3 else 'gravity', i % 3)] = part
    # Non-biological channels: random central-brain neurons.
    central = index(a[a.superclass.eq('cb_intrinsic')])
    nonbio = [c for c in CHANNELS if c[0] in ('goal', 'goal_confident', 'previous_action')]
    chosen = rng.choice(central, size=len(nonbio) * NONBIOLOGICAL_PER_CHANNEL, replace=False)
    for c, part in zip(nonbio, np.split(chosen, len(nonbio))):
        groups[c] = np.sort(part)
    readout = np.sort(index(a[a.superclass.eq('descending_neuron')]))
    return groups, readout


def random_map(groups, readout, n, rng):
    total = sum(len(v) for v in groups.values()) + len(readout)
    drawn = rng.choice(n, size=total, replace=False)
    out, start = {}, 0
    for c in CHANNELS:
        size = len(groups[c])
        out[c] = np.sort(drawn[start:start + size])
        start += size
    return out, np.sort(drawn[start:])


READOUT_FRACTION = 0.5  # ticks must let each channel reach half the readout


def hop_distance(A, sources, targets, limit=12):
    """Hops from a channel until it first reaches any readout neuron, and until it
    reaches READOUT_FRACTION of them (None if beyond limit)."""
    target = np.zeros(A.shape[0], bool)
    target[targets] = True
    seen = np.zeros(A.shape[0], bool)
    seen[sources] = True
    frontier = np.asarray(sources)
    first = 0 if target[frontier].any() else None
    reached_count = int(target[frontier].sum())
    for d in range(1, limit + 1):
        if reached_count >= READOUT_FRACTION * len(targets):
            return first, d - 1
        reached = np.unique(A[frontier].indices)
        reached = reached[~seen[reached]]
        if not len(reached):
            return first, None
        seen[reached] = True
        reached_count += int(target[reached].sum())
        if first is None and target[reached].any():
            first = d
        frontier = reached
    return first, d if reached_count >= READOUT_FRACTION * len(targets) else None


def audit(groups, readout, graphs):
    rows = {}
    for name, A in graphs.items():
        rows[name] = {f'{c[0]}[{c[1]}]': list(hop_distance(A, groups[c], readout)) for c in CHANNELS if len(groups[c])}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eye-map', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    a = pd.read_feather(ANNOTATIONS)
    a = a[a.superclass.notna()]
    ids = np.load(BODY_IDS)
    position = {b: i for i, b in enumerate(ids)}
    with open(args.eye_map) as f:
        eye_map = json.load(f)
    rng = np.random.default_rng(POPULATION_SEED)
    groups, readout = anatomical_map(a, position, eye_map, rng)
    random_groups, random_readout = random_map(groups, readout, len(ids), rng)

    used = np.concatenate([*groups.values(), readout])
    random_used = np.concatenate([*random_groups.values(), random_readout])
    graphs = {'real': load_graph(GRAPH), **{p.split('/')[-1][:-4]: load_graph(p) for p in REWIRED}}
    hops = {'anatomical': audit(groups, readout, graphs), 'random': audit(random_groups, random_readout, graphs)}
    worst = {
        m: max((h[1] for g in rows.values() for h in g.values() if h[1] is not None), default=None)
        for m, rows in hops.items()
    }
    unreachable = {m: sorted({k for g in rows.values() for k, h in g.items() if h[1] is None}) for m, rows in hops.items()}
    empty = [f'{c[0]}[{c[1]}]' for c in CHANNELS if not len(groups[c])]
    result = {
        'version': 1,
        'inputs': {p: sha256(p) for p in (ANNOTATIONS, GRAPH, BODY_IDS, args.eye_map, *REWIRED)},
        'populationSeed': POPULATION_SEED,
        'channels': [f'{c[0]}[{c[1]}]' for c in CHANNELS],
        'anatomical': {
            'channels': {f'{c[0]}[{c[1]}]': groups[c].tolist() for c in CHANNELS},
            'readout': readout.tolist(),
        },
        'random': {
            'channels': {f'{c[0]}[{c[1]}]': random_groups[c].tolist() for c in CHANNELS},
            'readout': random_readout.tolist(),
        },
        'audit': {
            'disjoint': {
                'anatomical': len(np.unique(used)) == len(used),
                'random': len(np.unique(random_used)) == len(random_used),
            },
            'emptyAnatomicalChannels': empty,
            'channelSizes': {f'{c[0]}[{c[1]}]': int(len(groups[c])) for c in CHANNELS},
            'readoutSize': int(len(readout)),
            'readoutFractionForTicks': READOUT_FRACTION,
            'maxHopsToReadoutFraction': worst,
            'uncoveredDepthCellsDroppedForAllArms': empty,
            'unreachableWithin12Hops': unreachable,
            'hopsByGraph': hops,
        },
        'ticksPerControlStep': max(v for v in worst.values() if v is not None),
        'droppedChannels': empty,
    }
    with open(args.out, 'w') as f:
        json.dump(result, f)
    sizes = result['audit']['channelSizes']
    depth_sizes = [sizes[f'depth[{i}]'] for i in range(64)]
    print(json.dumps({
        'disjoint': result['audit']['disjoint'],
        'empty': empty,
        'depthNeuronsPerCell_min_median_max': [min(depth_sizes), int(np.median(depth_sizes)), max(depth_sizes)],
        'otherSizes': {k: v for k, v in sizes.items() if not k.startswith('depth')},
        'readout': result['audit']['readoutSize'],
        'maxHopsToHalfReadout': worst,
        'unreachable': unreachable,
        'ticksPerControlStep': result['ticksPerControlStep'],
        'graphs': list(graphs),
    }, indent=1))


if __name__ == '__main__':
    main()
