"""CP1: estimate each optic-lobe hex column's viewing direction from MaleCNS data.

Brain axes come from independent landmarks, not assumed conventions: antennal-lobe
local neurons are anterior to Kenyon cells, dorsal endocrine cells are above the
ventral DNg cluster, and left/right come from annotated soma sides.

Viewing directions use a regular hex lattice (axial coordinates, axes 120 degrees
apart, DELTA_PHI_DEG per step). The data fix only what it robustly supports: a planar
fit of lamina (L1-L5) soma position against hex coordinates gives the in-sheet
orientation of the lattice, i.e. which lattice direction points backward and which
up (mirror-consistent across eyes). Anchors are declared: each eye's anterior lattice
boundary near the equator looks at --front-azimuth (0 = straight ahead; negative =
binocular overlap, chosen by the CP1 coverage rule, never by flight performance)
and the eye's middle row is the equator (elevation 0). Medulla positions are never
used, because the first optic chiasm mirrors the anterior-posterior axis.

Rejected first (recorded in the CP1 report): a single sphere centre (eyes asymmetric,
wrapping past +-180 degrees); local normals of a cubic sheet (about half the columns
faced inward); and absolute directions extrapolated from the planar normal, because
lamina somata with positions cover only one corner of each eye. DELTA_PHI_DEG is a declared
literature value, not a fit. The upgrade path is registering the Zhao et al. microCT
eye map onto MaleCNS hex coordinates.

Usage: python -m scripts.connectome_flight.eye_map --out results/connectome_flight/cp1/eye_map.json
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

ANNOTATIONS = 'data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather'
GRAPH = 'data/processed/malecns.npz'
BODY_IDS = 'data/processed/malecns_body_ids.npy'
LAMINA = ['L1', 'L2', 'L3', 'L4', 'L5']


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def centroid(frame):
    frame = frame[frame.somaLocation.notna()]
    return np.stack(frame.somaLocation.values).astype(float).mean(0), len(frame)


def body_axes(a):
    """Unit brain-frame vectors for the fly's forward, left and up."""
    kc, n_kc = centroid(a[a['class'].eq('Kenyon_Cell')])
    al, n_al = centroid(a[a['class'].eq('ALLN')])
    dorsal, n_d = centroid(a[a.superclass.eq('cb_endocrine')])
    ventral, n_v = centroid(a[a.type.str.startswith('DNg', na=False)])
    side = a.somaSide.fillna(a.rootSide)
    left, n_l = centroid(a[side.eq('L') & a.assignedOlHex1.notna()])
    right, n_r = centroid(a[side.eq('R') & a.assignedOlHex1.notna()])
    raw = {'forward': al - kc, 'up': dorsal - ventral, 'left': left - right}
    # Orthonormalise, keeping each axis's dominant sign from its own landmark pair.
    forward = raw['forward'] / np.linalg.norm(raw['forward'])
    up = raw['up'] - forward * (raw['up'] @ forward)
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    if left @ raw['left'] <= 0:
        raise SystemExit('Landmark axes are not right-handed; refusing to guess.')
    cosines = {k: float(v @ n / np.linalg.norm(v)) for (k, v), n in zip(raw.items(), (forward, up, left))}
    counts = {'kenyon': n_kc, 'alln': n_al, 'cb_endocrine': n_d, 'DNg': n_v, 'left_ol': n_l, 'right_ol': n_r}
    return np.stack([forward, left, up]), cosines, counts


DELTA_PHI_DEG = 5.0  # declared literature value; sensitivity range 4-6


def cv_errors(X, positions, rng):
    folds = rng.permutation(len(X)) % 5
    errors = []
    for k in range(5):
        train, test = folds != k, folds == k
        W, *_ = np.linalg.lstsq(X[train], positions[train], rcond=None)
        errors.append(np.linalg.norm(X[test] @ W - positions[test], axis=1))
    return np.concatenate(errors)


def fit_eye(lamina, columns, axes, rng, brain_centre, delta_phi_deg, sign, front_azimuth_deg=0.0):
    hexes = lamina[['assignedOlHex1', 'assignedOlHex2']].to_numpy(float)
    positions = np.stack(lamina.somaLocation.values).astype(float)
    centre_hex = hexes.mean(0)
    X = np.hstack([np.ones((len(hexes), 1)), hexes - centre_hex])
    cv = cv_errors(X, positions, rng)
    W, *_ = np.linalg.lstsq(X, positions, rcond=None)
    t1, t2 = W[1], W[2]
    normal = np.cross(t1, t2)
    normal /= np.linalg.norm(normal)
    if (positions.mean(0) - brain_centre) @ normal < 0:
        normal = -normal
    # Hex coordinates are axial: neighbours are (+-1,0), (0,+-1) and +-(1,1), so a
    # regular lattice has its two axes 120 degrees apart. Orientation comes from the
    # data (h1's in-sheet direction and the side h2 lies on); spacing and the
    # 120-degree geometry come from the lattice definition, not the noisy soma fit.
    e1 = t1 - normal * (t1 @ normal)
    e1 /= np.linalg.norm(e1)
    perpendicular = np.cross(normal, e1)
    if perpendicular @ t2 < 0:
        perpendicular = -perpendicular
    e2 = -0.5 * e1 + np.sqrt(3) / 2 * perpendicular
    # Place the regular lattice in the eye's own frame: "back" is the in-sheet
    # direction most aligned with the fly's posterior, "up" completes the frame.
    to_body_vec = lambda v: v @ axes.T
    lattice = np.stack([columns[:, 0], columns[:, 1]], 1)
    xy_e1, xy_e2 = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
    plane = lattice[:, :1] * xy_e1 + lattice[:, 1:] * xy_e2  # regular hex, 1 = one step
    # Express the plane's x/y axes in body coordinates via e1 and the perpendicular.
    x_body, y_body = to_body_vec(e1), to_body_vec(perpendicular)
    backward = -np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    # Least-squares 2x2 map from plane coords to (backward, up) body components.
    basis = np.array([[x_body @ backward, y_body @ backward], [x_body @ up, y_body @ up]])
    back_up = plane @ basis.T
    posterior, dorsal = back_up[:, 0], back_up[:, 1]
    # Anchor: the anterior boundary near the equator looks straight ahead (azimuth 0);
    # the eye's middle row is the equator (elevation 0).
    equator = np.median(dorsal)
    near_equator = np.abs(dorsal - equator) <= 2
    front = np.percentile(posterior[near_equator], 2)
    azimuth = sign * (delta_phi_deg * (posterior - front) + front_azimuth_deg)
    elevation = delta_phi_deg * (dorsal - equator)
    to_body = lambda v: (v / np.linalg.norm(v)) @ axes.T
    centre_body = to_body(normal)
    return azimuth, elevation, {
        'laminaSomata': int(len(positions)),
        'planarFitCvMedianVoxels': float(np.median(cv)),
        'voxelsPerHexStep': [float(np.linalg.norm(t1)), float(np.linalg.norm(t2))],
        'hexAxisAngleDegrees': float(np.degrees(np.arccos(t1 @ t2 / np.linalg.norm(t1) / np.linalg.norm(t2)))),
        'hexAxesInBody': {'h1': to_body(t1).round(3).tolist(), 'h2': to_body(t2).round(3).tolist()},
        'meanViewAzimuthDegrees': float(np.degrees(np.arctan2(centre_body[1], centre_body[0]))),
        'meanViewElevationDegrees': float(np.degrees(np.arcsin(centre_body[2]))),
        'fittedSomaHexCentroid': centre_hex.round(2).tolist(),
    }


def dorsal_rim_columns(a, side):
    """Columns receiving dorsal-rim photoreceptor (R7d/R8d) input on one side."""
    ids = np.load(BODY_IDS)
    position = {b: i for i, b in enumerate(ids)}
    z = np.load(GRAPH)
    A = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
    s = a.somaSide.fillna(a.rootSide)
    dra = [position[b] for b in a.bodyId[a.type.isin(['R7d', 'R8d']) & s.eq(side)] if b in position]
    targets = a[a.assignedOlHex1.notna() & s.eq(side)]
    target_index = np.array([position[b] for b in targets.bodyId])
    weights = np.asarray(A[dra][:, target_index].sum(0)).ravel()
    hit = targets[weights > 0]
    return set(zip(hit.assignedOlHex1.astype(int), hit.assignedOlHex2.astype(int)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--delta-phi', type=float, default=DELTA_PHI_DEG)
    parser.add_argument('--front-azimuth', type=float, default=0.0,
                        help='azimuth of the anterior boundary; negative = binocular overlap')
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    a = pd.read_feather(ANNOTATIONS)
    a = a[a.superclass.notna()]
    axes, cosines, counts = body_axes(a)
    brain_centre, _ = centroid(a[a.superclass.eq('cb_intrinsic')])
    side_of = a.somaSide.fillna(a.rootSide)
    eyes, audits = {}, {}
    for side in ['L', 'R']:
        on_side = a[side_of.eq(side) & a.assignedOlHex1.notna()]
        columns = np.unique(on_side[['assignedOlHex1', 'assignedOlHex2']].to_numpy(float), axis=0)
        lamina = on_side[on_side.type.isin(LAMINA) & on_side.somaLocation.notna()]
        azimuth, elevation, fit = fit_eye(lamina, columns, axes, rng, brain_centre, args.delta_phi, 1.0 if side == 'L' else -1.0, args.front_azimuth)
        dra = dorsal_rim_columns(a, side)
        is_dra = np.array([(int(c[0]), int(c[1])) in dra for c in columns])
        fit.update({
            'columns': int(len(columns)),
            'azimuthRangeDegrees': [float(azimuth.min()), float(azimuth.max())],
            'elevationRangeDegrees': [float(elevation.min()), float(elevation.max())],
            'dorsalRimColumns': int(is_dra.sum()),
            'dorsalRimMeanElevationDegrees': float(elevation[is_dra].mean()) if is_dra.any() else None,
            'allColumnsMeanElevationDegrees': float(elevation.mean()),
            'meanAzimuthDegrees': float(azimuth.mean()),
            'azimuthPercentiles5_50_95': np.percentile(azimuth, [5, 50, 95]).round(1).tolist(),
            'elevationPercentiles5_50_95': np.percentile(elevation, [5, 50, 95]).round(1).tolist(),
        })
        audits[side] = fit
        eyes[side] = [
            {'hex': [int(c[0]), int(c[1])], 'azimuthDeg': round(float(az), 3), 'elevationDeg': round(float(el), 3)}
            for c, az, el in zip(columns, azimuth, elevation)
        ]
    # A left eye must look mostly left (positive azimuth) and vice versa.
    audits['mirror'] = {
        'leftMeanAzimuth': audits['L']['meanAzimuthDegrees'],
        'rightMeanAzimuth': audits['R']['meanAzimuthDegrees'],
        'passes': audits['L']['meanAzimuthDegrees'] > 0 > audits['R']['meanAzimuthDegrees'],
    }
    for side in ['L', 'R']:
        rim = audits[side]['dorsalRimMeanElevationDegrees']
        audits[side]['dorsalRimAboveMean'] = rim is not None and rim > audits[side]['allColumnsMeanElevationDegrees']
    out = {
        'version': 1,
        'method': __doc__.split('\n\n')[1].replace('\n', ' '),
        'inputs': {p: sha256(p) for p in (ANNOTATIONS, GRAPH, BODY_IDS)},
        'bodyAxesInBrainFrame': {'forward': axes[0].tolist(), 'left': axes[1].tolist(), 'up': axes[2].tolist()},
        'landmarkAxisCosines': cosines,
        'landmarkCounts': counts,
        'audits': audits,
        'deltaPhiDegrees': args.delta_phi,
        'frontAzimuthDegrees': args.front_azimuth,
        'convention': 'azimuth positive = fly left, 0 = straight ahead; elevation positive = up',
        'eyes': eyes,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'axesCosines': cosines, 'audits': audits}, indent=1))


if __name__ == '__main__':
    main()
