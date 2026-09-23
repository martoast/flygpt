# CP1 — wiring map and observation contract

2026-09-23. **Gate passed.** The contract is frozen as `connectome-pilot-v1.json`. No
model was trained and no flight was run.

## Body axes from landmarks

Axes come from `scripts/connectome_flight/eye_map.py`:

| Fly direction | Brain axis | Landmarks |
|---|---|---|
| Forward | −z | antennal-lobe local neurons lie anterior to Kenyon cells |
| Up | −y | dorsal endocrine cells lie above the ventral DNg cluster |
| Left | +x | annotated soma sides |

The axes are right-handed. The script exits rather than guess if they aren't.

## Eye map

**Final method:**
- **Lattice:** hex coordinates are axial, meaning the six neighbours are (±1,0), (0,±1)
  and ±(1,1). A regular lattice therefore has its axes 120° apart.
- **Orientation from data:** a planar fit of lamina (L1–L5) soma positions shows h2
  points backward and h1 points up. This is mirror-consistent across the two eyes.
- **Declared anchors:**
  - each eye's anterior boundary at the equator looks straight ahead (0°)
  - the middle row is the equator
  - Δφ = 5° per step, a commonly cited literature value (4.5–5°), with 4° and 6°
    sensitivity maps
- **Result at Δφ = 5°:**
  - left eye azimuth +7° to +127° and right eye −106° to −1° (5th–95th percentile)
  - elevation about ±50°
  - dorsal-rim columns (R7d/R8d targets) at +28° and +36°, above the eye mean
- **Medulla positions are never used,** because the first optic chiasm mirrors the
  anterior-posterior axis.

**Rejected on the way:**
1. **Single sphere centre.** The eyes came out asymmetric and wrapped past ±180°.
2. **Local normals of a cubic sheet.** About half the columns faced inward.
3. **Absolute directions extrapolated from the planar normal.** Lamina somata with
   positions cover only one dorsal-posterior corner of each eye (their centroid is at
   hex (27, 31), against the eye's (19, 20)). Both eyes ended up centred near 0°
   azimuth.

**Upgrade path:** register the Zhao et al. microCT eye map (Janelia figshare 29111339,
CC-BY, raw scans only) onto MaleCNS hex coordinates.

## Input and readout map

Built by `scripts/connectome_flight/io_map.py` into `io_map.json`.

- **Depth:** 59 of 64 cells get L1/L2 neurons of the columns viewing their bin, a
  median of 30 neurons (0 to 52).
- **Five upper-front cells** (depth 51, 52, 59, 60, 61) have no eye coverage. The same
  five drop at Δφ = 4° and 6°. Binocular overlap up to 30° does not fill them, because
  the lattice's dorsal edge sits laterally. They are **dropped for every arm, MLP
  included**. Inflating the overlap to hide the gap was rejected.
- **Body rates:** haltere afferents (205 neurons), 3 disjoint parts of 68–69.
- **Velocity and gravity:** Johnston's-organ wind/gravity neurons (475), 6 disjoint
  parts of 79–80.
- **Goal and previous action (non-biological):** 20 random `cb_intrinsic` neurons per
  channel.
- **Readout:** all 1,314 descending neurons. A subclass filter would have dropped DNp01
  (`lt`), DNa02 and DNb01.
- **Random-I/O map:** the same sizes drawn uniformly from the whole graph, seed 2026.
  Both maps are disjoint.

## Ticks per control step: 4

**Rule:** the maximum, over channels and over the real graph plus 10 degree-rewired
controls (`rewired_777–781`, `877–881`), of the hops each channel needs to reach 50% of
the readout.

**Results:**
- Anatomical map: 4. Random map: 3. Every channel reaches the threshold in every graph.
- The shortest path to any DN is only 1–2 hops, but that was rejected as the rule: it
  would allow ticks = 2, too few for the visual pathway.
- At Δφ = 4° ticks is 4 and at 6° it is 3, so 4 covers all three.

## Weight models

Declared here, built in CP3.

- **Trained:** the FlyGPT setup.
- **Frozen biological:** synapse count × transmitter sign, using `consensus_nt`.
  - acetylcholine is excitatory; GABA, glutamate and histamine are inhibitory
  - unclear falls back to the cell-type prediction
  - modulators get zero fast weight
  - 166,522 of 166,700 neurons are covered
  - a single global gain gives a spectral radius of 0.9 before any flight
- **Rewired frozen controls** carry each edge's weight and sign.

## Stage A output

A 3-dimensional velocity correction, the same output as V5's learned net, feeds the
authored velocity controller. On drone-env v2 that is a velocity → ANGLE-mode stick
loop. Direct stick output is CP7.

## Hashes (sha256, prefix)

| File | Hash |
|---|---|
| `eye_map.json` | `882543a3b460d424` |
| `eye_map_dphi4.json` | `70edfc6868d58de0` |
| `eye_map_dphi6.json` | `33b3a2a7c422e7fc` |
| `io_map.json` | `fbd5120161021e63` |
| `eye_map.py` | `7ef7b604db20cbba` |
| `io_map.py` | `f24a0d4fea19db1b` |

Full input hashes are inside the JSON files. `body-neurotransmitters-male-cns-v1.0.feather`
was added to `download_malecns.py`.
