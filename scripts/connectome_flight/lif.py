"""Leaky integrate-and-fire MaleCNS brain for the frozen-weights arm.

Replicates the Shiu et al. 2024 whole-brain Drosophila model
(github.com/philshiu/Drosophila_brain_model, model.py, MIT licence) on the MaleCNS graph:

    dv/dt = (v_0 - v + g) / t_mbr        (frozen while refractory)
    dg/dt = -g / tau                     (frozen while refractory)
    spike when v > v_th: v = v_rst, g = 0, refractory t_rfc
    presynaptic spike -> g_post += w after t_dly, w = sign(pre) * synapses * w_syn
    stimulus: Poisson input on v with weight w_syn * f_poi (Shiu drop the refractory
    period of stimulated sensory neurons; ours are interneurons and keep it, see
    calibrated_params)

Parameters are the published defaults except w_syn, which Shiu et al. fitted as a
free parameter on FlyWire 783. MaleCNS neurons receive more synapses (median total
input 346 vs FlyWire's 200, both tables unthresholded), so w_syn is calibrated to
give the median neuron the same total synaptic drive: 0.275 mV * 200/346 = 0.159 mV
(DATASET_CALIBRATION; a property of the two datasets, never of flight). dt = 0.1 ms
(brian2 default) with exact
integration of the linear dynamics between events. Weights are fixed: nothing in
the brain is trained. Signs: acetylcholine +, GABA / glutamate / histamine -, other
transmitters 0 (connectome-pilot-v1 sign table).
"""
import numpy as np
import scipy.sparse as sp
from numba import njit, prange

PARAMS = {
    'v_0': -52.0, 'v_rst': -52.0, 'v_th': -45.0,  # mV
    't_mbr': 20.0, 'tau': 5.0, 't_rfc': 2.2, 't_dly': 1.8,  # ms
    'w_syn': 0.275, 'f_poi': 250.0, 'r_poi_max': 150.0,  # mV, -, Hz (published)
    'dt': 0.1,  # ms
}


@njit(cache=True)
def _run(steps, v, g, refractory, buffer, t0, indptr, indices, weights,
         stim_neurons, stim_rates, counts, seed_state,
         v_0, v_rst, v_th, dt, e_m, e_s, c_gv, rfc_steps, dly_steps, w_poi):
    """Advance `steps` of dt. Returns the new step counter and xorshift state.
    counts[i] += spikes of neuron i during these steps."""
    n = v.shape[0]
    slots = buffer.shape[0]
    state = seed_state
    stim_prob = stim_rates * dt * 1e-3
    for step in range(steps):
        t = t0 + step
        slot = t % slots
        # 1. synaptic arrivals scheduled for this step
        arrivals = buffer[slot]
        for i in range(n):
            if arrivals[i] != 0.0:
                g[i] += arrivals[i]
                arrivals[i] = 0.0
        # 2. Poisson stimulus onto v
        for k in range(stim_neurons.shape[0]):
            if stim_prob[k] > 0.0:
                state ^= state << np.uint64(13)  # xorshift64 (uint64 wraps)
                state ^= state >> np.uint64(7)
                state ^= state << np.uint64(17)
                if (state >> np.uint64(11)) * (1.0 / 9007199254740992.0) < stim_prob[k]:
                    v[stim_neurons[k]] += w_poi
        # 3. exact integration of the linear membrane and synapse between events
        for i in range(n):
            if refractory[i] > t:
                continue
            vi = v[i] - v_0
            gi = g[i]
            v[i] = v_0 + vi * e_m + gi * c_gv
            g[i] = gi * e_s
        # 4. threshold, reset, schedule synaptic events after the delay
        target = (t + dly_steps) % slots
        for i in range(n):
            if refractory[i] > t or v[i] <= v_th:
                continue
            v[i] = v_rst
            g[i] = 0.0
            refractory[i] = t + rfc_steps[i]
            counts[i] += 1
            for e in range(indptr[i], indptr[i + 1]):
                buffer[target, indices[e]] += weights[e]
    return t0 + steps, state


CHUNKS = 64  # neuron blocks per parallel pass (more than cores, for balance)


@njit(cache=True, parallel=True)
def _run_parallel(steps, v, g, refractory, buffer, chunk_spikes, chunk_count, t0,
                  indptr, indices, weights, stim_neurons, stim_rates, counts, seed_state,
                  v_0, v_rst, v_th, dt, e_m, e_s, c_gv, rfc_steps, dly_steps, w_poi):
    """Multi-core equivalent of _run. Each step: serial Poisson stimulus (one RNG
    stream), one parallel pass over neuron blocks (arrivals, integration, threshold;
    each block lists its own spikes in index order), then serial delivery in block
    order -- the same order as _run, so floating-point sums and spike trains are
    identical."""
    n = v.shape[0]
    slots = buffer.shape[0]
    blocks = chunk_count.shape[0]
    size = (n + blocks - 1) // blocks
    state = seed_state
    stim_prob = stim_rates * dt * 1e-3
    for step in range(steps):
        t = t0 + step
        slot = t % slots
        for k in range(stim_neurons.shape[0]):
            if stim_prob[k] > 0.0:
                state ^= state << np.uint64(13)
                state ^= state >> np.uint64(7)
                state ^= state << np.uint64(17)
                if (state >> np.uint64(11)) * (1.0 / 9007199254740992.0) < stim_prob[k]:
                    v[stim_neurons[k]] += w_poi
        for b in prange(blocks):
            found = 0
            for i in range(b * size, min(n, (b + 1) * size)):
                a = buffer[slot, i]
                if a != 0.0:
                    g[i] += a
                    buffer[slot, i] = 0.0
                if refractory[i] > t:
                    continue
                vi = v[i] - v_0
                gi = g[i]
                v[i] = v_0 + vi * e_m + gi * c_gv
                g[i] = gi * e_s
                if v[i] > v_th:
                    v[i] = v_rst
                    g[i] = 0.0
                    refractory[i] = t + rfc_steps[i]
                    counts[i] += 1
                    chunk_spikes[b, found] = i
                    found += 1
            chunk_count[b] = found
        # Deliver serially in block order = neuron-index order, exactly as in _run.
        # (A second parallel region per step for delivery was measured slower: at
        # 10,000 steps per simulated second, thread launch overhead dominates.)
        target = (t + dly_steps) % slots
        for b in range(blocks):
            for k in range(chunk_count[b]):
                i = chunk_spikes[b, k]
                for e in range(indptr[i], indptr[i + 1]):
                    buffer[target, indices[e]] += weights[e]
    return t0 + steps, state


class LifBrain:
    def __init__(self, graph_csr_pre, signs, stim_neurons, params=PARAMS, seed=1, engine='parallel'):
        """graph_csr_pre: CSR with rows = presynaptic neurons, data = synapse counts.
        engine: 'parallel' (multi-core, default) or 'dense' (single-core reference).
        Both give identical spike trains."""
        self.engine = engine
        p = params
        self.p = p
        A = graph_csr_pre.tocsr()
        self.indptr = A.indptr.astype(np.int64)
        self.indices = A.indices.astype(np.int64)
        # w = sign(pre) * synapses * w_syn (row i = presynaptic i)
        row_sign = np.repeat(signs.astype(np.float64), np.diff(A.indptr))
        self.weights = (row_sign * A.data * p['w_syn']).astype(np.float64)
        self.n = A.shape[0]
        self.stim_neurons = np.asarray(stim_neurons, dtype=np.int64)
        dt = p['dt']
        self.dt = dt
        self.e_m = np.exp(-dt / p['t_mbr'])
        self.e_s = np.exp(-dt / p['tau'])
        # v response to g over dt (exact for tau != t_mbr): g*tau/(tau - t_mbr)*(e_s - e_m)
        self.c_gv = p['tau'] / (p['tau'] - p['t_mbr']) * (self.e_s - self.e_m)
        self.dly_steps = int(round(p['t_dly'] / dt))
        rfc = np.full(self.n, int(round(p['t_rfc'] / dt)), dtype=np.int64)
        if p.get('stim_refractory', True) is False:
            rfc[self.stim_neurons] = 0  # Shiu: no refractory period for Poisson targets
        self.rfc_steps = rfc
        self.w_poi = p['w_syn'] * p['f_poi']
        mixed = (0x9E3779B97F4A7C15 ^ (seed * 0x2545F4914F6CDD1D)) & 0xFFFFFFFFFFFFFFFF
        self.seed = mixed or 1
        self.reset()

    def reset(self):
        p = self.p
        self.v = np.full(self.n, p['v_0'], dtype=np.float64)
        self.g = np.zeros(self.n, dtype=np.float64)
        self.refractory = np.zeros(self.n, dtype=np.int64)
        slots = self.dly_steps + 1
        self.buffer = np.zeros((slots, self.n), dtype=np.float64)
        self.chunk_count = np.zeros(CHUNKS, dtype=np.int64)
        self.chunk_spikes = np.zeros((CHUNKS, (self.n + CHUNKS - 1) // CHUNKS), dtype=np.int64)
        self.t = 0
        self.state = np.uint64(self.seed)

    def run(self, milliseconds, stim_rates_hz):
        """Run with per-stimulated-neuron Poisson rates; returns spike counts of all neurons."""
        steps = int(round(milliseconds / self.dt))
        counts = np.zeros(self.n, dtype=np.int64)
        rates = np.asarray(stim_rates_hz, dtype=np.float64)
        if self.engine == 'parallel':
            self.t, self.state = _run_parallel(
                steps, self.v, self.g, self.refractory, self.buffer, self.chunk_spikes,
                self.chunk_count, self.t, self.indptr, self.indices, self.weights,
                self.stim_neurons, rates, counts,
                self.state, self.p['v_0'], self.p['v_rst'], self.p['v_th'], self.dt,
                self.e_m, self.e_s, self.c_gv, self.rfc_steps, self.dly_steps, self.w_poi,
            )
            return counts
        self.t, self.state = _run(
            steps, self.v, self.g, self.refractory, self.buffer, self.t,
            self.indptr, self.indices, self.weights,
            self.stim_neurons, rates,
            counts, self.state,
            self.p['v_0'], self.p['v_rst'], self.p['v_th'], self.dt,
            self.e_m, self.e_s, self.c_gv, self.rfc_steps, self.dly_steps, self.w_poi,
        )
        return counts


# Median total input synapses per neuron: FlyWire 783 (Shiu's table) vs MaleCNS v1.0.
DATASET_CALIBRATION = {'flywireMedianInput': 200.0, 'malecnsMedianInput': 346.0}


def calibrated_params():
    """Published parameters with w_syn scaled for MaleCNS. The Poisson stimulus keeps
    the published absolute kick (w_syn * f_poi = 68.75 mV), which always makes the
    stimulated neuron spike."""
    p = dict(PARAMS)
    scale = DATASET_CALIBRATION['flywireMedianInput'] / DATASET_CALIBRATION['malecnsMedianInput']
    p['f_poi'] = PARAMS['f_poi'] / scale
    p['w_syn'] = PARAMS['w_syn'] * scale
    # Shiu drop the refractory period of stimulated *sensory* neurons, which have
    # little recurrent input. Our stimulated L1/L2 are recurrently driven interneurons;
    # without a refractory period they fired every 0.1 ms step (~4,800 Hz, 3/4 of all
    # spikes), so they keep the standard 2.2 ms like every other neuron.
    p['stim_refractory'] = True
    return p


def load_brain(stim_neurons, seed=1, params=None, engine='parallel'):
    """MaleCNS graph + connectome-pilot-v1 signs, as a LifBrain (calibrated w_syn)."""
    from scripts.connectome_flight.cp2_benchmark import GRAPH, presynaptic_signs
    z = np.load(GRAPH)
    A = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))  # A[pre, post]
    signs, _ = presynaptic_signs(A.shape[0])
    return LifBrain(A, signs, stim_neurons, params=params or calibrated_params(), seed=seed, engine=engine)
