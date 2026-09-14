"""Prespecified paired-seed analysis; no pilot pooling or case pseudoreplication."""
import itertools
import numpy as np
from scipy import stats

CONTRASTS = [('B_soft', 'A_hard'), ('C_hidden', 'D_hidden_shuffled'),
             ('E_relational', 'F_relational_shuffled'), ('C_hidden', 'A_hard'),
             ('E_relational', 'A_hard')]


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.] * len(values)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[i]))
        adjusted[i] = running
    return adjusted


def paired(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need finite paired observations')
    mean = float(x.mean()); sd = float(x.std(ddof=1)); sem = sd / len(x)**.5
    if sem:
        p = float(2*stats.t.sf(abs(mean/sem), len(x)-1))
    else:
        p = 1. if mean == 0 else 0.
    radius = float(stats.t.ppf(.975, len(x)-1)*sem)
    permutations = [abs(float(np.mean(x*np.asarray(s)))) for s in itertools.product([-1, 1], repeat=len(x))]
    exact = sum(v >= abs(mean)-1e-12 for v in permutations) / len(permutations)
    return dict(n=len(x), paired_benefits=x.tolist(), mean_benefit=mean, sd=sd,
                unadjusted_95_t_interval=[mean-radius, mean+radius], paired_t_p=p,
                exact_two_sided_sign_flip_p=exact, degenerate_zero_variance=sem == 0)


def analyze(by_seed):
    if len(by_seed) != 5:
        raise ValueError('Analysis requires all five preregistered seeds')
    results = []
    for method, control in CONTRASTS:
        for endpoint in ['accuracy', 'response_ce']:
            sign = 1 if endpoint == 'accuracy' else -1
            values = [sign*(models[method][endpoint]-models[control][endpoint]) for _, models in sorted(by_seed.items())]
            results.append(dict(method=method, control=control, endpoint=endpoint, **paired(values)))
    for row, p in zip(results, holm([row['paired_t_p'] for row in results])):
        row['holm_p'] = p
        row['supported_under_preregistered_t_analysis'] = row['mean_benefit'] > 0 and p < .05
    return dict(seeds=sorted(by_seed), comparisons=results, correction='Holm across all ten paired-t tests',
                caveat='Five paired seeds; t assumptions uncertain; exact two-sided sign-flip minimum p=.0625; fixed teacher and corpus')
