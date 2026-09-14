"""Fixed ten-seed primary analysis; saved prefix remains secondary."""
from scripts.compiler_confirmation_analysis import CONTRASTS, paired, holm


def analyze(by_seed):
    if set(by_seed) != {str(s) for s in range(500,510)}:
        raise ValueError('All ten registered seeds required')
    results = []
    for method, control in CONTRASTS:
        for endpoint in ['accuracy', 'response_ce']:
            sign = 1 if endpoint == 'accuracy' else -1
            values = [sign*(models[method][endpoint]-models[control][endpoint]) for _,models in sorted(by_seed.items())]
            results.append(dict(method=method, control=control, endpoint=endpoint, **paired(values)))
    for row,p in zip(results,holm([row['paired_t_p'] for row in results])):
        row['holm_p'] = p
        row['supported_under_preregistered_t_analysis'] = row['mean_benefit']>0 and p<.05
    return dict(seeds=sorted(by_seed),comparisons=results,correction='Holm across ten primary paired-t tests at 256 updates',
        caveat='Ten paired seeds, one fixed teacher/task/corpus; t assumptions and exact sign-flip sensitivity both reported. Prefix at 128 is secondary.')
