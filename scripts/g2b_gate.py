"""Execute the prospective G2b teacher gate on every holdout/operation cell."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.g2b_experiment import ROOT,PROTOCOL,EXECUTION,config,verify_gate
from src.provenance import save_json,sha256


def assess_category(teachers,baselines,axis,thresholds,repetitions,rng):
    seeds=len(teachers);cases=len(teachers[0]);reference=teachers[0]
    for family in [teachers,*baselines.values()]:
        assert len(family)==seeds
        for rows in family:
            assert len(rows)==cases
            for a,b in zip(reference,rows):
                assert a['case_index']==b['case_index'] and a['subject_attribute']==b['subject_attribute'] and a['subject_relation']==b['subject_relation']
    def accuracy(family):return float(np.mean([[r['exact'] for r in rows] for rows in family]))
    def ce(family):
        if 'nll' not in family[0][0]:return None
        return sum(r['nll'] for rows in family for r in rows)/sum(r['response_bytes'] for rows in family for r in rows)
    teacher_accuracy=accuracy(teachers);teacher_ce=ce(teachers)
    baseline_accuracy={k:accuracy(v) for k,v in baselines.items()}
    baseline_ce={k:ce(v) for k,v in baselines.items() if ce(v) is not None}
    groups=[tuple(r['subject_attribute']) if axis=='attribute' else tuple(r['subject_relation']) if axis=='relation' else tuple(r['subject_attribute']+r['subject_relation']) for r in reference]
    unique=sorted(set(groups));index=np.array([unique.index(g) for g in groups]);count=len(unique)
    t=np.array([[r['exact'] for r in rows] for rows in teachers])
    bs=[np.array([[r['exact'] for r in rows] for rows in family]) for family in baselines.values()]
    differences=[]
    for _ in range(repetitions):
        si=rng.integers(seeds,size=seeds);gi=rng.integers(count,size=count)
        weights=np.bincount(gi,minlength=count)[index]
        def weighted(values):return float((values[si]*weights[None]).sum()/(seeds*weights.sum()))
        differences.append(weighted(t)-max(weighted(b) for b in bs))
    lower,upper=np.percentile(differences,[2.5,97.5]);best_accuracy=max(baseline_accuracy.values());best_ce=min(baseline_ce.values())
    checks={
        'teacher_accuracy':teacher_accuracy>=thresholds['teacher_mean_exact_response_accuracy_min'],
        'accuracy_margin':teacher_accuracy-best_accuracy>=thresholds['teacher_mean_exact_response_accuracy_margin_over_best_baseline_min'],
        'ce_margin':best_ce>0 and teacher_ce<=(1-thresholds['teacher_mean_response_ce_reduction_vs_best_probabilistic_baseline_min_fraction'])*best_ce,
        'bootstrap_margin':float(lower)>thresholds['paired_seed_and_composition_bootstrap_accuracy_margin_lower_bound_min']}
    return {'passed':all(checks.values()),'checks':checks,'teacher_accuracy':teacher_accuracy,'teacher_ce':teacher_ce,
            'baseline_accuracy':baseline_accuracy,'baseline_ce':baseline_ce,
            'accuracy_margin_over_best':teacher_accuracy-best_accuracy,'relative_ce_reduction_vs_best':1-teacher_ce/best_ce if best_ce>0 else None,
            'paired_seed_composition_margin_95_ci':[float(lower),float(upper)],'seeds':seeds,'cases_per_seed':cases,'composition_groups':count}


def main(split):
    protocol,execution=config()
    if split=='qualification':verify_gate('validation')
    seeds=protocol['seeds'];folder=ROOT/split
    paths=[folder/f'{kind}_{s}.json' for kind in ('teacher','gru') for s in seeds]
    paths += [folder/f'{name}.json' for name in ('ngram1','ngram4','ngram8','ngram16','retrieval')]
    evidence={p.stem:json.loads(p.read_text()) for p in paths}
    inputs={str(p):sha256(p) for p in paths}
    for r in evidence.values():
        for path,digest in r['inputs'].items():
            assert sha256(path)==digest,'Gate input changed'
            inputs[path]=digest
    rng=np.random.default_rng(execution['bootstrap_seed']);categories={}
    for axis in protocol['holdout_axes']:
        for operation in protocol['operations']:
            category=f'{axis}_{operation}'
            teachers=[evidence[f'teacher_{s}']['rows'][category] for s in seeds]
            baselines={'gru':[evidence[f'gru_{s}']['rows'][category] for s in seeds]}
            baselines.update({name:[evidence[name]['rows'][category] for s in seeds] for name in ('ngram1','ngram4','ngram8','ngram16','retrieval')})
            categories[category]=assess_category(teachers,baselines,axis,protocol['gate_each_category'],execution['bootstrap_replicates'],rng)
    result={'stage':split,'passed':all(r['passed'] for r in categories.values()),'categories':categories,
            'protocol_sha256':sha256(PROTOCOL),'execution_sha256':sha256(EXECUTION),'inputs':inputs,
            'gate_script_sha256':sha256(__file__),
            'scope':'Teacher qualification gate, not a confirmatory claim of architecture superiority; all six categories required'}
    save_json(ROOT/f'gate_{split}.json',result)
    lines=[f'# G2b teacher gate: {split}','',f"Passed: **{result['passed']}**. All six categories must pass every threshold.",'',
           '| Category | Teacher exact | Best baseline exact | Teacher CE | Best baseline CE | Pass |',
           '|---|---:|---:|---:|---:|---|']
    for category,r in categories.items():
        lines.append(f"| {category} | {r['teacher_accuracy']:.3f} | {max(r['baseline_accuracy'].values()):.3f} | {r['teacher_ce']:.4f} | {min(r['baseline_ce'].values()):.4f} | {r['passed']} |")
    (ROOT/f'GATE_{split.upper()}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'stage':split,'passed':result['passed'],'checks':{k:v['checks'] for k,v in categories.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['validation','qualification']);main(parser.parse_args().split)
