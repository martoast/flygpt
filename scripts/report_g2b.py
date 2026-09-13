"""Report both knowledge-transfer and topology contrasts without selecting axes."""
import json
import numpy as np
from scripts.g2b_experiment import ROOT,config,verify_gate
from src.provenance import save_json


def main():
    verify_gate('validation');verify_gate('qualification');protocol,execution=config()
    seeds=execution['student_seeds'];rng=np.random.default_rng(44)
    conditions=['teacher','gru',*execution['student_conditions']]
    records={(c,s):json.loads((ROOT/'test'/f'{c}_{s}.json').read_text()) for c in conditions for s in seeds}
    comparisons=[];lines=['# G2b held-out student test','',
                         'Teacher passed separate validation and qualification gates before any student training. Final student test uses different composition pairs. Five paired seeds and one partition remain exploratory.','',
                         '| Category | Model | Response CE | Exact response accuracy |',
                         '|---|---|---:|---:|']
    for axis in protocol['holdout_axes']:
        for operation in protocol['operations']:
            category=f'{axis}_{operation}'
            for c in conditions:
                m=[records[(c,s)]['metrics'][category] for s in seeds]
                lines.append(f"| {category} | {c} | {np.mean([v['ce'] for v in m]):.4f} | {np.mean([v['exact_accuracy'] for v in m]):.3f} |")
            for name,a,b,metric in [('knowledge_transfer_ce','real_ce','real_kd','ce'),
                                    ('knowledge_transfer_kl','real_ce','real_kd','teacher_kl'),
                                    ('topology_ce','rewired_kd','real_kd','ce')]:
                differences=np.array([records[(a,s)]['metrics'][category][metric]-records[(b,s)]['metrics'][category][metric] for s in seeds])
                boot=differences[rng.integers(len(seeds),size=(10000,len(seeds)))].mean(1)
                comparisons.append({'category':category,'contrast':name,'positive_means':'lower loss for real KD',
                                    'paired_seed_differences':differences.tolist(),'mean':float(differences.mean()),
                                    'paired_seed_bootstrap_95_ci':np.percentile(boot,[2.5,97.5]).tolist()})
    lines += ['', 'Positive CE-only minus KD CE and KL together support useful teacher transfer; positive rewired-KD minus real-KD CE supports a topology difference under these matched budgets. All exact-answer, baseline, ablation and per-case results remain available. No category is dropped after observing results.']
    save_json(ROOT/'comparisons.json',{'comparisons':comparisons,'scope':'exploratory paired-seed comparisons; multiple categories, one held-out composition partition'})
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
