"""Summarize only completed experiments; never impute unfinished conditions."""
import json
from pathlib import Path
import numpy as np
from src.provenance import save_json

ROOT=Path('results/malecns_v1')


def paired_summary(values):
    x=np.asarray(values,float);rng=np.random.default_rng(42)
    means=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    return {'n_seeds':len(x),'values':x.tolist(),'mean':float(x.mean()),
            'bootstrap_95_ci':np.percentile(means,[2.5,97.5]).tolist() if len(x)>=5 else None,
            'interpretation':'exploratory paired seed bootstrap; not confirmatory' if len(x)>=5 else 'insufficient seeds; descriptive only'}


def main():
    out={'evidence_domain':'computational, MaleCNS versus synthetic controls; no wetware','phases':{}}
    text=['# FlyGPT measured results','', 'Only completed JSON records are summarized. Missing runs are not negative results.','']
    for phase in ['language','distill']:
        records={}
        for path in sorted((ROOT/phase).glob('*.json')):
            r=json.loads(path.read_text());records[(r['config']['condition'],r['config']['seed'])]=r
        rows=[]
        for condition in ['real','rewired','configuration','er','gru']:
            selected=[(s,r) for (c,s),r in records.items() if c==condition]
            if not selected:continue
            ce=[r['validation']['ce'] for s,r in selected]
            rows.append({'condition':condition,'seeds':[s for s,r in selected],**paired_summary(ce),
                         'total_runtime_seconds':sum(r['runtime_seconds'] for s,r in selected)})
        matched=sorted(s for c,s in records if c=='real' and ('rewired',s) in records)
        # Performance = -CE, so A_bio = CE_rewired - CE_real.
        difference=[records['rewired',s]['validation']['ce']-records['real',s]['validation']['ce'] for s in matched]
        comparison=None if not difference else paired_summary(difference)
        out['phases'][phase]={'conditions':rows,'A_bio_negative_CE':comparison}
        text+= [f'## {phase}', '', '| Condition | Completed seeds | Mean validation CE |','|---|---:|---:|']
        text +=[f'| {r["condition"]} | {len(r["seeds"])} | {r["mean"]:.4f} |' for r in rows]
        text+=['','A_bio uses performance = -CE. Positive values favor real topology.']
        if comparison:text +=[f'Paired A_bio: {comparison["mean"]:.4f} nats/byte; n={comparison["n_seeds"]}. {comparison["interpretation"]}.']
        else:text+=['No completed paired real/rewired comparison yet.']
        text+=['']
    target=ROOT/'target'/'real_0.json'
    if target.exists():
        r=json.loads(target.read_text());out['target']=r
        text+=['## Teacher-gap continuation','','| Additional steps | Validation CE | Gap to teacher |','|---:|---:|---:|']
        text += [f'| {m["step"]} | {m["ce"]:.4f} | {m["ce"]-r["teacher"]["ce"]:.4f} |' for m in r['evaluations']]
        text+=['',f'Complete: {r["complete"]}. Single-seed feasibility only; no topology-superiority conclusion.','']
    save_json(ROOT/'summary.json',out);(ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
