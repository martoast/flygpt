"""Descriptive paired-seed and paired-example analyses, keeping units explicit."""
import numpy as np
from scipy.stats import t

def seed_summary(real,rewired):
    a=np.asarray(real,dtype=float);b=np.asarray(rewired,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)==0:raise ValueError('Require nonempty paired vectors')
    d=a-b;n=len(d)
    result={'n':n,'real_mean':float(a.mean()),'rewired_mean':float(b.mean()),'gap_mean':float(d.mean()),
        'gap_min':float(d.min()),'gap_max':float(d.max()),'positive_gaps':int((d>0).sum())}
    if n>1:
        va=float(a.var(ddof=1));vb=float(b.var(ddof=1));sd=float(d.std(ddof=1));half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
        result.update(real_variance=va,rewired_variance=vb,real_sd=va**.5,rewired_sd=vb**.5,gap_sd=sd,
            gap_mean_t95=[float(d.mean()-half),float(d.mean()+half)],variance_ratio_rewired_over_real=vb/va if va else None)
    return result

def case_summary(real_rows,rewired_rows,seed=0):
    if [r['id'] for r in real_rows]!=[r['id'] for r in rewired_rows]:raise ValueError('Case IDs/order differ')
    a=np.array([r['exact'] for r in real_rows]);b=np.array([r['exact'] for r in rewired_rows]);d=a-b
    rng=np.random.default_rng(94100+seed);samples=rng.integers(len(d),size=(10000,len(d)))
    return {'cases':len(d),'real_correct':int(a.sum()),'rewired_correct':int(b.sum()),
        'both_correct':int(((a==1)&(b==1)).sum()),'real_only':int(((a==1)&(b==0)).sum()),
        'rewired_only':int(((a==0)&(b==1)).sum()),'both_wrong':int(((a==0)&(b==0)).sum()),
        'gap':float(d.mean()),'paired_case_bootstrap_95':np.quantile(d[samples].mean(1),[.025,.975]).tolist()}

def learning_summary(real,rewired):
    ar={r['step']:r['validation'] for r in real['evaluations']};br={r['step']:r['validation'] for r in rewired['evaluations']}
    steps=sorted(set(ar)&set(br))
    for s in steps:
        if [r['id'] for r in ar[s]['rows']]!=[r['id'] for r in br[s]['rows']]:raise ValueError('Validation windows differ')
    result={'steps':steps,'domain':'32 fixed validation cases; normalized trapezoidal area only over shared observed updates'}
    for metric in ('accuracy','response_ce'):
        a=np.array([ar[s]['metrics'][metric] for s in steps]);b=np.array([br[s]['metrics'][metric] for s in steps])
        result[metric]={'real':a.tolist(),'rewired':b.tolist()}
        if len(steps)>1:
            area_a=float(np.trapezoid(a,steps)/(steps[-1]-steps[0]));area_b=float(np.trapezoid(b,steps)/(steps[-1]-steps[0]))
            result[metric].update(real_normalized_area=area_a,rewired_normalized_area=area_b,area_gap_real_minus_rewired=area_a-area_b)
    return result
