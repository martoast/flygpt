"""Live compiler-study report from measured artifacts, never projected successes."""
import datetime as dt
import json
from pathlib import Path
import numpy as np
ROOT=Path('results/compiler_v1')
def read(p):return json.loads(p.read_text())
def pc(v):return f'{100*v:.1f}%'
def main():
    lines=['# Compiler benchmark report','',f'Updated {dt.datetime.now(dt.timezone.utc).isoformat()}. Computational full MaleCNS experiments; no living tissue.','',
        'The substitution dataset, teacher, 128-instance primary test (`test_extension`), recurrent architecture and optimization budgets are fixed. This test was already inspected: the tournament is exploratory, not fresh confirmatory evaluation. Exact complete-answer accuracy is primary.','']
    if (ROOT/'failure.json').exists():lines+=['**Execution stopped:** '+read(ROOT/'failure.json')['error'],'']
    elif (ROOT/'completion.json').exists():lines+=['**Status:** frozen study completed.','']
    else:lines+=['**Status:** running / queued. Empty result cells are pending, not failures.','']
    backup=ROOT/'backup_status.json'
    if backup.exists() and not read(backup)['external_connected']:
        lines+=['**Backup status:** Seagate is offline. New checkpoints are preserved in local staging with a 3 GiB free-space reserve; they are not yet externally backed up. Migration and hash verification resume automatically when the drive returns.','']
    audit=ROOT/'teacher_label_audit.json'
    if audit.exists():
        a=read(audit);lines += [f'Teacher-only generation audit: {a["matches"]}/{a["cases"]} generated training answers equal the original answers. '+('Consequently hard-target CE has exactly the same objective as supervised CE. A separate run checks implementation equivalence; this does not establish improved sample efficiency or direct parameter translation.' if a['all_identical'] else 'Targets are retained exactly as generated; original answers are not used to repair them.'),'']
    lines+=['## Paired CE-only replication','', '| Seed | MaleCNS CE exact | Rewired CE exact | Difference (pp) | MaleCNS zero-edge exact |','|---|---:|---:|---:|---:|']
    pairs=[]
    for p in sorted(ROOT.glob('paired_ce/seed_*/comparison.json')):
        r=read(p);pairs.append(r);m=r['metrics']
        lines.append(f'| {r["seed"]} | {pc(m["real_ce"]["accuracy"])} | {pc(m["rewired_ce"]["accuracy"])} | {100*r["A_topology_CE"]:+.2f} | {pc(m["real_ce"]["zero_edge_exact"])} |')
    if not pairs:lines.append('| pending | — | — | — | — |')
    lines+=['','Seed zero reuses the original real-CE checkpoint; it is not counted as a new independent replication. Rewired seed zero is newly trained. Seeds 1–4 are fresh paired replications.','']
    for label,rs in [('all available seeds',pairs),('fresh seeds 1–4',[r for r in pairs if r['seed']>0])]:
        if rs:
            acc=np.array([r['metrics']['real_ce']['accuracy'] for r in rs]);delta=np.array([r['A_topology_CE'] for r in rs])
            text=f'{label}: n={len(rs)}, mean MaleCNS CE {pc(acc.mean())}, mean paired topology difference {100*delta.mean():+.2f} pp.'
            if len(rs)>1:text+=f' Seed SD: {100*acc.std(ddof=1):.2f} pp (accuracy), {100*delta.std(ddof=1):.2f} pp (difference).'
            lines += [text,'']
    historical=Path('results/g2c_overnight/substitute_6/full/seed_0/comparison_1024.json')
    if pairs and historical.exists():
        zero=next((r for r in pairs if r['seed']==0),None)
        if zero:
            kd=read(historical)['A_topology'];ce=zero['A_topology_CE']
            lines += [f'Seed-zero topology difference is {100*ce:+.2f} pp under CE versus {100*kd:+.2f} pp under KD; their difference is {100*(ce-kd):+.2f} pp. This is a descriptive topology-by-objective comparison, not a replicated interaction estimate.','']
    lines+=['## Compiler outcomes','', '| Method / seed | Exact | CE reference exact | Difference (pp) | Response CE | Teacher KL | Zero-edge exact |','|---|---:|---:|---:|---:|---:|---:|']
    equivalence_notes=[]
    for p in sorted(ROOT.glob('hard_comparisons/seed_*.json')):
        r=read(p)
        if 'A_transfer' not in r:continue
        m=r['metrics'];lines.append(f'| C1 hard / {r["seed"]} | {pc(m["accuracy"])} | {pc(r["CE_metrics"]["accuracy"])} | {100*r["A_transfer"]:+.2f} | {m["response_ce"]:.4f} | {m["teacher_kl"]:.4f} | {pc(m["zero_edge_exact"])} |')
        equivalence_notes.append(f'Hard seed {r["seed"]} model/optimizer/RNG exact equivalence: {r["state_equivalence"]["all_equal"]}.')
    baseline=ROOT/'paired_ce/seed_0/real_ce_test.json';base=read(baseline)['metrics']['accuracy'] if baseline.exists() else None
    for p in sorted(ROOT.glob('screening/*_test.json')):
        m=read(p)['metrics'];lines.append(f'| {p.stem.removesuffix("_test")} / 0 | {pc(m["accuracy"])} | {pc(base) if base is not None else "pending"} | {100*(m["accuracy"]-base):+.2f} | {m["response_ce"]:.4f} | {m["teacher_kl"]:.4f} | {pc(m["zero_edge_exact"])} |')
    for p in sorted(ROOT.glob('selected_replications/*.json')):
        m=read(p)['metrics'];seed=int(p.stem.rsplit('_',1)[1]);b=read(ROOT/'paired_ce'/f'seed_{seed}'/'real_ce_test.json')['metrics']['accuracy']
        lines.append(f'| {p.stem} | {pc(m["accuracy"])} | {pc(b)} | {100*(m["accuracy"]-b):+.2f} | {m["response_ce"]:.4f} | {m["teacher_kl"]:.4f} | {pc(m["zero_edge_exact"])} |')
    lines+=['',*equivalence_notes,'']
    selection=ROOT/'screening/selection.json'
    lines+=['','Batch-one objective selection: '+(read(selection)['method'] if selection.exists() else 'pending; full-validation exact accuracy, then CE, then name determines selection before candidate test evaluation.'),'',
        'C5 uses a separate batch-two cohort with ground-truth CE, hard-teacher CE and T=2 KD controls, each 512 updates / 1,024 examples. It cannot be compared to batch-one results as if optimization were identical.','']
    for p in sorted(ROOT.glob('relational/seed_*/comparison.json')):
        r=read(p);lines += [f'Relational seed {r["seed"]}: '+', '.join(k+' '+pc(m['accuracy']) for k,m in r['metrics'].items())+'.','']
    lines+=['## Execution and provenance','']
    active=ROOT/'active.json'
    if active.exists() and not read(active).get('idle'):
        a=read(active);lines += [f'Current job: `{a["log"]}` (PID {a["pid"]}).','']
    for p in sorted(ROOT.glob('**/progress.json')):
        r=read(p);ev=r.get('evaluations',[]);m=ev[-1]['validation']['metrics'] if ev else {}
        lines.append(f'- `{p.parent}`: {r["step"]} updates; {r["response_symbols"]} response symbols; {r["wall_seconds"]/60:.1f} min; validation exact {pc(m.get("accuracy",0))}.')
    lines+=['','Every checkpoint is preserved with hashes. Archives go to Seagate when available and to bounded local staging while disconnected; staged files are migrated and verified on reconnect. Final checkpoint references may be symlinks to conserve internal storage. Losses, hidden-state norms/saturation, gradients, optimizer state, RNG, input hashes and per-instance evaluation outputs are preserved. Source/protocol: `results/compiler_v1/frozen_plan.json` and `configs/compiler_v1.json`.','',
        'The unresolved questions are robustness across independent seeds, topology effects under CE, and whether teacher-derived objectives match or improve supervised training. “Matches” is a descriptive two-percentage-point target, not a formal noninferiority conclusion. Positive screening results alone are not robust transfer evidence.','']
    p=Path('COMPILER_REPORT.md');tmp=p.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines));tmp.replace(p)
if __name__=='__main__':main()
