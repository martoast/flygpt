"""Regenerate a concise report directly from saved G2c measurements."""
import datetime as dt
import json
from pathlib import Path
import numpy as np

ROOT=Path('results/g2c_overnight')

def read(path):return json.loads(path.read_text())
def pct(x):return f'{100*x:.1f}%'

def main():
    now=dt.datetime.now(dt.timezone.utc).isoformat();lines=['# G2c overnight report','',f'Updated: {now}. All results are computational; no living tissue was used.','',
        'This is adaptive exploratory research. The primary outcome is exact autoregressive complete-answer accuracy, including the end marker, on unseen input instances. Response CE is in nats per task symbol, not nats per byte.','',
        'The first task applies a fixed one-to-one substitution to six symbols from a four-symbol alphabet. Training has 2,048 distinct inputs; validation 256; each qualification and final-test partition has 128. Instance IDs are allocated without replacement. This tests unseen instances at a fixed length, not length generalization or natural-language ability.','']
    failure=ROOT/'failure.json';complete=ROOT/'completion.json'
    if failure.exists():lines+=['**Execution stopped:** '+read(failure)['error'],'']
    elif complete.exists():lines+=['**Run status:** '+read(complete)['reason'],'']
    else:lines+=['**Run status: in progress.** Pending results must not be interpreted as failures or successes.','']
    teachers=[]
    for p in sorted(ROOT.glob('*/teacher_*/validation.json')):
        job=p.parent;v=read(p)['metrics'];q=job/'qualification.json';qm=read(q)['metrics'] if q.exists() else None
        teachers.append((job,v,qm))
    lines+=['## Did we establish a teacher that genuinely generalizes?','']
    if teachers:
        lines+=['| Task / candidate | Validation exact | Locked qualification exact | Gate |','|---|---:|---:|---|']
        for job,v,q in teachers:lines.append(f'| {job.parent.name}/{job.name} | {pct(v["accuracy"])} | {pct(q["accuracy"]) if q else "unopened"} | {"PASS" if q and q["accuracy"]>=.95 else "not passed"} |')
        lines+=['','A pass establishes ≥95% observed exact accuracy on this finite held-out sample. It does not prove correctness on every possible input. Thresholds were fixed before qualification.','']
    else:lines+=['Teacher evaluation pending.','']
    for p in sorted(ROOT.glob('*/baseline_validation.json')):
        b=read(p);lines+=[f'Baselines for **{p.parent.name}**, on validation: GRU {pct(b["gru"]["accuracy"])} exact; '+', '.join(k+' '+pct(v['accuracy']) for k,v in b['ngram'].items())+f'. Teacher parameters: {b["teacher_parameters"]:,}; GRU: {b["gru_parameters"]:,}.','']
    comparisons=[read(p) for p in sorted(ROOT.glob('*/**/comparison_*.json'))]
    lines+=['## Matched locked-test results','']
    if comparisons:
        lines+=['| Task / substrate | Seed | Updates | KD exact | CE-only exact | Rewired KD exact | Transfer (pp) | Topology (pp) | KD zero-edge exact |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in comparisons:
            m=r['metrics'];lines.append(f'| {r["task"]} / {r["substrate"]} | {r["seed"]} | {r["updates"]} | {pct(m["real_kd"]["accuracy"])} | {pct(m["real_ce"]["accuracy"])} | {pct(m["rewired_kd"]["accuracy"])} | {r["A_transfer"]*100:+.2f} | {r["A_topology"]*100:+.2f} | {pct(m["real_kd"]["zero_edge_exact"])} |')
        lines+=['','Each comparison has matched training examples/order, updates, optimizer, initialization rule, I/O populations and architecture. KD necessarily adds teacher forward-pass overhead; wall-clock times are recorded rather than claimed identical. “Full” uses all 166,700 MaleCNS neurons and 25,582,938 directed edges; induced subgraphs are explicitly labeled.','',
            '| Task / substrate / seed / updates | Condition | Response CE | Teacher KL | Exact teacher agreement | Zero-edge CE |','|---|---|---:|---:|---:|---:|']
        for r in comparisons:
            for c,m in r['metrics'].items():lines.append(f'| {r["task"]}/{r["substrate"]}/{r["seed"]}/{r["updates"]} | {c} | {m["response_ce"]:.4f} | {m["teacher_kl"]:.4f} | {pct(m["exact_teacher_agreement"])} | {m["zero_edge_ce"]:.4f} |')
        lines+=['']
    else:lines+=['No matched cohort has completed final testing yet. The final test remains locked until every condition reaches the same budget.','']
    full=[r for r in comparisons if r['substrate']=='full']
    lines+=['## Answers to the critical questions','']
    if not full:
        lines+=['- **Did MaleCNS itself generalize?** Pending full-graph held-out exact-answer evaluation.',
            '- **Did distillation improve over CE-only?** Pending matched results.',
            '- **Did recurrent-edge ablation destroy capability?** Pending intact-versus-zero-edge exact-answer evaluation.',
            '- **Did biological topology differ from rewiring?** Pending matched results.',
            '- **Which hypothesis is supported/falsified?** A qualified teacher removes the teacher-capability bottleneck for the passed task; no student-transfer conclusion yet.',
            '- **Single highest-value next experiment:** Finish the frozen full-MaleCNS KD / CE-only / rewired KD cohort and its locked exact-answer test.','']
    else:
        r=full[-1];m=r['metrics'];kd=m['real_kd'];ce=m['real_ce']
        lines += [f'- **Did MaleCNS itself generalize?** At {r["updates"]} updates, seed {r["seed"]}, distilled exact accuracy is {pct(kd["accuracy"])} and CE-only is {pct(ce["accuracy"])} on {kd["cases"]} unseen instances. Partial accuracy is not mastery.',
            f'- **Did distillation improve over CE-only?** Observed paired difference: {100*r["A_transfer"]:+.2f} percentage points. '+('This is an exploratory lead requiring independent seed replication.' if r['A_transfer']>0 else 'This comparison does not support a distillation advantage.'),
            f'- **Did recurrent-edge ablation destroy capability?** Distilled exact accuracy changes from {pct(kd["accuracy"])} to {pct(kd["zero_edge_exact"])}; response CE changes from {kd["response_ce"]:.4f} to {kd["zero_edge_ce"]:.4f}. '+('Without strong intact capability, this cannot establish destruction of a mastered function.' if kd['accuracy']<.95 else 'The measured ablation assesses necessity of recurrence for this checkpoint.'),
            f'- **Did biological topology differ from rewiring?** Observed difference: {100*r["A_topology"]:+.2f} percentage points. A single seed is insufficient evidence of a robust topology advantage.',
            '- **Which hypothesis is supported/falsified?** '+('The observed positive KD-minus-CE difference is consistent with functional transfer on this specific task, pending replication and uncertainty assessment.' if r['A_transfer']>0 else 'The current fixed training configuration has not shown a KD generalization advantage. This does not falsify representability or all possible compilation procedures.'),
            '- **Single highest-value next experiment:** '+('Complete independent matched seed replication of the transfer difference.' if r['A_transfer']>0 else 'Use matched training/validation probes and the preplanned budget extension to distinguish basic learnability from KD-specific failure.'),'']
        groups={}
        for x in full:groups.setdefault((x['task'],x['updates']),[]).append(x)
        for key,rs in groups.items():
            if len(rs)>1:
                vals=np.array([x['A_transfer'] for x in rs]);lines += [f'{key[0]} at {key[1]} updates: {len(rs)} independent paired seeds; mean transfer {100*vals.mean():+.2f} pp, seed SD {100*vals.std(ddof=1):.2f} pp. Replication may have been selected after the exploratory seed-zero result; this is not confirmatory evidence.','']
    lines+=['## Learning curves and provenance','',
        'Every saved progress file contains per-update loss, gradient norm, response symbols seen, hidden-state RMS/saturation, validation exact accuracy/CE/KL at fixed checkpoints and cumulative wall-clock time. Final evaluation JSON retains per-instance predictions for paired analysis.','']
    for p in sorted(ROOT.glob('*/**/progress.json')):
        r=read(p);cfg=r['config']
        if cfg['kind']!='graph':continue
        last=r.get('evaluations',[]);v=last[-1]['validation']['metrics'] if last else {}
        lines.append(f'- `{p.parent}`: {r["step"]} updates, {r["response_symbols"]:,} response symbols, {r["wall_seconds"]/60:.1f} min; latest validation exact {pct(v.get("accuracy",0))}, CE {v.get("response_ce",0):.4f}.')
    lines+=['','Protocol: `configs/g2c_overnight_v1.json`. Qualification receipts, frozen cohort specs, code/input hashes, teacher/graph hashes, optimizer and sampling RNG are preserved. Checkpoints are copied and hash-verified on `/Volumes/Seagate/FlyGPT Backups/G2c-overnight/`; raw graphs/checkpoints stay out of Git. Code and small results are committed/pushed at stage boundaries.','',
        'G1 remains a finite-function encoding result, not novel-composition generalization. G2/G2b negative results remain unchanged. G2c does not support claims about living brains, natural language, or a global minimum substrate capacity.','']
    path=Path('OVERNIGHT_REPORT.md');tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines));tmp.replace(path)

if __name__=='__main__':main()
