"""Full, automatically finalized paired-CE report from completed matched tests."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.stats import spearmanr
from scripts.paired_ce_analysis import seed_summary,case_summary,learning_summary
from src.provenance import sha256,save_json

ROOT=Path('results/compiler_v1');OUT=ROOT/'full_report';REPORT=Path('PAIRED_CE_REPORT.md')

def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}'

def collect():
    plan=read(ROOT/'frozen_plan.json');pairs=[];inputs={};curves={};progress={}
    for seed in range(5):
        specs=plan['paired_ce'][str(seed)];curves[seed]={};progress[seed]={}
        for condition,spec_path in specs.items():
            job=Path(read(spec_path)['job_dir']);p=job/'progress.json'
            if p.exists():progress[seed][condition]=read(p);curves[seed][condition]=read(p)['evaluations']
        p=ROOT/'paired_ce'/f'seed_{seed}'/'comparison.json'
        if not p.exists():continue
        r=read(p);assert r['updates']==1024 and r['seed']==seed;raw={}
        for condition in ('real_ce','rewired_ce'):
            q=p.parent/f'{condition}_test.json';v=read(q);raw[condition]=v
            assert len(v['rows'])==128
            assert abs(np.mean([x['exact'] for x in v['rows']])-v['metrics']['accuracy'])<1e-12
            assert abs(np.mean([x['zero_edge_exact'] for x in v['rows']])-v['metrics']['zero_edge_exact'])<1e-12
            assert v['metrics']==r['metrics'][condition]
            assert progress[seed][condition]['step']==1024
            inputs[str(q)]=sha256(q)
        inputs[str(p)]=sha256(p)
        pairs.append({**r,'case_analysis':case_summary(raw['real_ce']['rows'],raw['rewired_ce']['rows'],seed),
            'learning':learning_summary(progress[seed]['real_ce'],progress[seed]['rewired_ce'])})
    return pairs,curves,progress,inputs

def graph_correlations(pairs):
    if len(pairs)!=5:return {'status':'pending all five final outcomes'}
    files=[OUT/'graph_properties'/f'rewired_{s}.json' for s in range(5)]
    if not all(p.exists() for p in files):return {'status':'pending graph diagnostics'}
    graphs=[read(p) for p in files];y=np.array([r['metrics']['rewired_ce']['accuracy'] for r in pairs]);correlations={}
    for key in ('giant_scc_fraction','reciprocal_fraction','binary_perron_estimate','input_output_mean_min_hops','output_reachable_fraction'):
        values=[g[key] for g in graphs]
        if any(v is None for v in values) or np.ptp(values)==0 or np.ptp(y)==0:
            correlations[key]={'pearson':None,'spearman':None,'reason':'constant or missing feature/outcome','values':values}
        else:
            correlations[key]={'pearson':float(np.corrcoef(values,y)[0,1]),'spearman':float(spearmanr(values,y).statistic),'values':values}
    return {'status':'computed','n':5,'correlations':correlations,
        'limitation':'Post-hoc, multiple exploratory correlations; graph, initialization and data sampling seeds co-vary. No causal attribution or reliable predictive model.'}

def audit(pairs):
    out=OUT/'artifact_audit.json'
    if out.exists():return read(out)
    if len(pairs)!=5:return {'status':'pending completion of all five pairs'}
    plan=read(ROOT/'frozen_plan.json');expected=dict(plan['inputs'])
    for seed in range(5):
        unlock=ROOT/'unlocks'/f'ce_seed_{seed}.json';receipt=read(unlock)
        assert receipt['choices_frozen'] and receipt['split']=='test_extension'
        expected.update(receipt['inputs']);expected[str(unlock)]=sha256(unlock)
        for c in ('real_ce','rewired_ce'):
            p=ROOT/'paired_ce'/f'seed_{seed}'/f'{c}_test.json';result=read(p)
            expected.update(result['inputs']);expected[str(p)]=sha256(p)
    missing=[];mismatch=[];verified={}
    for path,digest in expected.items():
        p=Path(path)
        if not p.exists():missing.append(path);continue
        actual=sha256(p)
        if actual!=digest:mismatch.append(path)
        else:verified[path]=actual
    result={'status':'passed' if not missing and not mismatch else 'incomplete_or_failed','missing':missing,'mismatch':mismatch,
        'verified':verified,'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    if result['status']=='passed':save_json(out,result)
    return result

def figures(pairs,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True);colors={'real_ce':'#1769aa','rewired_ce':'#d97925'}
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');done={r['seed']:r for r in pairs}
    for seed in range(5):
        if seed in done:
            r=done[seed]
            for offset,c in [(-.18,'real_ce'),(.18,'rewired_ce')]:
                label='MaleCNS CE' if c=='real_ce' else 'Rewired CE'
                val=100*r['metrics'][c]['accuracy'];axes[0].bar(seed+offset,val,.34,color=colors[c],label=label if seed==0 else None)
                axes[0].text(seed+offset,val+1,f'{val:.1f}',ha='center',fontsize=8)
            gap=100*r['A_topology_CE'];axes[1].scatter(seed,gap,color='#1769aa');axes[1].text(seed,gap+2,f'{gap:+.1f}',ha='center',fontsize=9)
        else:
            for ax in axes:ax.text(seed,5,'pending',ha='center',fontsize=8,color='gray')
    axes[0].set(title='Final exact-answer accuracy',ylabel='Percent correct',ylim=(0,105));axes[0].legend(fontsize=8)
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set(title='Paired topology gap: real minus rewired',ylabel='Percentage points',ylim=(-100,100))
    for ax in axes:ax.set(xticks=range(5),xlabel='Paired seed',xlim=(-.5,4.5));ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'CE-only substitution, 1,024 updates — {len(pairs)}/5 pairs complete')
    for ext in ('png','pdf'):fig.savefig(OUT/f'final_accuracy.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(10,12),layout='constrained')
    for seed in range(5):
        for c,ev in curves[seed].items():
            steps=[r['step'] for r in ev]
            for col,key in enumerate(('response_ce','accuracy')):
                values=[r['validation']['metrics'][key] for r in ev]
                axes[seed,col].plot(steps,values,label=c,color=colors[c],ls='-' if c=='real_ce' else '--',marker='.',ms=4)
        axes[seed,0].set(ylabel=f'Seed {seed}: response CE',xlim=(0,1024),ylim=(0,2.1));axes[seed,1].set(ylabel='Exact accuracy',xlim=(0,1024),ylim=(0,1))
        for ax in axes[seed]:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Optimizer updates')
    fig.suptitle('Validation learning curves: 32 fixed cases, distinct from the 128-case final test')
    for ext in ('png','pdf'):fig.savefig(OUT/f'learning_curves.{ext}',dpi=130)
    plt.close(fig)

def render():
    pairs,curves,progress,inputs=collect();complete=len(pairs)==5
    graph=graph_correlations(pairs);verification=audit(pairs)
    final=complete and graph['status']=='computed' and verification['status']=='passed'
    groups={}
    for label,rs in [('all_seeds',pairs),('fresh_seeds',[r for r in pairs if r['seed']>0])]:
        if rs:groups[label]=seed_summary([r['metrics']['real_ce']['accuracy'] for r in rs],[r['metrics']['rewired_ce']['accuracy'] for r in rs])
    summary={'status':'complete' if final else 'interim','completed_pairs':len(pairs),'pairs':pairs,'groups':groups,
        'graph_analysis':graph,'artifact_audit_status':verification['status'],'inputs':inputs,
        'analysis_spec_sha256':sha256('configs/paired_ce_analysis_v1.json'),'report_source_sha256':sha256(__file__)}
    save_json(OUT/'summary.json',summary);figures(pairs,curves)
    lines=['# Paired CE topology replication — full report','',f'Updated {dt.datetime.now(dt.timezone.utc).isoformat()}.', '',
        f'**Status: {"all five final comparisons and artifact audit complete" if final else "INTERIM — "+str(len(pairs))+"/5 matched final comparisons complete"}.** The separate compiler-method tournament is not declared complete by this report.','',
        '## Main result','']
    if pairs:
        g=groups['all_seeds'];lines += [f'Across {len(pairs)} completed pairs, MaleCNS averages **{pct(g["real_mean"])}** exact accuracy and degree-preserving rewiring averages **{pct(g["rewired_mean"])}**. The mean paired gap is **{pp(g["gap_mean"])} percentage points**, with {g["positive_gaps"]}/{g["n"]} positive gaps.','',
            'These are computational results on a fixed-length deterministic task. They do not establish arbitrary computation, length generalization, natural-language competence or implementation in living tissue. The test was previously inspected, and seed zero motivated this replication; inference is exploratory.','']
    lines+=['| Seed | MaleCNS correct / 128 | Rewired correct / 128 | MaleCNS exact | Rewired exact | Gap (pp) |','|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        c=r['case_analysis'];m=r['metrics'];lines.append(f'| {r["seed"]} | {c["real_correct"]} | {c["rewired_correct"]} | {pct(m["real_ce"]["accuracy"])} | {pct(m["rewired_ce"]["accuracy"])} | {pp(c["gap"])} |')
    for seed in sorted(set(range(5))-{r['seed'] for r in pairs}):lines.append(f'| {seed} | pending | pending | pending | pending | pending |')
    lines+=['','![Final accuracies and paired gaps](results/compiler_v1/full_report/final_accuracy.png)','',
        '## Uncertainty and variance','',
        'The primary replication unit is the paired seed. The same 128 test instances appear in every pair; they must not be treated as 640 independent replications. Seed zero reuses the historical MaleCNS checkpoint. Fresh seeds 1–4 are also summarized separately.','',
        '| Group | n | Mean gap (pp) | Gap SD (pp) | MaleCNS SD (pp) | Rewired SD (pp) | Rewired / MaleCNS variance ratio |','|---|---:|---:|---:|---:|---:|---:|']
    interval_notes=[]
    for label,g in groups.items():
        fmt=lambda k:f'{100*g[k]:.2f}' if k in g else 'n/a'
        ratio=g.get('variance_ratio_rewired_over_real');lines.append(f'| {label} | {g["n"]} | {pp(g["gap_mean"])} | {fmt("gap_sd")} | {fmt("real_sd")} | {fmt("rewired_sd")} | {ratio:.2f} |' if ratio is not None else f'| {label} | {g["n"]} | {pp(g["gap_mean"])} | {fmt("gap_sd")} | {fmt("real_sd")} | {fmt("rewired_sd")} | undefined |')
        if complete and 'gap_mean_t95' in g:interval_notes.append(f'{label}: approximate 95% Student-t interval for the mean paired gap: [{pp(g["gap_mean_t95"][0])}, {pp(g["gap_mean_t95"][1])}] pp.')
    lines+=['',*interval_notes]
    lines+=['','Seed-level confidence intervals are reported after all planned pairs finish. They assume approximately independent, normally distributed paired seed differences; n=5 (or four fresh seeds) is small. Variance ratios are descriptive; no variance significance test is used. Conditional paired-example bootstrap intervals below assess sensitivity to test instances for a fixed checkpoint pair, not reliability across training seeds.','',
        '| Seed | Real-only correct | Rewired-only correct | Both correct | Both wrong | Paired-example bootstrap gap interval (pp) |','|---|---:|---:|---:|---:|---|']
    for r in pairs:
        c=r['case_analysis'];lo,hi=c['paired_case_bootstrap_95'];lines.append(f'| {r["seed"]} | {c["real_only"]} | {c["rewired_only"]} | {c["both_correct"]} | {c["both_wrong"]} | [{pp(lo)}, {pp(hi)}] |')
    lines+=['','## Causal ablation and distributional metrics','',
        '| Seed | Condition | Response CE | Teacher KL | Exact teacher agreement | Zero-edge exact | Zero-edge CE |','|---|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        for c,m in r['metrics'].items():lines.append(f'| {r["seed"]} | {c} | {m["response_ce"]:.5f} | {m["teacher_kl"]:.5f} | {pct(m["exact_teacher_agreement"])} | {pct(m["zero_edge_exact"])} | {m["zero_edge_ce"]:.5f} |')
    lines+=['','CE and KL are nats per response symbol, including the end marker. Exact answers are greedy autoregressive generations from prompts with no reference answer prefix. CE/KL use teacher forcing for probability evaluation only. Zero-edge tests remove all recurrent weights from the trained checkpoint, reset state and regenerate; input/output populations remain disjoint. This tests the necessity of recurrence for the measured capability, not biological uniqueness.','',
        '## Learning dynamics','',
        'Both conditions use the same 32 validation instances at the same checkpoints. Normalized trapezoidal area summarizes updates 64–1,024; lower CE area and higher accuracy area are better. It is not extrapolated to update zero or beyond the training budget.','',
        '| Seed | Real CE area | Rewired CE area | Real accuracy area | Rewired accuracy area |','|---|---:|---:|---:|---:|']
    for r in pairs:
        l=r['learning'];lines.append(f'| {r["seed"]} | {l["response_ce"]["real_normalized_area"]:.4f} | {l["response_ce"]["rewired_normalized_area"]:.4f} | {pct(l["accuracy"]["real_normalized_area"])} | {pct(l["accuracy"]["rewired_normalized_area"])} |')
    lines+=['','![Validation learning curves](results/compiler_v1/full_report/learning_curves.png)','',
        'Greater accuracy consistency at a fixed budget would be compatible with more reliable optimization. It cannot prove a better optimization landscape or distinguish optimization speed from ultimate representational capacity; no asymptotic performance is measured.','',
        '## Structural diagnostics and exploratory correlations','',
        '| Graph | Giant SCC fraction | Reciprocal fraction | Binary Perron estimate | Eigenvector residual | Mean minimum I/O hops | Reachable outputs |','|---|---:|---:|---:|---:|---:|---:|']
    for p in sorted((OUT/'graph_properties').glob('*.json')):
        r=read(p);h=r['input_output_mean_min_hops'];lines.append(f'| {p.stem} | {r["giant_scc_fraction"]:.6f} | {r["reciprocal_fraction"]:.6f} | {r["binary_perron_estimate"]:.4f} | {r["perron_relative_residual"]:.2e} | {h:.4f} | {r["output_reachable_fraction"]:.6f} |' if h is not None else f'| {p.stem} | unavailable I/O paths | | | | | |')
    lines+=['','Spectral values describe binary structural adjacency, not trained signed weights or the nonlinear recurrent Jacobian. I/O distances are shortest paths from any input neuron to each output neuron, using the actual fixed population assignment. Degree distributions, node count and edge count are matched and cannot explain variation among the rewired controls.','']
    if graph['status']=='computed':
        lines+=['| Feature vs rewired exact accuracy (n=5) | Pearson r | Spearman rho |','|---|---:|---:|']
        for k,v in graph['correlations'].items():
            lines.append(f'| {k} | {v["pearson"]:.3f} | {v["spearman"]:.3f} |' if v['pearson'] is not None else f'| {k} | undefined | undefined |')
    else:lines += [f'Correlation analysis: {graph["status"]}.']
    lines+=['','These features and correlations were specified after two completed pairs were known. Five points, multiple features and co-varying rewiring/initialization/sampling seeds make these exploratory diagnostics only. A large correlation cannot identify a helpful biological motif. A crossed rewiring-seed × initialization-seed experiment with controlled sampling order would be needed to separate graph structure from optimization variability. No such extra training is launched here.','',
        '## Frozen methods and provenance','',
        '- Task: six input symbols from an alphabet of four, transformed by `(symbol + 1) mod 4`, then an end marker. Training: 2,048 distinct instances; validation: 256; qualification: 128; primary test: 128 disjoint instances. Teacher qualification and this test have 100% exact teacher accuracy in the observed results.',
        '- Teacher: two transformer layers, four attention heads, embedding width 64, 102,656 parameters; 1,200 training updates at batch 32. The 102,861-parameter GRU also reached 100% exact accuracy on 256 validation examples; n-gram orders 1, 3 and 6 reached 0%. These baseline figures are validation results, not extra final-test measurements. See `results/g2c_overnight/substitute_6/baseline_validation.json`.',
        '- Full graph: 166,700 neurons and 25,582,938 directed edges from the MaleCNS preprocessing pipeline. V1 uses signed trainable weights on fixed anatomical topology, not transmitter-constrained dynamics. Degree-rewired graphs preserve each node’s in/out degree; existing graph identities and rewiring audits are retained.',
        '- Student: vocabulary 8, embedding dimension 16, disjoint 1,025-neuron input and output populations, leak 0.65, two recurrent updates per symbol, degree-normalized initial recurrent weights. No arbitrary recurrent edges or decoder bypass.',
        '- Training: AdamW, LR 0.001, weight decay 0.01, gradient clipping 1, batch one, 1,024 updates / 7,168 response symbols per condition. Sampling is with replacement; 1,024 draws are not 1,024 unique training examples. Paired seeds share sampling order and initialization rules. Full optimizer, RNG and checkpoint series are retained.',
        '- Hardware: Apple M1 CPU with four PyTorch threads and SciPy sparse propagation. Never a dense 166,700 × 166,700 adjacency. Actual per-run wall time is retained; matched update budgets do not imply identical wall-clock duration.',
        f'- Final artifact audit: **{verification["status"]}**. Verification covers frozen code/data/teacher/graph inputs, evaluation receipts and checkpoint hashes. The original failed USB launch and storage-only amendment remain recorded.',
        '- Analysis specification: [paired_ce_analysis_v1.json](configs/paired_ce_analysis_v1.json). Numeric results and input hashes: [summary.json](results/compiler_v1/full_report/summary.json). Training freeze: [frozen_plan.json](results/compiler_v1/frozen_plan.json). Figures also have standalone PDF versions in the full_report directory.','',
        '## What this establishes—and what remains open','',
        'The completed checkpoints support task-specific generalization on a real connectome-constrained substrate. A consistent positive paired gap across all seeds would support a topology-dependent advantage under this training budget. It would not establish biological superiority on other tasks or prove that rewired networks lack sufficient capacity.','',
        'Current vanilla KD was worse than supervised CE in the earlier seed-zero comparison. The teacher generates exactly the same 2,048 training answers as the original labels, so teacher-hard CE is mathematically the same objective; its separate implementation-equivalence run is part of the queued compiler study. No compiler success is inferred solely from these CE replications.','',
        'Next scheduled work remains the frozen compiler tournament. Architecture-independence experiments and harder functions stay gated. See [COMPILER_REPORT.md](COMPILER_REPORT.md) and [DECISION_TREE.md](DECISION_TREE.md).','']
    temp=REPORT.with_suffix('.md.tmp');temp.write_text('\n'.join(lines));temp.replace(REPORT)
    return final

def signature():
    paths=list(ROOT.glob('paired_ce/seed_*/*.json'))+list(ROOT.glob('paired_ce/seed_*/*/progress.json'))+list((OUT/'graph_properties').glob('*.json'))
    return [(str(p),p.stat().st_mtime_ns) for p in sorted(paths)]

def publish():
    paths=[REPORT,OUT,Path('configs/paired_ce_analysis_v1.json'),Path('scripts/paired_ce_report.py'),Path('scripts/paired_ce_analysis.py'),Path('scripts/paired_graph_diagnostics.py'),Path('tests/test_paired_ce_analysis.py')]
    drive=Path('/Volumes/Seagate')
    if drive.is_mount():
        backup=drive/'FlyGPT Backups/G2c-overnight'
        for top in paths:
            for p in (top.rglob('*') if top.is_dir() else [top]):
                if p.is_file():dest=backup/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    subprocess.run(['git','add','--',*map(str,paths)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:subprocess.run(['git','commit','-m','Finalize five-seed paired CE report with uncertainty and graph diagnostics'],check=True)
    result=subprocess.run(['git','push'],capture_output=True,text=True,timeout=120)
    save_json(OUT/'publication.json',{'github_push_succeeded':result.returncode==0,'push_error':result.stderr if result.returncode else None,'external_report_copy':drive.is_mount()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    last=None
    while True:
        sig=signature()
        if sig!=last:
            final=render();last=sig
            if final and args.watch:
                publish()
                from scripts.compiler_notify import notify
                notify('full_paired_report_ready','The complete five-seed CE report, confidence intervals and graph diagnostics are ready: PAIRED_CE_REPORT.md.')
                return
        if not args.watch:return
        if (ROOT/'failure.json').exists():return
        time.sleep(30)

if __name__=='__main__':main()
