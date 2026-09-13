"""Preregistered fresh-seed CE confirmation; test stays closed until all training ends."""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy import stats
from src.provenance import save_json, sha256

ROOT = Path('results/topology_confirmation_v1')
CFG = Path('configs/topology_confirmation_v1.json')
PY = str(Path('.venv/bin/python').absolute())
BACKUP = Path('/Volumes/Seagate/FlyGPT Backups/G2c-overnight')

def read(p): return json.loads(Path(p).read_text())
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def hashes(paths): return {str(p): sha256(p) for p in paths}
def freeze(p, value):
    p = Path(p)
    if p.exists():
        if read(p) != value: raise RuntimeError(f'Frozen value changed: {p}')
    else: save_json(p, value)
    return str(p)
def verify(values):
    for p, h in values.items():
        if sha256(p) != h: raise RuntimeError(f'Frozen input changed: {p}')

def assign_test(old, count, salt):
    used = [int(r['id']) for rows in old.values() for r in rows]
    if len(used) != len(set(used)): raise RuntimeError('Original split overlap')
    available = set(range(4096)) - set(used)
    if len(available) < count: raise RuntimeError('Insufficient unused domain')
    ids = sorted(available, key=lambda i: hashlib.sha256(f'{salt}:{i}'.encode()).digest())[:count]
    return ids, len(available)

def prepare_data(cfg):
    from scripts.g2c_engine import BOS, SEP, EOS, transform
    source = Path(cfg['source_dataset']); dest = Path(cfg['dataset']); dest.mkdir(parents=True, exist_ok=True)
    meta = read(source/'manifest.json'); old = {}
    for name, info in meta['counts'].items():
        p = source/f'{name}.json'
        if sha256(p) != info['sha256']: raise RuntimeError('Original dataset changed')
        old[name] = read(p)
    ids, available = assign_test(old, cfg['test_count'], cfg['test_assignment_salt'])
    for name in ['train', 'validation']:
        p = dest/f'{name}.json'
        if p.exists() and sha256(p) != sha256(source/p.name): raise RuntimeError('Dataset collision')
        shutil.copy2(source/p.name, p)
    rows = []
    for identifier in ids:
        digits = [(identifier // (4**i)) % 4 for i in reversed(range(6))]
        rows.append(dict(id=identifier, input=digits, prompt=[BOS,*digits,SEP], response=[*transform(digits,'substitute'),EOS]))
    freeze(dest/'test_confirm.json', rows)
    freeze(dest/'manifest.json', dict(version='topology-confirmation-v1', task='substitute', length=6,
        counts={n:dict(count=len(old[n]) if n in old else len(rows), sha256=sha256(dest/f'{n}.json')) for n in ['train','validation','test_confirm']},
        exclusion='All previously allocated splits excluded', available_before_selection=available,
        original_inputs=hashes([source/'manifest.json', *[source/f'{n}.json' for n in old]])))
    freeze(ROOT/'leakage_audit.json', dict(passed=True, unused_domain_size=available,
        test_count=len(ids), disjoint_from_all_previous_splits=True, test_sha256=sha256(dest/'test_confirm.json')))

def analysis(real, rewired):
    real=np.asarray(real,dtype=float); rewired=np.asarray(rewired,dtype=float)
    if real.shape != (10,) or rewired.shape != (10,) or not np.isfinite([real,rewired]).all():
        raise ValueError('Exactly ten finite paired results required')
    gaps=real-rewired; mean=float(gaps.mean()); sd=float(gaps.std(ddof=1)); se=sd/np.sqrt(10)
    p=float(2*stats.t.sf(abs(mean/se),9)) if se else (0. if mean else 1.)
    width=float(stats.t.ppf(.975,9)*se)
    null=np.abs(np.asarray(list(itertools.product([-1,1],repeat=10)))@gaps/10)
    return dict(n=10,real_mean=float(real.mean()),rewired_mean=float(rewired.mean()),gaps=gaps.tolist(),
        mean_gap=mean,gap_sd=sd,ci95=[mean-width,mean+width],primary_p=p,
        primary_pass=bool(mean>0 and p<.05),sign_flip_p=float(np.mean(null>=abs(mean)-1e-12)),
        positive_pairs=int((gaps>0).sum()),practical_secondary_lower_exceeds_10pp=bool(mean-width>.10),
        real_sd=float(real.std(ddof=1)),rewired_sd=float(rewired.std(ddof=1)))

def preserve(message):
    from scripts.compiler_storage import flush_pending
    flush_pending()
    paths=[ROOT,CFG,Path('TOPOLOGY_CONFIRMATION_PREREGISTRATION.md'),Path('scripts/run_topology_confirmation.py'),Path('tests/test_topology_confirmation.py'),Path('data/raw/topology_confirmation_v1'),Path('results/compiler_v1/priority_handoff.json')]
    report=Path('TOPOLOGY_CONFIRMATION_REPORT.md')
    if report.exists(): paths.append(report)
    if Path('/Volumes/Seagate').is_mount():
        for root in paths:
            for p in (root.rglob('*') if root.is_dir() else [root]):
                if p.is_file() and p.suffix in ('.json','.jsonl','.md','.py'):
                    d=BACKUP/p;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    subprocess.run(['git','add','--',*map(str,paths)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:
        subprocess.run(['git','commit','-m',message],check=True)
    subprocess.run(['git','push'],check=True,timeout=120)

def run():
    import fcntl
    from scripts import run_compiler_v1 as v
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())])
    try:
        # Retain the active hard-label child; prevent a second memory-heavy job.
        handoff=read('results/compiler_v1/priority_handoff.json'); pid=handoff['retained_active_child']['pid']
        save_json(ROOT/'status.json',dict(stage='waiting_for_existing_hard_label_run',utc=now(),pid=os.getpid()))
        while True:
            state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'): break
            time.sleep(20)
        # Complete existing C1 evaluation/equivalence before postponing its queue.
        v.hard_run(0,read(v.ROOT/'frozen_plan.json'),read(v.CFG))
        save_json(v.ROOT/'postponed.json',dict(utc=now(),reason='Fresh topology confirmation has user priority',resume='Original frozen plan remains intact'))
        v.ROOT=ROOT;v.CFG=CFG;v.refresh=lambda:None
        cfg=read(CFG)
        registration=read(ROOT/'registration.json');verify(registration['inputs'])
        prepare_data(cfg)
        save_json(ROOT/'status.json',dict(stage='preparing_fresh_rewired_graphs',utc=now(),pid=os.getpid()))
        for seed in cfg['rewiring_seeds']:
            graph=Path(f'data/processed/controls/rewired_{seed}.npz'); audit=Path(f'results/malecns_v1/controls/rewired_{seed}.json')
            if not audit.exists():
                if graph.exists(): raise RuntimeError('Unaudited graph exists; inspect before retry')
                v.child('src.graph_controls',['--graph','data/processed/malecns.npz','--condition','rewired','--seed',seed,'--swap-factor',10],ROOT/f'graph_{seed}.log')
            a=read(audit);verify(a['inputs']);assert sha256(graph)==a['graph_sha256'] and a['successful_swaps']==10*a['n_edge_parameters']
            freeze(ROOT/'graph_audits'/audit.name,a)
        pairs={}; specs=[]; base=read(cfg['reference_spec'])
        for seed,rseed in zip(cfg['seeds'],cfg['rewiring_seeds']):
            pair={}
            for condition in ['real_ce','rewired_ce']:
                job=ROOT/'pairs'/f'seed_{seed}'/condition
                s={**base,'seed':seed,'sampling_seed':75000+seed,'dataset':cfg['dataset'],'job_dir':str(job),
                   'condition':condition,'graph':'data/processed/malecns.npz' if condition=='real_ce' else f'data/processed/controls/rewired_{rseed}.npz',
                   'objective':'ce','budgets':[1024],'checkpoints':[64,128,256,512,768,1024],
                   'evidence_tier':'Preregistered fresh-seed fixed-corpus confirmation'}
                pair[condition]=freeze(job/'spec.json',s);specs.append(pair[condition])
            pairs[str(seed)]=pair
        inputs=[*registration['inputs'],ROOT/'registration.json',*specs,*Path(cfg['dataset']).glob('*.json'),*Path(ROOT/'graph_audits').glob('*.json')]
        inputs += [read(p)['graph'] for p in specs]
        plan=freeze(ROOT/'frozen_plan.json',dict(pairs=pairs,inputs=hashes(sorted(set(map(str,inputs)))),test_gate='all_twenty_checkpoints_complete'))
        preserve('Preregister topology confirmation: freeze ten pairs and untouched test before training')
        checkpoints={}
        for seed,pair in pairs.items():
            verify(read(plan)['inputs'])
            checkpoints[seed]={}
            for condition,p in pair.items():
                save_json(ROOT/'status.json',dict(stage='training',seed=int(seed),condition=condition,utc=now(),pid=os.getpid(),final_test_opened=False))
                checkpoints[seed][condition]=v.train(p,'scripts.g2c_engine')
            save_json(ROOT/'completed_training.json',checkpoints)
            preserve(f'Topology confirmation: preserve trained seed {seed}; test remains closed')
        verify(read(plan)['inputs'])
        # The evaluator validates this receipt before loading ANY final test rows.
        receipt=freeze(ROOT/'unlock.json',dict(split='test_confirm',choices_frozen=True,
            inputs=hashes([plan,*[p for pair in checkpoints.values() for p in pair.values()]]),all_twenty_complete=True))
        outputs={}
        for seed,pair in pairs.items():
            save_json(ROOT/'status.json',dict(stage='final_evaluation',seed=int(seed),utc=now(),pid=os.getpid()))
            scores={c:v.evaluate(p,checkpoints[seed][c],'test_confirm',ROOT/'pairs'/f'seed_{seed}'/f'{c}_test.json',receipt) for c,p in pair.items()}
            outputs[seed]=v.compare(scores);save_json(ROOT/'pairs'/f'seed_{seed}'/'comparison.json',outputs[seed])
            preserve(f'Topology confirmation: preserve final paired seed {seed}')
        summary=analysis([r['metrics']['real_ce']['accuracy'] for r in outputs.values()],[r['metrics']['rewired_ce']['accuracy'] for r in outputs.values()])
        verify(read(plan)['inputs']);verify(read(receipt)['inputs'])
        save_json(ROOT/'analysis.json',summary)
        lines=['# Topology confirmation v1','',f"Primary criterion passed: **{summary['primary_pass']}**.",'',
            '| Seed | MaleCNS | Rewired | Gap (pp) |','|---|---:|---:|---:|']
        for seed,r in outputs.items():
            a=r['metrics']['real_ce']['accuracy'];b=r['metrics']['rewired_ce']['accuracy'];lines.append(f'| {seed} | {100*a:.2f}% | {100*b:.2f}% | {100*(a-b):+.2f} |')
        lines += ['',f"Mean gap {100*summary['mean_gap']:+.2f} pp; 95% paired t interval [{100*summary['ci95'][0]:+.2f}, {100*summary['ci95'][1]:+.2f}] pp; primary p={summary['primary_p']:.6g}; sign-flip sensitivity p={summary['sign_flip_p']:.6g}.",'',
            'All twenty checkpoints completed before test evaluation. Input and checkpoint audit passed. Full per-case CE/KL, agreement and recurrent ablations are retained in each paired result; training curves remain in progress files.',
            'This confirms or fails to confirm a fixed-corpus, fixed-budget substitution effect; it does not identify a structural mechanism or establish distillation benefit.']
        Path('TOPOLOGY_CONFIRMATION_REPORT.md').write_text('\n'.join(lines)+'\n')
        save_json(ROOT/'completion.json',dict(utc=now(),artifact_audit_passed=True,primary_pass=summary['primary_pass']))
        save_json(ROOT/'status.json',dict(stage='complete',utc=now()))
        preserve('Topology confirmation: publish complete preregistered results')
        subprocess.run(['osascript','-e','display notification "Ten paired seeds and locked final tests complete." with title "FlyGPT confirmation ready"'],check=False)
    except BaseException as exc:
        import traceback
        save_json(ROOT/'failure.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()));raise
    finally: awake.terminate()

def register():
    cfg=read(CFG);base=read(cfg['reference_spec'])
    inputs=[CFG,Path(__file__),Path('TOPOLOGY_CONFIRMATION_PREREGISTRATION.md'),Path('tests/test_topology_confirmation.py'),
            'scripts/g2c_engine.py','scripts/run_compiler_v1.py','scripts/compiler_storage.py','scripts/run_g2c_overnight.py',
            *Path('src').glob('*.py'),cfg['reference_spec'],base['teacher_spec'],base['teacher_checkpoint'],base['qualification_receipt'],
            'data/processed/malecns.npz',*Path(cfg['source_dataset']).glob('*.json')]
    p=ROOT/'registration.json'
    if p.exists(): verify(read(p)['inputs'])
    else: freeze(p,dict(utc=now(),inputs=hashes(inputs),registered_before_new_data_and_training=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    if args.register: register()
    else: run()
