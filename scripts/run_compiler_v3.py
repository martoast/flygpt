"""Prepare and execute the frozen six-method pilot on one machine."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch
from scripts import compiler_v3_engine as c
from scripts import g2c_engine as e
from scripts import run_compiler_v1 as v
from src.provenance import save_json,sha256

ROOT=Path('results/compiler_v3');CFG=Path('configs/compiler_v3.json')
def read(p):return json.loads(Path(p).read_text())
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def verify(inputs):
    for p,h in inputs.items():
        if sha256(p)!=h:raise RuntimeError(f'Frozen input changed: {p}')
def paths_hashes(paths):return {str(p):sha256(p) for p in paths}

def prepare_data(cfg):
    source=Path(cfg['source_dataset']);dest=Path(cfg['dataset']);dest.mkdir(parents=True,exist_ok=True)
    old=read(source/'manifest.json');used=set();inputs=[source/'manifest.json']
    for name,info in old['counts'].items():
        p=source/f'{name}.json';assert sha256(p)==info['sha256'];inputs.append(p)
        ids={r['id'] for r in read(p)}
        if used&ids:raise RuntimeError('Original split leakage')
        used.update(ids)
    confirmation=Path('data/raw/topology_confirmation_v2/test_confirm.json');inputs.append(confirmation)
    confirm_ids={r['id'] for r in read(confirmation)}
    if used&confirm_ids:raise RuntimeError('Confirmation test overlaps original splits')
    used.update(confirm_ids)
    remaining=set(range(4096))-used
    ids=sorted(remaining,key=lambda i:hashlib.sha256(f"{cfg['test_salt']}:{i}".encode()).digest())[:cfg['test_count']]
    if len(ids)!=cfg['test_count']:raise RuntimeError('Insufficient unused domain')
    rows=[]
    for i in ids:
        digits=[i//4**j%4 for j in reversed(range(6))]
        rows.append(dict(id=i,input=digits,prompt=[e.BOS,*digits,e.SEP],response=[*e.transform(digits,'substitute'),e.EOS]))
    c.freeze(dest/f"{cfg['primary_split']}.json",rows)
    for name in ['train','validation']:
        p=dest/f'{name}.json'
        if p.exists() and sha256(p)!=sha256(source/p.name):raise RuntimeError('Dataset collision')
        shutil.copy2(source/p.name,p)
    c.freeze(dest/'manifest.json',dict(version='compiler-v3',counts={name:dict(count=len(read(dest/f'{name}.json')),sha256=sha256(dest/f'{name}.json')) for name in ['train','validation',cfg['primary_split']]},
        source_inputs=paths_hashes(inputs),task='substitute',length=6))
    c.freeze(ROOT/'leakage_audit.json',dict(passed=True,unused_before_selection=len(remaining),new_test_count=len(ids),
        new_test_sha256=sha256(dest/f"{cfg['primary_split']}.json"),excluded_original_and_confirmation_ids=True))

def publish(message):
    backup=Path('/Volumes/Seagate/FlyGPT Backups/G2c-overnight')
    if not Path('/Volumes/Seagate').is_mount():raise RuntimeError('Pilot requires connected Seagate')
    paths=[ROOT,CFG,Path('COMPILER_V3_PREREGISTRATION.md'),Path('scripts/compiler_v3_engine.py'),Path('scripts/run_compiler_v3.py'),Path('tests/test_compiler_v3.py'),Path(read(CFG)['dataset'])]
    if Path('COMPILER_V3_REPORT.md').exists():paths.append(Path('COMPILER_V3_REPORT.md'))
    for root in paths:
        for p in (root.rglob('*') if root.is_dir() else [root]):
            if p.is_file() and (p.suffix in ('.json','.jsonl','.md','.py') or p==ROOT/'teacher_cache.pt'):
                dest=backup/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    subprocess.run(['git','add','--',*map(str,paths)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:subprocess.run(['git','commit','-m',message],check=True)
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    if branch!='experiment/compiler-v3':raise RuntimeError('Pilot must use its assigned branch')
    subprocess.run(['git','push','-u','origin',branch],check=True,timeout=120)

def make_spec(cfg,seed,method):
    base=read(cfg['reference_spec']);job=ROOT/f'seed_{seed}'/method
    base.update(seed=seed,sampling_seed=75000+seed,group_seed=125000+seed,correspondence_seed=175000+seed,
        method=method,objective=method,projection_seed=225000+seed,job_dir=str(job),dataset=cfg['dataset'],
        batch=cfg['batch'],budgets=[cfg['updates']],checkpoints=cfg['checkpoints'],curve_cases=cfg['curve_cases'],
        temperature=cfg['temperature'],alignment_weight=cfg['alignment_weight'],evidence_tier=cfg['evidence'],
        training_targets=cfg['training_targets'],training_targets_sha256=sha256(cfg['training_targets']),
        cache=str(ROOT/'teacher_cache.pt'),cache_receipt=str(ROOT/'teacher_cache_receipt.json'),
        schedule=str(ROOT/f'schedule_{seed}.json'),schedule_audit=str(ROOT/f'shuffle_audit_{seed}.json'))
    return c.freeze(job/'spec.json',base)

def prepare():
    cfg=read(CFG);prepare_data(cfg);c.prepare_cache(cfg)
    cache=torch.load(ROOT/'teacher_cache.pt',weights_only=True,map_location='cpu');methods={};orders={}
    for seed in cfg['seeds']:
        schedule=c.schedule(len(cache['ids']),cfg['batch'],cfg['updates'],seed)
        schedule_path=c.freeze(ROOT/f'schedule_{seed}.json',schedule)
        audit=c.audit_schedule(schedule,cache['features'],cache['mask'],cfg['shuffle_epsilon'])
        audit.update(cache_sha256=sha256(ROOT/'teacher_cache.pt'),schedule_sha256=sha256(schedule_path))
        c.freeze(ROOT/f'shuffle_audit_{seed}.json',audit)
        if not audit['passed']:raise RuntimeError('Frozen shuffle audit failed; preserve failure and stop without resampling')
        methods[str(seed)]={method:make_spec(cfg,seed,method) for method in cfg['methods']}
        order=list(cfg['methods']);random.Random(33000+seed).shuffle(order);orders[str(seed)]=order
    # One disposable update tests full-graph batch-eight memory; never a candidate.
    if not (ROOT/'memory_probe.json').exists():
        probe=read(methods[str(cfg['seeds'][0])]['C_hidden']);probe.update(seed=90001,job_dir=str(ROOT/'memory_probe'),budgets=[1],checkpoints=[1],curve_cases=1)
        path=c.freeze(ROOT/'memory_probe/spec.json',probe)
        v.ROOT=ROOT;v.refresh=lambda:None
        checkpoint=v.train(path,'scripts.compiler_v3_engine')
        p=read(ROOT/'memory_probe/progress.json')
        save_json(ROOT/'memory_probe.json',dict(passed=True,batch=8,max_rss_bytes=p['max_rss_bytes'],wall_seconds=p['wall_seconds'],
            checkpoint_sha256=sha256(checkpoint),role='One-update hardware feasibility check; no performance-based selection'))
    inputs=[CFG,'COMPILER_V3_PREREGISTRATION.md','scripts/compiler_v3_engine.py','scripts/run_compiler_v3.py','tests/test_compiler_v3.py',
        'scripts/g2c_engine.py','scripts/compiler_storage.py','scripts/run_compiler_v1.py','scripts/run_g2c_overnight.py',*Path('src').glob('*.py'),
        cfg['teacher_spec'],cfg['teacher_checkpoint'],cfg['qualification_receipt'],cfg['training_targets'],cfg['training_prompts'],
        read(cfg['reference_spec'])['graph'],ROOT/'teacher_cache.pt',ROOT/'teacher_cache_receipt.json',*Path(cfg['dataset']).glob('*.json'),
        *[p for group in methods.values() for p in group.values()],*ROOT.glob('schedule_*.json'),*ROOT.glob('shuffle_audit_*.json')]
    c.freeze(ROOT/'frozen_plan.json',dict(methods=methods,execution_order=orders,inputs=paths_hashes(inputs),
        final_test_gate='All six conditions complete before evaluation',created_before_full_budget_training=True))
    publish('Compiler v3: freeze six conditions, fixed derangements and qualified numerical audits')
    print('PREPARED: publish this branch to GitHub and provide its publication receipt before starting',flush=True)

def run():
    import fcntl
    cfg=read(CFG);plan=read(ROOT/'frozen_plan.json');verify(plan['inputs'])
    publication=read(ROOT/'publication_receipt.json');assert publication['frozen_plan_sha256']==sha256(ROOT/'frozen_plan.json')
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())]);v.ROOT=ROOT;v.refresh=lambda:None
    try:
        checkpoints={}
        for seed,order in plan['execution_order'].items():
            checkpoints[seed]={}
            for method in order:
                verify(plan['inputs']);save_json(ROOT/'status.json',dict(stage='training',seed=int(seed),method=method,pid=os.getpid(),utc=now(),final_test_opened=False))
                p=plan['methods'][seed][method];checkpoints[seed][method]=v.train(p,'scripts.compiler_v3_engine')
                save_json(ROOT/'completed_training.json',checkpoints)
                publish(f'Compiler v3: preserve {method} seed {seed}; final test stays closed')
        verify(plan['inputs'])
        receipt=c.freeze(ROOT/'unlock.json',dict(split=cfg['primary_split'],choices_frozen=True,
            inputs=paths_hashes([ROOT/'frozen_plan.json',ROOT/'publication_receipt.json',*[p for group in checkpoints.values() for p in group.values()]])))
        comparisons={}
        for seed,methods in plan['methods'].items():
            results={};curves={}
            for method,p in methods.items():
                save_json(ROOT/'status.json',dict(stage='evaluation',seed=int(seed),method=method,utc=now(),pid=os.getpid()))
                out=ROOT/f'seed_{seed}'/f'{method}_test.json';already=out.exists();eval_start=time.perf_counter()
                results[method]=v.evaluate(p,checkpoints[seed][method],cfg['primary_split'],out,receipt)
                timing=out.with_name(out.stem+'_timing.json')
                if not timing.exists():save_json(timing,dict(evaluation_wall_seconds=None if already else time.perf_counter()-eval_start,
                    includes_teacher_student_loading_generation_and_ablation=True,existing_result_reused=already))
                curves[method]=c.learning_summary(read(Path(read(p)['job_dir'])/'progress.json'),cfg['threshold'])
            result=v.compare(results);result['learning']=curves;metrics=result['metrics'];hard=metrics['A_hard']['accuracy']
            result['gap_to_hard']={m:r['accuracy']-hard for m,r in metrics.items()}
            result['representation_contrasts']={a:dict(vs_hard=metrics[a]['accuracy']-hard,vs_shuffled=metrics[a]['accuracy']-metrics[b]['accuracy']) for a,b in [('C_hidden','D_hidden_shuffled'),('E_relational','F_relational_shuffled')]}
            result['evidence']=cfg['evidence'];comparisons[seed]=result;save_json(ROOT/f'seed_{seed}'/'comparison.json',result)
        verify(plan['inputs']);verify(read(receipt)['inputs'])
        lines=['# Compiler v3: six-condition pilot','',cfg['evidence'],'','All six models trained before opening the fresh compiler test. No statistical significance or replication claim is made.','']
        for seed,result in comparisons.items():
            lines+=['Seed '+seed,'','| Method | Exact | Gap to hard (pp) | Response CE | KL | Zero-edge exact | Acquisition min | End-to-end min | First ≥85% validation update |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for m,r in result['metrics'].items():
                curve=result['learning'][m];cross=curve['first_observed_crossing'];threshold=cross['step'] if cross else '>512 (censored)'
                lines.append(f"| {m} | {100*r['accuracy']:.2f}% | {100*result['gap_to_hard'][m]:+.2f} | {r['response_ce']:.5f} | {r['teacher_kl']:.5f} | {100*r['zero_edge_exact']:.2f}% | {curve['acquisition_seconds']/60:.1f} | {curve['end_to_end_seconds']/60:.1f} | {threshold} |")
        lines+=['','A representation-specific pilot signal requires beating both A and its matched shuffled control; even then independent seeds are needed. Full learning curves and threshold intervals are preserved in comparison.json.',
                'Acquisition time charges full standalone teacher-answer generation and, where used, extraction/cache cost. End-to-end time additionally includes validation and checkpoint I/O. Teacher pretraining is excluded. Final-test evaluation overhead is recorded separately in evaluation manifests/logs and is not a training-efficiency advantage.']
        Path('COMPILER_V3_REPORT.md').write_text('\n'.join(lines)+'\n');save_json(ROOT/'completion.json',dict(utc=now(),artifact_audit_passed=True,evidence=cfg['evidence']))
        save_json(ROOT/'status.json',dict(stage='complete',utc=now()));publish('Compiler v3: preserve all six pilot outcomes and matched-control analysis')
    except BaseException as exc:
        import traceback
        save_json(ROOT/'failure.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()));raise
    finally:awake.terminate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--data-only',action='store_true');a=p.parse_args()
    if a.data_only:prepare_data(read(CFG))
    elif a.prepare:prepare()
    else:run()
