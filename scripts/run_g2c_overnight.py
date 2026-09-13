"""Frozen-gate, matched-cohort overnight controller. Runs jobs sequentially on M1."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts import g2c_engine as e
from src.provenance import save_json,sha256

ROOT=Path('results/g2c_overnight')
CONFIG=Path('configs/g2c_overnight_v1.json')
PYTHON=str(Path('.venv/bin/python').absolute())
EXTERNAL=Path('/Volumes/Seagate/FlyGPT Backups/G2c-overnight')


def read(p):return json.loads(Path(p).read_text())

def frozen(p,value):
    p=Path(p)
    if p.exists():assert read(p)==value, f'Frozen artifact differs: {p}'
    else:save_json(p,value)
    return str(p)

def hashes(paths):return {str(p):sha256(p) for p in paths}

def event(message,**extra):
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'message':message,**extra})+'\n')
    print(message,extra,flush=True)
    report()

def report():
    subprocess.run([PYTHON,'-m','scripts.overnight_report'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)

def preserve(message):
    if not Path('/Volumes/Seagate').is_mount():raise RuntimeError('Seagate backup disconnected')
    for top in (ROOT,Path('data/raw/g2c_v1'),Path('configs'),Path('scripts'),Path('tests'),Path('src')):
        for p in top.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.py','.md'):
                dest=EXTERNAL/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    if Path('OVERNIGHT_REPORT.md').exists():shutil.copy2('OVERNIGHT_REPORT.md',EXTERNAL/'OVERNIGHT_REPORT.md')
    paths=['scripts/g2c_engine.py','scripts/run_g2c_overnight.py','scripts/overnight_report.py','tests/test_g2c.py',str(CONFIG),str(ROOT),'data/raw/g2c_v1','OVERNIGHT_REPORT.md']
    subprocess.run(['git','add','--',*paths],check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet']).returncode
    if changed:subprocess.run(['git','commit','-m',message],check=True)
    pushed=subprocess.run(['git','push'],capture_output=True,text=True,timeout=120)
    if pushed.returncode:event('GitHub push failed; local commit and USB backup retained',error=pushed.stderr[-2000:])


def child(args,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        proc=subprocess.Popen([PYTHON,'-m','scripts.g2c_engine',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'NUMBA_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
        save_json(ROOT/'active.json',{'pid':proc.pid,'args':list(map(str,args)),'log':str(log),'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        while proc.poll() is None:
            report();time.sleep(20)
        if proc.returncode:raise RuntimeError(f'Engine exited {proc.returncode}: {log}')
    save_json(ROOT/'active.json',{'idle':True})

def train(spec,until):
    job=Path(read(spec)['job_dir']);progress=job/'progress.json'
    if progress.exists() and read(progress)['step']>=until:return
    child(['train','--spec',spec,'--until',until],job/'train.log')

def evaluate(spec,split,out,checkpoint=None,unlock=None):
    if Path(out).exists():return read(out)
    args=['evaluate','--spec',spec,'--checkpoint',checkpoint or str(Path(read(spec)['job_dir'])/'model.pt'),'--split',split,'--out',out]
    if unlock:args+=['--unlock',unlock]
    child(args,Path(out).with_suffix('.log'))
    return read(out)


def teacher(task,cfg):
    dataset=e.make_data(task,cfg['input_length']);root=ROOT/dataset.name
    for attempt,a in enumerate(cfg['teacher_attempts']):
        job=root/f'teacher_{attempt}';spec=frozen(job/'spec.json',{
            'kind':'teacher','seed':cfg['teacher_seed_base']+attempt,'sampling_seed':cfg['teacher_sampling_base']+attempt,
            'model':{'vocab_size':8,'d_model':a['embedding'],'n_head':4,'n_layer':a['layers'],'max_len':32,'dropout':0.},
            'objective':'ce','lr':a['lr'],'batch':cfg['teacher_batch'],'budgets':[a['steps']],
            'checkpoints':sorted(set([a['steps']//3,2*a['steps']//3,a['steps']])),
            'dataset':str(dataset),'job_dir':str(job)})
        train(spec,a['steps']);val=evaluate(spec,'validation',job/'validation.json')
        if val['metrics']['accuracy']<.95:
            event('Teacher failed validation; locked qualification and final test remain unopened',task=task,attempt=attempt,metrics=val['metrics']);continue
        qual=evaluate(spec,f'qualification_{attempt}',job/'qualification.json')
        passed=qual['metrics']['accuracy']>=.95
        receipt=frozen(job/'qualification_receipt.json',{'passed':passed,'threshold':.95,'accuracy':qual['metrics']['accuracy'],
            'task':task,'attempt':attempt,'inputs':hashes([spec,job/'model.pt',job/'validation.json',job/'qualification.json',dataset/'manifest.json',dataset/'train.json',CONFIG,'scripts/g2c_engine.py'])})
        event('Teacher qualification passed' if passed else 'Teacher qualification failed',task=task,attempt=attempt,accuracy=qual['metrics']['accuracy'])
        if passed:return {'spec':spec,'checkpoint':str(job/'model.pt'),'receipt':receipt,'dataset':str(dataset),'task':task}
    return None


def baselines(t,cfg):
    root=Path(t['spec']).parent.parent;out=root/'baseline_validation.json'
    if out.exists():return
    ts=read(t['spec']);n=sum(p.numel() for p in e.build(ts).parameters())
    hidden=min(range(8,512),key=lambda h:abs(3*h*h+62*h+136-n))
    job=root/'gru';spec=frozen(job/'spec.json',{**ts,'kind':'gru','job_dir':str(job),'model':{'hidden':hidden,'embed':16}})
    train(spec,ts['budgets'][-1]);val=evaluate(spec,'validation',job/'validation.json')
    save_json(out,{'teacher_parameters':n,'gru_parameters':sum(p.numel() for p in e.build(read(spec)).parameters()),
        'teacher':read(Path(t['spec']).parent/'validation.json')['metrics'],'gru':val['metrics'],'ngram':e.baseline(t['dataset']),
        'qualification_rule':'Teacher must exceed 95% exact; baseline superiority is not required'})
    event('Teacher and simple baselines measured',task=t['task'],metrics=read(out))


def graph_paths(seed):
    real='data/processed/malecns.npz';rewired=f'data/processed/controls/rewired_{777+seed}.npz'
    expected=read(f'results/malecns_v1/controls/rewired_{777+seed}.json')
    assert sha256(real)==expected['inputs'][real] and sha256(rewired)==expected['graph_sha256']
    return real,rewired


def specs(t,cfg,seed,label='full',graphs=None):
    real,rewired=graphs or graph_paths(seed);cohort=ROOT/Path(t['dataset']).name/label/f'seed_{seed}'
    paths=[]
    for condition in cfg['student_conditions']:
        job=cohort/condition;spec=frozen(job/'spec.json',{'kind':'graph','seed':seed,'sampling_seed':cfg['student_sampling_base']+seed,
            'model':cfg['model'],'objective':'ce' if condition=='real_ce' else 'kd',
            'alpha':cfg['alpha'],'temperature':cfg['temperature'],'lr':cfg['student_lr'],'batch':cfg['student_batch'],
            'budgets':cfg['student_budgets'],'checkpoints':cfg['learning_curve_steps'],'curve_cases':cfg['learning_curve_validation_cases'],
            'dataset':t['dataset'],'job_dir':str(job),'graph':real if condition!='rewired_kd' else rewired,
            'qualification_receipt':t['receipt'],'teacher_spec':t['spec'],'teacher_checkpoint':t['checkpoint'],
            'condition':condition,'substrate':label,'initialization':'same model seed, same input/output population seed; graph-dependent edges and degree normalization',
            'evidence_tier':'exploratory'})
        paths.append(spec)
    freeze=cohort/'frozen_choices.json'
    frozen(freeze,{'seed':seed,'label':label,'specs':paths,'inputs':hashes([*paths,t['receipt'],CONFIG,'scripts/g2c_engine.py','scripts/run_g2c_overnight.py'])})
    return paths,str(freeze)


def cohort(t,cfg,seed,until=512,label='full',graphs=None,split='test_main'):
    paths,freeze=specs(t,cfg,seed,label,graphs);root=Path(freeze).parent;summary=root/f'comparison_{until}.json'
    if summary.exists():return read(summary)
    start=time.monotonic();event('Starting matched cohort',label=label,seed=seed,updates=until,split=split)
    # All conditions are fixed together before any condition starts.
    for p in paths:train(p,until)
    checkpoints={}
    for p in paths:
        progress=read(Path(read(p)['job_dir'])/'progress.json');assert progress['step']==until
        saved=next(a for a in progress['archives'] if a['step']==until)
        assert sha256(saved['path'])==saved['sha256'];checkpoints[p]=saved['path']
    unlock=frozen(root/f'unlock_{until}.json',{'split':split,'choices_frozen':True,'matched_updates':until,
        'inputs':hashes([freeze,*paths,*checkpoints.values(),t['checkpoint'],t['spec'],Path(t['dataset'])/'manifest.json'])})
    results={}
    for p in paths:
        condition=read(p)['condition'];out=root/f'{condition}_{until}_{split}.json'
        results[condition]=evaluate(p,split,out,checkpoint=checkpoints[p],unlock=unlock)
    teacher_out=root/f'teacher_{until}_{split}.json'
    teach=evaluate(t['spec'],split,teacher_out,unlock=unlock)
    # Paired instance bootstrap is descriptive. Independent seed replication remains essential.
    rng=np.random.default_rng(87000+seed);n=len(results['real_kd']['rows']);index=rng.integers(n,size=(10000,n))
    order=[r['id'] for r in results['real_kd']['rows']]
    for r in results.values():assert [x['id'] for x in r['rows']]==order
    kd=np.array([r['exact'] for r in results['real_kd']['rows']]);ce=np.array([r['exact'] for r in results['real_ce']['rows']]);rw=np.array([r['exact'] for r in results['rewired_kd']['rows']])
    result={'task':t['task'],'substrate':label,'seed':seed,'updates':until,'split':split,
        'metrics':{k:v['metrics'] for k,v in results.items()},'teacher':teach['metrics'],
        'A_transfer':float((kd-ce).mean()),'A_topology':float((kd-rw).mean()),
        'paired_case_bootstrap_95_transfer':np.quantile((kd-ce)[index].mean(1),[.025,.975]).tolist(),
        'paired_case_bootstrap_95_topology':np.quantile((kd-rw)[index].mean(1),[.025,.975]).tolist(),
        'inference_limit':'Descriptive paired-case intervals; single/adaptively selected seeds do not establish a robust advantage',
        'elapsed_seconds':time.monotonic()-start,'freeze':freeze,'unlock':unlock,'checkpoint_paths':checkpoints}
    save_json(summary,result);event('Matched cohort completed',summary=str(summary),A_transfer=result['A_transfer'],A_topology=result['A_topology'])
    preserve(f'G2c: preserve {label} seed {seed} at {until} updates')
    return result


def capacity_graph(n,fraction,seed=0):
    from scipy import sparse
    from src.graph_controls import make_control
    label=f'induced_{n}_edges_{str(fraction).replace(".","p")}';root=Path('data/processed/g2c_capacity')/label
    root.mkdir(parents=True,exist_ok=True);bio=root/'real.npz';rw=root/'rewired.npz';meta=ROOT/'capacity_graphs'/f'{label}.json'
    if meta.exists():
        m=read(meta);assert sha256(bio)==m['real_sha256'] and sha256(rw)==m['rewired_sha256'];return label,(str(bio),str(rw))
    source=sparse.load_npz('data/processed/malecns.npz').tocsr();rng=np.random.default_rng(88000)
    selected=np.sort(rng.permutation(source.shape[0])[:n]);sub=source[selected][:,selected].tocoo();del source
    edge_rng=np.random.default_rng(89000);keep=edge_rng.random(sub.nnz)<fraction
    s=sub.row[keep].astype(np.int64);d=sub.col[keep].astype(np.int64);del sub
    sparse.save_npz(bio,sparse.csr_matrix((np.ones(len(s),np.float32),(s,d)),shape=(n,n)))
    rs,rd,info=make_control(n,s,d,'rewired',seed=90000+seed,swap_factor=10)
    assert np.array_equal(np.bincount(s,minlength=n),np.bincount(rs,minlength=n))
    assert np.array_equal(np.bincount(d,minlength=n),np.bincount(rd,minlength=n))
    sparse.save_npz(rw,sparse.csr_matrix((np.ones(len(rs),np.float32),(rs,rd)),shape=(n,n)))
    save_json(meta,{'label':label,'n':n,'edges':len(s),'edge_fraction':fraction,'selected_original_indices':selected.tolist(),
        'source_sha256':sha256('data/processed/malecns.npz'),'real_sha256':sha256(bio),'rewired_sha256':sha256(rw),
        'control':info,'domain':'induced MaleCNS subgraph; not the full graph','selection_seed':88000,'edge_seed':89000})
    for p in (bio,rw):
        dest=EXTERNAL/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha256(p)==sha256(dest)
    return label,(str(bio),str(rw))


def remaining(cfg):return (dt.datetime.fromisoformat(cfg['deadline_utc'])-dt.datetime.now(dt.timezone.utc)).total_seconds()


def run():
    ROOT.mkdir(parents=True,exist_ok=True);cfg=read(CONFIG)
    lock=ROOT/'controller.lock'
    if lock.exists():
        old=read(lock)
        try:os.kill(old['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'Controller already live: {old}')
    save_json(lock,{'pid':os.getpid(),'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())])
    try:
        event('G2c overnight controller started',deadline=cfg['deadline_utc'])
        frozen(ROOT/'protocol_source.json',{'inputs':hashes([CONFIG,'scripts/g2c_engine.py','scripts/run_g2c_overnight.py'])})
        t=None
        for task in cfg['tasks_in_order']:
            t=teacher(task,cfg)
            if t:break
        if not t:
            event('No teacher qualified; no connectome training authorized by the gate');return
        baselines(t,cfg)
        paths,freeze=specs(t,cfg,0)
        event('Student architecture and matched opportunities frozen',freeze=freeze)
        preserve('G2c: lock qualified teacher and matched full-connectome experiment')
        initial=cohort(t,cfg,0)
        cost=max(initial['elapsed_seconds'],600)
        if initial['A_transfer']>0:
            for seed in cfg['student_seeds'][1:]:
                if remaining(cfg)<cost*1.15+180:break
                cohort(t,cfg,seed)
        else:
            # Diagnose learnability before any expensive matched extension; no tuning on test.
            for p in paths:evaluate(p,'training_probe',Path(read(p)['job_dir'])/'training_probe_512.json')
            label,graphs=capacity_graph(4096,1.)
            cohort(t,cfg,0,label=label,graphs=graphs,split='test_capacity')
            if remaining(cfg)>cost*1.2+180:cohort(t,cfg,0,until=1024,split='test_extension')
        # Complete the preregistered small substrate grid with matched conditions.
        for n in cfg['capacity_grid']['neurons']:
            for frac in cfg['capacity_grid']['edge_fractions']:
                if remaining(cfg)<600:break
                label,graphs=capacity_graph(n,frac)
                cohort(t,cfg,0,label=label,graphs=graphs,split='test_capacity')
        # Independently qualify additional functions, preserving all failures.
        for task in cfg['tasks_in_order']:
            if task==t['task'] or remaining(cfg)<180:continue
            next_teacher=teacher(task,cfg)
            if next_teacher:
                baselines(next_teacher,cfg)
                if remaining(cfg)>cost*1.2+180:cohort(next_teacher,cfg,0)
                elif remaining(cfg)>600:
                    label,graphs=capacity_graph(4096,1.)
                    cohort(next_teacher,cfg,0,label=label,graphs=graphs,split='test_capacity')
        # Spend remaining feasible full-cohort time on independent replication, not architecture tuning.
        for seed in cfg['student_seeds'][1:]:
            if remaining(cfg)>cost*1.15+180:cohort(t,cfg,seed)
        event('All currently feasible planned comparisons completed',remaining_seconds=remaining(cfg))
        save_json(ROOT/'completion.json',{'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'reason':'Completed feasible frozen cohorts; no incomplete comparison launched','remaining_seconds':remaining(cfg)})
    except BaseException as exc:
        save_json(ROOT/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        event('Controller stopped on error; no test or qualification rule relaxed',error=repr(exc))
        raise
    finally:
        report()
        try:preserve('G2c: update overnight evidence and report')
        finally:awake.terminate();lock.unlink(missing_ok=True)

if __name__=='__main__':run()
