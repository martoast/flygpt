"""Finite, automatically advancing compiler study; full-graph jobs run sequentially."""
import datetime as dt
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch
import numpy as np
from scripts import g2c_engine as e
from scripts.run_g2c_overnight import frozen,hashes,graph_paths
from src.provenance import save_json,sha256
from scripts.compiler_storage import flush_pending

ROOT=Path('results/compiler_v1');CFG=Path('configs/compiler_v1.json')
PY=str(Path('.venv/bin/python').absolute());DISK=Path('/Volumes/Seagate')
BACKUP=DISK/'FlyGPT Backups/G2c-overnight'

def read(p):return json.loads(Path(p).read_text())
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def refresh():
    result=subprocess.run([PY,'-m','scripts.compiler_report'],capture_output=True,text=True)
    if result.returncode:print('Report refresh failed; training state is unaffected:',result.stderr[-2000:],flush=True)
def event(message,**extra):
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'message':message,**extra})+'\n')
    print(message,extra,flush=True);refresh()

def preserve(message):
    flush_pending()
    save_json(ROOT/'backup_status.json',{'utc':now(),'external_connected':DISK.is_mount(),'offline_policy':'Preserve every checkpoint locally with 3 GiB free-space reserve; migrate and verify on reconnect'})
    paths=[ROOT,CFG,Path('scripts/compiler_engine.py'),Path('scripts/compiler_targets.py'),Path('scripts/run_compiler_v1.py'),Path('scripts/compiler_storage.py'),Path('scripts/compiler_report.py'),Path('tests/test_compiler.py'),Path('COMPILER_BENCHMARK.md'),Path('COMPILER_REPORT.md')]
    if DISK.is_mount():
        for root in paths:
            for p in (root.rglob('*') if root.is_dir() else [root]):
                if p.is_file() and p.suffix in ('.py','.json','.jsonl','.md'):
                    dest=BACKUP/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    subprocess.run(['git','add','--',*map(str,paths)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:
        subprocess.run(['git','commit','-m',message],check=True)
    push=subprocess.run(['git','push'],capture_output=True,text=True,timeout=120)
    if push.returncode:event('Push failed; local commit and USB artifacts retained',error=push.stderr[-2000:])

def child(module,args,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    if args[0]=='train':args=['--engine',module,*args];module='scripts.compiler_storage'
    with log.open('a') as stream:
        p=subprocess.Popen([PY,'-m',module,*map(str,args)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'4','NUMBA_NUM_THREADS':'4'})
        save_json(ROOT/'active.json',{'pid':p.pid,'module':module,'args':list(map(str,args)),'log':str(log),'started_utc':now()})
        while p.poll() is None:refresh();time.sleep(20)
        if p.returncode:raise RuntimeError(f'Child failed with {p.returncode}: {log}')
    save_json(ROOT/'active.json',{'idle':True})

def spec(cfg,condition,seed,batch=1,objective='ce',temp=2.,hard=False):
    base=read(cfg['reference_spec']);real,rw=graph_paths_cached(seed)
    job=ROOT/('paired_ce' if condition in ('real_ce','rewired_ce') else 'methods')/f'seed_{seed}'/condition
    base.update(seed=seed,sampling_seed=75000+seed,job_dir=str(job),condition=condition,
        graph=rw if condition=='rewired_ce' else real,objective=objective,batch=batch,
        budgets=[cfg['updates']//batch],checkpoints=[x//batch for x in (64,128,256,512,768,1024)],
        temperature=temp,total_updates=cfg['updates']//batch,alignment_weight=1.,
        evidence_tier='Exploratory frozen tournament on previously inspected test',substrate='full')
    if hard:
        p=ROOT/'teacher_training_targets.json';base.update(training_targets=str(p),training_targets_sha256=sha256(p))
    return frozen(job/'spec.json',base)

GRAPHS={}
def graph_paths_cached(seed):
    if seed not in GRAPHS:GRAPHS[seed]=graph_paths(seed)
    return GRAPHS[seed]

def plan(cfg):
    paired={};hard={};screen={};relational={}
    for seed in cfg['seeds']:
        paired[str(seed)]={c:(cfg['reference_spec'] if c=='real_ce' and seed==0 else spec(cfg,c,seed)) for c in ('real_ce','rewired_ce')}
        hard[str(seed)]=spec(cfg,'C1_hard',seed,hard=True)
        screen[str(seed)]={f'C2_T{t:g}':spec(cfg,f'C2_T{t:g}',seed,objective='kd',temp=t,hard=True) for t in cfg['tournament']['temperatures']}
        screen[str(seed)]['C3_curriculum']=spec(cfg,'C3_curriculum',seed,objective='curriculum',hard=True)
        screen[str(seed)]['C4_hidden']=spec(cfg,'C4_hidden',seed,objective='hidden',hard=True)
        relational[str(seed)]={
            'CE_batch2':spec(cfg,'CE_batch2',seed,batch=2),
            'C1_batch2':spec(cfg,'C1_batch2',seed,batch=2,hard=True),
            'C0_batch2':spec(cfg,'C0_batch2',seed,batch=2,objective='kd',hard=True),
            'C5_relational':spec(cfg,'C5_relational',seed,batch=2,objective='relational',hard=True)}
    paths=[p for group in (paired,screen,relational) for cohort in group.values() for p in cohort.values()]+list(hard.values())
    source_paths=[CFG,'scripts/compiler_engine.py','scripts/compiler_targets.py','scripts/run_compiler_v1.py','scripts/g2c_engine.py',*Path('src').glob('*.py'),ROOT/'teacher_training_targets.json',ROOT/'teacher_targets_provenance.json',cfg['teacher_checkpoint'],cfg['qualification_receipt'],*Path(cfg['dataset']).glob('*.json'),*paths]
    source_paths+=['scripts/compiler_storage.py',*sorted({p for pair in GRAPHS.values() for p in pair})]
    return frozen(ROOT/'frozen_plan.json',{'paired_ce':paired,'hard':hard,'screen':screen,'relational':relational,
        'created_before_new_training':True,'inputs':hashes(source_paths),'historical_seed_zero':'Reused transparently, not an independent replication'})

def validate_plan(p):
    for path,digest in read(p)['inputs'].items():
        if sha256(path)!=digest:raise RuntimeError(f'Frozen input changed: {path}')

def train(path,engine):
    s=read(path);job=Path(s['job_dir']);budget=s['budgets'][-1];progress=job/'progress.json'
    if not progress.exists() or read(progress)['step']<budget:
        child(engine,['train','--spec',path,'--until',budget],job/'train.log')
    r=read(progress);assert r['step']==budget
    archive=next(x for x in r['archives'] if x['step']==budget)
    if not Path(archive['path']).exists():
        local=job/'model.pt'
        if local.exists() and sha256(local)==archive['sha256']==r['checkpoint_sha256']:
            return str(local)
        raise RuntimeError('Neither archived checkpoint nor verified original local copy is accessible')
    assert sha256(archive['path'])==archive['sha256']==r['checkpoint_sha256']
    # Free only the replaceable local copy after verified off-machine preservation.
    local=job/'model.pt'
    if str(job).startswith(str(ROOT)) and not local.is_symlink():
        assert sha256(local)==archive['sha256']
        link=job/'model.pt.link';link.symlink_to(archive['path']);link.replace(local)
    return archive['path']

def evaluate(path,checkpoint,split,out,unlock=None):
    if Path(out).exists():return read(out)
    args=['evaluate','--spec',path,'--checkpoint',checkpoint,'--split',split,'--out',out]
    if unlock:args+=['--unlock',unlock]
    child('scripts.g2c_engine',args,Path(out).with_suffix('.log'));return read(out)

def unlock(paths,checkpoints,tag,cfg):
    for path in paths:
        p=read(Path(read(path)['job_dir'])/'progress.json');assert p['step']==read(path)['budgets'][-1]
    return frozen(ROOT/'unlocks'/f'{tag}.json',{'split':cfg['primary_split'],'choices_frozen':True,
        'warning':'Previously inspected fixed benchmark; no claim of fresh test blindness',
        'inputs':hashes([ROOT/'frozen_plan.json',*paths,*checkpoints.values()])})

def compare(outputs):
    first=next(iter(outputs.values()));ids=[r['id'] for r in first['rows']]
    for r in outputs.values():assert [v['id'] for v in r['rows']]==ids
    return {'metrics':{k:r['metrics'] for k,r in outputs.items()},'cases':len(ids)}

def checkpoint_equal(a,b):
    x=torch.load(a,weights_only=True,map_location='cpu');y=torch.load(b,weights_only=True,map_location='cpu')
    def equal(u,v):
        if torch.is_tensor(u):return torch.is_tensor(v) and torch.equal(u,v)
        if isinstance(u,dict):return u.keys()==v.keys() and all(equal(u[k],v[k]) for k in u)
        if isinstance(u,(list,tuple)):return len(u)==len(v) and all(equal(i,j) for i,j in zip(u,v))
        return u==v
    result={k:equal(x[k],y[k]) for k in ('model','optimizer','rng')};del x,y;gc.collect()
    return {**result,'all_equal':all(result.values()),'inputs':hashes([a,b])}

def paired_ce(seed,plans,cfg):
    out=ROOT/'paired_ce'/f'seed_{seed}'/'comparison.json'
    if out.exists():return read(out)
    paths=plans['paired_ce'][str(seed)];event('Paired CE replication started',seed=seed)
    ck={k:train(p,'scripts.g2c_engine') for k,p in paths.items()}
    receipt=unlock(list(paths.values()),ck,f'ce_seed_{seed}',cfg)
    scores={k:evaluate(p,ck[k],cfg['primary_split'],out.parent/f'{k}_test.json',receipt) for k,p in paths.items()}
    result=compare(scores);result.update(seed=seed,updates=1024,A_topology_CE=result['metrics']['real_ce']['accuracy']-result['metrics']['rewired_ce']['accuracy'])
    save_json(out,result);event('Paired CE replication completed',seed=seed,metrics=result['metrics'])
    preserve(f'Compiler benchmark: preserve paired CE seed {seed}')
    return result

def hard_run(seed,plans,cfg):
    job=ROOT/'hard_comparisons';out=job/f'seed_{seed}.json'
    if out.exists():return read(out)
    p=plans['hard'][str(seed)];cp=train(p,'scripts.compiler_engine')
    ce=plans['paired_ce'][str(seed)]['real_ce'];ceck=train(ce,'scripts.g2c_engine')
    receipt=unlock([p,ce],{'hard':cp,'ce':ceck},f'hard_seed_{seed}',cfg)
    result=evaluate(p,cp,cfg['primary_split'],job/f'seed_{seed}_test.json',receipt)
    baseline=read(ROOT/'paired_ce'/f'seed_{seed}'/'real_ce_test.json')
    summary={'seed':seed,'metrics':result['metrics'],'CE_metrics':baseline['metrics'],
        'A_transfer':result['metrics']['accuracy']-baseline['metrics']['accuracy'],'state_equivalence':checkpoint_equal(cp,ceck)}
    save_json(out,summary);event('Teacher-hard transfer completed',seed=seed,summary=summary)
    preserve(f'Compiler benchmark: preserve teacher-hard seed {seed}')
    return summary

def screen_batch_one(plans,cfg):
    paths=plans['screen']['0'];root=ROOT/'screening';root.mkdir(exist_ok=True)
    scores={};checkpoints={}
    for name,p in paths.items():
        event('Compiler objective screening',method=name,seed=0)
        checkpoints[name]=train(p,'scripts.compiler_engine')
        scores[name]=evaluate(p,checkpoints[name],'validation',root/f'{name}_validation.json')
        preserve(f'Compiler benchmark: preserve {name} validation before test')
    winner=min(scores,key=lambda k:(-scores[k]['metrics']['accuracy'],scores[k]['metrics']['response_ce'],k))
    frozen(root/'selection.json',{'method':winner,'criterion':cfg['tournament']['selection'],
        'inputs':hashes([root/f'{name}_validation.json' for name in paths]),'selected_before_candidate_test_evaluation':True})
    receipt=unlock(list(paths.values()),checkpoints,'screening_seed_0',cfg)
    for name,p in paths.items():evaluate(p,checkpoints[name],cfg['primary_split'],root/f'{name}_test.json',receipt)
    event('Compiler screen completed; validation-selected method frozen',method=winner)
    preserve('Compiler benchmark: preserve complete objective screen and selection')
    return winner

def selected_replication(method,seed,plans,cfg):
    p=plans['screen'][str(seed)][method];cp=train(p,'scripts.compiler_engine')
    receipt=unlock([p],{method:cp},f'selected_{method}_seed_{seed}',cfg)
    result=evaluate(p,cp,cfg['primary_split'],ROOT/'selected_replications'/f'{method}_seed_{seed}.json',receipt)
    event('Selected compiler replication completed',method=method,seed=seed,metrics=result['metrics'])
    preserve(f'Compiler benchmark: preserve selected {method} seed {seed}')

def relational(seed,plans,cfg):
    root=ROOT/'relational'/f'seed_{seed}';out=root/'comparison.json'
    if out.exists():return read(out)
    paths=plans['relational'][str(seed)];ck={};vals={}
    for name,p in paths.items():
        ck[name]=train(p,'scripts.compiler_engine');vals[name]=evaluate(p,ck[name],'validation',root/f'{name}_validation.json')
    replicate=vals['C5_relational']['metrics']['accuracy']>vals['C1_batch2']['metrics']['accuracy']
    decision=frozen(root/'validation_decision.json',{'replicate':replicate,'inputs':hashes([root/f'{name}_validation.json' for name in paths])})
    receipt=unlock(list(paths.values()),ck,f'relational_seed_{seed}',cfg)
    scores={k:evaluate(p,ck[k],cfg['primary_split'],root/f'{k}_test.json',receipt) for k,p in paths.items()}
    result=compare(scores);result.update(seed=seed,validation_replication_decision=replicate,decision=decision,updates=512,batch=2)
    save_json(out,result);event('Relational matched cohort completed',seed=seed,metrics=result['metrics'])
    preserve(f'Compiler benchmark: preserve relational cohort seed {seed}')
    return result

def run():
    ROOT.mkdir(exist_ok=True);cfg=read(CFG);lock=ROOT/'controller.lock'
    if lock.exists():
        try:os.kill(read(lock)['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Compiler controller already active')
    save_json(lock,{'pid':os.getpid(),'started_utc':now()});awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())])
    try:
        e.qualified(cfg['qualification_receipt']);plan_path=plan(cfg);validate_plan(plan_path);plans=read(plan_path)
        event('Compiler study started; all conditions frozen before new training')
        preserve('Compiler benchmark: freeze paired replication and objective tournament')
        for seed in cfg['seeds']:paired_ce(seed,plans,cfg)
        hard=hard_run(0,plans,cfg)
        exact=read(ROOT/'teacher_label_audit.json')['all_identical'] and hard['state_equivalence']['all_equal']
        if not exact:
            for seed in cfg['seeds'][1:]:hard_run(seed,plans,cfg)
        else:event('Hard-label objective and complete training state equal CE; redundant deterministic reruns omitted, not counted as measured replications')
        winner=screen_batch_one(plans,cfg)
        relational_zero=relational(0,plans,cfg)
        for seed in cfg['seeds'][1:]:selected_replication(winner,seed,plans,cfg)
        if relational_zero['validation_replication_decision']:
            for seed in cfg['seeds'][1:]:relational(seed,plans,cfg)
        save_json(ROOT/'completion.json',{'utc':now(),'status':'Frozen compiler study completed; harder tasks remain gated'})
        event('Compiler study completed')
    except BaseException as exc:
        import traceback
        save_json(ROOT/'failure.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});event('Stopped on execution error',error=repr(exc));raise
    finally:
        try:refresh();preserve('Compiler benchmark: update report and preserved evidence')
        finally:awake.terminate();lock.unlink(missing_ok=True)

if __name__=='__main__':run()
