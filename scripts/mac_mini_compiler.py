"""Independent, branch-isolated five-method worker for the second Mac."""
import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time

from src.provenance import save_json, sha256

ROOT=Path('results/macmini_compiler_v2')
CFG=Path('configs/mac_mini_compiler_v2.json')
DEPENDENCIES=Path('configs/mac_mini_inputs.json')

def read(p): return json.loads(Path(p).read_text())
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def verify(inputs):
    for p,h in inputs.items():
        if not Path(p).is_file() or sha256(p)!=h: raise RuntimeError(f'Missing or changed input: {p}')
def freeze(p,d):
    if Path(p).exists():
        if read(p)!=d: raise RuntimeError(f'Frozen record changed: {p}')
    else: save_json(p,d)
    return str(p)

def preflight(benchmark_only=False):
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    if not branch.startswith(read(CFG)['branch_prefix']): raise RuntimeError('Switch to the assigned Mac mini branch first')
    verify(read(DEPENDENCIES)['inputs'])
    if platform.python_version_tuple()[:2]!=('3','12'): raise RuntimeError('This cohort requires Python 3.12')
    packages={}
    for line in Path('requirements-mac-mini.txt').read_text().splitlines():
        name,version=line.split('==');packages[name]=importlib.metadata.version(name)
        if packages[name]!=version: raise RuntimeError(f'Version mismatch: {name}')
    reserve_gib=4 if benchmark_only else 16
    if not (ROOT/'frozen_plan.json').exists() and shutil.disk_usage('.').free < reserve_gib*1024**3:
        raise RuntimeError(f'At least {reserve_gib} GiB free disk required before starting')
    return dict(utc=now(),branch=branch,platform=platform.platform(),python=platform.python_version(),
        packages=packages,hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string','hw.memsize'],text=True).strip(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())

def publish(message):
    subprocess.run(['git','add','--',str(ROOT)],check=True)
    if Path('MAC_MINI_REPORT.md').exists():subprocess.run(['git','add','--','MAC_MINI_REPORT.md'],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:
        subprocess.run(['git','commit','-m',message],check=True)
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    subprocess.run(['git','push','-u','origin',branch],check=True,timeout=120)

def make_return(checkpoints):
    dest=Path('artifacts/mac_mini_return');dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'flygpt-mac-mini-final-checkpoints.tar.gz'
    records={name:dict(member=f'checkpoints/{name}.pt',sha256=sha256(p),bytes=Path(p).stat().st_size) for name,p in checkpoints.items()}
    if archive.exists(): raise RuntimeError('Return archive already exists without completion; verify before restarting packaging')
    temporary=archive.with_suffix('.partial')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tf:
        for name,p in checkpoints.items():tf.add(Path(p).resolve(),arcname=records[name]['member'],recursive=False)
    temporary.replace(archive)
    digest=sha256(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{digest}  {archive.name}\n')
    save_json(ROOT/'return_manifest.json',dict(archive_name=archive.name,archive_sha256=digest,checkpoints=records))

def benchmark():
    """Disposable timing run; never a confirmatory checkpoint or model selection."""
    import fcntl
    from scripts import run_compiler_v1 as v
    hardware=preflight(benchmark_only=True);ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v.ROOT=ROOT;v.refresh=lambda:None
    awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())])
    try:
        job=ROOT/'hardware_benchmark';base=read(read(CFG)['reference_spec'])
        base.update(seed=90000,sampling_seed=165000,batch=1,objective='ce',budgets=[64],checkpoints=[64],
            total_updates=64,job_dir=str(job),condition='hardware_benchmark',
            evidence_tier='Disposable hardware timing; not part of any accuracy comparison')
        base.pop('training_targets',None);base.pop('training_targets_sha256',None)
        spec=freeze(job/'spec.json',base)
        start=time.perf_counter();v.train(spec,'scripts.g2c_engine');elapsed=time.perf_counter()-start
        progress=read(job/'progress.json')
        save_json(ROOT/'benchmark.json',dict(hardware=hardware,steps=64,batch=1,
            full_graph_sha256=sha256(base['graph']),wall_seconds=progress['wall_seconds'],
            seconds_through_first_validation=progress['evaluations'][0]['wall_seconds'],
            invocation_wall_seconds=elapsed,includes_validation_and_checkpoint_overhead=True,
            next_action='Report timing to primary agent for complete-pair allocation; do not independently start confirmation'))
        publish('Mac mini: preserve full MaleCNS timing benchmark for compute allocation')
    finally:awake.terminate()

def run():
    import fcntl
    from scripts import run_compiler_v1 as v
    hardware=preflight();ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())])
    v.ROOT=ROOT;v.refresh=lambda:None
    try:
        if (ROOT/'completion.json').exists(): return
        save_json(ROOT/'sessions'/f'{os.getpid()}.json',hardware)
        cfg=read(CFG);base=read(cfg['reference_spec']);paths={}
        objectives={'CE':'ce','hard':'ce','soft_KD':'kd','hidden':'hidden','relational':'relational'}
        for name in cfg['methods']:
            spec={**base,'seed':cfg['seed'],'sampling_seed':cfg['sampling_seed'],'job_dir':str(ROOT/'methods'/name),
                'condition':name,'objective':objectives[name],'evidence_tier':cfg['evidence']}
            if name=='CE':
                spec.pop('training_targets',None);spec.pop('training_targets_sha256',None)
            assert spec['batch']==cfg['batch'] and spec['budgets']==[cfg['updates']]
            paths[name]=freeze(ROOT/'methods'/name/'spec.json',spec)
        inputs={**read(DEPENDENCIES)['inputs'],**{str(p):sha256(p) for p in [CFG,DEPENDENCIES,Path(__file__),*paths.values()]}}
        plan=freeze(ROOT/'frozen_plan.json',dict(inputs=inputs,methods=paths,primary_split=cfg['primary_split']))
        verify(inputs);publish('Mac mini: freeze matched representation screen before training')
        checkpoints={}
        for name,p in paths.items():
            verify(inputs);save_json(ROOT/'status.json',dict(utc=now(),stage='training',method=name,pid=os.getpid()))
            checkpoints[name]=v.train(p,'scripts.compiler_engine')
            save_json(ROOT/'completed_training.json',checkpoints)
            publish(f'Mac mini: preserve {name} checkpoint trajectory')
        verify(inputs)
        receipt=freeze(ROOT/'unlock.json',dict(split=cfg['primary_split'],choices_frozen=True,
            inputs={str(p):sha256(p) for p in [plan,*checkpoints.values()]},all_five_conditions_complete=True,
            evidence='Original previously inspected compiler test, not confirmation test'))
        results={}
        for name,p in paths.items():
            save_json(ROOT/'status.json',dict(utc=now(),stage='evaluation',method=name,pid=os.getpid()))
            v.evaluate(p,checkpoints[name],'validation',ROOT/f'{name}_validation.json')
            results[name]=v.evaluate(p,checkpoints[name],cfg['primary_split'],ROOT/f'{name}_test.json',receipt)
            publish(f'Mac mini: preserve {name} final evaluation and ablation')
        result=v.compare(results);hard=result['metrics']['hard']['accuracy']
        result['gap_to_hard']={n:r['accuracy']-hard for n,r in result['metrics'].items()}
        result['hardware_scope']='Same-machine matched batch-two cohort; do not pool with topology confirmation'
        result['hard_ce_state_equivalence']=v.checkpoint_equal(checkpoints['CE'],checkpoints['hard'])
        verify(inputs);verify(read(receipt)['inputs']);save_json(ROOT/'comparison.json',result)
        lines=['# Mac mini compiler screen','',cfg['evidence'],'',
            '| Method | Exact | Gap to hard (pp) | Response CE | Teacher KL | Zero-edge exact |',
            '|---|---:|---:|---:|---:|---:|']
        for n,r in result['metrics'].items():
            lines.append(f"| {n} | {100*r['accuracy']:.2f}% | {100*(r['accuracy']-hard):+.2f} | {r['response_ce']:.5f} | {r['teacher_kl']:.5f} | {100*r['zero_edge_exact']:.2f}% |")
        lines+=['','One exploratory seed. Teacher-hard labels equal original labels; parity alone is not evidence of stronger transfer. No method was selected or tuned on these final results.',
            'All intermediate checkpoints remain archived on the mini. The final checkpoint return package must be transferred and verified before declaring an off-machine backup.']
        Path('MAC_MINI_REPORT.md').write_text('\n'.join(lines)+'\n')
        if not (ROOT/'return_manifest.json').exists():make_return(checkpoints)
        save_json(ROOT/'completion.json',dict(utc=now(),artifact_audit_passed=True,methods=list(paths)))
        save_json(ROOT/'status.json',dict(utc=now(),stage='complete'))
        publish('Mac mini: complete matched compiler screen and return manifest')
    except BaseException as exc:
        import traceback
        save_json(ROOT/'failure.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()));raise
    finally:awake.terminate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--benchmark',action='store_true');a=p.parse_args()
    if a.preflight:print(json.dumps(preflight(benchmark_only=a.benchmark),indent=2))
    elif a.benchmark:benchmark()
    else:run()
