"""Additive hardware amendment; never changes frozen learning inputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from src.provenance import sha256, save_json

ROOT = Path('results/compiler_efficiency')
REMOTE = '/home/alex/flygpt_efficiency'
DOC = Path('COMPILER_EFFICIENCY_HARDWARE_AMENDMENT.md')
SSH = ['ssh', '-o', 'UserKnownHostsFile='+str(Path.home()/'.ssh/flygpt_omarchy_known_hosts'),
       '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       'alex@omarchy.tail61505e.ts.net']


def owner(seed):
    if seed not in range(500, 510):
        raise ValueError('Seed outside frozen cohort')
    return 'macbook' if seed >= 505 else 'omarchy'


def read(path):
    return json.loads(Path(path).read_text())


def send(source, relative):
    """Atomic, hash-verified import. Retry connectivity without deleting source."""
    source = Path(source); relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe relative path')
    digest = sha256(source)
    # Immutable checkpoint conflicts fail; mutable progress is published atomically.
    code = f'''import sys,hashlib,os
from pathlib import Path
p=Path({str(Path(REMOTE)/relative)!r});p.parent.mkdir(parents=True,exist_ok=True)
t=p.with_name(p.name+'.m1-incoming')
with t.open('wb') as f:
 while b:=sys.stdin.buffer.read(8388608):f.write(b)
 f.flush();os.fsync(f.fileno())
assert hashlib.file_digest(t.open('rb'),'sha256').hexdigest()=={digest!r}
if p.suffix=='.pt' and p.exists():assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()=={digest!r}
t.replace(p)
'''
    for attempt in range(1000000):
        with source.open('rb') as stream:
            result = subprocess.run([*SSH, REMOTE+'/.venv/bin/python -c '+shlex.quote(code)], stdin=stream)
        if result.returncode == 0:
            return digest
        if result.returncode != 255:
            raise RuntimeError('Remote import failed; preserve source and inspect')
        print('Connection unavailable; retaining local artifact and retrying', flush=True)
        time.sleep(30)


def archive(checkpoint, spec, step):
    relative = Path('artifacts/compiler_pending')/spec['job_dir']/f'step_{step:05d}.pt'
    digest = send(checkpoint, relative)
    return dict(step=step, path=str(Path(REMOTE)/relative), sha256=digest,
                storage='hash_verified_omarchy_from_macbook')


def train_one(spec_path):
    from scripts import compiler_v3_engine as c
    spec = read(spec_path)
    assert owner(spec['seed']) == 'macbook'
    c.archive = archive
    c.train(spec_path, spec['budgets'][-1])


def worker():
    from scripts.run_compiler_v3 import verify
    plan = read(ROOT/'frozen_plan.json')
    verify(plan['inputs'])
    env = ROOT/'distribution/macbook_environment.json'
    from scripts.run_compiler_efficiency import environment_versions
    import platform
    value = dict(versions=environment_versions(), platform=platform.platform(),
                 backend='CPU SciPy four threads', amendment_sha256=sha256(DOC))
    if env.exists():
        assert read(env) == value
    else:
        save_json(env, value)
    send(env, env)
    for job in plan['jobs']:
        if owner(job['seed']) != 'macbook':
            continue
        verify(plan['inputs'])
        spec = read(job['spec']); directory = Path(spec['job_dir'])
        receipt = directory/'macbook_complete.json'
        if receipt.exists():
            continue
        save_json(ROOT/'distribution/macbook_status.json', dict(stage='training', **job, final_test_opened=False))
        with (directory/'train.log').open('a') as log:
            subprocess.run([sys.executable, '-m', 'scripts.compiler_efficiency_distribution',
                            'train', '--spec', job['spec']], stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ,'OMP_NUM_THREADS':'4','NUMBA_NUM_THREADS':'4'}, check=True)
        progress = read(directory/'progress.json')
        assert progress['step'] == 256
        for name in ['progress.json','train.log','engine_source.py']:
            send(directory/name, directory/name)
        save_json(receipt, dict(seed=job['seed'],method=job['method'],
            progress_sha256=sha256(directory/'progress.json'), checkpoint_sha256=progress['checkpoint_sha256'],
            amendment_sha256=sha256(DOC), environment_sha256=sha256(env), final_test_opened=False))
        send(receipt, receipt)
        # All checkpoints already exist on Linux; its existing relay backs up to Seagate.
        (directory/'model.pt').unlink()
    save_json(ROOT/'distribution/macbook_status.json', dict(stage='training_complete',models=30,final_test_opened=False))


def controller(wait_pid):
    from scripts import run_compiler_efficiency as r
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while wait_pid and Path(f'/proc/{wait_pid}').exists():
        time.sleep(10)
    old_train = r.v.train
    def routed_train(spec_path, engine):
        spec = read(spec_path)
        if owner(spec['seed']) == 'macbook':
            directory = Path(spec['job_dir']); receipt = directory/'macbook_complete.json'
            while not receipt.exists():
                save_json(ROOT/'distribution/coordinator_status.json', dict(stage='waiting_for_macbook',spec=spec_path))
                time.sleep(30)
            proof = read(receipt)
            assert proof['amendment_sha256'] == sha256(DOC)
            assert proof['progress_sha256'] == sha256(directory/'progress.json')
            progress = read(directory/'progress.json')
            assert progress['step'] == 256 and progress['checkpoint_sha256'] == proof['checkpoint_sha256']
            for a in progress['archives']:
                assert sha256(a['path']) == a['sha256']
            latest = next(a for a in progress['archives'] if a['step']==256)
            model = directory/'model.pt'
            if not model.exists():model.symlink_to(latest['path'])
        return old_train(spec_path,engine)
    r.v.train = routed_train
    old_publish = r.publish
    def publish(message):
        subprocess.run(['git','add',str(DOC),__file__],check=True)
        old_publish(message)
    r.publish = publish
    old_report = r.report
    def report(by_seed, analysis, learning):
        old_report(by_seed,analysis,learning)
        text = r.REPORT.read_text().replace('All 60 models completed on Omarchy Linux before final-test evaluation.',
            'All 60 models completed before final-test evaluation: seeds 500–504 on Omarchy, 505–509 on MacBook M1, under the published hardware amendment.')
        r.REPORT.write_text(text+'\nHardware was amended after interim validation inspection at user request. Each six-method seed block stayed on one host. Report timing within host; do not interpret pooled seconds as a compiler speedup. See COMPILER_EFFICIENCY_HARDWARE_AMENDMENT.md.\n')
        save_json(ROOT/'distribution/host_results.json', {host:{s:m for s,m in by_seed.items() if owner(int(s))==host} for host in ['omarchy','macbook']})
    r.report = report
    old_freeze = r.c.freeze
    def freeze(path,value):
        if str(path)==str(ROOT/'unlock.json'):
            value['inputs'].update({str(DOC):sha256(DOC),str(Path(__file__).relative_to(Path.cwd())):sha256(__file__),
                                  str(ROOT/'distribution/macbook_environment.json'):sha256(ROOT/'distribution/macbook_environment.json')})
        return old_freeze(path,value)
    r.c.freeze = freeze
    r.run()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','train','controller']);p.add_argument('--spec');p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args()
    if a.mode=='train':train_one(a.spec)
    elif a.mode=='worker':
        f=(ROOT/'distribution/worker.lock');f.parent.mkdir(parents=True,exist_ok=True)
        lock=f.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);worker()
    else:controller(a.wait_pid)
