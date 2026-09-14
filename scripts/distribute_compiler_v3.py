"""Two-host orchestration; original frozen student and evaluator are unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from scripts import run_compiler_v3 as runner
from scripts import run_compiler_v1 as v
from scripts import mirror_m1_checkpoints as mirror
from src.provenance import save_json, sha256

ROOT = Path('results/compiler_v3')
DIST = ROOT / 'distributed'
REMOTE = '/Users/alex/flygpt_compiler_v3'
M1 = ('D_hidden_shuffled', 'C_hidden')
M4 = ('A_hard', 'E_relational', 'F_relational_shuffled', 'B_soft')


def read(p):
    return json.loads(Path(p).read_text())


def remote_python(code):
    return subprocess.check_output([*mirror.SSH, REMOTE + '/.venv/bin/python -c ' + shlex.quote(code)], text=True, timeout=120)


def upload(path):
    """Publish a small immutable worker artifact atomically with a hash check."""
    path = Path(path)
    dest = REMOTE + '/' + str(path)
    code = ('from pathlib import Path; import sys,hashlib,os; '
            f'p=Path({dest!r}); p.parent.mkdir(parents=True,exist_ok=True); '
            'b=sys.stdin.buffer.read(); '
            f'assert hashlib.sha256(b).hexdigest()=={sha256(path)!r}; '
            'q=p.with_name(p.name+".incoming"); q.write_bytes(b); os.replace(q,p)')
    with path.open('rb') as stream:
        subprocess.run([*mirror.SSH, REMOTE + '/.venv/bin/python -c ' + shlex.quote(code)], stdin=stream, check=True, timeout=120)


def check_allocation():
    receipt = read(DIST / 'allocation.json')
    runner.verify(receipt['inputs'])
    runner.verify(read(ROOT / 'frozen_plan.json')['inputs'])
    publication = read(DIST / 'publication.json')
    assert publication['allocation_sha256'] == sha256(DIST / 'allocation.json')


def backup_job(method):
    job = ROOT / 'seed_300' / method
    progress = read(job / 'progress.json')
    receipt_path = DIST / (method + '_imports.json')
    receipts = read(receipt_path) if receipt_path.exists() else {}
    for archive in progress['archives']:
        key = str(archive['step'])
        if key in receipts:
            assert receipts[key]['sha256'] == archive['sha256']
            continue
        path = Path(archive['path'])
        assert sha256(path) == archive['sha256']
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        external = mirror.copy(relative, archive['sha256'])
        receipts[key] = dict(path=external, sha256=archive['sha256'])
        save_json(receipt_path, receipts)
    return receipts


def laptop():
    probe = read('results/compiler_v3_m1_probe/progress.json')
    assert probe['step'] == 1
    v.ROOT = ROOT
    for method in M1:
        check_allocation()
        job = ROOT / 'seed_300' / method
        save_json(DIST / 'm1_status.json', dict(method=method, stage='training', utc=runner.now()))
        # Each published checkpoint is backed up while the next segment trains.
        def refresh():
            if (job / 'progress.json').exists():
                try:
                    backup_job(method)
                except (subprocess.SubprocessError, OSError) as exc:
                    save_json(DIST / 'backup_retry.json', dict(error=repr(exc), utc=runner.now()))
        v.refresh = refresh
        v.train(job / 'spec.json', 'scripts.compiler_v3_engine')
        receipts = backup_job(method)
        progress = read(job / 'progress.json')
        assert progress['step'] == 512 and receipts['512']['sha256'] == progress['checkpoint_sha256']
        artifacts = [job / 'progress.json', job / 'engine_source.py', job / 'train.log', DIST / (method + '_imports.json')]
        for path in artifacts:
            upload(path)
        done = DIST / (method + '_complete.json')
        save_json(done, dict(method=method, step=512, host='M1 MacBook', checkpoint=receipts['512'],
                            inputs=runner.paths_hashes(artifacts), spec_sha256=sha256(job / 'spec.json'), utc=runner.now()))
        upload(done)
    save_json(DIST / 'm1_complete.json', dict(utc=runner.now(), methods=M1))
    upload(DIST / 'm1_complete.json')


def imported_checkpoint(method):
    receipt = read(DIST / (method + '_complete.json'))
    assert receipt['method'] == method and receipt['step'] == 512
    runner.verify(receipt['inputs'])
    job = ROOT / 'seed_300' / method
    assert receipt['spec_sha256'] == sha256(job / 'spec.json')
    progress = read(job / 'progress.json')
    assert progress['step'] == 512
    for archive in read(DIST / (method + '_imports.json')).values():
        assert sha256(archive['path']) == archive['sha256']
    checkpoint = receipt['checkpoint']
    assert checkpoint['sha256'] == progress['checkpoint_sha256'] == sha256(checkpoint['path'])
    return checkpoint['path']


def mini(existing_child):
    check_allocation()
    # Adopt completed A artifacts without restarting its live training process.
    while existing_child:
        state = subprocess.run(['ps', '-p', str(existing_child), '-o', 'stat='], capture_output=True, text=True)
        if state.returncode or state.stdout.strip().startswith('Z'):
            break
        time.sleep(20)
    if existing_child:
        assert read(ROOT / 'seed_300/A_hard/progress.json')['step'] == 512
    v.ROOT = ROOT
    v.refresh = lambda: None
    original_train = v.train
    for method in M4:
        check_allocation()
        save_json(ROOT / 'status.json', dict(stage='training', method=method, host='M4', utc=runner.now(), final_test_opened=False))
        original_train(ROOT / 'seed_300' / method / 'spec.json', 'scripts.compiler_v3_engine')
        runner.publish('Compiler v3: preserve M4 ' + method + ' under published hardware amendment')
    for method in M1:
        while not (DIST / (method + '_complete.json')).exists():
            if (DIST / 'm1_failure.json').exists():
                raise RuntimeError('M1 worker failed; preserve all results and stop')
            save_json(ROOT / 'status.json', dict(stage='awaiting_m1', method=method, utc=runner.now(), final_test_opened=False))
            time.sleep(20)
        imported_checkpoint(method)
    def assigned_train(path, engine):
        method = read(path)['method']
        return imported_checkpoint(method) if method in M1 else original_train(path, engine)
    v.train = assigned_train
    warning = ('\nHardware amendment: C/D trained on M1; A/B/E/F on M4. Cross-machine raw times do not establish compiler efficiency. '
               'Cached teacher preparation was measured on M4 for all methods. C/D versus A is additionally hardware-confounded; '
               'real/shuffled pairs share hardware. See COMPILER_V3_HARDWARE_AMENDMENT.md.\n')
    original_publish = runner.publish
    def publish_with_hardware(message):
        report = Path('COMPILER_V3_REPORT.md')
        if report.exists() and warning.strip() not in report.read_text():
            report.write_text(report.read_text() + warning)
        original_publish(message)
    runner.publish = publish_with_hardware
    # The unchanged coordinator now sees six finished jobs, then unlocks/evaluates.
    runner.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=['m1', 'm4'])
    parser.add_argument('--existing-child', type=int, default=0)
    args = parser.parse_args()
    DIST.mkdir(parents=True, exist_ok=True)
    lock = (DIST / (args.host + '.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    awake = subprocess.Popen(['caffeinate', '-is', '-w', str(os.getpid())])
    try:
        (laptop() if args.host == 'm1' else mini(args.existing_child))
    except BaseException as exc:
        path = DIST / (args.host + '_failure.json')
        save_json(path, dict(error=repr(exc), utc=runner.now()))
        if args.host == 'm1':
            try:
                upload(path)
            except Exception:
                pass
        raise
    finally:
        awake.terminate()


if __name__ == '__main__':
    main()
