"""Frozen five-seed compiler replication on one M4; reuse the v3 engine unchanged."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch
from scripts import compiler_v3_engine as c, g2c_engine as e, run_compiler_v1 as v
from scripts.run_compiler_v3 import now, read, verify, paths_hashes
from scripts.compiler_confirmation_analysis import analyze
from src.provenance import save_json, sha256

ROOT = Path('results/compiler_confirmation')
CFG = Path('configs/compiler_confirmation.json')
DOC = Path('COMPILER_CONFIRMATION_PREREGISTRATION.md')
REPORT = Path('COMPILER_CONFIRMATION_REPORT.md')
BRANCH = 'experiment/compiler-confirmation'
SOURCES = [Path('scripts/run_compiler_confirmation.py'), Path('scripts/compiler_confirmation_analysis.py'),
           Path('tests/test_compiler_confirmation.py')]


def select_test_ids(used, count, salt):
    remaining = set(range(4096)) - set(used)
    if not set(used) <= set(range(4096)) or len(remaining) < count:
        raise ValueError('Invalid exclusions or insufficient unused inputs')
    return sorted(remaining, key=lambda i: hashlib.sha256(f'{salt}:{i}'.encode()).digest())[:count]


def prepare_data(cfg):
    source = Path(cfg['source_dataset']); dest = Path(cfg['dataset'])
    inputs = [source/'manifest.json']; old = read(inputs[0]); used = set()
    for name, info in old['counts'].items():
        p = source/f'{name}.json'; assert sha256(p) == info['sha256']; inputs.append(p)
    inputs += [Path('data/raw/topology_confirmation_v2/test_confirm.json'), Path('data/raw/compiler_v3/test_compiler_v3.json')]
    for p in inputs[1:]:
        rows = read(p); ids = {r['id'] for r in rows}
        assert len(ids) == len(rows) and not used & ids
        used.update(ids)
    assert len(used) == 3200
    ids = select_test_ids(used, cfg['test_count'], cfg['test_salt'])
    rows = []
    for i in ids:
        digits = [i//4**j % 4 for j in reversed(range(6))]
        rows.append(dict(id=i, input=digits, prompt=[e.BOS, *digits, e.SEP], response=[*e.transform(digits, 'substitute'), e.EOS]))
    dest.mkdir(parents=True, exist_ok=True)
    c.freeze(dest/f"{cfg['primary_split']}.json", rows)
    for name in ['train', 'validation']:
        p = dest/f'{name}.json'
        if p.exists(): assert sha256(p) == sha256(source/p.name)
        else: shutil.copy2(source/p.name, p)
    c.freeze(dest/'manifest.json', dict(version='compiler-confirmation', task='substitute', length=6,
        counts={n:dict(count=len(read(dest/f'{n}.json')), sha256=sha256(dest/f'{n}.json')) for n in ['train', 'validation', cfg['primary_split']]},
        source_inputs=paths_hashes(inputs)))
    c.freeze(ROOT/'leakage_audit.json', dict(passed=True, excluded=3200, available_before_selection=896,
        test_count=len(ids), remaining_unused=896-len(ids), test_sha256=sha256(dest/f"{cfg['primary_split']}.json")))


def publish(message):
    if not Path('/Volumes/Seagate').is_mount(): raise RuntimeError('Seagate required')
    paths = [ROOT, CFG, DOC, *SOURCES, Path(read(CFG)['dataset'])]
    if REPORT.exists(): paths.append(REPORT)
    backup = Path('/Volumes/Seagate/FlyGPT Backups/G2c-overnight')
    for root in paths:
        for p in root.rglob('*') if root.is_dir() else [root]:
            if p.is_file() and (p.suffix in ('.json', '.jsonl', '.md', '.py', '.log') or p == ROOT/'teacher_cache.pt'):
                dest = backup/p; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH
    subprocess.run(['git', 'add', '--', *map(str, paths)], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode:
        subprocess.run(['git', 'commit', '-m', message], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True, timeout=120)


def prepare():
    cfg = read(CFG); prepare_data(cfg); c.prepare_cache(cfg)
    cache = torch.load(ROOT/'teacher_cache.pt', map_location='cpu', weights_only=True)
    jobs = []; audits = []
    for seed in cfg['seeds']:
        schedule = c.schedule(len(cache['ids']), cfg['batch'], cfg['updates'], seed)
        schedule_path = c.freeze(ROOT/f'schedule_{seed}.json', schedule)
        audit = c.audit_schedule(schedule, cache['features'], cache['mask'], cfg['shuffle_epsilon'])
        audit.update(cache_sha256=sha256(ROOT/'teacher_cache.pt'), schedule_sha256=sha256(schedule_path))
        audit_path = c.freeze(ROOT/f'shuffle_audit_{seed}.json', audit); audits += [schedule_path, audit_path]
        if not audit['passed']: raise RuntimeError('Frozen correspondence audit failed; stop all seeds without resampling')
        order = list(cfg['methods']); random.Random(33000+seed).shuffle(order)
        for method in order:
            spec = read(f'results/compiler_v3/seed_300/{method}/spec.json')
            spec.update(seed=seed, sampling_seed=75000+seed, group_seed=125000+seed, correspondence_seed=175000+seed,
                projection_seed=225000+seed, dataset=cfg['dataset'], job_dir=str(ROOT/f'seed_{seed}'/method),
                cache=str(ROOT/'teacher_cache.pt'), cache_receipt=str(ROOT/'teacher_cache_receipt.json'),
                schedule=schedule_path, schedule_audit=audit_path, evidence_tier=cfg['evidence'])
            path = c.freeze(Path(spec['job_dir'])/'spec.json', spec)
            jobs.append(dict(seed=seed, method=method, spec=path))
    inputs = [CFG, DOC, *SOURCES, *Path('src').glob('*.py'),
        'scripts/compiler_v3_engine.py', 'scripts/compiler_storage.py', 'scripts/g2c_engine.py',
        'scripts/run_compiler_v1.py', 'scripts/run_compiler_v3.py', 'scripts/run_g2c_overnight.py',
        cfg['teacher_spec'], cfg['teacher_checkpoint'], cfg['qualification_receipt'], cfg['training_targets'], cfg['training_prompts'],
        'data/processed/malecns.npz', ROOT/'teacher_cache.pt', ROOT/'teacher_cache_receipt.json',
        *Path(cfg['dataset']).glob('*.json'), *audits, *[j['spec'] for j in jobs]]
    c.freeze(ROOT/'frozen_plan.json', dict(jobs=jobs, inputs=paths_hashes(inputs), final_test_gate='All 30 final checkpoints verified before any final evaluation'))
    publish('Preregister all thirty compiler replication models and passed shuffle audits')
    print('PREPARED; publish frozen plan to GitHub before training', flush=True)


def report(by_seed, analysis, learning):
    lines = ['# Compiler v3 independent replication', '', read(CFG)['evidence'], '',
        'All 30 models completed on the M4 before final-test evaluation. The pilot is excluded.', '',
        '| Seed | Method | Exact % | Response CE | Zero-edge exact % |', '|---|---|---:|---:|---:|']
    for seed, methods in sorted(by_seed.items()):
        for method, m in sorted(methods.items()):
            lines.append(f"| {seed} | {method} | {100*m['accuracy']:.3f} | {m['response_ce']:.6f} | {100*m['zero_edge_exact']:.3f} |")
    lines += ['', 'Positive benefit means higher accuracy or lower CE. Accuracy benefits below are fractions, not percentage points.', '',
        '| Method vs control | Endpoint | Mean benefit | 95% t interval (unadjusted) | Holm p | Exact sign-flip p | Meets registered t criterion |',
        '|---|---|---:|---|---:|---:|---|']
    for r in analysis['comparisons']:
        lines.append(f"| {r['method']} vs {r['control']} | {r['endpoint']} | {r['mean_benefit']:.6f} | {r['unadjusted_95_t_interval']} | {r['holm_p']:.5g} | {r['exact_two_sided_sign_flip_p']:.5g} | {r['supported_under_preregistered_t_analysis']} |")
    lines += ['', analysis['caveat'], 'Representation benefit requires both shuffled and hard-answer contrasts on the same endpoint. No pooled pilot inference.',
        'Complete curves, threshold intervals/censoring and standalone acquisition/end-to-end timing are in learning.json. Final-test evaluation times are separate. No update-efficiency claim follows from coincident observed threshold crossings.']
    REPORT.write_text('\n'.join(lines)+'\n')


def run():
    cfg = read(CFG); plan = read(ROOT/'frozen_plan.json'); verify(plan['inputs'])
    assert read(ROOT/'publication.json')['frozen_plan_sha256'] == sha256(ROOT/'frozen_plan.json')
    v.ROOT = ROOT; v.refresh = lambda: None
    checkpoints = {}
    for job in plan['jobs']:
        verify(plan['inputs'])
        save_json(ROOT/'status.json', dict(stage='training', **job, utc=now(), final_test_opened=False))
        checkpoint = v.train(job['spec'], 'scripts.compiler_v3_engine')
        checkpoints[job['spec']] = checkpoint
        save_json(ROOT/'completed_training.json', checkpoints)
        publish(f"Compiler replication: preserve {job['method']} seed {job['seed']}")
    assert len(checkpoints) == 30
    verify(plan['inputs'])
    unlock = c.freeze(ROOT/'unlock.json', dict(split=cfg['primary_split'], choices_frozen=True,
        inputs=paths_hashes([ROOT/'frozen_plan.json', ROOT/'publication.json', *checkpoints.values()])))
    by_seed = {}; learning = {}
    for job in plan['jobs']:
        verify(plan['inputs'])
        save_json(ROOT/'status.json', dict(stage='evaluation', **job, utc=now()))
        out = ROOT/f"seed_{job['seed']}"/f"{job['method']}_test.json"
        existing = out.exists(); started = time.perf_counter()
        result = v.evaluate(job['spec'], checkpoints[job['spec']], cfg['primary_split'], out, unlock)
        timing = out.with_name(out.stem+'_timing.json')
        if not timing.exists(): save_json(timing, dict(seconds=None if existing else time.perf_counter()-started, reused=existing))
        seed = str(job['seed']); by_seed.setdefault(seed, {})[job['method']] = result['metrics']
        progress = read(Path(read(job['spec'])['job_dir'])/'progress.json')
        learning.setdefault(seed, {})[job['method']] = c.learning_summary(progress, cfg['threshold'])
        publish(f"Compiler replication: preserve final evaluation {job['method']} seed {seed}")
    verify(plan['inputs']); verify(read(unlock)['inputs'])
    analysis = analyze(by_seed); save_json(ROOT/'analysis.json', analysis); save_json(ROOT/'learning.json', learning)
    report(by_seed, analysis, learning)
    save_json(ROOT/'completion.json', dict(utc=now(), artifact_audit_passed=True, models=30, evidence=cfg['evidence']))
    save_json(ROOT/'status.json', dict(stage='complete', utc=now()))
    publish('Complete all compiler replication outcomes and preregistered paired analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--data-only', action='store_true'); args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    awake = subprocess.Popen(['caffeinate', '-is', '-w', str(os.getpid())])
    try:
        if args.data_only: prepare_data(read(CFG))
        elif args.prepare: prepare()
        else: run()
    except BaseException as exc:
        import traceback
        save_json(ROOT/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), utc=now())); raise
    finally: awake.terminate()
