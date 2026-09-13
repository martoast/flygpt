"""User-amended serial handoff; preserve active child, finish G2 seed zero."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from src.provenance import save_json,sha256

ROOT=Path('results/g2_v1')


def run(command, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable,*command],stdout=stream,stderr=subprocess.STDOUT,check=True)


def completed(condition):
    p=ROOT/f'{condition}_0.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    if not r.get('complete'):return False
    assert r['evaluations'][-1]['step']==512
    assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
    return True


def freeze():
    from scripts.g2_span_diagnostics import inspect
    from scripts.report_g2_relation import main as report
    for condition in ['real_kd','rewired_kd','real_ce']:
        assert completed(condition)
        out=ROOT/'span_diagnostics'/f'{condition}_0_step0512.json'
        # Existing watcher may finish this before us. Use a separate output
        # directory if it is still busy, then publish after its process finishes.
        if not out.exists():
            private=ROOT/'frozen_seed0'/f'{condition}_span.json'
            inspect(ROOT/f'{condition}_0.pt',private)
            if not out.exists():save_json(out,json.loads(private.read_text()))
    report()
    relation=json.loads((ROOT/'relation_focus.json').read_text())
    lines=['# G2 exploratory seed-zero result','',
           'User-directed truncation after teacher weakness and real seed-zero outcomes; not the original five-seed stopping plan. G2 test data remain locked. All three axes are preserved in the original per-run records.','',
           'The table uses the fixed eight relation validation sentences and final 512-update checkpoints. Relation focus and span KL are post hoc diagnostics.','',
           '| Student | Relation span CE | Relation span KL |','|---|---:|---:|']
    for row in relation['rows']:
        lines.append(f"| {row['condition']} | {row['span_ce']:.4f} | {row['span_kl']:.4f} |")
    lines += ['',f"Teacher relation span CE: {relation['teacher_span_ce']:.4f}. N-gram span CE: {relation['ngram_span_ce']:.4f}.",'',
              'CE-only minus KD differences (positive means KD improves):']
    for comparison in relation['paired_objective_comparisons']:
        lines.append(f"- {comparison['topology']}, seed {comparison['seed']}: CE {comparison['ce_only_minus_kd_span_ce']:.4f}; KL {comparison['ce_only_minus_kd_span_kl']:.4f}; both improve: {comparison['both_improve_with_kd']}.")
    lines += ['', 'This single-seed validation comparison cannot establish replicated generalization or a biological topology advantage. It documents the distinction between ordinary sequence prediction and the specific held-out behavior sought for transfer.']
    (ROOT/'G2_EXPLORATORY_SEED_ZERO.md').write_text('\n'.join(lines)+'\n')
    checkpoint_files=[ROOT/f'{c}_0.pt' for c in ['teacher','real_kd','rewired_kd','real_ce']]
    graph_files=[Path('data/processed/malecns.npz'),Path('data/processed/controls/rewired_777.npz')]
    sources=checkpoint_files+graph_files+list(Path('data/raw/grammar_v2').glob('*'))
    sources += list(ROOT.glob('*.json'))+list((ROOT/'span_diagnostics').glob('*_step0512.json'))
    sources += [Path('configs/g2_v1.json'),Path('configs/g2_seed0_amendment.json')]
    sources += list(ROOT.glob('*.md'))+list(Path('src').glob('*.py'))
    sources += [Path('scripts/g2_span_diagnostics.py'),Path('scripts/report_g2_relation.py')]
    disk=Path('/Volumes/Seagate'); backups=[]
    for source in sources:
        entry={'source':str(source),'sha256':sha256(source)}
        if disk.is_mount():
            dest=disk/'FlyGPT Backups/G2-exploratory-seed-zero'/source
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with source.open('rb') as src,dest.open('xb') as dst:
                    shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
            assert sha256(dest)==entry['sha256'],f'Backup differs: {dest}'
            entry['external_copy']=str(dest)
        backups.append(entry)
    summary={'experiment':'G2 exploratory seed-zero','status':'frozen',
             'amendment':json.loads(Path('configs/g2_seed0_amendment.json').read_text()),
             'files':backups,'external_backup_verified':disk.is_mount(),
             'test_evaluated':False,'relation_comparison':json.loads((ROOT/'relation_focus.json').read_text())}
    save_json(ROOT/'frozen_seed0/manifest.json',summary)
    if disk.is_mount():save_json(disk/'FlyGPT Backups/G2-exploratory-seed-zero/manifest.json',summary)
    save_json(ROOT/'queue.json',{'status':'complete','scope':'amended seed-zero subset only',
                              'remaining_runs':'postponed by user','freeze_manifest':str(ROOT/'frozen_seed0/manifest.json')})


def main(active_pid):
    save_json(ROOT/'handoff.json',{'status':'waiting for active rewired seed zero','active_pid':active_pid,
                                'amendment_sha256':sha256('configs/g2_seed0_amendment.json')})
    while True:
        command=subprocess.run(['ps','-p',str(active_pid),'-o','command='],capture_output=True,text=True).stdout
        if '-m src.g2 train --condition rewired_kd --seed 0' not in command:break
        time.sleep(10)
    if not completed('rewired_kd'):
        raise RuntimeError('Active rewired run ended without a verified final checkpoint; inspect before continuing')
    if not completed('real_ce'):
        save_json(ROOT/'queue.json',{'status':'train','condition':'real_ce','seed':0,'scope':'user-amended final G2 run'})
        run(['-m','src.g2','train','--condition','real_ce','--seed','0'],ROOT/'train_real_ce_0.log')
    if not (ROOT/'frozen_seed0/manifest.json').exists():freeze()
    save_json(ROOT/'handoff.json',{'status':'G2 frozen; starting G2b qualification','amendment':'configs/g2_seed0_amendment.json'})
    run(['-m','scripts.run_g2b'],Path('results/g2b_v1/controller.log'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--active-pid',type=int,required=True);args=parser.parse_args()
    try:main(args.active_pid)
    except Exception as error:
        save_json(ROOT/'handoff.json',{'status':'error','error':str(error)});raise
