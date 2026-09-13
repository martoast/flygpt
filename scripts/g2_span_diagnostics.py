"""Post hoc, read-only span KL; does not alter G2 training or stopping.

Use the SAME 8 validation sentences per axis and SAME target byte indices as
the frozen G2 evaluator. No test data. Watch only matched 256/512 checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import torch
from torch import nn
from src.g2 import datasets, batch, span_indices, logits
from src.train_memory import build_model
from src.query_flygpt import load_model
from src.tinygpt import TinyGPT
from src.provenance import manifest, save_json, sha256

ROOT = Path('results/g2_v1')
OUT = ROOT/'span_diagnostics'


def restricted_metrics(student, teacher, targets, selected):
    """Each returned sum is over target positions, with a full 256-byte KL."""
    s = student[0,selected]; t = teacher[0,selected]; y = targets[0,selected]
    return {'student_nll_sum':float(nn.functional.cross_entropy(s,y,reduction='sum')),
            'teacher_nll_sum':float(nn.functional.cross_entropy(t,y,reduction='sum')),
            'teacher_kl_sum':float(nn.functional.kl_div(nn.functional.log_softmax(s,-1),nn.functional.softmax(t,-1),reduction='sum')),
            'teacher_kl_T2_scaled_sum':float(nn.functional.kl_div(nn.functional.log_softmax(s/2,-1),nn.functional.softmax(t/2,-1),reduction='sum')*4),
            'teacher_agreement_count':int((s.argmax(-1)==t.argmax(-1)).sum()),
            'bytes':len(selected)}


def inspect(snapshot, output):
    torch.set_num_threads(2)
    start = time.perf_counter()
    ck = torch.load(snapshot,weights_only=True,map_location='cpu')
    condition, seed, step = ck['condition'],ck['seed'],ck['completed_steps']
    graph = ck['graph']; checkpoint_hash = sha256(snapshot)
    if condition=='gru_ce':
        assert sha256(graph)==ck['graph_sha256']
        model = build_model(graph,'gru',seed,ck['config']); model.load_state_dict(ck['model'])
    else:
        model = load_model(graph,snapshot,'cpu')
    del ck
    model.eval()
    tc = torch.load(ROOT/'teacher_0.pt',weights_only=True,map_location='cpu')
    teacher = TinyGPT(**tc['config']).eval(); teacher.load_state_dict(tc['model']); del tc
    record = manifest({'analysis':'post hoc span diagnostic, not a training change',
                       'condition':condition,'seed':seed,'step':step,'sentences_per_axis':8,
                       'span':'exact frozen color/verb word bytes including trailing separator',
                       'temperature':1,'checkpoint_sha256':checkpoint_hash},
                      [snapshot,graph,ROOT/'teacher_0.pt','configs/g2_v1.json','data/raw/grammar_v2/manifest.json'])
    record['diagnostic_script_sha256'] = sha256(__file__)
    results = {}
    with torch.no_grad():
        for name, rows in datasets('validation',8).items():
            details = []
            for index, row in enumerate(rows):
                x,y = batch([row],[0]); s = logits(model,x); t = teacher(x)
                attribute = span_indices(row['text'],2); relation = span_indices(row['text'],4)
                selected = attribute if name.endswith('attribute') else relation if name.endswith('relation') else attribute+relation
                details.append({'sentence_index':index,'target_indices':selected,
                                'target_bytes':y[0,selected].tolist(),
                                'subject_attribute':row['subject_attribute'],'subject_relation':row['subject_relation'],
                                **restricted_metrics(s,t,y,selected)})
            count = sum(d['bytes'] for d in details)
            results[name] = {'span_bytes':count,'sentences':len(details),
                             'student_span_ce':sum(d['student_nll_sum'] for d in details)/count,
                             'teacher_span_ce':sum(d['teacher_nll_sum'] for d in details)/count,
                             'teacher_span_kl':sum(d['teacher_kl_sum'] for d in details)/count,
                             'teacher_span_kl_T2_scaled':sum(d['teacher_kl_T2_scaled_sum'] for d in details)/count,
                             'teacher_span_agreement':sum(d['teacher_agreement_count'] for d in details)/count,
                             'rows':details}
    record.update(metrics=results,diagnostic_wall_seconds=time.perf_counter()-start,
                  interpretation='Teacher similarity and ground-truth likelihood are separate. Lower KL can reproduce a teacher failure; no binary success threshold was preregistered for this added diagnostic.')
    save_json(output,record)
    print(condition,seed,step,{k:{m:v[m] for m in ['student_span_ce','teacher_span_ce','teacher_span_kl']} for k,v in results.items()},flush=True)


def archive(snapshot):
    """Keep diagnostic checkpoints on the authorized external disk if mounted."""
    disk = Path('/Volumes/Seagate')
    if not disk.is_mount(): return False
    destination = disk/'FlyGPT Backups/G2-span-diagnostics'
    destination.mkdir(parents=True,exist_ok=True)
    target = destination/snapshot.name; digest = sha256(snapshot)
    if not target.exists():
        tmp = target.with_suffix('.pt.partial')
        with snapshot.open('rb') as src, tmp.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024); dst.flush(); os.fsync(dst.fileno())
        assert sha256(tmp)==digest; tmp.rename(target)
    assert sha256(target)==digest
    save_json(OUT/'backups'/snapshot.with_suffix('.json').name,
              {'destination':str(target),'sha256':digest,'storage':'verified external USB copy; graph and teacher identities recorded in diagnostic'})
    snapshot.unlink()  # Only our hardlink; the training checkpoint is untouched.
    return True


def scan():
    OUT.mkdir(parents=True,exist_ok=True); snapshots = OUT/'snapshots'; snapshots.mkdir(exist_ok=True)
    cfg = json.loads(Path('configs/g2_v1.json').read_text())
    for seed in cfg['seeds']:
        for condition in cfg['conditions']:
            path = ROOT/f'{condition}_{seed}.pt'
            progress = path.with_suffix('.json')
            if not path.exists() or not progress.exists(): continue
            result = json.loads(progress.read_text())
            step = result['evaluations'][-1]['step']
            if step not in (256,512): continue
            output = OUT/f'{condition}_{seed}_step{step:04d}.json'
            if output.exists(): continue
            snapshot = snapshots/output.with_suffix('.pt').name
            if not snapshot.exists(): os.link(path,snapshot)
            if sha256(snapshot)!=result['checkpoint_sha256']:
                snapshot.unlink(); continue  # Atomic publication raced; retry later.
            inspect(snapshot,output)
            archive(snapshot)
    report()


def report():
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob('*_step*.json'))]
    lines = ['# G2: post hoc teacher KL on held-out spans','',
             'Read-only validation diagnostic added after observing the real seed-zero 256 result. Training, stopping, and test access are unchanged. Matching checkpoints: 256 and 512. Lower KL means greater teacher similarity, not necessarily better generalization.','',
             '| Condition | Seed | Step | Holdout | Student span CE | Teacher span CE | Span KL (T=1) | Span agreement |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in records:
        c = r['config']
        for name,m in r['metrics'].items():
            lines.append(f"| {c['condition']} | {c['seed']} | {c['step']} | {name} | {m['student_span_ce']:.4f} | {m['teacher_span_ce']:.4f} | {m['teacher_span_kl']:.4f} | {m['teacher_span_agreement']:.3f} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--watch',action='store_true'); args = parser.parse_args()
    while True:
        scan()
        if not args.watch or json.loads((ROOT/'queue.json').read_text()).get('status') in ('complete','error'): break
        time.sleep(20)
