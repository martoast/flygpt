"""Post hoc relation focus, without suppressing the other frozen G2 axes."""
import json
import argparse
import time
from pathlib import Path
from src.provenance import save_json


def main():
    root = Path('results/g2_v1'); axis = 'validation_relation'
    teacher = json.loads((root/'teacher_0.json').read_text())['evaluations'][-1]['metrics'][axis]['metrics']
    rows = []
    for p in sorted((root/'span_diagnostics').glob('*_step0512.json')):
        r = json.loads(p.read_text()); m = r['metrics'][axis]
        rows.append({'condition':r['config']['condition'],'seed':r['config']['seed'],
                     'span_ce':m['student_span_ce'],'teacher_span_ce':m['teacher_span_ce'],
                     'span_kl':m['teacher_span_kl']})
    comparisons = []
    for topology in ['real','rewired']:
        for seed in range(5):
            kd = next((r for r in rows if r['condition']==topology+'_kd' and r['seed']==seed),None)
            ce = next((r for r in rows if r['condition']==topology+'_ce' and r['seed']==seed),None)
            if kd and ce:
                comparisons.append({'topology':topology,'seed':seed,'ce_only_minus_kd_span_ce':ce['span_ce']-kd['span_ce'],
                                    'ce_only_minus_kd_span_kl':ce['span_kl']-kd['span_kl'],
                                    'both_improve_with_kd':kd['span_ce']<ce['span_ce'] and kd['span_kl']<ce['span_kl']})
    save_json(root/'relation_focus.json',{'status':'post hoc exploratory focus selected after teacher validation; all other axes remain reported',
              'checkpoint':512,'validation_sentences':8,'teacher_span_ce':teacher['span_ce'],
              'ngram_span_ce':teacher['ngram4_span_ce'],'rows':rows,'paired_objective_comparisons':comparisons})
    lines = ['# G2 relation-span comparison at 512','',
             'Post hoc focus on the fixed eight relation validation sentences. This does not replace the three-axis protocol or create a new confirmatory endpoint. Missing controls remain pending.','',
             '| Model | Seed | Span CE | Span KL to teacher |','|---|---:|---:|---:|',
             f"| Teacher | 0 | {teacher['span_ce']:.4f} | 0 |",
             f"| N-gram order 4 | — | {teacher['ngram4_span_ce']:.4f} | not measured |"]
    for r in rows: lines.append(f"| {r['condition']} | {r['seed']} | {r['span_ce']:.4f} | {r['span_kl']:.4f} |")
    lines += ['', 'Positive CE-only minus KD differences in both span CE and span KL would support beneficial distillation on these validation positions. Replication and untouched evaluation are needed for a generalization claim. Teacher advantage over the n-gram alone does not establish teacher superiority over retrieval or a simple RNN.']
    (root/'RELATION_FOCUS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--watch',action='store_true'); args=parser.parse_args()
    previous=None
    while True:
        files=tuple(sorted(str(p) for p in Path('results/g2_v1/span_diagnostics').glob('*_step0512.json')))
        if files!=previous: main(); previous=files
        if not args.watch or json.loads(Path('results/g2_v1/queue.json').read_text()).get('status') in ('complete','error'): break
        time.sleep(20)
