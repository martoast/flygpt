"""Paired G2 seed comparisons, preserving per-axis and per-composition detail."""
import json
from pathlib import Path
import numpy as np
from src.provenance import save_json


def main():
    root = Path('results/g2_v1'); cfg = json.loads(Path('configs/g2_v1.json').read_text())
    records = {(c,s):json.loads((root/'final'/f'{c}_{s}.json').read_text())
               for c in cfg['conditions'] for s in cfg['seeds']}
    rng = np.random.default_rng(42); comparisons = []
    for split in ('validation','test'):
        for axis in ('attribute','relation','combined'):
            for objective in ('kd','ce'):
                for metric in ('ce','span_ce'):
                    diffs = np.array([records[(f'rewired_{objective}',s)]['metrics'][f'{split}_{axis}']['metrics'][metric]-
                                      records[(f'real_{objective}',s)]['metrics'][f'{split}_{axis}']['metrics'][metric] for s in cfg['seeds']])
                    boot = diffs[rng.integers(len(diffs),size=(10000,len(diffs)))].mean(1)
                    comparisons.append({'split':split,'axis':axis,'objective':objective,'metric':metric,
                                        'A_bio_negative_loss':float(diffs.mean()),'seed_differences':diffs.tolist(),
                                        'paired_seed_bootstrap_95_ci':np.percentile(boot,[2.5,97.5]).tolist()})
                    grouped = []
                    for s in cfg['seeds']:
                        real_rows = records[(f'real_{objective}',s)]['metrics'][f'{split}_{axis}']['rows']
                        rewired_rows = records[(f'rewired_{objective}',s)]['metrics'][f'{split}_{axis}']['rows']
                        groups = {}
                        for real, rewired in zip(real_rows,rewired_rows,strict=True):
                            assert real['subject_attribute']==rewired['subject_attribute'] and real['subject_relation']==rewired['subject_relation']
                            key = tuple(real['subject_attribute']) if axis=='attribute' else tuple(real['subject_relation']) if axis=='relation' else tuple(real['subject_attribute']+real['subject_relation'])
                            weight = real['bytes'] if metric=='ce' else real['span_bytes']
                            groups.setdefault(key,[]).append((rewired[metric]-real[metric],weight))
                        grouped.append([np.average([v for v,w in groups[k]],weights=[w for v,w in groups[k]]) for k in sorted(groups)])
                    matrix = np.array(grouped)
                    si = rng.integers(len(cfg['seeds']),size=(5000,len(cfg['seeds']),1))
                    gi = rng.integers(matrix.shape[1],size=(5000,1,matrix.shape[1]))
                    group_boot = matrix[si,gi].mean(axis=(1,2))
                    comparisons[-1].update(group_balanced_A_bio=float(matrix.mean()),
                                          crossed_seed_composition_bootstrap_95_ci=np.percentile(group_boot,[2.5,97.5]).tolist(),
                                          composition_groups=matrix.shape[1])
    save_json(root/'comparisons.json',{'comparisons':comparisons,
              'interpretation':'Exploratory five-seed, one corpus partition. Multiple endpoints, no confirmatory significance claim. Per-pair-group observations retained in final records.'})
    lines = ['# G2 — Finite-Grammar Compositional Generalization','',
             'All scheduled conditions and seeds completed the fixed budget. Positive A_bio means lower loss for the real topology. Five seeds on one composition partition remain exploratory.','',
             '| Split | Axis | Objective | Metric | A_bio | Paired-seed 95% interval |',
             '|---|---|---|---|---:|---|']
    for r in comparisons:
        lo,hi = r['paired_seed_bootstrap_95_ci']
        lines.append(f"| {r['split']} | {r['axis']} | {r['objective']} | {r['metric']} | {r['A_bio_negative_loss']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ['', 'The JSON also reports a separate composition-group-balanced estimate and crossed seed/composition bootstrap. It uses equal group weights rather than the primary byte weights. Intervals do not establish confirmatory significance.', '',
              '## Test behavior across models','',
              '| Axis | Condition | Mean CE | Mean held-out span CE |',
              '|---|---|---:|---:|']
    teacher = json.loads((root/'final/teacher_0.json').read_text())
    for axis in ('attribute','relation','combined'):
        tm = teacher['metrics'][f'test_{axis}']['metrics']
        lines.append(f"| {axis} | teacher | {tm['ce']:.4f} | {tm['span_ce']:.4f} |")
        for c in cfg['conditions']:
            metrics = [records[(c,s)]['metrics'][f'test_{axis}']['metrics'] for s in cfg['seeds']]
            lines.append(f"| {axis} | {c} | {np.mean([m['ce'] for m in metrics]):.4f} | {np.mean([m['span_ce'] for m in metrics]):.4f} |")
        for label in ('unigram','ngram4','unrestricted_grammar_oracle'):
            lines.append(f"| {axis} | {label} | {tm[label+'_ce']:.4f} | {tm[label+'_span_ce']:.4f} |")
    lines += ['', 'Read this as three separate questions: does the teacher assign useful probability to unseen compositions, do KD students transfer that behavior beyond CE-only students, and does real topology improve on rewiring? Agreement alone does not answer the first question. A uniform unrestricted grammar admits combinations deliberately excluded from training; its oracle is an external reference, not a learned baseline. No automatic algorithm-discovery claim follows from average byte loss.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
