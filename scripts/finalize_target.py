"""Finish read-only reporting after the fixed baseline and diagnostic watcher.
Does not start a variant, select by test results, or overwrite pilot weights.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from src.provenance import save_json


def main():
    root=Path('results/malecns_v1/target')
    for step in [64,128,256,512]:
        while not (root/'trajectory'/f'step_{step:04d}.json').exists():time.sleep(5)
        subprocess.run([sys.executable,'-m','scripts.plot_trajectory'],check=True)
        subprocess.run([sys.executable,'-m','scripts.report_screen'],check=True)
    while True:
        run=json.loads((root/'real_0.json').read_text())
        if 'gap_to_teacher' in run:break
        time.sleep(5)
    if not (root/'final_test.json').exists():
        subprocess.run([sys.executable,'-m','scripts.final_baseline_evaluation'],check=True)
    rows=json.loads((root/'trajectory.json').read_text())['records']
    first=rows[0];last=rows[-1];losses=[r['validation_nats_per_byte'] for r in rows]
    recent=losses[-3:]
    stalled=len(recent)==3 and max(recent)-min(recent)<.05
    if last['validation_nats_per_byte']<=.425:
        regime='within the predefined near-teacher validation threshold; inspect independent test and ablations'
    elif stalled:
        regime='plateau far above teacher under this configuration; capacity versus optimization unresolved'
    elif losses[-1]<losses[0]-.05 and last['teacher_kl_nats_per_byte']<first['teacher_kl_nats_per_byte']:
        regime='learning, still far from teacher; no established asymptotic convergence or capacity limit'
    else:
        regime='no clear sustained improvement under this budget'
    summary={'regime':regime,'classification':'descriptive; heuristic plateau tolerance 0.05 nats across final three saved checkpoints',
             'validation_losses':losses,'final_teacher_gap':last['gap_to_teacher'],
             'final_teacher_kl':last['teacher_kl_nats_per_byte'],'final_argmax_agreement':last['teacher_argmax_agreement'],
             'final_ablation_penalty':last['ablation_minus_intact'],'training_seeds':1,
             'topology_advantage':'not tested by this single real-graph continuation',
             'test':json.loads((root/'final_test.json').read_text())['metrics']}
    save_json(root/'conclusion.json',summary)
    (root/'CONCLUSION.md').write_text('# Fixed baseline A outcome\n\n'+regime+'.\n\n'+
        'Validation trajectory (nats/byte): '+ ' → '.join(f'{x:.4f}' for x in losses)+'.\n\n'+
        f'Final teacher gap: {last["gap_to_teacher"]:.4f}. Final teacher KL: {last["teacher_kl_nats_per_byte"]:.4f}. '+
        f'Argmax agreement: {last["teacher_argmax_agreement"]:.1%}. Zero-edge ablation penalty: {last["ablation_minus_intact"]:.4f} nats/byte.\n\n'+
        'This is one seed on synthetic grammar with the real MaleCNS graph. It does not establish biological topology superiority or a capacity bound. See final_test.json for the separate final-checkpoint test evaluation.\n')
    subprocess.run([sys.executable,'-m','scripts.report_screen'],check=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
