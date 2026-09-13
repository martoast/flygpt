"""Conditionally extend the same baseline; never launch architecture variants.

Policy was added in response to the user's instruction to let a descending
curve run. It is adaptive feasibility work, not a confirmatory stopping rule.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.trajectory import inspect,report
from src.provenance import save_json

ROOT=Path('results/malecns_v1/target')


def refresh():
    report()
    subprocess.run([sys.executable,'-m','scripts.plot_trajectory'],check=True)
    subprocess.run([sys.executable,'-m','scripts.report_screen'],check=True)


def main():
    # Wait for all baseline-A diagnostics, plus the trainer's final ablation.
    while not (ROOT/'trajectory'/'step_0512.json').exists():time.sleep(5)
    while True:
        run=json.loads((ROOT/'real_0.json').read_text())
        if 'gap_to_teacher' in run:break
        time.sleep(5)
    cumulative=run['runtime_seconds_this_session']
    pilot=json.loads(Path('results/malecns_v1/language/real_0.json').read_text())
    decisions=[]
    for target_steps in [1024,2048]:
        rows=json.loads((ROOT/'trajectory.json').read_text())['records'];latest=rows[-1];previous=rows[-2]
        improvement=previous['validation_nats_per_byte']-latest['validation_nats_per_byte']
        kl_improvement=previous['teacher_kl_nats_per_byte']-latest['teacher_kl_nats_per_byte']
        continue_run=latest['validation_nats_per_byte']>.425 and improvement>=.05 and kl_improvement>=.05
        decisions.append({'after_step':latest['continuation_step'],'next_step':target_steps,
                          'ce_improvement':improvement,'kl_improvement':kl_improvement,'continue':continue_run})
        save_json(ROOT/'extension_decisions.json',{'policy':'adaptive feasibility, unchanged model/optimizer/data RNG; not confirmatory',
                                                 'decisions':decisions,'maximum_continuation_updates':2048})
        if not continue_run:break
        log=ROOT/f'extension_{target_steps}.log'
        with log.open('w') as stream:
            subprocess.run([sys.executable,'-m','scripts.train_target','--resume','--steps',str(target_steps)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        run=json.loads((ROOT/'real_0.json').read_text());cumulative+=run['runtime_seconds_this_session']
        snapshot=ROOT/'snapshots'/f'real_0_step_{target_steps:04d}.pt'
        if not snapshot.exists():os.link(ROOT/'real_0.pt',snapshot)
        inspect(snapshot,ROOT/'trajectory'/f'step_{target_steps:04d}.json',target_steps,pilot['runtime_seconds']+cumulative,run['trace'])
        refresh()
    # Test stays untouched while continuation decisions depend on validation.
    final_step=json.loads((ROOT/'trajectory.json').read_text())['records'][-1]['continuation_step']
    subprocess.run([sys.executable,'-m','scripts.final_baseline_evaluation','--step',str(final_step)],check=True)
    rows=json.loads((ROOT/'trajectory.json').read_text())['records'];losses=[r['validation_nats_per_byte'] for r in rows]
    last=rows[-1];recent=losses[-3:]
    if last['validation_nats_per_byte']<=.425:regime='within the predefined near-teacher validation threshold; inspect held-out test and ablations'
    elif max(recent)-min(recent)<.05:regime='plateau far above teacher; capacity versus optimization unresolved'
    elif losses[-1]<losses[0]-.05:regime='learning, still far from teacher; convergence and capacity limit remain unestablished'
    else:regime='no clear sustained improvement under this budget'
    conclusion={'regime':regime,'final_step':final_step,'validation_trajectory':losses,'teacher_gap':last['gap_to_teacher'],
                'single_seed':True,'topology_advantage':'not established','extension_decisions':decisions,
                'test':json.loads((ROOT/'final_test.json').read_text())['metrics']}
    save_json(ROOT/'conclusion.json',conclusion)
    (ROOT/'CONCLUSION.md').write_text('# Baseline A outcome\n\n'+regime+'.\n\n'+
        'Measured validation CE: '+' → '.join(f'{v:.4f}' for v in losses)+'.\n\n'+
        'The same graph, populations, ticks, leak, optimizer settings and RNG trajectory were retained. Extensions were validation-driven feasibility decisions, not confirmatory tests. No capacity or topology-superiority claim follows from this one seed.\n')
    refresh();print(json.dumps(conclusion,indent=2),flush=True)


if __name__=='__main__':main()
