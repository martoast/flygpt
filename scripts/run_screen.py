"""Resumable serial training ladder. Control construction may run separately.
Each subprocess releases model/optimizer memory before the next experiment.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from src.provenance import save_json

PYTHON=sys.executable
ROOT=Path('results/malecns_v1')


def run(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN', ' '.join(command),flush=True)
    with log.open('w') as stream:
        subprocess.run([PYTHON,*command],stdout=stream,stderr=subprocess.STDOUT,check=True)


def controls():
    for seed in range(5):
        for condition in ['rewired','configuration','er']:
            out=ROOT/'controls'/f'{condition}_{777+seed}.json'
            if out.exists():continue
            # The first rewired graph may already be under construction.
            if condition=='rewired' and seed==0:
                while not out.exists():time.sleep(5)
                continue
            run(['-m','src.graph_controls','--graph','data/processed/malecns.npz','--condition',condition,'--seed',str(777+seed)],out.with_suffix('.log'))


def ladder():
    cfg=json.loads(Path('configs/screen_v1.json').read_text())
    for phase in ['language','memory','distill']:
        if phase=='distill':
            while not (ROOT/'teacher.json').exists():time.sleep(5)
        for seed in cfg['seeds']:
            # Real first, then baseline, leaving producer time to build controls.
            for condition in ['real','gru','rewired','configuration','er']:
                out=ROOT/phase/f'{condition}_{seed}.json'
                if out.exists():continue
                graph='data/processed/malecns.npz'
                if condition not in ('real','gru'):
                    graph=f'data/processed/controls/{condition}_{seed+777}.npz'
                    while not (ROOT/'controls'/f'{condition}_{seed+777}.json').exists():time.sleep(5)
                module='src.train_memory' if phase=='memory' else 'src.train_language'
                command=['-m',module,'--graph',graph,'--condition',condition,'--seed',str(seed),'--steps',str(cfg['memory' if phase=='memory' else 'language']['steps']),'--out',str(out)]
                if phase=='distill':command+=['--teacher',str(ROOT/'teacher.pt')]
                if phase=='language' and condition=='real' and seed==0:command+=['--checkpoint','results/fly_real.pt']
                run(command,out.with_suffix('.log'))
                save_json(ROOT/'progress.json',{'last_completed':str(out),'updated_unix':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controls','ladder']);a=p.parse_args()
    controls() if a.mode=='controls' else ladder()
