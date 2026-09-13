"""Serial G2 queue, gated on completion of the clean G1 seed-zero comparison."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from src.provenance import save_json, sha256

ROOT = Path('results/g2_v1')


def run(mode, condition, seed):
    if shutil.disk_usage('.').free < 2*1024**3:
        raise RuntimeError('Less than 2 GiB free; stop before risking an incomplete checkpoint')
    save_json(ROOT/'queue.json',{'status':mode,'condition':condition,'seed':seed})
    with (ROOT/f'{mode}_{condition}_{seed}.log').open('a') as stream:
        subprocess.run([sys.executable,'-m','src.g2',mode,'--condition',condition,'--seed',str(seed)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    if Path('configs/g2_seed0_amendment.json').exists():
        raise RuntimeError('Original G2 queue retired by user amendment; use scripts.finish_g2_start_g2b, not the five-seed schedule')
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg = json.loads(Path('configs/g2_v1.json').read_text())
    save_json(ROOT/'queue.json',{'status':'waiting for G1 matched rewired comparison',
                              'protocol_sha256':sha256('configs/g2_v1.json')})
    while True:
        control = Path('results/malecns_v1/target/rewired_queue.json')
        if control.exists() and json.loads(control.read_text()).get('status')=='complete': break
        time.sleep(10)
    # Finish the G1 test comparison with precisely the same windows and metrics.
    control_test = Path('results/malecns_v1/target/rewired_final_test.json')
    if not control_test.exists():
        final_step = json.loads(Path('results/malecns_v1/target/conclusion.json').read_text())['final_step']
        assert final_step==1024, 'G1 target is frozen at 1024'
        save_json(ROOT/'queue.json',{'status':'final testing of G1 matched rewired control'})
        with (ROOT/'g1_control_test.log').open('a') as stream:
            subprocess.run([sys.executable,'-m','scripts.final_baseline_evaluation','--step','1024','--condition','rewired'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    schedule = [('teacher',cfg['teacher_seed'])]+[(c,s) for s in cfg['seeds'] for c in cfg['conditions']]
    for condition, seed in schedule:
        path = ROOT/f'{condition}_{seed}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('complete'):
                assert sha256(path.with_suffix('.pt'))==previous['checkpoint_sha256']
                assert previous['inputs']['configs/g2_v1.json']==sha256('configs/g2_v1.json')
                continue
        run('train',condition,seed)
    # Only now may model evaluation read the fixed test sentences.
    for condition, seed in schedule:
        path = ROOT/'final'/f'{condition}_{seed}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            assert previous['inputs']['configs/g2_v1.json']==sha256('configs/g2_v1.json')
            checkpoint = str(ROOT/f'{condition}_{seed}.pt')
            assert previous['inputs'][checkpoint]==sha256(checkpoint)
            continue
        run('evaluate',condition,seed)
    subprocess.run([sys.executable,'-m','scripts.report_g2'],check=True)
    save_json(ROOT/'queue.json',{'status':'complete','training_runs':len(schedule),
                              'report':str(ROOT/'REPORT.md')})


if __name__=='__main__':
    try: main()
    except Exception as error:
        save_json(ROOT/'queue.json',{'status':'error','error':str(error),'action':'inspect error; do not change budgets based on results'})
        raise
