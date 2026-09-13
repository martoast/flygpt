"""Teacher qualification first; never launch connectome students on a failed gate."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.g2b_experiment import ROOT,PROTOCOL,EXECUTION,config,verify_gate,student_schedule
from src.provenance import save_json,sha256


def run(module,args,label):
    if shutil.disk_usage('.').free<2*1024**3:raise RuntimeError('Less than 2 GiB free; paused before another checkpoint allocation')
    ROOT.mkdir(parents=True,exist_ok=True)
    save_json(ROOT/'queue.json',{'status':'running','stage':label})
    with (ROOT/f'{label}.log').open('a') as stream:
        subprocess.run([sys.executable,'-m',module,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)


def valid_result(path):
    if not path.exists():return False
    r=json.loads(path.read_text())
    for source,digest in r['inputs'].items():assert sha256(source)==digest,f'Stale result: {path}'
    return True


def train_and_validate(kind,seed):
    progress=ROOT/f'{kind}_{seed}.json'
    done=valid_result(progress) and json.loads(progress.read_text()).get('complete')
    if done:assert sha256(progress.with_suffix('.pt'))==json.loads(progress.read_text())['checkpoint_sha256']
    else:run('scripts.g2b_experiment',['train','--kind',kind,'--seed',str(seed)],f'train_{kind}_{seed}')
    out=ROOT/'validation'/f'{kind}_{seed}.json'
    if not valid_result(out):run('scripts.g2b_experiment',['evaluate','--kind',kind,'--seed',str(seed),'--split','validation'],f'validation_{kind}_{seed}')


def main():
    ROOT.mkdir(parents=True,exist_ok=True);protocol,execution=config()
    # G2 exploratory subset must be preserved before allocating the next stage.
    if not Path('results/g2_v1/frozen_seed0/manifest.json').exists():raise RuntimeError('Freeze G2 seed zero before starting G2b')
    for seed in protocol['seeds']:
        for kind in ('teacher','gru'):train_and_validate(kind,seed)
    for split in ('validation','qualification'):
        if split=='qualification':
            verify_gate('validation')
            for seed in protocol['seeds']:
                for kind in ('teacher','gru'):
                    out=ROOT/split/f'{kind}_{seed}.json'
                    if not valid_result(out):run('scripts.g2b_experiment',['evaluate','--kind',kind,'--seed',str(seed),'--split',split],f'{split}_{kind}_{seed}')
        if not all(valid_result(ROOT/split/f'{name}.json') for name in ('ngram1','ngram4','ngram8','ngram16','retrieval')):
            run('scripts.g2b_experiment',['baselines','--split',split],f'{split}_baselines')
        gate=ROOT/f'gate_{split}.json'
        if not gate.exists():run('scripts.g2b_gate',[split],f'{split}_gate')
        result=json.loads(gate.read_text())
        assert result['protocol_sha256']==sha256(PROTOCOL) and result['execution_sha256']==sha256(EXECUTION)
        for p,h in result['inputs'].items():assert sha256(p)==h
        if not result['passed']:
            save_json(ROOT/'queue.json',{'status':'teacher_not_qualified','failed_stage':split,
                                       'student_runs_launched':0,'next_action':'Report failed gate; no automatic rescue or connectome training'})
            return
    verify_gate('validation');verify_gate('qualification')
    for kind,seed in student_schedule():
        progress=ROOT/f'{kind}_{seed}.json'
        if not (valid_result(progress) and json.loads(progress.read_text()).get('complete')):
            run('scripts.g2b_experiment',['train','--kind',kind,'--seed',str(seed)],f'train_{kind}_{seed}')
        assert sha256(progress.with_suffix('.pt'))==json.loads(progress.read_text())['checkpoint_sha256']
    # Test is separate from the qualification split and opens only now.
    for seed in protocol['seeds']:
        for kind in ('teacher','gru',*execution['student_conditions']):
            out=ROOT/'test'/f'{kind}_{seed}.json'
            if not valid_result(out):run('scripts.g2b_experiment',['evaluate','--kind',kind,'--seed',str(seed),'--split','test'],f'test_{kind}_{seed}')
    run('scripts.g2b_experiment',['baselines','--split','test'],'test_baselines')
    run('scripts.report_g2b',[],'final_report')
    save_json(ROOT/'queue.json',{'status':'complete','teacher_qualified':True,'student_runs':len(student_schedule())})


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True)
        save_json(ROOT/'queue.json',{'status':'error','error':str(error)});raise
