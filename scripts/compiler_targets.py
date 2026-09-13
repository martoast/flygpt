"""Separate prompt extraction, teacher-only labeling, and independent label audit."""
import json
from pathlib import Path
import torch
from scripts import g2c_engine as e
from src.provenance import save_json,sha256

ROOT=Path('results/compiler_v1')

def prepare(cfg):
    torch.set_num_threads(4);ROOT.mkdir(parents=True,exist_ok=True)
    source=Path(cfg['dataset'])/'train.json'
    # This extraction is the only step that reads original training records.
    # The labeling function receives a file with no response/label field.
    prompts=[{'id':r['id'],'prompt':r['prompt']} for r in json.loads(source.read_text())]
    out=ROOT/'teacher_training_prompts.json';save_json(out,prompts)
    return label(cfg,out)

def label(cfg,prompt_path):
    e.qualified(cfg['qualification_receipt'])
    prompts=json.loads(Path(prompt_path).read_text())
    assert all(set(r)=={'id','prompt'} for r in prompts)
    teacher=e.load(json.loads(Path(cfg['teacher_spec']).read_text()),cfg['teacher_checkpoint'])
    records=[]
    with torch.no_grad():
        for r in prompts:
            answer=e.generate(teacher,r['prompt'],8)
            if len(answer)!=7 or answer[-1]!=e.EOS or any(v not in range(4) for v in answer[:-1]):
                raise RuntimeError('Teacher answer format failure; preserve failed candidate rather than replacing with ground truth')
            records.append({**r,'response':answer})
    target=ROOT/'teacher_training_targets.json';save_json(target,records)
    save_json(ROOT/'teacher_targets_provenance.json',{'teacher_only':True,'generation':'greedy autoregressive from prompt; no reference answer prefix',
        'inputs':{str(p):sha256(p) for p in [prompt_path,cfg['teacher_checkpoint'],cfg['teacher_spec'],'scripts/compiler_targets.py']},
        'output':str(target),'sha256':sha256(target),'count':len(records)})
    return target

def audit(cfg,target):
    # Separate reporting-only audit. Nothing is substituted into teacher targets.
    original=e.data(cfg['dataset'],'train');generated=json.loads(Path(target).read_text())
    assert [r['id'] for r in original]==[r['id'] for r in generated]
    same=sum(a['response']==b['response'] for a,b in zip(original,generated))
    result={'matches':same,'cases':len(original),'all_identical':same==len(original),
        'target_sha256':sha256(target),'original_sha256':sha256(Path(cfg['dataset'])/'train.json'),
        'interpretation':'Identical labels imply identical hard-target CE objective, not an optimization advantage or compressed parameter translation'}
    save_json(ROOT/'teacher_label_audit.json',result);return result

if __name__=='__main__':
    cfg=json.loads(Path('configs/compiler_v1.json').read_text());path=prepare(cfg);print(audit(cfg,path))
