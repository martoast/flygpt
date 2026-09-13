"""Fixed-budget longer-context continuation toward the frozen grammar teacher.

This is a feasibility extension, not a topology-advantage conclusion. Use the
same seed, stage budgets, data sampling, and learning rate for every condition.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.query_flygpt import load_model,generate
from src.train_language import bytes_from_file,sample_batch,evaluate,TeacherAdapter
from src.tinygpt import TinyGPT
from src.provenance import manifest,sha256,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',default='data/processed/malecns.npz')
    p.add_argument('--initial',default='results/fly_real.pt');p.add_argument('--condition',default='real');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=512);p.add_argument('--out',default='results/malecns_v1/target/real_0.json');p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);ckpath=out.with_suffix('.pt')
    source=ckpath if a.resume and ckpath.exists() else Path(a.initial)
    ck=torch.load(source,map_location='cpu',weights_only=True);model=load_model(a.graph,source,'cpu');cfg=ck['config']
    tck=torch.load('results/malecns_v1/teacher.pt',map_location='cpu',weights_only=True)
    teacher=TinyGPT(**tck['config']).eval()
    teacher.load_state_dict(tck['model'])
    for p in teacher.parameters():p.requires_grad_(False)
    tr=bytes_from_file('data/raw/grammar_v1/train.txt');va=bytes_from_file('data/raw/grammar_v1/validation.txt')
    starts=np.random.default_rng(9100).integers(0,len(va)-33,size=8).tolist()
    result=manifest({**vars(a),'block':32,'batch':1,'lr':.001,'alpha':.5,'temperature':2.,'model':cfg,
                     'optimizer':'AdamW','weight_decay':.01,'initial_optimizer':'reset after 64-step CE pilot',
                     'validation_schedule':[0,64,128,256,512],'close_threshold_nats':.425,
                     'scope':'single-seed feasibility continuation; same recipe required for controls'},
                    [a.graph,a.initial,'results/malecns_v1/teacher.pt','data/raw/grammar_v1/train.txt','data/raw/grammar_v1/validation.txt'])
    start=time.perf_counter();result['teacher']=evaluate(TeacherAdapter(teacher),va,starts,32)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
    gen=torch.Generator().manual_seed(20000+a.seed);done=0;trace=[];evaluations=[]
    if a.resume and source==ckpath:
        opt.load_state_dict(ck['optimizer']);gen.set_state(ck['data_rng']);done=ck['completed_steps'];trace=ck['trace'];evaluations=ck['evaluations']
    else:evaluations.append({'step':0,**evaluate(model,va,starts,32)})
    result['evaluations']=evaluations
    for step in range(done,a.steps):
        model.train();x,y=sample_batch(tr,1,32,generator=gen)
        with torch.no_grad():tz=teacher(x)
        sz,_=model(x)
        ce=nn.functional.cross_entropy(sz.reshape(-1,256),y.flatten())
        kl=nn.functional.kl_div(nn.functional.log_softmax(sz/2,-1),nn.functional.softmax(tz/2,-1),reduction='sum')/32*4
        loss=.5*ce+.5*kl
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if not np.isfinite(loss.item()):raise RuntimeError('Nonfinite loss')
        trace.append({'step':step+1,'ce':ce.item(),'kl':kl.item(),'loss':loss.item(),'gradient_norm':float(norm)})
        if (step+1)%16==0:print(f'{a.condition} target step={step+1} ce={ce.item():.4f} kl={kl.item():.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        if step+1 in [64,128,256,512] or step+1==a.steps:
            model.eval();metric=evaluate(model,va,starts,32);evaluations.append({'step':step+1,**metric})
            checkpoint={'model':{k:v for k,v in model.state_dict().items() if k not in ('core.src','core.dst')},
                        'topology_buffers_external':True,'config':cfg,'graph':a.graph,'graph_sha256':sha256(a.graph),
                        'condition':a.condition,'seed':a.seed,'completed_steps':step+1,'optimizer':opt.state_dict(),
                        'data_rng':gen.get_state(),'trace':trace,'evaluations':json.loads(json.dumps(evaluations)),
                        'code_commit':result['code_commit'],'evidence_domain':'MaleCNS-based' if a.condition=='real' else 'synthetic control'}
            tmp=ckpath.with_suffix('.pt.tmp');torch.save(checkpoint,tmp);tmp.replace(ckpath)
            result.update(trace=trace,completed_steps=step+1,evaluations=evaluations,checkpoint=str(ckpath),
                          runtime_seconds_this_session=time.perf_counter()-start,complete=step+1==a.steps)
            save_json(out,result);print('VALIDATION',metric,flush=True)
    # Edge ablation after the fixed budget; no topology or initial-state edits.
    weights=model.core.edge_w.detach().clone()
    with torch.no_grad():model.core.edge_w.zero_()
    result['edge_ablated']=evaluate(model,va,starts,32)
    with torch.no_grad():model.core.edge_w.copy_(weights)
    torch.manual_seed(12345);result['sample']=generate(model,b'the cat ',max_new=96,top_k=8,temperature=.7).decode('utf8',errors='replace')
    result['gap_to_teacher']=evaluations[-1]['ce']-result['teacher']['ce']
    result['checkpoint_sha256']=sha256(ckpath);save_json(out,result)


if __name__=='__main__':main()
