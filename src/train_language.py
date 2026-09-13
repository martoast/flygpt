"""Byte-level fixed-topology training with paired sampling and causal evaluation."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .train_memory import build_model
from .provenance import manifest, save_json, sha256


def bytes_from_file(path):return torch.tensor(list(Path(path).read_bytes()),dtype=torch.long)


def sample_batch(data,batch,block,device='cpu',generator=None):
    if len(data)<=block:raise ValueError('Corpus must exceed block length')
    ix=torch.randint(0,len(data)-block,(batch,),generator=generator)
    x=torch.stack([data[i:i+block] for i in ix]).to(device)
    y=torch.stack([data[i+1:i+block+1] for i in ix]).to(device)
    return x,y


def evaluate(model,data,starts,block):
    total=count=correct=0
    with torch.no_grad():
        for start in starts:
            x=data[start:start+block][None];y=data[start+1:start+block+1][None]
            z,_=model(x)
            total+=float(nn.functional.cross_entropy(z.reshape(-1,256),y.flatten(),reduction='sum'))
            correct+=int((z.argmax(-1)==y).sum());count+=y.numel()
    return {'ce':total/count,'bits_per_byte':total/count/np.log(2),'byte_accuracy':correct/count,'n_bytes':count}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--condition',default='real')
    p.add_argument('--text',default='data/raw/grammar_v1/train.txt');p.add_argument('--validation',default='data/raw/grammar_v1/validation.txt')
    p.add_argument('--steps',type=int,default=64);p.add_argument('--block',type=int,default=8);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,default=.003);p.add_argument('--out',required=True)
    p.add_argument('--checkpoint');p.add_argument('--teacher');p.add_argument('--alpha',type=float,default=.5);p.add_argument('--temperature',type=float,default=2.)
    a=p.parse_args();torch.set_num_threads(4)
    config=dict(embed_dim=16,leak=.65,edge_scale=.9,input_fraction=.00615,output_fraction=.00615,inner_steps=2,
                backend='scipy',population_seed=2026,degree_normalize=True)
    result=manifest({**vars(a),'model':config,'optimizer':'AdamW','weight_decay':.01,'evaluation':'fixed 8 windows of 32 bytes; state resets between windows',
                     'tier':'exploratory, hardware-limited; no hyperparameter search'},[a.graph,a.text,a.validation]+([a.teacher] if a.teacher else []))
    start=time.perf_counter();model=build_model(a.graph,a.condition,a.seed,config)
    tr=bytes_from_file(a.text);va=bytes_from_file(a.validation)
    starts=np.random.default_rng(9100).integers(0,len(va)-33,size=8).tolist()
    result['evaluation_starts']=starts;result['untrained']=evaluate(model,va,starts,32)
    freq=torch.bincount(tr,minlength=256).float()+1;freq/=freq.sum()
    validation_targets=torch.cat([va[s+1:s+33] for s in starts])
    result['unigram_ce']=float(-freq[validation_targets].log().mean())
    teacher=None
    if a.teacher:
        from .tinygpt import TinyGPT
        checkpoint=torch.load(a.teacher,map_location='cpu',weights_only=True)
        teacher=TinyGPT(**checkpoint['config']);teacher.load_state_dict(checkpoint['model']);teacher.eval()
        for param in teacher.parameters():param.requires_grad_(False)
        result['teacher']=evaluate(TeacherAdapter(teacher),va,starts,32)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,foreach=False)
    generator=torch.Generator().manual_seed(1000+a.seed);trace=[];model.train()
    for step in range(a.steps):
        x,y=sample_batch(tr,a.batch,a.block,generator=generator);z,_=model(x)
        ce=nn.functional.cross_entropy(z.reshape(-1,256),y.flatten());loss=ce;kl=None
        if teacher is not None:
            with torch.no_grad():tz=teacher(x)
            kl=nn.functional.kl_div(nn.functional.log_softmax(z/a.temperature,-1),nn.functional.softmax(tz/a.temperature,-1),reduction='sum')/x.numel()*a.temperature**2
            loss=a.alpha*ce+(1-a.alpha)*kl
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        trace.append({'step':step+1,'ce':ce.item(),'loss':loss.item(),'kl':None if kl is None else kl.item(),'gradient_norm':float(norm)})
        if not np.isfinite(loss.item()):raise RuntimeError('Nonfinite loss')
        if (step+1)%16==0:print(f'{a.condition} seed={a.seed} step={step+1} ce={ce.item():.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    model.eval();result['validation']=evaluate(model,va,starts,32)
    if a.condition!='gru':
        weights=model.core.edge_w.detach().clone()
        with torch.no_grad():model.core.edge_w.zero_()
        result['edge_ablated']=evaluate(model,va,starts,32)
        with torch.no_grad():model.core.edge_w.copy_(weights)
        del weights
        # Same suffix, differing prefix, fresh state; zero edges must erase identity.
        from .query_flygpt import generate
        torch.manual_seed(12345)
        result['sample']=generate(model,b'the cat ',max_new=64,top_k=8,temperature=.7).decode('utf8',errors='replace')
        with torch.no_grad():
            z1,h1=model(torch.tensor([list(b'the cat sees a ')]));z2,h2=model(torch.tensor([list(b'the dog sees a ')]))
        result['same_suffix_probe']={'prompts':['the cat sees a ','the dog sees a '],
                                     'logit_l2':float((z1[:,-1]-z2[:,-1]).norm()),'state_l2':float((h1-h2).norm()),
                                     'note':'dependence diagnostic; not semantic correctness'}
        ckpt=Path(a.checkpoint or str(Path(a.out).with_suffix('.pt')));ckpt.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'config':config,'graph':a.graph,'graph_sha256':sha256(a.graph),
                    'condition':a.condition,'seed':a.seed,'training':result['config'],'code_commit':result['code_commit'],
                    'evidence_domain':'MaleCNS-based' if a.condition=='real' else 'synthetic topology control'},ckpt)
        result['checkpoint']={'path':str(ckpt),'sha256':sha256(ckpt)}
    if teacher is not None:
        denominator=result['untrained']['ce']-result['teacher']['ce']
        result['distillation_efficiency']=None if denominator<=0 else (result['untrained']['ce']-result['validation']['ce'])/denominator
        with torch.no_grad():
            kl_sum=0.;count=0
            for s in starts:
                x=va[s:s+32][None];tz=teacher(x);sz,_=model(x)
                kl_sum+=float(nn.functional.kl_div(nn.functional.log_softmax(sz,-1),nn.functional.softmax(tz,-1),reduction='sum'));count+=x.numel()
            result['teacher_student_kl_per_byte']=kl_sum/count
    result.update(parameters=sum(p.numel() for p in model.parameters()),training_tokens=a.steps*a.batch*a.block,trace=trace,
                  runtime_seconds=time.perf_counter()-start)
    save_json(a.out,result);print({k:result[k] for k in ['validation','runtime_seconds']},flush=True)


class TeacherAdapter(nn.Module):
    def __init__(self,teacher):super().__init__();self.teacher=teacher
    def forward(self,x,h=None):return self.teacher(x),None


if __name__=='__main__':main()
