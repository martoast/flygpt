"""Matched-budget multitask memory screening on a full fixed graph or GRU.

One model is trained on a mixture of six tasks, then evaluated at each delay.
These short screens measure feasibility; they are not confirmatory estimates.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .graphs import load_npz
from .graph_lm import GraphLanguageModel
from .provenance import manifest, save_json

TASKS=['delayed_bit','delayed_symbol','copy','associative','grammar','parity']
ANSWERS={'delayed_bit':(32,2),'delayed_symbol':(40,8),'copy':(48,4),'associative':(72,4),'grammar':(88,3),'parity':(104,2)}


class GRULanguageModel(nn.Module):
    def __init__(self,hidden=512,embed_dim=16):
        super().__init__();self.embed=nn.Embedding(256,embed_dim);self.core=nn.GRU(embed_dim,hidden,batch_first=True);self.head=nn.Linear(hidden,256)
    def forward(self,x,h=None):
        z,h=self.core(self.embed(x),h);return self.head(z),h


def parameter_matched_gru(target,embed_dim=16):
    # GRU: 3H^2 + 3EH + 6H; decoder: 256H+256; embedding: 256E.
    linear=3*embed_dim+6+256
    return max(1,int(round((-linear+np.sqrt(linear**2+12*(target-256*embed_dim-256)))/6)))


def build_model(graph,condition,seed,config):
    torch.manual_seed(seed)
    n,s,d,_=load_npz(graph)
    if condition=='gru':
        ni=max(1,int(n*config['input_fraction']));no=max(1,int(n*config['output_fraction']))
        target=len(s)+2*n+ni*config['embed_dim']+no*256+256+256*config['embed_dim']
        return GRULanguageModel(parameter_matched_gru(target,config['embed_dim']),config['embed_dim'])
    return GraphLanguageModel(n,s,d,**config)


def task_batch(task,delay,batch,rng):
    """Targets cannot be read from the shared final query symbol (byte 250)."""
    if task=='delayed_bit':
        values=rng.integers(0,2,batch);x=np.full((batch,delay+2),240);x[:,0]=values+32;x[:,-1]=250
        y=values[:,None]+32;positions=[x.shape[1]-1];chance=.5
    elif task=='delayed_symbol':
        values=rng.integers(0,8,batch);x=np.full((batch,delay+2),240);x[:,0]=values+40;x[:,-1]=250
        y=values[:,None]+40;positions=[x.shape[1]-1];chance=.125
    elif task=='copy':
        values=rng.integers(0,4,(batch,3))+48;x=np.full((batch,3+delay+3),240);x[:,:3]=values;x[:,-3:]=250
        y=values;positions=list(range(x.shape[1]-3,x.shape[1]));chance=.25
    elif task=='associative':
        keys=np.array([64,65]);values=np.stack([rng.permutation(4)[:2]+72 for _ in range(batch)])
        query=rng.integers(0,2,batch);x=np.full((batch,6+delay),240)
        x[:,0]=keys[0];x[:,1]=values[:,0];x[:,2]=keys[1];x[:,3]=values[:,1]
        x[:,-2]=keys[query];x[:,-1]=250;y=values[np.arange(batch),query,None];positions=[x.shape[1]-1];chance=.25
    elif task=='grammar':
        # Three-state modular automaton; query predicts the terminal state.
        transitions=rng.integers(0,3,(batch,delay+1));x=np.concatenate([transitions+80,np.full((batch,1),250)],axis=1)
        y=(transitions.sum(axis=1)%3+88)[:,None];positions=[x.shape[1]-1];chance=1/3
    elif task=='parity':
        bits=rng.integers(0,2,(batch,delay+1));x=np.concatenate([bits+96,np.full((batch,1),250)],axis=1)
        y=(bits.sum(axis=1)%2+104)[:,None];positions=[x.shape[1]-1];chance=.5
    else:raise ValueError(task)
    # Task identifier precedes all content; same across train/test and conditions.
    x=np.concatenate([np.full((batch,1),200+TASKS.index(task)),x],axis=1)
    return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long),[p+1 for p in positions],chance


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--condition',default='real');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=24);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    config=dict(embed_dim=16,leak=.65,edge_scale=.9,input_fraction=.00615,output_fraction=.00615,inner_steps=2,
                backend='scipy',population_seed=2026,degree_normalize=True)
    result=manifest({**vars(a),'model':config,'train_delays':[0,1,2],'eval_delays':[0,2,4,8],
                     'batch':1,'evaluation_batch':8,'lr':.003,'optimizer':'AdamW','weight_decay':.01,
                     'interpretation':'budget-limited multitask feasibility screen; not confirmatory'},[a.graph])
    start=time.perf_counter();m=build_model(a.graph,a.condition,a.seed,config);opt=torch.optim.AdamW(m.parameters(),lr=.003,foreach=False)
    rng=np.random.default_rng(1000+a.seed);trace=[];tokens=0
    for step in range(a.steps):
        task=TASKS[step%len(TASKS)];delay=int(rng.choice([0,1,2]));x,y,pos,chance=task_batch(task,delay,1,rng)
        z,_=m(x);base,classes=ANSWERS[task];loss=nn.functional.cross_entropy(z[:,pos,base:base+classes].reshape(-1,classes),y.flatten()-base)
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();tokens+=x.numel()
        trace.append({'step':step+1,'task':task,'delay':delay,'loss':loss.item(),'gradient_norm':float(norm)})
        if not np.isfinite(loss.item()): raise RuntimeError('Nonfinite loss')
    m.eval();scores=[]
    # Independent, fixed paired evaluation RNG across graph conditions.
    ev=np.random.default_rng(9000+a.seed)
    with torch.no_grad():
        for task in TASKS:
            for delay in [0,2,4,8]:
                x,y,pos,chance=task_batch(task,delay,8,ev);z,_=m(x);base,classes=ANSWERS[task];logits=z[:,pos,base:base+classes];y=y-base
                scores.append({'task':task,'delay':delay,'n_targets':y.numel(),'chance':chance,
                               'accuracy':float((logits.argmax(-1)==y).float().mean()),
                               'ce':float(nn.functional.cross_entropy(logits.reshape(-1,classes),y.flatten()))})
    result.update(parameters=sum(p.numel() for p in m.parameters()),train_tokens=tokens,trace=trace,scores=scores,
                  runtime_seconds=time.perf_counter()-start)
    save_json(a.out,result);print(f'{a.condition} seed {a.seed}: {result["runtime_seconds"]:.1f}s',flush=True)


if __name__=='__main__':main()
