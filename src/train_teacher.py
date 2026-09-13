"""Small reproducible byte transformer teacher, with compatible checkpoints."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from .tinygpt import TinyGPT
from .train_language import bytes_from_file, sample_batch, evaluate, TeacherAdapter
from .provenance import manifest,save_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--text',default='data/raw/grammar_v1/train.txt');p.add_argument('--validation',default='data/raw/grammar_v1/validation.txt')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--block',type=int,default=64);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--out',default='results/malecns_v1/teacher.pt');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed);start=time.perf_counter()
    config=dict(vocab_size=256,d_model=128,n_head=4,n_layer=3,max_len=64,dropout=0.)
    result=manifest({**vars(a),'model':config,'lr':.0003},[a.text,a.validation])
    tr=bytes_from_file(a.text);va=bytes_from_file(a.validation);m=TinyGPT(**config)
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,foreach=False);generator=torch.Generator().manual_seed(5000+a.seed);trace=[]
    for step in range(a.steps):
        x,y=sample_batch(tr,a.batch,a.block,generator=generator);z=m(x);loss=torch.nn.functional.cross_entropy(z.reshape(-1,256),y.flatten())
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if (step+1)%100==0:trace.append({'step':step+1,'ce':loss.item()});print(trace[-1],flush=True)
    m.eval();starts=np.random.default_rng(9100).integers(0,len(va)-33,size=8).tolist()
    result.update(validation=evaluate(TeacherAdapter(m),va,starts,32),trace=trace,training_tokens=a.steps*a.batch*a.block,
                  parameters=sum(p.numel() for p in m.parameters()),runtime_seconds=time.perf_counter()-start)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':m.state_dict(),'config':config,'manifest':result},out)
    result['checkpoint_sha256']=sha256(out);save_json(out.with_suffix('.json'),result);print(result['validation'],flush=True)


if __name__=='__main__':main()
