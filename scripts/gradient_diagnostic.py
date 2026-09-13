"""Read-only component gradient diagnostics at a saved fixed-graph checkpoint."""
import argparse
import time
from pathlib import Path
import torch
from torch import nn
from src.query_flygpt import load_model
from src.tinygpt import TinyGPT
from src.train_language import bytes_from_file
from src.provenance import manifest,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,default=256);a=p.parse_args();torch.set_num_threads(4)
    path=Path(f'results/malecns_v1/target/snapshots/real_0_step_{a.step:04d}.pt')
    model=load_model('data/processed/malecns.npz',path,'cpu')
    tc=torch.load('results/malecns_v1/teacher.pt',map_location='cpu',weights_only=True)
    teacher=TinyGPT(**tc['config']).eval();teacher.load_state_dict(tc['model']);del tc
    data=bytes_from_file('data/raw/grammar_v1/validation.txt')
    start=time.perf_counter();x=data[:32][None];y=data[1:33]
    with torch.no_grad():tz=teacher(x)
    z,_=model(x);ce=nn.functional.cross_entropy(z[0],y)
    kl=nn.functional.kl_div(nn.functional.log_softmax(z/2,-1),nn.functional.softmax(tz/2,-1),reduction='sum')/32*4
    (.5*ce+.5*kl).backward()
    stats={}
    for name,param in model.named_parameters():
        g=param.grad
        stats[name]={'parameters':param.numel(),'weight_l2':float(param.detach().norm()),
                     'gradient_l2':None if g is None else float(g.norm()),
                     'gradient_rms':None if g is None else float(g.square().mean().sqrt()),
                     'nonzero_gradient_fraction':0. if g is None else float((g!=0).float().mean()),
                     'gradient_finite':True if g is None else bool(torch.isfinite(g).all())}
    record=manifest({'checkpoint':str(path),'validation_bytes':'first 32 input bytes; fixed diagnostic, no update',
                     'loss':'0.5 CE + 0.5 T^2 KL at T=2'},[path,'data/raw/grammar_v1/validation.txt'])
    record.update(ce=float(ce.detach()),teacher_kl_temperature2_scaled=float(kl.detach()),components=stats,
                  weights_updated=False,runtime_seconds=time.perf_counter()-start)
    save_json(f'results/malecns_v1/target/gradient_components_{a.step}.json',record)


if __name__=='__main__':main()
