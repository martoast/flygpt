import argparse, json
from pathlib import Path
import numpy as np, torch
from .graphs import load_npz, degree_preserving_rewire
from .graph_lm import GraphLanguageModel
from .tinygpt import TinyGPT
from .train_language import bytes_from_file, sample_batch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--graph',required=True); p.add_argument('--text',required=True); p.add_argument('--teacher',required=True); p.add_argument('--control',choices=['real','rewired'],default='real'); p.add_argument('--steps',type=int,default=1000); p.add_argument('--block',type=int,default=64); p.add_argument('--batch',type=int,default=8); p.add_argument('--seed',type=int,default=0); p.add_argument('--temperature',type=float,default=2.0); p.add_argument('--alpha',type=float,default=.5); p.add_argument('--out',default='results/distill.json'); a=p.parse_args()
    torch.manual_seed(a.seed); dev='cuda' if torch.cuda.is_available() else 'cpu'; data=bytes_from_file(a.text); cut=int(.9*len(data)); tr,va=data[:cut],data[cut:]
    teacher=TinyGPT().to(dev); teacher.load_state_dict(torch.load(a.teacher,map_location=dev)); teacher.eval()
    n,s,d,_=load_npz(a.graph)
    if a.control=='rewired': s,d=degree_preserving_rewire(s,d,n_swaps=min(10*len(s),5_000_000),seed=a.seed+777)
    student=GraphLanguageModel(n,s,d).to(dev); opt=torch.optim.AdamW(student.parameters(),lr=1e-3); T=a.temperature
    for step in range(a.steps):
        x,y=sample_batch(tr,a.batch,a.block,dev)
        with torch.no_grad(): tz=teacher(x)
        sz,_=student(x)
        ce=torch.nn.functional.cross_entropy(sz.reshape(-1,256),y.reshape(-1))
        kl=torch.nn.functional.kl_div(torch.log_softmax(sz/T,-1),torch.softmax(tz/T,-1),reduction='batchmean')*(T*T)/x.shape[1]
        loss=a.alpha*ce+(1-a.alpha)*kl; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),1); opt.step()
    ce_s=[]; ce_t=[]; klv=[]
    with torch.no_grad():
        for _ in range(20):
            x,y=sample_batch(va,a.batch,a.block,dev); tz=teacher(x); sz,_=student(x)
            ce_s.append(torch.nn.functional.cross_entropy(sz.reshape(-1,256),y.reshape(-1)).item()); ce_t.append(torch.nn.functional.cross_entropy(tz.reshape(-1,256),y.reshape(-1)).item()); klv.append((torch.nn.functional.kl_div(torch.log_softmax(sz/T,-1),torch.softmax(tz/T,-1),reduction='batchmean')*(T*T)/x.shape[1]).item())
    out={'control':a.control,'student_ce':float(np.mean(ce_s)),'teacher_ce':float(np.mean(ce_t)),'teacher_student_kl':float(np.mean(klv))}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
