import argparse, json, random
from pathlib import Path
import numpy as np, torch
from scipy import sparse
from .graphs import load_npz, degree_preserving_rewire, er_graph
from .graph_lm import GraphLanguageModel

def bytes_from_file(path): return torch.tensor(list(Path(path).read_bytes()),dtype=torch.long)
def sample_batch(data,batch,block,device):
    ix=torch.randint(0,len(data)-block-1,(batch,))
    x=torch.stack([data[i:i+block] for i in ix]).to(device); y=torch.stack([data[i+1:i+block+1] for i in ix]).to(device)
    return x,y

def main():
    p=argparse.ArgumentParser(); p.add_argument('--graph',required=True); p.add_argument('--text',required=True); p.add_argument('--control',choices=['real','rewired','er'],default='real'); p.add_argument('--steps',type=int,default=1000); p.add_argument('--block',type=int,default=64); p.add_argument('--batch',type=int,default=8); p.add_argument('--seed',type=int,default=0); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--out',default='results/lm.json'); p.add_argument('--checkpoint',default=None); p.add_argument('--embed-dim',type=int,default=64); p.add_argument('--leak',type=float,default=.8); p.add_argument('--input-fraction',type=float,default=.25); p.add_argument('--output-fraction',type=float,default=.25); p.add_argument('--inner-steps',type=int,default=3); a=p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed); dev='cuda' if torch.cuda.is_available() else 'cpu'
    n,s,d,_=load_npz(a.graph)
    if a.control=='rewired': s,d=degree_preserving_rewire(s,d,n_swaps=min(10*len(s),5_000_000),seed=a.seed+777)
    elif a.control=='er': s,d=er_graph(n,len(s),seed=a.seed+777)
    data=bytes_from_file(a.text); cut=int(.9*len(data)); tr,va=data[:cut],data[cut:]
    m=GraphLanguageModel(n,s,d,embed_dim=a.embed_dim,leak=a.leak,input_fraction=a.input_fraction,output_fraction=a.output_fraction,inner_steps=a.inner_steps).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr)
    for step in range(a.steps):
        x,y=sample_batch(tr,a.batch,a.block,dev); z,_=m(x); loss=torch.nn.functional.cross_entropy(z.reshape(-1,256),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    m.eval(); vals=[]
    with torch.no_grad():
        for _ in range(20):
            x,y=sample_batch(va,a.batch,a.block,dev); z,_=m(x); vals.append(torch.nn.functional.cross_entropy(z.reshape(-1,256),y.reshape(-1)).item())
    out={'control':a.control,'seed':a.seed,'n_nodes':n,'n_edges':len(s),'val_ce':float(np.mean(vals)),'bits_per_byte':float(np.mean(vals)/np.log(2))}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2));
    ckpt_path=a.checkpoint or str(Path(a.out).with_suffix('.pt')); Path(ckpt_path).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':m.state_dict(),'config':{'vocab_size':256,'embed_dim':a.embed_dim,'leak':a.leak,'edge_scale':.02,'input_fraction':a.input_fraction,'output_fraction':a.output_fraction,'inner_steps':a.inner_steps},'graph':a.graph,'control':a.control,'seed':a.seed},ckpt_path); out['checkpoint']=ckpt_path; Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
