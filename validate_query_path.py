import json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from src.graphs import er_graph
from src.model import SparseGraphRNN

SEQS=["Q:a?\nA:x\n","Q:b?\nA:y\n"]
chars=sorted(set(''.join(SEQS)))
stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}

class CharBrain(torch.nn.Module):
    def __init__(self,n,src,dst):
        super().__init__(); self.emb=torch.nn.Embedding(len(chars),8)
        self.core=SparseGraphRNN(n,src,dst,8,len(chars),leak=.65,edge_scale=.08,input_fraction=.25,output_fraction=.25,inner_steps=3)
    def forward(self,x,h=None): return self.core(self.emb(x),h)

seed=123
torch.manual_seed(seed); np.random.seed(seed)
n=32; e=320
src,dst=er_graph(n,e,seed)
A=sparse.coo_matrix((np.ones(e,dtype=np.float32),(src,dst)),shape=(n,n)).tocsr()
sparse.save_npz('/mnt/data/flygpt_v01/data/processed/query_path_demo.npz',A)
X=torch.tensor([[stoi[c] for c in s[:-1]] for s in SEQS],dtype=torch.long)
Y=torch.tensor([[stoi[c] for c in s[1:]] for s in SEQS],dtype=torch.long)
m=CharBrain(n,src,dst); opt=torch.optim.AdamW(m.parameters(),lr=.02,weight_decay=0)
losses=[]
for step in range(1200):
    z,_=m(X); loss=torch.nn.functional.cross_entropy(z.reshape(-1,len(chars)),Y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),2); opt.step()
    if step%100==0: losses.append(float(loss))
    if loss.item()<0.002: break

def next_char(prompt, model):
    x=torch.tensor([[stoi[c] for c in prompt]],dtype=torch.long)
    with torch.no_grad(): z,h=model(x)
    return itos[int(z[0,-1].argmax())], z[0,-1].softmax(-1)

normal={}
for p in ['Q:a?\nA:','Q:b?\nA:']:
    c,pr=next_char(p,m); normal[p]=c
saved=m.core.edge_w.detach().clone()
with torch.no_grad(): m.core.edge_w.zero_()
ablated={}
for p in ['Q:a?\nA:','Q:b?\nA:']:
    c,pr=next_char(p,m); ablated[p]=c
with torch.no_grad(): m.core.edge_w.copy_(saved)

out={'seed':seed,'nodes':n,'edges':e,'vocab':chars,'steps':step+1,'final_loss':float(loss),'normal':normal,'edge_ablated':ablated,'loss_trace':losses}
Path('/mnt/data/flygpt_v01/results/query_path_validation.json').write_text(json.dumps(out,indent=2))
torch.save({'model':m.state_dict(),'stoi':stoi,'itos':itos,'graph':'data/processed/query_path_demo.npz','config':{'nodes':n,'edges':e}},'/mnt/data/flygpt_v01/results/query_path_demo.pt')
print(json.dumps(out,indent=2))
