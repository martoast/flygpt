import json, time, torch, numpy as np
from src.graphs import clustered_graph, degree_preserving_rewire, er_graph
from src.model import SparseGraphRNN
from src.tasks import delayed_bit_batch

def run(kind,seed,steps=50,n=192,e=1400):
    torch.manual_seed(seed); np.random.seed(seed)
    base_s,base_d=clustered_graph(n,e,communities=8,seed=123)
    if kind=='clustered': s,d=base_s,base_d
    elif kind=='rewired': s,d=degree_preserving_rewire(base_s,base_d,n_swaps=5*e,seed=321)
    else: s,d=er_graph(n,e,seed=321)
    m=SparseGraphRNN(n,s,d,2,2,leak=.7,edge_scale=.04)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5)
    for step in range(steps):
        x,y=delayed_bit_batch(32,seq_len=8)
        z,_=m(x); loss=torch.nn.functional.cross_entropy(z[:,-1],y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    with torch.no_grad():
        acc=[]; losses=[]
        for _ in range(12):
            x,y=delayed_bit_batch(64,seq_len=8); z,_=m(x); l=torch.nn.functional.cross_entropy(z[:,-1],y)
            losses.append(l.item()); acc.append((z[:,-1].argmax(-1)==y).float().mean().item())
    return {'graph':kind,'seed':seed,'val_loss':float(np.mean(losses)),'val_acc':float(np.mean(acc)),'nodes':n,'edges':e,'steps':steps}

if __name__=='__main__':
    t=time.time(); rows=[]
    for k in ['clustered','rewired','er']:
        for seed in [0,1,2]:
            r=run(k,seed); rows.append(r); print(r,flush=True)
    import pathlib, csv
    pathlib.Path('results').mkdir(exist_ok=True)
    with open('results/smoke.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={k:{'acc_mean':float(np.mean([r['val_acc'] for r in rows if r['graph']==k])), 'loss_mean':float(np.mean([r['val_loss'] for r in rows if r['graph']==k]))} for k in ['clustered','rewired','er']}
    summary['runtime_s']=time.time()-t
    pathlib.Path('results/smoke_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
