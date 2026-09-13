"""Run each backend in a separate process to bound memory and isolate failures."""
import argparse
import resource
import time
import numpy as np
import torch
from src.graphs import load_npz
from src.model import SparseGraphRNN
from src.provenance import manifest, save_json

p=argparse.ArgumentParser();p.add_argument('--backend',choices=['scipy','scatter','coo','csr'],required=True)
p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=1);p.add_argument('--threads',type=int,default=4)
a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(0)
path='data/processed/malecns.npz';n,s,d,_=load_npz(path)
result=manifest(vars(a),[path]);result['evidence_domain']='MaleCNS full annotated-neuron graph'
try:
    m=SparseGraphRNN(n,s,d,8,256,input_fraction=.001,output_fraction=.001,backend='scipy' if a.backend=='scipy' else 'scatter',degree_normalize=True).to(a.device)
    h=torch.randn(a.batch,n,device=a.device,requires_grad=True)
    def step():
        if a.backend in ('coo','csr'):
            adj=torch.sparse_coo_tensor(torch.stack((m.dst,m.src)),m.edge_w,(n,n)).coalesce()
            if a.backend=='csr':adj=adj.to_sparse_csr()
            y=torch.sparse.mm(adj,h.T).T
        else:y=m.recurrent(h)
        y.square().mean().backward();m.zero_grad(set_to_none=True);h.grad=None
        if a.device=='mps':torch.mps.synchronize()
    start=time.perf_counter();step();result['warmup_seconds']=time.perf_counter()-start
    times=[]
    for _ in range(3):
        start=time.perf_counter();step();times.append(time.perf_counter()-start)
    result.update(seconds_forward_backward_tick=times,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,n_nodes=n,n_edges=len(s))
except Exception as e:
    result['error']=f'{type(e).__name__}: {str(e)[:1500]}'
save_json(f'results/malecns_v1/benchmark_{a.backend}_{a.device}_b{a.batch}_t{a.threads}.json',result);print(result,flush=True)
