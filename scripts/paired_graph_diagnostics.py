"""Post-hoc binary topology diagnostics; never modifies training graphs or models."""
import gc
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
import torch
from src.provenance import sha256,save_json,manifest

OUT=Path('results/compiler_v1/full_report/graph_properties')

def characterize(path,label):
    out=OUT/f'{label}.json'
    if out.exists():
        r=json.loads(out.read_text());assert r['inputs'][path]==sha256(path);return
    start=time.monotonic()
    with np.load(path) as stored:
        if 'src' in stored:
            n=int(stored['n']);s=stored['src'];d=stored['dst']
            a=sparse.csr_matrix((np.ones(len(s),np.float32),(s,d)),shape=(n,n))
            del s,d
        else:a=sparse.load_npz(path).tocsr()
    a.data=np.ones(a.nnz,np.float32);n=a.shape[0]
    nscc,labels=csgraph.connected_components(a,directed=True,connection='strong');sizes=np.bincount(labels)
    loops=int(np.count_nonzero(a.diagonal()));reciprocal=int(a.multiply(a.T).sum())-loops
    x=np.ones(n)/np.sqrt(n);estimate=0.
    for _ in range(80):
        y=a@x;estimate=float(np.linalg.norm(y));x=y/max(estimate,1e-30)
    residual=float(np.linalg.norm(a@x-estimate*x)/max(estimate,1e-30))
    population=torch.randperm(n,generator=torch.Generator().manual_seed(2026)).numpy();count=max(1,int(n*.00615))
    inputs=population[:count];outputs=population[-count:]
    dist=csgraph.dijkstra(a,directed=True,unweighted=True,indices=inputs,min_only=True)[outputs]
    finite=dist[np.isfinite(dist)];values,counts=np.unique(finite,return_counts=True)
    result=manifest({'label':label,'binary':True,'power_iterations':80,'population_seed':2026,
        'scope':'Structural estimates, not trained recurrent Jacobian stability'},[path,'configs/paired_ce_analysis_v1.json',__file__])
    result.update(n_nodes=n,n_edges=a.nnz,strongly_connected_components=nscc,giant_scc_fraction=float(sizes.max()/n),
        reciprocal_fraction=reciprocal/max(a.nnz-loops,1),binary_perron_estimate=estimate,perron_relative_residual=residual,
        input_output_mean_min_hops=float(finite.mean()) if len(finite) else None,output_reachable_fraction=float(len(finite)/count),
        input_output_min_hop_counts={str(int(v)):int(c) for v,c in zip(values,counts)},runtime_seconds=time.monotonic()-start)
    save_json(out,result);print(label,{k:result[k] for k in ('giant_scc_fraction','reciprocal_fraction','binary_perron_estimate','perron_relative_residual','input_output_mean_min_hops','output_reachable_fraction','runtime_seconds')},flush=True)
    del a;gc.collect()

if __name__=='__main__':
    torch.set_num_threads(1)
    characterize('data/processed/malecns.npz','real')
    for seed in range(5):characterize(f'data/processed/controls/rewired_{777+seed}.npz',f'rewired_{seed}')
