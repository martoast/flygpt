"""Sparse structural characterization; approximations are explicitly labelled."""
import argparse
import time
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from .provenance import manifest, save_json


def distribution(x):
    v,c=np.unique(x,return_counts=True)
    return {'min':float(np.min(x)), 'mean':float(np.mean(x)), 'max':float(np.max(x)),
            'quantiles':dict(zip(['0','25','50','75','90','95','99','100'],np.percentile(x,[0,25,50,75,90,95,99,100]).tolist())),
            'histogram':{'value':v.tolist(),'count':c.tolist()}}


def characterize(path,out,seed=0,sources=32,wedges=20000):
    start=time.perf_counter();rng=np.random.default_rng(seed)
    a=sparse.load_npz(path).tocsr();n=a.shape[0];binary=a.copy();binary.data=np.ones(a.nnz,dtype=np.float32)
    indeg=np.asarray(binary.sum(axis=0)).ravel();outdeg=np.diff(binary.indptr)
    nscc,labels=csgraph.connected_components(binary,directed=True,connection='strong')
    sizes=np.bincount(labels); giant=np.flatnonzero(labels==sizes.argmax())
    sub=binary[giant][:,giant].tocsr()
    sample=rng.choice(len(giant),min(sources,len(giant)),replace=False)
    distances=csgraph.shortest_path(sub,directed=True,unweighted=True,indices=sample)
    finite=distances[np.isfinite(distances)&(distances>0)]
    # Monte Carlo global transitivity on the undirected simple projection.
    u=binary.maximum(binary.T);u.setdiag(0);u.eliminate_zeros();u.sort_indices()
    degree=np.diff(u.indptr);counts=degree.astype(float)*(degree-1)/2
    centers=rng.choice(n,size=wedges,p=counts/counts.sum());closed=0
    for center in centers:
        neighbors=u.indices[u.indptr[center]:u.indptr[center+1]]
        x,y=rng.choice(neighbors,2,replace=False)
        row=u.indices[u.indptr[x]:u.indptr[x+1]];where=np.searchsorted(row,y)
        closed+=int(where<len(row) and row[where]==y)
    # Perron estimate of binary adjacency, and operator 2-norm estimate.
    x=np.ones(n)/np.sqrt(n);radius=0.
    for _ in range(80):
        y=binary@x;radius=float(np.linalg.norm(y));x=y/max(radius,1e-30)
    residual=float(np.linalg.norm(binary@x-radius*x)/max(radius,1e-30))
    x=np.ones(n)/np.sqrt(n)
    for _ in range(40):
        y=binary.T@(binary@x);x=y/max(np.linalg.norm(y),1e-30)
    norm=float(np.linalg.norm(binary@x))
    loops=int(np.count_nonzero(binary.diagonal()))
    reciprocal=int(binary.multiply(binary.T).sum())-loops
    result=manifest({'seed':seed,'shortest_path_sources':sources,'wedge_samples':wedges},[path])
    result.update(evidence_domain='MaleCNS-based computational graph',n_nodes=n,n_edges=a.nnz,
                  synaptic_contacts=int(a.data.sum()),self_edges=loops,density_excluding_loops=(a.nnz-loops)/(n*(n-1)),
                  in_degree=distribution(indeg),out_degree=distribution(outdeg),synapse_count=distribution(a.data),
                  strongly_connected_components=nscc,scc_size_distribution=distribution(sizes),giant_scc_size=int(sizes.max()),
                  isolated_nodes=int(np.sum((indeg==0)&(outdeg==0))),
                  shortest_path={'scope':'giant SCC, directed, unweighted, sampled sources, excludes self distances',
                                 'sources_body_indices':giant[sample].tolist(),'mean':float(finite.mean()),'max_sampled':float(finite.max())},
                  motifs={'reciprocal_directed_edges_excluding_loops':reciprocal,'reciprocal_dyads':reciprocal//2,
                          'undirected_global_transitivity_estimate':closed/wedges,'wedge_samples':wedges,
                          'monte_carlo_standard_error':float(np.sqrt((closed/wedges)*(1-closed/wedges)/wedges)),
                          'scope':'dyads exact; wedge closure sampled; full directed triad census not computed'},
                  stability={'binary_adjacency_perron_estimate':radius,'relative_residual':residual,
                             'binary_operator_2norm_estimate':norm,'note':'structural only; not trained Jacobian stability'},
                  runtime_seconds=time.perf_counter()-start)
    save_json(out,result)
    print({k:result[k] for k in ['n_nodes','n_edges','synaptic_contacts','giant_scc_size','runtime_seconds']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--out',default='results/malecns_v1/graph_statistics.json')
    a=p.parse_args();characterize(a.graph,a.out)
