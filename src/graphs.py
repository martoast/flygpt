import numpy as np
from scipy import sparse

def er_graph(n, e, seed=0):
    rng=np.random.default_rng(seed)
    pairs=set()
    while len(pairs)<e:
        s=int(rng.integers(n)); d=int(rng.integers(n))
        if s!=d: pairs.add((s,d))
    a=np.array(list(pairs),dtype=np.int64)
    return a[:,0],a[:,1]

def clustered_graph(n, e, communities=8, seed=0, p_in=.85):
    rng=np.random.default_rng(seed); c=np.arange(n)%communities; pairs=set()
    members=[np.where(c==i)[0] for i in range(communities)]
    while len(pairs)<e:
        s=int(rng.integers(n))
        if rng.random()<p_in:
            d=int(rng.choice(members[c[s]]))
        else: d=int(rng.integers(n))
        if s!=d: pairs.add((s,d))
    a=np.array(list(pairs),dtype=np.int64); return a[:,0],a[:,1]

def degree_preserving_rewire(src,dst,n_swaps=None,seed=0):
    # directed double-edge swaps: (a->b,c->d) => (a->d,c->b), preserving in/out degree
    rng=np.random.default_rng(seed); src=src.copy(); dst=dst.copy(); m=len(src)
    if n_swaps is None: n_swaps=10*m
    edges=set(zip(src.tolist(),dst.tolist())); done=0
    for _ in range(n_swaps*5):
        if done>=n_swaps: break
        i,j=rng.integers(m,size=2)
        if i==j: continue
        a,b=int(src[i]),int(dst[i]); c,d=int(src[j]),int(dst[j])
        if a==d or c==b: continue
        if (a,d) in edges or (c,b) in edges: continue
        edges.remove((a,b)); edges.remove((c,d)); edges.add((a,d)); edges.add((c,b))
        dst[i]=d; dst[j]=b; done+=1
    return src,dst

def load_npz(path, binary=True):
    with np.load(path) as data:
        if 'src' in data:
            src=data['src'].astype(np.int64);dst=data['dst'].astype(np.int64)
            return int(data['n']),src,dst,np.ones(len(src),dtype=np.float32)
    mat=sparse.load_npz(path).tocoo()
    src=mat.row.astype(np.int64); dst=mat.col.astype(np.int64)
    val=np.ones_like(mat.data,dtype=np.float32) if binary else mat.data.astype(np.float32)
    return mat.shape[0],src,dst,val
