"""Explicit directed nulls. Configuration control is a stub-matched multigraph."""
import argparse
import time
from pathlib import Path
import numpy as np
from numba import njit
from .graphs import load_npz
from .provenance import manifest, save_json, sha256


@njit(cache=True)
def swaps(src,dst,n,count,seed,rebuild_every=-1):
    np.random.seed(seed)
    present=set(src.astype(np.int64)*n+dst)
    done=0;attempts=0;m=len(src)
    if rebuild_every<0:rebuild_every=m
    while done<count and attempts<count*20:
        attempts+=1;i=np.random.randint(m);j=np.random.randint(m)
        a=src[i];b=dst[i];c=src[j];d=dst[j]
        if i==j or a==b or c==d or a==d or c==b: continue
        k1=a*n+d;k2=c*n+b
        if k1 in present or k2 in present: continue
        present.remove(a*n+b);present.remove(c*n+d);present.add(k1);present.add(k2)
        dst[i]=d;dst[j]=b;done+=1
        if rebuild_every>0 and done%rebuild_every==0:
            # Numba sets resize by live size, not tombstone fill. Rebuild the
            # exact same membership set; this consumes no random numbers.
            present=set(src.astype(np.int64)*n+dst)
            print('successful swaps',done,'of',count)
    return dst,done,attempts


def make_control(n,src,dst,condition,seed,swap_factor=10):
    rng=np.random.default_rng(seed);s=src.copy();d=dst.copy();info={}
    if condition=='rewired':
        d,done,attempts=swaps(s,d,n,swap_factor*len(s),seed)
        if done<swap_factor*len(s): raise RuntimeError('Swap budget not reached')
        info.update(successful_swaps=done,attempts=attempts,swap_factor=swap_factor,self_loops='preserved in place')
    elif condition=='configuration':
        d=rng.permutation(d)
        info.update(multigraph=True,note='Independent directed stub matching; parallel edges retain independent parameters; loops allowed')
    elif condition=='er':
        # Uniform G(N,M), conditional on the biological number of self-loops.
        m=len(s);loops=int(np.sum(s==d));keys=np.empty(0,dtype=np.int64)
        while len(keys)<m-loops:
            new=rng.integers(0,n*n,size=int((m-loops-len(keys))*1.05)+100)
            new=new[new//n!=new%n];keys=np.unique(np.concatenate([keys,new]))
        keys=rng.choice(keys,size=m-loops,replace=False)
        self_nodes=rng.choice(n,size=loops,replace=False)
        keys=np.concatenate([keys,self_nodes*n+self_nodes]);s=keys//n;d=keys%n
        info.update(self_loop_count=loops)
    else: raise ValueError(condition)
    if condition in ('rewired','configuration'):
        assert np.array_equal(np.bincount(s,minlength=n),np.bincount(src,minlength=n))
        assert np.array_equal(np.bincount(d,minlength=n),np.bincount(dst,minlength=n))
    assert len(s)==len(src)
    original=np.sort(src.astype(np.int64)*n+dst);new=s.astype(np.int64)*n+d
    unique=np.unique(new)
    info.update(n_nodes=n,n_edge_parameters=len(s),unique_directed_pairs=len(unique),self_edges=int(np.sum(s==d)),
                original_edge_overlap=float(np.isin(unique,original,assume_unique=True).sum()/len(src)))
    return s,d,info


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--condition',choices=['rewired','configuration','er'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--swap-factor',type=int,default=10);a=p.parse_args()
    start=time.perf_counter();result=manifest(vars(a),[a.graph]);n,s,d,_=load_npz(a.graph);s,d,info=make_control(n,s,d,a.condition,a.seed,a.swap_factor)
    path=Path(f'data/processed/controls/{a.condition}_{a.seed}.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,n=np.array(n),src=s.astype(np.int32),dst=d.astype(np.int32))
    result.update(info);result.update(graph_sha256=sha256(path),runtime_seconds=time.perf_counter()-start)
    save_json(f'results/malecns_v1/controls/{a.condition}_{a.seed}.json',result);print(info,flush=True)


if __name__=='__main__':main()
