import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def detect_columns(df):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    pre_candidates = ['body_pre','bodypre','pre','source','bodyid_pre','bodyidpre']
    post_candidates = ['body_post','bodypost','post','target','bodyid_post','bodyidpost']
    w_candidates = ['weight','syn_count','count','synapse_count','n_synapses','roi_weight']
    def pick(cands):
        for k in cands:
            if k in low: return low[k]
        for c in cols:
            cl=c.lower()
            if any(k in cl for k in cands): return c
        return None
    return pick(pre_candidates), pick(post_candidates), pick(w_candidates)


def prepare(path, out, min_weight=1):
    df = pd.read_feather(path)
    pre, post, weight = detect_columns(df)
    if pre is None or post is None:
        raise ValueError(f'Could not identify pre/post columns. Columns: {list(df.columns)}')
    if weight is None:
        df['_weight'] = 1
        weight = '_weight'
    df = df[[pre, post, weight]].dropna()
    df = df[df[weight] >= min_weight]
    # aggregate in case table has multiple rows per pair
    df = df.groupby([pre, post], as_index=False)[weight].sum()
    ids = np.unique(np.concatenate([df[pre].to_numpy(), df[post].to_numpy()]))
    id_to_idx = {int(v): i for i,v in enumerate(ids)}
    src = df[pre].map(id_to_idx).to_numpy(np.int64)
    dst = df[post].map(id_to_idx).to_numpy(np.int64)
    w = df[weight].to_numpy(np.float32)
    mat = sparse.coo_matrix((w,(src,dst)), shape=(len(ids),len(ids))).tocsr()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out, mat)
    np.save(out.with_name(out.stem+'_body_ids.npy'), ids)
    meta = {'n_nodes': int(mat.shape[0]), 'n_edges': int(mat.nnz), 'weight_sum': float(mat.sum()),
            'source': str(path), 'min_weight': min_weight, 'pre_col': pre, 'post_col': post, 'weight_col': weight}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',default='data/processed/malecns_graph.npz'); p.add_argument('--min-weight',type=float,default=1)
    a=p.parse_args(); prepare(a.input,a.output,a.min_weight)
