"""Stream the full segment table into the complete annotated-neuron graph.

The official v1.0 annotation universe is superclass.notna(): 166,700 neurons.
Retain isolates. All other segments (including glia and fragments) are excluded
explicitly. Do not infer neuron identity from participation in the raw table.
Adjacency storage orientation is [source, destination].
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy import sparse
from .provenance import manifest, save_json, sha256


def prepare(path, out, annotations, min_weight=1):
    start=time.perf_counter()
    annotation=pd.read_feather(annotations)
    selected=annotation[annotation.superclass.notna()].copy()
    ids=np.sort(selected.bodyId.to_numpy(np.int64))
    if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate annotated body IDs')
    n=len(ids)
    graph=sparse.csr_matrix((n,n),dtype=np.int64)
    rows=contacts=retained_rows=retained_contacts=0
    reader=ipc.open_file(pa.memory_map(str(path),'r'))
    required=('body_pre','body_post','weight')
    if not all(c in reader.schema.names for c in required):
        raise ValueError(f'Unexpected Feather schema: {reader.schema}')
    # Aggregate retained records in bounded chunks; threshold AFTER aggregation.
    ss=[]; dd=[]; ww=[]
    for k in range(reader.num_record_batches):
        batch=reader.get_batch(k)
        pre,post,w=(batch.column(batch.schema.get_field_index(c)).to_numpy() for c in required)
        if np.any(w<0): raise ValueError('Negative anatomical counts')
        rows+=len(w); contacts+=int(w.sum())
        si=np.searchsorted(ids,pre); di=np.searchsorted(ids,post)
        keep=(si<n)&(di<n)
        keep &= (ids[np.minimum(si,n-1)]==pre)&(ids[np.minimum(di,n-1)]==post)
        ss.append(si[keep].astype(np.int32));dd.append(di[keep].astype(np.int32));ww.append(w[keep])
        retained_rows+=int(keep.sum()); retained_contacts+=int(w[keep].sum())
        if (k+1)%128==0 or k+1==reader.num_record_batches:
            part=sparse.coo_matrix((np.concatenate(ww),(np.concatenate(ss),np.concatenate(dd))),shape=(n,n)).tocsr()
            graph=graph+part; ss.clear();dd.clear();ww.clear()
        if (k+1)%512==0: print(f'{k+1}/{reader.num_record_batches} batches; {retained_rows:,} retained rows',flush=True)
    graph.data[graph.data<min_weight]=0; graph.eliminate_zeros();graph.sort_indices()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(out,graph)
    np.save(out.with_name(out.stem+'_body_ids.npy'),ids)
    meta=manifest({'dataset':'MaleCNS v1.0 minconf 0.5','node_rule':'annotation superclass is non-null; includes isolates',
                   'min_pair_weight':min_weight,'orientation':'source,destination'},[path,annotations])
    meta.update(evidence_domain='MaleCNS-based computational graph; no wetware',n_nodes=n,n_edges=int(graph.nnz),
                synaptic_contacts=int(graph.data.sum()),raw_segment_pair_rows=rows,raw_synaptic_contacts=contacts,
                retained_pair_rows_before_aggregation=retained_rows,retained_contacts_before_threshold=retained_contacts,
                excluded_segment_pair_rows=rows-retained_rows,graph_sha256=sha256(out),
                annotation_coverage={c:selected[c].fillna('UNANNOTATED').value_counts().to_dict() for c in ['superclass','status','somaNeuromere']},
                cell_type_annotated_fraction=float(selected.type.notna().mean()),runtime_seconds=time.perf_counter()-start)
    save_json(out.with_suffix('.json'),meta)
    save_json('results/malecns_v1/graph_provenance.json',meta)
    print(f'Saved {n:,} nodes, {graph.nnz:,} edges, {int(graph.data.sum()):,} contacts',flush=True)
    return graph


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='data/processed/malecns.npz')
    p.add_argument('--annotations',default='data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather')
    p.add_argument('--min-weight',type=int,default=1)
    a=p.parse_args();prepare(a.input,a.output,a.annotations,a.min_weight)
