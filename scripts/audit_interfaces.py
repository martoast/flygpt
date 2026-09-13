"""Characterize the frozen input/output populations without changing the model."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse import csgraph
from src.provenance import manifest,save_json


def main():
    graph='data/processed/malecns.npz';checkpoint='results/fly_real.pt'
    c=torch.load(checkpoint,weights_only=True,map_location='cpu');state=c['model']
    inputs=state['core.input_nodes'].numpy();outputs=state['core.output_nodes'].numpy();del c,state
    a=sparse.load_npz(graph).tocsr();a.data=np.ones(a.nnz,dtype=np.float32);n=a.shape[0]
    assert not np.intersect1d(inputs,outputs).size
    distances=csgraph.dijkstra(a,directed=True,unweighted=True,indices=inputs,min_only=True)
    nearest=distances[outputs];finite=nearest[np.isfinite(nearest)]
    ids=np.load('data/processed/malecns_body_ids.npy')
    ann=pd.read_feather('data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId')
    outdegree=np.diff(a.indptr);indegree=np.asarray(a.sum(axis=0)).ravel()
    record=manifest({'population_seed':2026,'placement':'uniform random disjoint index populations, fixed across topology/weight seeds',
                     'path_metric':'minimum directed hop distance from any input neuron'},[graph,checkpoint])
    record.update(n_input=len(inputs),n_output=len(outputs),overlap=0,
                  direct_input_to_output_edges=int(a[inputs][:,outputs].nnz),
                  outputs_unreachable_from_any_input=int(np.sum(~np.isfinite(nearest))),
                  nearest_input_hops_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(finite,return_counts=True))},
                  populations={name:{'superclass_counts':ann.loc[ids[nodes],'superclass'].value_counts().to_dict(),
                                     'mean_in_degree':float(indegree[nodes].mean()),'mean_out_degree':float(outdegree[nodes].mean()),
                                     'body_ids':ids[nodes].tolist()} for name,nodes in [('input',inputs),('output',outputs)]},
                  interpretation='Digital interface design, not a demonstrated biological read/write interface. Full recurrence retains all graph nodes.')
    save_json('results/malecns_v1/interface_audit.json',record)
    print({k:record[k] for k in ['n_input','n_output','overlap','direct_input_to_output_edges','outputs_unreachable_from_any_input','nearest_input_hops_histogram']})


if __name__=='__main__':main()
