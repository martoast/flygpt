"""Causal, stability and precision diagnostics on a graph-bound checkpoint.
Perturbation is post-training: it does NOT establish achievable retraining,
biological precision requirements, or a physical synaptic writing mechanism.
"""
import argparse
import time
import numpy as np
import torch
from .query_flygpt import load_model
from .train_language import bytes_from_file,evaluate
from .provenance import manifest,save_json


def probes(model):
    prompts=[b'the cat sees a ',b'the dog sees a ']
    with torch.no_grad():
        za,ha=model(torch.tensor([list(prompts[0])]))
        zb,hb=model(torch.tensor([list(prompts[1])]))
    return {'prompts':[p.decode() for p in prompts],
            'logit_l2':float((za[:,-1]-zb[:,-1]).norm()),
            'readout_state_l2':float((ha[:,model.core.output_nodes]-hb[:,model.core.output_nodes]).norm())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',default='data/processed/malecns.npz');p.add_argument('--checkpoint',default='results/fly_real.pt')
    p.add_argument('--out',default='results/malecns_v1/checkpoint_audit.json');a=p.parse_args();torch.set_num_threads(4)
    result=manifest(vars(a),[a.graph,a.checkpoint,'data/raw/grammar_v1/validation.txt']);start=time.perf_counter()
    model=load_model(a.graph,a.checkpoint,'cpu');core=model.core
    weights=core.edge_w.detach().clone();va=bytes_from_file('data/raw/grammar_v1/validation.txt')
    starts=np.random.default_rng(9100).integers(0,len(va)-33,size=8).tolist()
    result['intact']={'performance':evaluate(model,va,starts,32),'probe':probes(model)}
    with torch.no_grad():core.edge_w.zero_()
    result['zero_edges']={'performance':evaluate(model,va,starts,32),'probe':probes(model)}
    if result['zero_edges']['probe']['logit_l2']!=0:raise RuntimeError('Causal no-bypass audit failed')
    result['perturbations']=[]
    # Fixed perturbation seed; a diagnostic only, not a multi-seed robustness claim.
    for scale in [.01,.1,.5,1.]:
        torch.manual_seed(4400)
        with torch.no_grad():core.edge_w.copy_(weights+torch.randn_like(weights)*weights.std()*scale)
        result['perturbations'].append({'kind':'gaussian_weight_noise','std_over_global_weight_std':scale,'seed':4400,
                                        'performance':evaluate(model,va,starts,32)})
    bound=float(weights.abs().max())
    for levels in [2,3,4,8,16]:
        with torch.no_grad():
            quantized=torch.round((weights+bound)/(2*bound)*(levels-1))/(levels-1)*(2*bound)-bound
            core.edge_w.copy_(quantized)
        result['perturbations'].append({'kind':'uniform_global_quantization','levels':levels,'range':[-bound,bound],
                                        'performance':evaluate(model,va,starts,32)})
    with torch.no_grad():core.edge_w.copy_(weights)
    # Restore subsets of learned edge changes to the exact seeded ANN initial state.
    # This is not anatomical programming: v1 has signed random initialization.
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    torch.manual_seed(checkpoint['seed']);torch.randn(256,checkpoint['config']['embed_dim']) # embedding initialization
    initial=torch.randn_like(weights)*checkpoint['config']['edge_scale']
    degree=torch.bincount(core.dst,minlength=core.n).clamp_min(1)
    initial/=degree[core.dst].sqrt()
    # Independent audit avoids asserting exact init reconstruction: compare against
    # a newly seeded model and abort if initializer implementation changes.
    from .train_memory import build_model
    reference=build_model(a.graph,'real',checkpoint['seed'],checkpoint['config'])
    torch.testing.assert_close(initial,reference.core.edge_w)
    del reference
    for fraction in [.01,.05,.1,.2,.5]:
        torch.manual_seed(4500)
        with torch.no_grad():
            mask=torch.rand_like(weights)<fraction;core.edge_w.copy_(torch.where(mask,weights,initial))
        result['perturbations'].append({'kind':'posthoc_partial_update_retention','expected_fraction':fraction,
                                        'actual_fraction':float(mask.float().mean()),'baseline':'seeded signed ANN initialization, not anatomy',
                                        'all_other_parameters':'kept trained','performance':evaluate(model,va,starts,32)})
    with torch.no_grad():core.edge_w.copy_(weights)
    result['scope']='One pilot checkpoint, one noise/mask seed, validation only. No wetware feasibility conclusion.'
    result['runtime_seconds']=time.perf_counter()-start;save_json(a.out,result)


if __name__=='__main__':main()
