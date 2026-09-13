"""Measure actual recurrent updates, separating them from AdamW weight decay."""
import json
import torch
from src.query_flygpt import load_model
from src.provenance import manifest,save_json


def main():
    torch.set_num_threads(4);step=256;path=f'results/malecns_v1/target/snapshots/real_0_step_{step:04d}.pt'
    m=load_model('data/processed/malecns.npz',path,'cpu')
    # Exact initialization order in GraphLanguageModel: embedding, then edge_w.
    torch.manual_seed(0);torch.randn(256,16)
    initial=torch.randn_like(m.core.edge_w)*.9
    degree=torch.bincount(m.core.dst,minlength=m.core.n).clamp_min(1)
    initial/=degree[m.core.dst].sqrt()
    decay=(1-.003*.01)**64*(1-.001*.01)**step
    with torch.no_grad():
        delta=m.core.edge_w-initial;optimization_delta=m.core.edge_w-initial*decay
        values={'initial_edge_weight_l2':float(initial.norm()),'trained_edge_weight_l2':float(m.core.edge_w.norm()),
                'edge_delta_l2':float(delta.norm()),'edge_delta_rms':float(delta.square().mean().sqrt()),
                'fraction_changed_beyond_decay_at_1e_6':float((optimization_delta.abs()>1e-6).float().mean()),
                'gradient_update_residual_l2_after_known_decay':float(optimization_delta.norm()),
                'expected_decay_only_factor':decay}
    result=manifest({'step':step,'seed':0,'initialization':'exact seeded embedding + degree-normalized edge draw',
                     'pilot_lr':.003,'continuation_lr':.001,'weight_decay':.01,'absolute_change_tolerance':1e-6},[path,'data/processed/malecns.npz'])
    result.update(values);result['interpretation']='Actual recurrent parameter changes, not a claim that every change improves predictions or is physically writable.'
    save_json('results/malecns_v1/target/weight_change_256.json',result);print(values)


if __name__=='__main__':main()
