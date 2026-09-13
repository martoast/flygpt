"""Read-only checkpoint diagnostics for an unchanged running baseline.

Training is never imported or modified. Snapshot links preserve every observed
checkpoint inode while the trainer atomically replaces its latest checkpoint.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.query_flygpt import load_model
from src.tinygpt import TinyGPT
from src.train_language import bytes_from_file,evaluate
from src.provenance import manifest,save_json,sha256

ROOT=Path('results/malecns_v1/target')


def inspect(checkpoint,out,step,training_seconds,trace,graph='data/processed/malecns.npz',condition='real'):
    torch.set_num_threads(4);start=time.perf_counter()
    model=load_model(graph,checkpoint,'cpu')
    tc=torch.load('results/malecns_v1/teacher.pt',map_location='cpu',weights_only=True)
    teacher=TinyGPT(**tc['config']).eval();teacher.load_state_dict(tc['model']);del tc
    data=bytes_from_file('data/raw/grammar_v1/validation.txt')
    starts=np.random.default_rng(9100).integers(0,len(data)-33,size=8).tolist()
    loss=teacher_loss=kl=kl_t2=agreement=correct=count=0
    norms=[];rms=[];readout_norms=[];saturated=0;states=0
    with torch.no_grad():
        for s in starts:
            x=data[s:s+32][None];y=data[s+1:s+33][None];tz=teacher(x);h=None;zs=[]
            for t in range(x.shape[1]):
                z,h=model(x[:,t:t+1],h);zs.append(z)
                norms.append(float(h.norm()));rms.append(float(h.square().mean().sqrt()))
                readout_norms.append(float(h[:,model.core.output_nodes].norm()))
                saturated+=int((h.abs()>.95).sum());states+=h.numel()
            z=torch.cat(zs,dim=1)
            loss+=float(nn.functional.cross_entropy(z.reshape(-1,256),y.flatten(),reduction='sum'))
            teacher_loss+=float(nn.functional.cross_entropy(tz.reshape(-1,256),y.flatten(),reduction='sum'))
            kl+=float(nn.functional.kl_div(nn.functional.log_softmax(z,-1),nn.functional.softmax(tz,-1),reduction='sum'))
            kl_t2+=float(nn.functional.kl_div(nn.functional.log_softmax(z/2,-1),nn.functional.softmax(tz/2,-1),reduction='sum')*4)
            agreement+=int((z.argmax(-1)==tz.argmax(-1)).sum());correct+=int((z.argmax(-1)==y).sum());count+=y.numel()
        model.core.edge_w.zero_()
    ablated=evaluate(model,data,starts,32)
    record=manifest({'checkpoint':str(checkpoint),'continuation_step':step,'evaluation_starts':starts,
                     'n_windows':8,'bytes_per_window':32,'saturation_definition':'abs(hidden state)>0.95 at byte boundaries'},
                    [checkpoint,graph,'results/malecns_v1/teacher.pt','data/raw/grammar_v1/validation.txt'])
    record.update(evidence_domain=('MaleCNS-based' if condition=='real' else 'synthetic topology control')+' computation on synthetic grammar; no wetware',
                  condition=condition,
                  continuation_step=step,validation_nats_per_byte=loss/count,teacher_nats_per_byte=teacher_loss/count,
                  gap_to_teacher=loss/count-teacher_loss/count,gap_to_rounded_0325=loss/count-.325,
                  teacher_kl_nats_per_byte=kl/count,teacher_kl_temperature2_scaled_per_byte=kl_t2/count,
                  teacher_argmax_agreement=agreement/count,byte_accuracy=correct/count,
                  zero_edge_ablation_nats_per_byte=ablated['ce'],ablation_minus_intact=ablated['ce']-loss/count,
                  gradient_norm={'last_unclipped':trace[-1]['gradient_norm'] if trace else None,
                                 'mean_last_16_unclipped':float(np.mean([r['gradient_norm'] for r in trace[-16:]])) if trace else None,
                                 'clipping_threshold':1.0},
                  hidden_state={'mean_l2_norm':float(np.mean(norms)),'max_l2_norm':max(norms),'mean_rms':float(np.mean(rms)),
                                'mean_readout_l2_norm':float(np.mean(readout_norms)),'saturation_fraction':saturated/states,
                                'observed_byte_boundary_states':len(norms),'neurons_per_state':model.core.n},
                  bytes_seen={'pilot':512,'continuation':step*32,'total':512+step*32},
                  training_wall_seconds=training_seconds,diagnostic_wall_seconds=time.perf_counter()-start,
                  validation_bytes=count,topology_verified_by_sha256=True)
    save_json(out,record);print(json.dumps({k:record[k] for k in ['continuation_step','validation_nats_per_byte','gap_to_teacher','teacher_kl_nats_per_byte','zero_edge_ablation_nats_per_byte']}),flush=True)


def report():
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'trajectory').glob('step_*.json'))]
    records.sort(key=lambda r:r['continuation_step'])
    save_json(ROOT/'trajectory.json',{'architecture':'Baseline A, unchanged throughout continuation','records':records})
    lines=['# Baseline A: fixed-MaleCNS teacher-gap trajectory','',
           'Same 256 validation bytes and frozen teacher at every checkpoint. Single-seed feasibility; no topology-superiority claim.','',
           '| Additional updates | Total bytes seen | Val nats/byte | Teacher gap | KL (T=1) | Teacher argmax agreement | Zero-edge loss | Last gradient norm | Hidden RMS | Saturation | Training seconds |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        grad=r['gradient_norm']['last_unclipped']
        lines.append(f'| {r["continuation_step"]} | {r["bytes_seen"]["total"]} | {r["validation_nats_per_byte"]:.4f} | {r["gap_to_teacher"]:.4f} | {r["teacher_kl_nats_per_byte"]:.4f} | {r["teacher_argmax_agreement"]:.3f} | {r["zero_edge_ablation_nats_per_byte"]:.4f} | {grad:.3f} | {r["hidden_state"]["mean_rms"]:.4f} | {r["hidden_state"]["saturation_fraction"]:.6f} | {r["training_wall_seconds"]:.1f} |')
    lines+=['','Gradient norms are measured before clipping to 1.0. Saturation is abs(h)>0.95, sampled at byte boundaries. KL is forward KL(teacher || student), in nats per byte at T=1; the T=2 training-scaled KL is also retained in JSON. Training wall time excludes these separate diagnostic processes but can include resource contention.','']
    (ROOT/'TRAJECTORY.md').write_text('\n'.join(lines))


def watch():
    ROOT.mkdir(parents=True,exist_ok=True);snapshots=ROOT/'snapshots';snapshots.mkdir(exist_ok=True)
    pilot=json.loads(Path('results/malecns_v1/language/real_0.json').read_text())
    initial=ROOT/'trajectory'/'step_0000.json'
    if not initial.exists():inspect('results/fly_real.pt',initial,0,pilot['runtime_seconds'],pilot['trace']);report()
    for step in [64,128,256,512]:
        out=ROOT/'trajectory'/f'step_{step:04d}.json'
        if out.exists():continue
        while True:
            progress=ROOT/'real_0.json'
            if progress.exists():
                r=json.loads(progress.read_text())
                if r['completed_steps']==step:break
                if r['completed_steps']>step:raise RuntimeError(f'Missed checkpoint {step}; refuse to relabel a later checkpoint')
            time.sleep(3)
        snapshot=snapshots/f'real_0_step_{step:04d}.pt'
        if not snapshot.exists():os.link(ROOT/'real_0.pt',snapshot)
        ck=torch.load(snapshot,map_location='cpu',weights_only=True)
        if ck['completed_steps']!=step:raise RuntimeError('Checkpoint/progress race: wrong step')
        trace=ck['trace'];del ck
        inspect(snapshot,out,step,pilot['runtime_seconds']+r['runtime_seconds_this_session'],trace);report()


if __name__=='__main__':watch()
