"""One final independent test-set evaluation after the fixed 512-update run.

Use the final checkpoint, never select it by test loss. No architecture edits,
no training, and no implication that a plateau identifies a capacity bound.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.query_flygpt import load_model
from src.tinygpt import TinyGPT
from src.train_language import bytes_from_file
from src.provenance import manifest,save_json


def main():
    torch.set_num_threads(4);root=Path('results/malecns_v1/target');ck=root/'snapshots'/'real_0_step_0512.pt'
    data_path='data/raw/grammar_v1/test.txt'
    record=manifest({'checkpoint':str(ck),'selection':'final fixed-budget checkpoint, not selected by test','window':32,'seed':42},[ck,data_path,'results/malecns_v1/teacher.pt'])
    start=time.perf_counter();model=load_model('data/processed/malecns.npz',ck,'cpu')
    tc=torch.load('results/malecns_v1/teacher.pt',weights_only=True,map_location='cpu');teacher=TinyGPT(**tc['config']).eval();teacher.load_state_dict(tc['model'])
    data=bytes_from_file(data_path);windows=list(range(0,len(data)-32,32));rows=[]
    with torch.no_grad():
        for s in windows:
            x=data[s:s+32][None];y=data[s+1:s+33];z,_=model(x);tz=teacher(x)
            rows.append({'start':s,'student_ce':float(nn.functional.cross_entropy(z[0],y)),
                         'teacher_ce':float(nn.functional.cross_entropy(tz[0],y)),
                         'teacher_kl':float(nn.functional.kl_div(nn.functional.log_softmax(z,-1),nn.functional.softmax(tz,-1),reduction='sum')/32),
                         'teacher_agreement':float((z.argmax(-1)==tz.argmax(-1)).float().mean())})
        model.core.edge_w.zero_()
        for row in rows:
            s=row['start'];x=data[s:s+32][None];y=data[s+1:s+33];z,_=model(x)
            row['zero_edge_ce']=float(nn.functional.cross_entropy(z[0],y))
    rng=np.random.default_rng(42);indices=rng.integers(0,len(rows),size=(10000,len(rows)))
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='start'}
    gap=np.array([r['student_ce']-r['teacher_ce'] for r in rows]);ablation=np.array([r['zero_edge_ce']-r['student_ce'] for r in rows])
    record.update(evidence_domain='MaleCNS-based computation on held-out synthetic grammar; one training seed',
                  metrics=means,n_test_bytes=32*len(rows),windows=rows,
                  paired_window_bootstrap={'gap_95_ci':np.percentile(gap[indices].mean(1),[2.5,97.5]).tolist(),
                                           'ablation_penalty_95_ci':np.percentile(ablation[indices].mean(1),[2.5,97.5]).tolist(),
                                           'limitation':'windows are not independent training seeds; neighboring text may be correlated'},
                  runtime_seconds=time.perf_counter()-start)
    save_json(root/'final_test.json',record);print(means,flush=True)


if __name__=='__main__':main()
