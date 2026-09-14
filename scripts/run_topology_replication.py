"""Independent ten-pair M1 topology replication; original training engine unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from scripts import run_compiler_v1 as v, g2c_engine as e
from scripts.run_topology_confirmation_v2 import assign_test
from scripts.run_compiler_v3 import read,now,verify,paths_hashes
from scripts.compiler_confirmation_analysis import paired
from scripts import mirror_m1_checkpoints as mirror
from src.provenance import save_json,sha256

ROOT=Path('results/topology_replication'); CFG=Path('configs/topology_replication.json')
DOC=Path('TOPOLOGY_REPLICATION_PREREGISTRATION.md'); REPORT=Path('TOPOLOGY_REPLICATION_REPORT.md')
BRANCH='experiment/topology-replication'
SOURCES=[Path('scripts/run_topology_replication.py'),Path('scripts/backup_topology_replication.py'),Path('tests/test_topology_replication.py')]
mirror.SSH[-1]='alex@alexs-mac-mini'


def freeze(p,value):
    p=Path(p)
    if p.exists():
        if read(p)!=value:raise RuntimeError('Frozen artifact changed: '+str(p))
    else:save_json(p,value)
    return str(p)


def prepare_data(cfg):
    source=Path(cfg['source_dataset']);dest=Path(cfg['dataset']);dest.mkdir(parents=True,exist_ok=True)
    manifest=read(source/'manifest.json');old={};inputs=[source/'manifest.json']
    for name,info in manifest['counts'].items():
        p=source/f'{name}.json';assert sha256(p)==info['sha256'];old[name]=read(p);inputs.append(p)
    for name in ['data/raw/topology_confirmation_v2/test_confirm.json','data/raw/compiler_v3/test_compiler_v3.json',
                 'data/raw/compiler_confirmation/test_compiler_confirmation.json','data/raw/compiler_efficiency/test_compiler_efficiency.json']:
        old[name]=read(name);inputs.append(name)
    ids,available=assign_test(old,cfg['test_count'],cfg['test_assignment_salt']);assert available==128 and len(ids)==128
    for name in ['train','validation']:
        p=dest/f'{name}.json'
        if p.exists():assert sha256(p)==sha256(source/p.name)
        else:shutil.copy2(source/p.name,p)
    rows=[]
    for i in ids:
        digits=[i//4**j%4 for j in reversed(range(6))]
        rows.append(dict(id=i,input=digits,prompt=[e.BOS,*digits,e.SEP],response=[*e.transform(digits,'substitute'),e.EOS]))
    freeze(dest/f"{cfg['primary_split']}.json",rows)
    freeze(dest/'manifest.json',dict(version='topology-replication',task='substitute',length=6,
        counts={n:dict(count=len(read(dest/f'{n}.json')),sha256=sha256(dest/f'{n}.json')) for n in ['train','validation',cfg['primary_split']]},
        source_inputs=paths_hashes(inputs)))
    freeze(ROOT/'leakage_audit.json',dict(passed=True,excluded=3968,test_count=128,remaining_unused=0,
        test_sha256=sha256(dest/f"{cfg['primary_split']}.json")))


def archive_file(path):
    path=Path(path);temporary=Path('artifacts/compiler_pending')/path
    temporary.parent.mkdir(parents=True,exist_ok=True)
    if not temporary.exists():os.link(path,temporary)
    digest=sha256(path);target=mirror.copy(temporary,digest);temporary.unlink()
    return dict(path=target,sha256=digest)


def publish(message):
    paths=[ROOT,CFG,DOC,*SOURCES,Path(read(CFG)['dataset'])]
    if REPORT.exists():paths.append(REPORT)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    subprocess.run(['git','add','--',*map(str,paths)],check=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:subprocess.run(['git','commit','-m',message],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True,timeout=120)
    # Off-machine immutable snapshot, including ignored locked test bytes.
    snapshot=Path('results/topology_replication_backup')/f'snapshot_{time.time_ns()}.tar.gz'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(snapshot,'w:gz',dereference=True) as archive:
        for root in paths:
            for p in root.rglob('*') if root.is_dir() else [root]:
                if p.is_file() and p.suffix in ('.json','.jsonl','.md','.py','.log'):archive.add(p,arcname=str(p))
    archive_file(snapshot);snapshot.unlink()


def prepare():
    cfg=read(CFG);prepare_data(cfg);v.ROOT=ROOT;v.refresh=lambda:None
    graphs=[];audits=[]
    for seed in cfg['rewiring_seeds']:
        graph=Path(f'data/processed/controls/rewired_{seed}.npz');audit=Path(f'results/malecns_v1/controls/rewired_{seed}.json')
        save_json(ROOT/'status.json',dict(stage='preparing_rewired_graph',graph_seed=seed,utc=now(),final_test_opened=False))
        if not audit.exists():
            if graph.exists():raise RuntimeError('Graph exists without audit; preserve and inspect')
            v.child('src.graph_controls',['--graph','data/processed/malecns.npz','--condition','rewired','--seed',seed,'--swap-factor',10],ROOT/f'graph_{seed}.log')
        a=read(audit);verify(a['inputs']);assert sha256(graph)==a['graph_sha256']
        assert a['successful_swaps']==10*a['n_edge_parameters']
        audits.append(freeze(ROOT/'graph_audits'/audit.name,a));graphs.append(graph)
        receipt=ROOT/'graph_backups'/f'{seed}.json'
        if not receipt.exists():save_json(receipt,archive_file(graph))
    base=read(cfg['reference_spec']);jobs=[]
    for seed,rseed in zip(cfg['seeds'],cfg['rewiring_seeds']):
        for condition in ['real_ce','rewired_ce']:
            spec={**base,'seed':seed,'sampling_seed':75000+seed,'dataset':cfg['dataset'],
                'job_dir':str(ROOT/'pairs'/f'seed_{seed}'/condition),'condition':condition,'objective':'ce','budgets':[1024],
                'graph':'data/processed/malecns.npz' if condition=='real_ce' else f'data/processed/controls/rewired_{rseed}.npz',
                'evidence_tier':'Independent preregistered ten-pair topology replication'}
            jobs.append(dict(seed=seed,condition=condition,spec=freeze(Path(spec['job_dir'])/'spec.json',spec)))
    inputs=[CFG,DOC,*SOURCES,*Path('src').glob('*.py'),'scripts/g2c_engine.py','scripts/compiler_storage.py',
        'scripts/run_compiler_v1.py','scripts/run_compiler_v3.py','scripts/run_topology_confirmation_v2.py',
        'scripts/compiler_confirmation_analysis.py','scripts/mirror_m1_checkpoints.py','scripts/distributed_confirmation.py',
        'scripts/run_g2c_overnight.py',base['teacher_spec'],base['teacher_checkpoint'],base['qualification_receipt'],
        'data/processed/malecns.npz',*graphs,*audits,*Path(cfg['dataset']).glob('*.json'),*[j['spec'] for j in jobs]]
    freeze(ROOT/'frozen_plan.json',dict(jobs=jobs,inputs=paths_hashes(inputs),test_gate='All 20 final checkpoints complete'))
    publish('Freeze ten independent topology pairs and new locked test before training')
    if not (ROOT/'publication.json').exists():
        freeze(ROOT/'publication.json',dict(frozen_plan_sha256=sha256(ROOT/'frozen_plan.json'),
            github_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=BRANCH,utc=now()))


def summarize(by_seed):
    import numpy as np
    if set(by_seed)!={str(s) for s in range(600,610)}:raise ValueError('All ten registered pairs required')
    real=[p['real_ce']['accuracy'] for _,p in sorted(by_seed.items())]
    rewired=[p['rewired_ce']['accuracy'] for _,p in sorted(by_seed.items())]
    result=paired([a-b for a,b in zip(real,rewired)])
    result.update(real_mean=float(np.mean(real)),rewired_mean=float(np.mean(rewired)),real_sd=float(np.std(real,ddof=1)),
        rewired_sd=float(np.std(rewired,ddof=1)),positive_pairs=sum(a>b for a,b in zip(real,rewired)),
        primary_pass=result['mean_benefit']>0 and result['paired_t_p']<.05)
    return result


def run():
    cfg=read(CFG);plan=read(ROOT/'frozen_plan.json');verify(plan['inputs'])
    assert read(ROOT/'publication.json')['frozen_plan_sha256']==sha256(ROOT/'frozen_plan.json')
    v.ROOT=ROOT;v.refresh=lambda:None;checkpoints={}
    for job in plan['jobs']:
        verify(plan['inputs']);save_json(ROOT/'status.json',dict(stage='training',**job,utc=now(),final_test_opened=False))
        checkpoints[job['spec']]=v.train(job['spec'],'scripts.g2c_engine')
        save_json(ROOT/'completed_training.json',checkpoints)
        publish(f"Topology replication: preserve {job['condition']} seed {job['seed']}")
    assert len(checkpoints)==20;verify(plan['inputs'])
    unlock=freeze(ROOT/'unlock.json',dict(split=cfg['primary_split'],choices_frozen=True,
        inputs=paths_hashes([ROOT/'frozen_plan.json',ROOT/'publication.json',*checkpoints.values()])))
    results={};curves={}
    for job in plan['jobs']:
        save_json(ROOT/'status.json',dict(stage='evaluation',**job,utc=now()))
        output=ROOT/'pairs'/f"seed_{job['seed']}"/f"{job['condition']}_test.json"
        result=v.evaluate(job['spec'],checkpoints[job['spec']],cfg['primary_split'],output,unlock)
        results.setdefault(str(job['seed']),{})[job['condition']]=result['metrics']
        p=read(Path(read(job['spec'])['job_dir'])/'progress.json');ev=p['evaluations']
        steps=[x['step'] for x in ev];acc=[x['validation']['metrics']['accuracy'] for x in ev]
        area=sum((b-a)*(x+y)/2 for a,b,x,y in zip(steps,steps[1:],acc,acc[1:]))/(steps[-1]-steps[0])
        curves.setdefault(str(job['seed']),{})[job['condition']]=dict(validation_accuracy_area=area,wall_seconds=p['wall_seconds'])
    verify(plan['inputs']);verify(read(unlock)['inputs']);analysis=summarize(results)
    save_json(ROOT/'analysis.json',analysis);save_json(ROOT/'learning_curves.json',curves)
    lines=['# Independent ten-pair topology replication','','Original five-pair and exploratory results are excluded from primary inference.','',
        '| Seed | MaleCNS % | Rewired % | Gap pp | Zero-edge real / rewired % |','|---|---:|---:|---:|---|']
    for seed,p in sorted(results.items()):
        a,b=p['real_ce'],p['rewired_ce'];lines.append(f"| {seed} | {100*a['accuracy']:.3f} | {100*b['accuracy']:.3f} | {100*(a['accuracy']-b['accuracy']):+.3f} | {100*a['zero_edge_exact']:.3f} / {100*b['zero_edge_exact']:.3f} |")
    lines+=['',f"Mean gap {100*analysis['mean_benefit']:+.3f} pp; primary paired-t p={analysis['paired_t_p']:.6g}; exact sign-flip sensitivity p={analysis['exact_two_sided_sign_flip_p']:.6g}; primary criterion passed: {analysis['primary_pass']}.",
        f"95% unadjusted t interval (accuracy fractions): {analysis['unadjusted_95_t_interval']}.",
        'Scope: fixed-budget substitution learning on one M1 machine and one corpus. Normality assumptions, graph/initialization coupling, fixed within-pair order, and the finite test domain limit interpretation. This does not identify a mechanism or establish compiler transfer.']
    REPORT.write_text('\n'.join(lines)+'\n')
    save_json(ROOT/'completion.json',dict(utc=now(),artifact_audit_passed=True,pairs=10));save_json(ROOT/'status.json',dict(stage='complete',utc=now()))
    publish('Complete independent ten-pair topology replication and paired analysis')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    awake=subprocess.Popen(['caffeinate','-is','-w',str(os.getpid())])
    try:
        if a.data_only:prepare_data(read(CFG))
        else:prepare();run()
    except BaseException as exc:
        import traceback
        save_json(ROOT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()));raise
    finally:awake.terminate()
