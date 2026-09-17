"""Six-way internal-representation transfer pilot; frozen batches and correspondence."""
import argparse
import json
import math
from pathlib import Path
import resource
import shutil
import time

import torch
from torch import nn
from scripts import g2c_engine as e
from scripts.compiler_storage import archive
from src.provenance import manifest,save_json,sha256

METHODS=('A_hard','B_soft','C_hidden','D_hidden_shuffled','E_relational','F_relational_shuffled')

def read(p):return json.loads(Path(p).read_text())
def freeze(p,value):
    if Path(p).exists():
        if read(p)!=value:raise RuntimeError(f'Frozen data collision: {p}')
    else:save_json(p,value)
    return str(p)

def schedule(n,batch,updates,seed):
    if batch<4 or n%batch:raise ValueError('Require batch >=4 and complete fixed groups')
    groups=torch.randperm(n,generator=torch.Generator().manual_seed(125000+seed)).reshape(-1,batch)
    g=torch.Generator().manual_seed(175000+seed);permutations=[]
    for _ in groups:
        order=torch.randperm(batch,generator=g);perm=torch.empty(batch,dtype=torch.long)
        perm[order]=order.roll(1)  # One cycle: no fixed examples or unordered pairs.
        permutations.append(perm.tolist())
    chosen=torch.randint(len(groups),(updates,),generator=torch.Generator().manual_seed(75000+seed))
    return dict(seed=seed,batch=batch,groups=groups.tolist(),derangements=permutations,
        update_groups=chosen.tolist(),rule='Fixed groups and one frozen within-group cycle per seed; no per-update reshuffling')

def geometry(features):
    z=nn.functional.normalize(features,dim=-1)
    return torch.einsum('btd,ctd->tbc',z,z)

def alignment_loss(student,teacher,mask,projection,kind):
    if not torch.equal(mask,mask[:1].expand_as(mask)):raise ValueError('This task requires aligned response positions')
    positions=mask[0];s=projection(student[:,positions]);t=teacher[:,positions]
    if kind=='hidden':
        return (nn.functional.normalize(s,dim=-1)-nn.functional.normalize(t,dim=-1)).square().sum(-1).mean()
    if kind!='relational':raise ValueError(kind)
    if s.shape[0]<4:raise ValueError('Relational control requires at least four examples')
    difference=geometry(s)-geometry(t)
    off_diagonal=~torch.eye(s.shape[0],dtype=torch.bool)
    return difference[:,off_diagonal].square().mean()

def audit_schedule(plan,features,mask,epsilon):
    if features.shape[0]!=len(plan['groups'])*plan['batch']:raise ValueError('Cache/group mismatch')
    if not torch.isfinite(features).all():raise ValueError('Nonfinite teacher features')
    relational=[];relational_float32=[];hidden=[];checked=[]
    for group_id in sorted(set(plan['update_groups'])):
        ids=torch.tensor(plan['groups'][group_id]);perm=torch.tensor(plan['derangements'][group_id])
        if sorted(perm.tolist())!=list(range(len(perm))) or (perm==torch.arange(len(perm))).any():
            raise ValueError('Correspondence must be a bijective derangement')
        m=mask[ids]
        if not torch.equal(m,m[:1].expand_as(m)):raise ValueError('Response masks differ')
        tf=features[ids][:,m[0]].double();original=geometry(tf);shuffled=original[:,perm][:,:,perm]
        rn=float(torch.linalg.vector_norm(original-shuffled))
        runtime_geometry=geometry(tf.float())
        relational_float32.append(float(torch.linalg.vector_norm(runtime_geometry-runtime_geometry[:,perm][:,:,perm])))
        hn=float(torch.linalg.vector_norm(nn.functional.normalize(tf,dim=-1)-nn.functional.normalize(tf[perm],dim=-1)))
        relational.append(rn);hidden.append(hn);checked.append(group_id)
    passed=bool(relational) and min(relational)>epsilon and min(relational_float32)>epsilon and min(hidden)>epsilon
    return dict(passed=passed,epsilon=epsilon,groups_checked=checked,scheduled_batches=len(plan['update_groups']),
        all_scheduled_batches_covered=True,relational_frobenius=relational,hidden_frobenius=hidden,
        relational_frobenius_float32=relational_float32,
        relational_min=min(relational) if relational else None,hidden_min=min(hidden) if hidden else None,
        interpretation='Response-position cosine matrices before/after fixed index permutation; no control-dependent resampling')

def prepare_cache(cfg):
    torch.set_num_threads(4);root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    receipt=root/'teacher_cache_receipt.json';out=root/'teacher_cache.pt'
    inputs={str(p):sha256(p) for p in [cfg['teacher_spec'],cfg['teacher_checkpoint'],cfg['qualification_receipt'],
        cfg['training_targets'],cfg['training_prompts'],'scripts/compiler_v3_engine.py']}
    if receipt.exists():
        r=read(receipt)
        assert r['inputs']==inputs and sha256(out)==r['cache_sha256']
        return
    if out.exists():raise RuntimeError('Cache without receipt; preserve and inspect before retry')
    e.qualified(cfg['qualification_receipt']);start=time.perf_counter()
    teacher=e.load(read(cfg['teacher_spec']),cfg['teacher_checkpoint']);prompts=read(cfg['training_prompts']);targets=read(cfg['training_targets'])
    if [r['id'] for r in prompts]!=[r['id'] for r in targets]:raise ValueError('Prompt/target order mismatch')
    # Replay teacher-only answer generation to charge its real one-time acquisition cost.
    with torch.no_grad():
        for prompt,target in zip(prompts,targets):
            if set(prompt)!={'id','prompt'}:raise ValueError('Prompt extraction contains answers')
            if e.generate(teacher,prompt['prompt'],8)!=target['response']:raise RuntimeError('Teacher answer replay mismatch')
    hard_seconds=time.perf_counter()-start;extract_start=time.perf_counter();features=[];logits=[];masks=[]
    with torch.no_grad():
        for offset in range(0,len(targets),cfg['batch']):
            indices=list(range(offset,min(offset+cfg['batch'],len(targets))));x,y=e.pack(targets,indices);captured=[]
            handle=teacher.ln.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
            try:z=teacher(x)
            finally:handle.remove()
            features.append(captured[0]);logits.append(z);masks.append(y!=-100)
    cache=dict(features=torch.cat(features),logits=torch.cat(logits),mask=torch.cat(masks),ids=[r['id'] for r in targets])
    torch.save(cache,out);representation_seconds=time.perf_counter()-extract_start
    save_json(receipt,dict(inputs=inputs,cache_sha256=sha256(out),hard_answer_generation_seconds=hard_seconds,
        representation_extraction_and_cache_seconds=representation_seconds,teacher_pretraining_excluded=True,
        timing_rule='Charge full one-time hard-answer generation to every method; B–F also charged full cache extraction. Shared-cache amortized figures are secondary.',
        teacher_only_targets=True,examples=len(targets),hardware=manifest({},[])['hardware']))

def objective(method,z,y,tf,sf,teacher_logits,projection,permutation,spec):
    mask=y!=-100;ce=nn.functional.cross_entropy(z[mask],y[mask]);alignment=None;kl=None
    if method=='A_hard':loss=ce
    elif method=='B_soft':
        t=spec['temperature']
        kl=nn.functional.kl_div(nn.functional.log_softmax(z[mask]/t,-1),nn.functional.softmax(teacher_logits[mask]/t,-1),reduction='sum')/mask.sum()*t*t
        loss=kl  # Pure soft target supervision; hard CE is logging only.
    else:
        target=tf[permutation] if method in ('D_hidden_shuffled','F_relational_shuffled') else tf
        kind='hidden' if method in ('C_hidden','D_hidden_shuffled') else 'relational'
        alignment=alignment_loss(sf,target,mask,projection,kind)
        loss=ce+spec['alignment_weight']*alignment
    return loss,ce,kl,alignment

def train(spec_path,until):
    session_start=time.perf_counter();spec=read(spec_path);torch.set_num_threads(4)
    if spec['method'] not in METHODS:raise ValueError('Unknown method')
    if until not in spec['budgets']:raise ValueError('Unregistered stopping point')
    e.qualified(spec['qualification_receipt']);audit=read(spec['schedule_audit'])
    if not audit['passed']:raise RuntimeError('Shuffled-control qualification failed')
    inputs={str(p):sha256(p) for p in [spec_path,spec['graph'],spec['teacher_checkpoint'],spec['teacher_spec'],spec['qualification_receipt'],
        spec['training_targets'],spec['cache'],spec['cache_receipt'],spec['schedule'],spec['schedule_audit'],
        Path(spec['dataset'])/'manifest.json',Path(spec['dataset'])/'validation.json',__file__]}
    assert inputs[spec['training_targets']]==spec['training_targets_sha256']
    if sha256(spec['schedule'])!=audit['schedule_sha256']:raise RuntimeError('Schedule differs from audited schedule')
    cache_receipt=read(spec['cache_receipt']);assert sha256(spec['cache'])==cache_receipt['cache_sha256']==audit['cache_sha256']
    cache=torch.load(spec['cache'],weights_only=True,map_location='cpu');records=read(spec['training_targets']);plan=read(spec['schedule'])
    if cache['ids']!=[r['id'] for r in records]:raise RuntimeError('Cache order differs from targets')
    assert plan['batch']==spec['batch'] and len(plan['update_groups'])>=until
    job=Path(spec['job_dir']);job.mkdir(parents=True,exist_ok=True);checkpoint=job/'model.pt';progress=job/'progress.json'
    if checkpoint.exists()!=progress.exists():raise RuntimeError('Partial checkpoint publication; inspect before restart')
    model=e.build(spec);teacher=e.load(read(spec['teacher_spec']),spec['teacher_checkpoint'])
    for p in teacher.parameters():p.requires_grad_(False)
    projection=None
    if spec['method'] in METHODS[2:]:
        with torch.random.fork_rng():
            torch.manual_seed(spec['projection_seed']);projection=nn.Linear(model.core.n_out,cache['features'].shape[-1],bias=False)
    parameters=list(model.parameters())+(list(projection.parameters()) if projection is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=spec['lr'],weight_decay=.01,foreach=False)
    result=manifest(spec,list(inputs));result['inputs']=inputs
    trace=[];evaluations=[];archives=[];done=0;training_seconds=0.;previous_wall=0.;seen_groups=set();setup_prior=0.
    if checkpoint.exists():
        result=read(progress)
        if result['inputs']!=inputs or sha256(checkpoint)!=result['checkpoint_sha256']:raise RuntimeError('Resume provenance mismatch')
        ck=torch.load(checkpoint,weights_only=True,map_location='cpu')
        missing,unexpected=model.load_state_dict(ck['model'],strict=False)
        assert set(missing)=={'core.src','core.dst'} and not unexpected
        if projection is not None:projection.load_state_dict(ck['auxiliary'])
        optimizer.load_state_dict(ck['optimizer']);done=ck['step'];torch.set_rng_state(ck['torch_rng'])
        trace=result['trace'];evaluations=result['evaluations'];archives=result['archives'];training_seconds=result['training_seconds'];previous_wall=result['engine_wall_seconds'];setup_prior=result['setup_seconds']
        seen_groups=set(plan['update_groups'][:done]);del ck
    else:shutil.copyfile(__file__,job/'engine_source.py')
    setup_seconds=setup_prior+time.perf_counter()-session_start
    preparation=cache_receipt['hard_answer_generation_seconds']+(0 if spec['method']=='A_hard' else cache_receipt['representation_extraction_and_cache_seconds'])
    model.train()
    for step in range(done+1,until+1):
        begin=time.perf_counter();group_id=plan['update_groups'][step-1];indices=plan['groups'][group_id];perm=torch.tensor(plan['derangements'][group_id]);seen_groups.add(group_id)
        x,y=e.pack(records,indices);sf=[];handle=None
        if projection is not None:handle=model.core.out_proj.register_forward_pre_hook(lambda module,args:sf.append(args[0]))
        forward_start=time.perf_counter()
        try:raw=model(x);z=raw[0]
        finally:
            if handle is not None:handle.remove()
        forward_seconds=time.perf_counter()-forward_start;loss_start=time.perf_counter()
        loss,ce,kl,alignment=objective(spec['method'],z,y,cache['features'][indices],torch.stack(sf,1) if sf else None,
            cache['logits'][indices],projection,perm,spec)
        objective_seconds=time.perf_counter()-loss_start;backward_start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite optimization state')
        optimizer.step();backward_seconds=time.perf_counter()-backward_start
        item=dict(step=step,group_id=group_id,ce=float(ce.detach()),loss=float(loss.detach()),kl=None if kl is None else float(kl.detach()),
            alignment=None if alignment is None else float(alignment.detach()),gradient_norm=float(grad),hidden_rms=float(raw[1].detach().square().mean().sqrt()),
            saturation=float((raw[1].detach().abs()>.95).float().mean()),examples_seen=step*spec['batch'],unique_examples_seen=len(seen_groups)*spec['batch'],
            student_forward_seconds=forward_seconds,objective_seconds=objective_seconds,backward_optimizer_seconds=backward_seconds)
        training_seconds+=time.perf_counter()-begin;item['acquisition_seconds']=preparation+setup_seconds+training_seconds;trace.append(item)
        if step%16==0:print(spec['method'],item,flush=True)
        if step in spec['checkpoints'] or step==until:
            acquisition_seconds=preparation+setup_seconds+training_seconds;validation_start=time.perf_counter()
            ev=e.score(model,e.data(spec['dataset'],'validation')[:spec['curve_cases']],teacher)
            evaluations.append(dict(step=step,validation=ev,examples_seen=step*spec['batch'],acquisition_seconds=acquisition_seconds,
                validation_seconds=time.perf_counter()-validation_start,wall_seconds=preparation+previous_wall+time.perf_counter()-session_start))
            print('VALIDATION',step,ev['metrics'],flush=True)
            ck=dict(model={k:v for k,v in model.state_dict().items() if k not in ('core.src','core.dst')},optimizer=optimizer.state_dict(),
                step=step,spec_sha256=sha256(spec_path),graph_sha256=sha256(spec['graph']),torch_rng=torch.get_rng_state(),schedule_sha256=sha256(spec['schedule']))
            if projection is not None:ck['auxiliary']=projection.state_dict()
            temporary=checkpoint.with_suffix('.pt.tmp');torch.save(ck,temporary);temporary.replace(checkpoint)
            archives.append(archive(checkpoint,spec,step));wall=previous_wall+time.perf_counter()-session_start
            result.update(step=step,trace=trace,evaluations=evaluations,archives=archives,checkpoint_sha256=sha256(checkpoint),
                training_seconds=training_seconds,setup_seconds=setup_seconds,preparation_seconds=preparation,engine_wall_seconds=wall,
                wall_seconds=preparation+wall,acquisition_seconds=acquisition_seconds,examples_seen=step*spec['batch'],
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save_json(progress,result);model.train()

def learning_summary(progress,threshold):
    ev=progress['evaluations'];cross=next((i for i,r in enumerate(ev) if r['validation']['metrics']['accuracy']>=threshold),None)
    steps=[r['step'] for r in ev];acc=[r['validation']['metrics']['accuracy'] for r in ev]
    area=sum((b-a)*(x+y)/2 for a,b,x,y in zip(steps,steps[1:],acc,acc[1:]))/(steps[-1]-steps[0]) if len(steps)>1 else None
    return dict(validation_accuracy_area=area,threshold=threshold,threshold_reached=cross is not None,
        first_observed_crossing=None if cross is None else {k:ev[cross][k] for k in ['step','examples_seen','acquisition_seconds','wall_seconds']},
        crossing_update_interval=None if cross is None else [0 if cross==0 else steps[cross-1],steps[cross]],
        right_censored_after_updates=steps[-1] if cross is None else None,final_accuracy=acc[-1],
        acquisition_seconds=progress['acquisition_seconds'],end_to_end_seconds=progress['wall_seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--until',type=int,required=True);a=p.parse_args();train(a.spec,a.until)
