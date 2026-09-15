import sys,os,json,math,time,hashlib
from pathlib import Path
sys.path.insert(0,os.getcwd())
from src.provenance import sha256
root=Path(sys.argv[1]);output=Path(sys.argv[2]);started=time.time()
read=lambda p:json.loads(Path(p).read_text())
plan=read(root/'frozen_plan.json');issues=[]
for path,digest in plan['inputs'].items():
 if not Path(path).exists() or sha256(path)!=digest:issues.append('Frozen input mismatch: '+path)
rows=[]
for job in plan['jobs']:
 spec=read(job['spec']);p=Path(spec['job_dir'])/'progress.json'
 if not p.exists():continue
 d=read(p);trace=d['trace'];step=d['step'];flags=[]
 if [t['step'] for t in trace]!=list(range(1,step+1)):flags.append('Noncontiguous training trace')
 for t in trace:
  for k,v in t.items():
   if isinstance(v,(int,float)) and not math.isfinite(v):flags.append('Nonfinite trace '+k)
  method=spec.get('method'); expected=t['ce']
  if method=='B_soft':expected=t['kl']
  elif method in ['C_hidden','D_hidden_shuffled','E_relational','F_relational_shuffled']:expected+=spec['alignment_weight']*t['alignment']
  if not math.isclose(t['loss'],expected,rel_tol=1e-5,abs_tol=1e-5):flags.append('Loss decomposition mismatch')
 if 'schedule' in spec:
  schedule=read(spec['schedule'])
  if any(t['group_id']!=schedule['update_groups'][t['step']-1] for t in trace):flags.append('Batch schedule mismatch')
  if d['examples_seen']!=step*spec['batch']:flags.append('Example budget mismatch')
 last=next(a for a in d['archives'] if a['step']==step)
 if sha256(last['path'])!=last['sha256'] or last['sha256']!=d['checkpoint_sha256']:flags.append('Checkpoint hash mismatch')
 ev=[]
 for item in d['evaluations']:
  m=item['validation']['metrics']
  if not all(math.isfinite(v) for v in m.values() if isinstance(v,(int,float))):flags.append('Nonfinite evaluation')
  ev.append(dict(step=item['step'],accuracy=m['accuracy'],ce=m['response_ce'],kl=m.get('teacher_kl')))
 record=dict(seed=job['seed'],condition=job.get('method',job.get('condition')),step=step,complete=step==spec['budgets'][-1],
   checkpoint_verified=True,loss_identity_verified=not any('Loss' in f for f in flags),
   gradient_max=max(t['gradient_norm'] for t in trace),hidden_rms_max=max(t['hidden_rms'] for t in trace),
   saturation_max=max(t['saturation'] for t in trace),evaluations=ev,issues=sorted(set(flags)))
 rows.append(record);issues.extend(f"{record['seed']}/{record['condition']}: {f}" for f in set(flags))
# Audit split separation by IDs only; no final-test model evaluation.
dataset=Path(read(plan['jobs'][0]['spec'])['dataset']);manifest=read(dataset/'manifest.json');seen=set();split_counts={}
for name,meta in manifest['counts'].items():
 p=dataset/f'{name}.json';records=read(p);ids={x['id'] for x in records}
 if sha256(p)!=meta['sha256'] or len(ids)!=len(records) or ids&seen:issues.append('Split integrity/overlap: '+name)
 seen.update(ids);split_counts[name]=len(ids)
checks=[]
for p in sorted(root.glob('shuffle_audit_*.json')):
 a=read(p)
 if not a['passed'] or min(a['relational_frobenius_float32'])<=a['epsilon']:issues.append('Shuffle audit failed '+str(p))
 checks.append(dict(file=p.name,passed=a['passed'],relational_min=a['relational_min']))
result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root=str(root),frozen_inputs_checked=len(plan['inputs']),
 planned_models=len(plan['jobs']),models_checked=len(rows),issues=issues,split_counts=split_counts,shuffle_audits=checks,
 jobs=rows,final_test_evaluated_by_audit=False,seconds=time.time()-started)
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='jobs'}))
