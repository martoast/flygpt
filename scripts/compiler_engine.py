"""Compiler-method fork; original G2c engine and query architecture are unchanged."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from src.tinygpt import TinyGPT
from src.graph_lm import GraphLanguageModel
from src.graphs import load_npz
from src.provenance import manifest,save_json,sha256

VOCAB=8;EOS=4;BOS=5;SEP=6


def transform(values,task):
    if task=='substitute':return [(x+1)%4 for x in values]
    if task=='reverse':return values[::-1]
    if task=='rotate':return values[2:]+values[:2]
    if task=='checksum':return [sum(values)%4]
    raise ValueError(task)


def make_data(task,length=6):
    root=Path(f'data/raw/g2c_v1/{task}_{length}');root.mkdir(parents=True,exist_ok=True)
    domain=4**length
    if domain<4096:raise ValueError('Primary G2c requires at least 4096 distinct instances')
    ids=sorted(range(domain),key=lambda i:hashlib.sha256(f'g2c-v1:{task}:{length}:{i}'.encode()).digest())
    counts={'train':2048,'validation':256,'qualification_0':128,'qualification_1':128,
            'test_main':128,'test_extension':128,'test_capacity':128}
    offset=0;records={};seen=set()
    for split,count in counts.items():
        rows=[]
        for identifier in ids[offset:offset+count]:
            digits=[(identifier//(4**i))%4 for i in reversed(range(length))]
            assert identifier not in seen;seen.add(identifier)
            rows.append({'id':identifier,'input':digits,'prompt':[BOS,*digits,SEP],
                         'response':[*transform(digits,task),EOS]})
        offset+=count;p=root/f'{split}.json';content=json.dumps(rows,separators=(',',':'))+'\n'
        if p.exists() and p.read_text()!=content:raise RuntimeError('Dataset version collision')
        p.write_text(content);records[split]={'count':count,'sha256':sha256(p)}
    save_json(root/'manifest.json',{'version':'G2c-v1','task':task,'length':length,'vocabulary':VOCAB,
             'domain':domain,'disjoint_ids':True,'split_assignment':'SHA256-sorted instance IDs, without replacement',
             'counts':records,'generator_source_sha256':sha256(__file__),
             'claim':'Unseen complete instances from a fixed-length deterministic function; not length generalization'})
    return root


def data(root,split):
    root=Path(root);meta=json.loads((root/'manifest.json').read_text());path=root/f'{split}.json'
    assert sha256(path)==meta['counts'][split]['sha256'];return json.loads(path.read_text())


def pack(records,indices):
    seq=[records[i]['prompt']+records[i]['response'] for i in indices]
    length=max(map(len,seq))-1;x=torch.zeros(len(seq),length,dtype=torch.long);y=torch.full_like(x,-100)
    for j,(i,s) in enumerate(zip(indices,seq)):
        p=len(records[i]['prompt']);x[j,:len(s)-1]=torch.tensor(s[:-1]);y[j,p-1:len(s)-1]=torch.tensor(s[p:])
    return x,y


class GRU(nn.Module):
    def __init__(self,hidden=128,embed=16):
        super().__init__();self.embed=nn.Embedding(VOCAB,embed);self.core=nn.GRU(embed,hidden,batch_first=True);self.head=nn.Linear(hidden,VOCAB)
    def forward(self,x,h=None):
        z,h=self.core(self.embed(x),h);return self.head(z),h


def build(spec):
    torch.manual_seed(spec['seed'])
    if spec['kind']=='teacher':return TinyGPT(**spec['model'])
    if spec['kind']=='gru':return GRU(**spec['model'])
    n,src,dst,_=load_npz(spec['graph']);return GraphLanguageModel(n,src,dst,**spec['model'])


def logits(model,x):
    result=model(x);return result[0] if isinstance(result,tuple) else result


def load(spec,checkpoint):
    ck=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if spec['kind']=='graph':assert ck['graph_sha256']==sha256(spec['graph'])
    model=build(spec);missing,unexpected=model.load_state_dict(ck['model'],strict=False)
    assert set(missing)==({'core.src','core.dst'} if spec['kind']=='graph' else set()) and not unexpected
    return model.eval()


def generate(model,prompt,max_new):
    prefix=torch.tensor([prompt]);answer=[];recurrent=not isinstance(model,TinyGPT)
    if recurrent:z,h=model(prefix)
    for _ in range(max_new):
        if not recurrent:z=model(prefix)
        token=int(z[0,-1].argmax());answer.append(token)
        if token==EOS:break
        byte=torch.tensor([[token]])
        if recurrent:z,h=model(byte,h)
        else:prefix=torch.cat([prefix,byte],1)
    return answer


def score(model,records,teacher=None,ablate=False):
    model.eval();rows=[]
    with torch.no_grad():
        for r in records:
            x,y=pack([r],[0]);mask=y!=-100;z=logits(model,x);count=int(mask.sum())
            answer=generate(model,r['prompt'],len(r['response'])+1)
            item={'id':r['id'],'symbols':count,'nll':float(nn.functional.cross_entropy(z[mask],y[mask],reduction='sum')),
                  'exact':int(answer==r['response']),'generated':answer,
                  'token_correct':int((z[mask].argmax(-1)==y[mask]).sum())}
            if teacher is not None:
                tz=teacher(x);ta=generate(teacher,r['prompt'],len(r['response'])+1)
                item.update(teacher_nll=float(nn.functional.cross_entropy(tz[mask],y[mask],reduction='sum')),
                            kl_sum=float(nn.functional.kl_div(nn.functional.log_softmax(z[mask],-1),nn.functional.softmax(tz[mask],-1),reduction='sum')),
                            exact_teacher_agreement=int(answer==ta),teacher_exact=int(ta==r['response']))
            rows.append(item)
        if ablate:
            weights=model.core.edge_w.detach().clone();model.core.edge_w.zero_()
            for item,r in zip(rows,records):
                x,y=pack([r],[0]);mask=y!=-100;z=logits(model,x)
                item['zero_edge_nll']=float(nn.functional.cross_entropy(z[mask],y[mask],reduction='sum'))
                item['zero_edge_exact']=int(generate(model,r['prompt'],len(r['response'])+1)==r['response'])
            model.core.edge_w.copy_(weights)
    count=sum(r['symbols'] for r in rows)
    result={'accuracy':float(np.mean([r['exact'] for r in rows])),'response_ce':sum(r['nll'] for r in rows)/count,
            'token_accuracy':sum(r['token_correct'] for r in rows)/count,'cases':len(rows),'response_symbols':count}
    for name in ('teacher_nll','kl_sum','zero_edge_nll'):
        if name in rows[0]:result[name.replace('nll','ce').replace('kl_sum','teacher_kl')]=sum(r[name] for r in rows)/count
    for name in ('exact_teacher_agreement','teacher_exact','zero_edge_exact'):
        if name in rows[0]:result[name]=float(np.mean([r[name] for r in rows]))
    return {'metrics':result,'rows':rows}


def qualified(path):
    receipt=json.loads(Path(path).read_text());assert receipt['passed'] and receipt['threshold']==.95
    for p,h in receipt['inputs'].items():assert sha256(p)==h
    return receipt


def archive(checkpoint,spec,step):
    disk=Path('/Volumes/Seagate')
    if not disk.is_mount():raise RuntimeError('External checkpoint archive unavailable; stopping before unpreserved continuation')
    destination=disk/'FlyGPT Backups/G2c-overnight'/Path(spec['job_dir'])/f'step_{step:05d}.pt'
    destination.parent.mkdir(parents=True,exist_ok=True);digest=sha256(checkpoint)
    if not destination.exists():
        tmp=destination.with_suffix('.pt.partial')
        with checkpoint.open('rb') as src,tmp.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
        assert sha256(tmp)==digest;tmp.rename(destination)
    assert sha256(destination)==digest
    return {'step':step,'path':str(destination),'sha256':digest}


def train(spec_path,until):
    spec=json.loads(Path(spec_path).read_text());torch.set_num_threads(4)
    if spec['kind']=='graph':qualified(spec['qualification_receipt'])
    assert until in spec['budgets'];job=Path(spec['job_dir']);job.mkdir(parents=True,exist_ok=True)
    checkpoint=job/'model.pt';progress=job/'progress.json'
    if checkpoint.exists()!=progress.exists():raise RuntimeError('Partial checkpoint publication; inspect before restart')
    model=build(spec);teacher=None
    if spec.get('teacher_spec'):
        ts=json.loads(Path(spec['teacher_spec']).read_text());teacher=load(ts,spec['teacher_checkpoint'])
        for p in teacher.parameters():p.requires_grad_(False)
    auxiliary=None
    if spec['objective']=='hidden':
        auxiliary=nn.Linear(model.core.n_out,teacher.tok.embedding_dim,bias=False)
    parameters=list(model.parameters())+(list(auxiliary.parameters()) if auxiliary is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=spec['lr'],weight_decay=.01,foreach=False)
    generator=torch.Generator().manual_seed(spec['sampling_seed'])
    if spec.get('training_targets'):
        assert sha256(spec['training_targets'])==spec['training_targets_sha256']
        records=json.loads(Path(spec['training_targets']).read_text())
    else:records=data(spec['dataset'],'train')
    inputs=[spec_path,Path(spec['dataset'])/'manifest.json',Path(spec['dataset'])/'train.json']
    if spec['kind']=='graph':inputs += [spec['graph'],spec['qualification_receipt']]
    if teacher is not None:inputs += [spec['teacher_checkpoint'],spec['teacher_spec']]
    if spec.get('training_targets'):inputs += [spec['training_targets']]
    result=manifest(spec,inputs);result['engine_sha256']=sha256(__file__)
    done=0;trace=[];evaluations=[];archives=[];seen=0;previous_wall=0
    if checkpoint.exists():
        old=json.loads(progress.read_text());assert old['inputs']==result['inputs'] and old['engine_sha256']==result['engine_sha256']
        assert sha256(checkpoint)==old['checkpoint_sha256']
        ck=torch.load(checkpoint,weights_only=True,map_location='cpu');missing,unexpected=model.load_state_dict(ck['model'],strict=False)
        assert set(missing)==({'core.src','core.dst'} if spec['kind']=='graph' else set()) and not unexpected
        if auxiliary is not None:auxiliary.load_state_dict(ck['auxiliary'])
        optimizer.load_state_dict(ck['optimizer']);generator.set_state(ck['rng']);done=ck['step']
        trace=old['trace'];evaluations=old['evaluations'];archives=old['archives'];seen=old['response_symbols'];previous_wall=old['wall_seconds'];result=old;del ck
    else:
        shutil.copyfile(__file__,job/'engine_source.py')
    start=time.perf_counter();model.train()
    for step in range(done+1,until+1):
        ids=torch.randint(len(records),(spec['batch'],),generator=generator).tolist();x,y=pack(records,ids);mask=y!=-100
        student_features=[];teacher_features=[];sh=th=None
        if spec['objective'] in ('hidden','relational'):
            sh=model.core.out_proj.register_forward_pre_hook(lambda module,args:student_features.append(args[0]))
            th=teacher.ln.register_forward_hook(lambda module,args,output:teacher_features.append(output.detach()))
        try:
            raw=model(x);z=raw[0] if isinstance(raw,tuple) else raw
            tz=None
            if spec['objective'] in ('kd','curriculum','hidden','relational'):
                with torch.no_grad():tz=teacher(x)
        finally:
            if sh is not None:sh.remove()
            if th is not None:th.remove()
        ce=nn.functional.cross_entropy(z[mask],y[mask]);loss=ce;kl=None
        alignment=None;alpha_used=None
        if spec['objective'] in ('kd','curriculum'):
            temp=spec['temperature'];kl=nn.functional.kl_div(nn.functional.log_softmax(z[mask]/temp,-1),nn.functional.softmax(tz[mask]/temp,-1),reduction='sum')/int(mask.sum())*temp**2
            alpha_used=spec['alpha'] if spec['objective']=='kd' else 1.-.5*(step-1)/(spec['total_updates']-1)
            loss=alpha_used*ce+(1-alpha_used)*kl
        if spec['objective'] in ('hidden','relational'):
            sf=torch.stack(student_features,1);tf=teacher_features[0]
            if spec['objective']=='hidden':
                alignment=(nn.functional.normalize(auxiliary(sf[mask]),dim=-1)-nn.functional.normalize(tf[mask],dim=-1)).square().sum(-1).mean()
            else:
                assert x.shape[0]==2,'Relational condition requires its matched batch-two cohort'
                both=mask[0]&mask[1]
                sn=nn.functional.normalize(sf,dim=-1);tn=nn.functional.normalize(tf,dim=-1)
                alignment=((sn[0,both]*sn[1,both]).sum(-1)-(tn[0,both]*tn[1,both]).sum(-1)).square().mean()
            loss=ce+spec['alignment_weight']*alignment
        optimizer.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite training state')
        optimizer.step();seen+=int(mask.sum())
        item={'step':step,'ce':float(ce.detach()),'loss':float(loss.detach()),'kl':None if kl is None else float(kl.detach()),
              'gradient_norm':float(grad),'response_symbols':seen,
              'alignment':None if alignment is None else float(alignment.detach()),'alpha_used':alpha_used}
        if spec['kind']=='graph':item.update(hidden_rms=float(raw[1].detach().square().mean().sqrt()),saturation=float((raw[1].detach().abs()>.95).float().mean()))
        trace.append(item)
        if step%32==0:print(Path(spec['job_dir']).name,item,flush=True)
        if step in spec['checkpoints'] or step==until:
            if spec['kind']=='graph':
                ev=score(model,data(spec['dataset'],'validation')[:spec['curve_cases']],teacher)
                evaluations.append({'step':step,'validation':ev,'wall_seconds':previous_wall+time.perf_counter()-start})
                print('VALIDATION',step,ev['metrics'],flush=True)
            ck={'model':{k:v for k,v in model.state_dict().items() if k not in ('core.src','core.dst')},
                'optimizer':optimizer.state_dict(),'rng':generator.get_state(),'step':step,'spec_sha256':sha256(spec_path)}
            if auxiliary is not None:ck['auxiliary']=auxiliary.state_dict()
            if spec['kind']=='graph':ck['graph_sha256']=sha256(spec['graph'])
            tmp=checkpoint.with_suffix('.pt.tmp');torch.save(ck,tmp);tmp.replace(checkpoint)
            archives.append(archive(checkpoint,spec,step))
            result.update(step=step,session_budget=until,trace=trace,evaluations=evaluations,archives=archives,response_symbols=seen,
                          checkpoint_sha256=sha256(checkpoint),wall_seconds=previous_wall+time.perf_counter()-start)
            save_json(progress,result);model.train()


def evaluate(spec_path,checkpoint,split,out,unlock=None):
    spec=json.loads(Path(spec_path).read_text());torch.set_num_threads(4)
    if split.startswith('test_'):
        if not unlock:raise RuntimeError('Final test locked without completed matched-cohort receipt')
        receipt=json.loads(Path(unlock).read_text());assert receipt['split']==split and receipt['choices_frozen']
        for p,h in receipt['inputs'].items():assert sha256(p)==h
    if split.startswith('qualification'):
        assert spec['kind']=='teacher'
        val=json.loads((Path(spec['job_dir'])/'validation.json').read_text())
        if val['metrics']['accuracy']<.95:raise RuntimeError('Qualification locked until validation accuracy reaches 95%')
    rows=data(spec['dataset'],'train' if split=='training_probe' else split)
    if split=='training_probe':rows=rows[:64]
    model=load(spec,checkpoint);teacher=None
    if spec.get('teacher_spec'):
        teacher=load(json.loads(Path(spec['teacher_spec']).read_text()),spec['teacher_checkpoint'])
    result=manifest({'spec':spec_path,'split':split,'teacher_forcing':'CE only; exact responses generated from prompt'},[spec_path,checkpoint,Path(spec['dataset'])/'manifest.json'])
    result.update(score(model,rows,teacher,spec['kind']=='graph' and split.startswith('test_')))
    save_json(out,result);print(split,result['metrics'],flush=True)


def baseline(dataset,split='validation'):
    if split.startswith('test_'):raise RuntimeError('Baseline final test requires the cohort evaluator')
    counts=[defaultdict(Counter) for _ in range(7)];training=data(dataset,'train')
    for r in training:
        seq=r['prompt']+r['response'];p=len(r['prompt'])
        for i in range(p,len(seq)):
            for k in range(min(i,6)+1):counts[k][tuple(seq[i-k:i])][seq[i]]+=1
    def probability(prefix,order):
        k=min(len(prefix),order)
        while k and tuple(prefix[-k:]) not in counts[k]:k-=1
        counter=counts[k][tuple(prefix[-k:]) if k else ()];v=np.full(VOCAB,.1)
        for token,count in counter.items():v[token]+=count
        return v/v.sum()
    reports={}
    for order in (1,3,6):
        ce=correct=total=0
        for r in data(dataset,split):
            prefix=r['prompt'].copy()
            for token in r['response']:ce-=math.log(probability(prefix,order)[token]);total+=1;prefix.append(token)
            prefix=r['prompt'].copy();answer=[]
            for _ in range(len(r['response'])+1):
                token=int(probability(prefix,order).argmax());answer.append(token);prefix.append(token)
                if token==EOS:break
            correct+=answer==r['response']
        reports[f'ngram{order}']={'accuracy':correct/len(data(dataset,split)),'response_ce':ce/total}
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--spec',required=True)
    p.add_argument('--until',type=int);p.add_argument('--checkpoint');p.add_argument('--split');p.add_argument('--out');p.add_argument('--unlock')
    a=p.parse_args()
    if a.mode=='train':train(a.spec,a.until)
    else:evaluate(a.spec,a.checkpoint,a.split,a.out,a.unlock)
