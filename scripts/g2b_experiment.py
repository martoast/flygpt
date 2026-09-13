"""G2b response-only training, baselines and locked qualification evaluation."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
from numba import njit,prange,set_num_threads
import torch
from torch import nn
from src.provenance import manifest,save_json,sha256
from src.tinygpt import TinyGPT
from src.train_memory import GRULanguageModel,parameter_matched_gru,build_model
from src.query_flygpt import load_model

ROOT=Path('results/g2b_v1');DATA=Path('data/raw/grammar_g2b_v1')
PROTOCOL=Path('configs/g2b_qualification_v1.json');EXECUTION=Path('configs/g2b_execution_v1.json')


def config():return json.loads(PROTOCOL.read_text()),json.loads(EXECUTION.read_text())


def student_schedule():
    _,execution=config()
    return [(kind,seed) for seed in execution['student_seeds'] for kind in execution['student_conditions']]


def verify_gate(split):
    path=ROOT/f'gate_{split}.json'
    if not path.exists():raise RuntimeError(f'No passing {split} qualification gate')
    gate=json.loads(path.read_text())
    if not gate['passed']:raise RuntimeError(f'Teacher failed the {split} qualification gate')
    assert gate['protocol_sha256']==sha256(PROTOCOL) and gate['execution_sha256']==sha256(EXECUTION)
    for source,digest in gate['inputs'].items():assert sha256(source)==digest,'Gate evidence changed'
    return gate


def read_data(split):
    if split=='qualification':verify_gate('validation')
    if split=='test':
        verify_gate('validation');verify_gate('qualification')
        for kind,seed in student_schedule():
            path=ROOT/f'{kind}_{seed}.json'
            if not path.exists() or not json.loads(path.read_text()).get('complete'):
                raise RuntimeError('Student test locked until all scheduled student training finishes')
    if split not in ('train','validation','qualification','test'):raise ValueError(split)
    meta=json.loads((DATA/'manifest.json').read_text());groups={}
    for path in sorted(DATA.glob(f'{split}_*.jsonl')):
        assert sha256(path)==meta['audit'][path.stem]['sha256']
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if split=='train':return records
        axis=path.stem.split('_',1)[1]
        for operation in ('swap','roles'):
            groups[f'{axis}_{operation}']=[r for r in records if r['operation']==operation]
    return groups


def pack(records,indices):
    sequences=[(records[i]['prompt']+records[i]['response']).encode() for i in indices]
    length=max(len(s)-1 for s in sequences)
    x=torch.zeros(len(indices),length,dtype=torch.long);y=torch.full_like(x,-100)
    for j,(i,s) in enumerate(zip(indices,sequences)):
        p=len(records[i]['prompt'].encode())
        x[j,:len(s)-1]=torch.tensor(list(s[:-1]))
        y[j,p-1:len(s)-1]=torch.tensor(list(s[p:]))
    return x,y


def logits(model,x):
    result=model(x)
    return result[0] if isinstance(result,tuple) else result


def teacher_config(protocol):
    t=protocol['teacher']
    return dict(vocab_size=256,d_model=t['embedding'],n_head=t['heads'],n_layer=t['layers'],max_len=t['context_bytes'],dropout=0.)


def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True);kind=ck['kind']
    if kind=='teacher':model=TinyGPT(**ck['model_config'])
    elif kind=='gru':model=GRULanguageModel(**ck['model_config'])
    else:return load_model(ck['graph'],path,'cpu')
    model.load_state_dict(ck['model']);return model.eval()


def train(kind,seed):
    torch.set_num_threads(4);torch.manual_seed(seed)
    protocol,execution=config();student=kind in execution['student_conditions']
    if student:verify_gate('validation');verify_gate('qualification')
    path=ROOT/f'{kind}_{seed}.pt';progress=path.with_suffix('.json');ROOT.mkdir(parents=True,exist_ok=True)
    if path.exists()!=progress.exists():raise RuntimeError('Incomplete checkpoint/manifest pair; inspect before retry')
    mc=teacher_config(protocol);teacher=None;graph=None
    if kind=='teacher':model=TinyGPT(**mc)
    elif kind=='gru':
        reference=TinyGPT(**mc);target=sum(p.numel() for p in reference.parameters());del reference
        torch.manual_seed(seed)
        mc={'hidden':parameter_matched_gru(target,128),'embed_dim':128};model=GRULanguageModel(**mc)
    elif student:
        graph=f'data/processed/controls/rewired_{777+seed}.npz' if kind.startswith('rewired') else 'data/processed/malecns.npz'
        if kind.startswith('rewired'):
            control=json.loads(Path(f'results/malecns_v1/controls/rewired_{777+seed}.json').read_text())
            assert sha256(graph)==control['graph_sha256']
        mc=execution['student_model'];model=build_model(graph,'real',seed,mc)
        teacher=load(ROOT/f'teacher_{seed}.pt')
        for p in teacher.parameters():p.requires_grad_(False)
    else:raise ValueError(kind)
    # Constructor RNG consumption can differ between model classes; sampling
    # uses its own generator, paired across conditions independently of that.
    generator=torch.Generator().manual_seed((execution['student_sampling_seed_base'] if student else execution['training_sampling_seed_base'])+seed)
    steps=execution['student_steps'] if student else protocol['updates']
    batch_size=execution['student_batch'] if student else protocol['batch']
    lr=execution['student_lr'] if student else protocol['learning_rate']
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=protocol['weight_decay'],foreach=False)
    records=read_data('train');inputs=[PROTOCOL,EXECUTION,DATA/'manifest.json',DATA/'train_familiar.jsonl']
    if student:inputs += [graph,ROOT/f'teacher_{seed}.pt',ROOT/'gate_qualification.json']
    record=manifest({'kind':kind,'seed':seed,'model':mc,'steps':steps,'batch':batch_size,'lr':lr},inputs)
    record['script_sha256']=sha256(__file__);record['parameters']=sum(p.numel() for p in model.parameters())
    done=0;trace=[];seen=0;wall=0
    if path.exists():
        old=json.loads(progress.read_text())
        assert old['inputs']==record['inputs'] and old['script_sha256']==record['script_sha256']
        assert old['config']==record['config'] and sha256(path)==old['checkpoint_sha256']
        if old['complete']:return
        ck=torch.load(path,weights_only=True,map_location='cpu')
        missing,unexpected=model.load_state_dict(ck['model'],strict=False)
        assert set(missing)==({'core.src','core.dst'} if student else set()) and not unexpected
        optimizer.load_state_dict(ck['optimizer']);generator.set_state(ck['sampling_rng'])
        done=ck['step'];trace=old['trace'];seen=old['response_bytes'];wall=old['runtime_seconds'];record=old;del ck
    start=time.perf_counter();model.train()
    for step in range(done+1,steps+1):
        indices=torch.randint(len(records),(batch_size,),generator=generator).tolist()
        x,y=pack(records,indices);z=logits(model,x);mask=y!=-100
        ce=nn.functional.cross_entropy(z.reshape(-1,256),y.flatten());loss=ce
        if kind.endswith('_kd'):
            with torch.no_grad():tz=teacher(x)
            temperature=execution['student_temperature'];alpha=execution['student_alpha']
            kl=nn.functional.kl_div(nn.functional.log_softmax(z/temperature,-1),nn.functional.softmax(tz/temperature,-1),reduction='none').sum(-1)[mask].mean()*temperature**2
            loss=alpha*ce+(1-alpha)*kl
        optimizer.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite training state')
        optimizer.step();seen+=int(mask.sum())
        trace.append({'step':step,'response_ce':float(ce.detach()),'loss':float(loss.detach()),'gradient_norm':float(grad),'response_bytes':seen})
        if step%25==0:print(kind,seed,trace[-1],flush=True)
        if step%100==0 or step==steps:
            ck={'model':{k:v for k,v in model.state_dict().items() if k not in ('core.src','core.dst')},
                'kind':kind,'model_config':mc,'config':mc,'seed':seed,'step':step,
                'optimizer':optimizer.state_dict(),'sampling_rng':generator.get_state(),
                'code_commit':record['code_commit'],'topology_buffers_external':student}
            if graph:ck.update(graph=graph,graph_sha256=sha256(graph))
            tmp=path.with_suffix('.pt.tmp');torch.save(ck,tmp);tmp.replace(path)
            record.update(trace=trace,response_bytes=seen,runtime_seconds=wall+time.perf_counter()-start,
                          complete=step==steps,checkpoint_sha256=sha256(path))
            save_json(progress,record)


def generate(model,prompt,max_new=64):
    sequence=torch.tensor([list(prompt.encode())],dtype=torch.long);answer=[]
    recurrent=not isinstance(model,TinyGPT)
    if recurrent:z,h=model(sequence)
    for _ in range(max_new):
        if not recurrent:z=model(sequence)
        token=int(z[0,-1].argmax());answer.append(token)
        if token==10:break
        next_byte=torch.tensor([[token]])
        if recurrent:z,h=model(next_byte,h)
        else:sequence=torch.cat((sequence,next_byte),dim=1)
    return bytes(answer)


def evaluate(kind,seed,split):
    torch.set_num_threads(4);groups=read_data(split);path=ROOT/f'{kind}_{seed}.pt'
    checkpoint=json.loads(path.with_suffix('.json').read_text());assert checkpoint['complete']
    assert checkpoint['checkpoint_sha256']==sha256(path)
    model=load(path);protocol,execution=config();student=kind in execution['student_conditions']
    teacher=load(ROOT/f'teacher_{seed}.pt') if student else None
    rows={};start=time.perf_counter()
    with torch.no_grad():
        for category,records in groups.items():
            details=[]
            for i,r in enumerate(records):
                x,y=pack([r],[0]);z=logits(model,x);mask=y!=-100;count=int(mask.sum())
                nll=float(nn.functional.cross_entropy(z[mask],y[mask],reduction='sum'))
                prediction=generate(model,r['prompt'])
                item={'case_index':i,'subject_attribute':r['subject_attribute'],'subject_relation':r['subject_relation'],
                      'response_bytes':count,'nll':nll,'ce':nll/count,'exact':int(prediction==r['response'].encode()),
                      'prediction_bytes':list(prediction)}
                if teacher is not None:
                    tz=teacher(x)
                    item['teacher_kl']=float(nn.functional.kl_div(nn.functional.log_softmax(z[mask],-1),nn.functional.softmax(tz[mask],-1),reduction='sum')/count)
                details.append(item)
            rows[category]=details
            print(kind,seed,split,category,aggregate(details),flush=True)
        if student:
            weights=model.core.edge_w.detach().clone();model.core.edge_w.zero_()
            for category,records in groups.items():
                for item,r in zip(rows[category],records):
                    x,y=pack([r],[0]);mask=y!=-100;z=logits(model,x)
                    item['zero_edge_ce']=float(nn.functional.cross_entropy(z[mask],y[mask]))
                    item['zero_edge_exact']=int(generate(model,r['prompt'])==r['response'].encode())
            model.core.edge_w.copy_(weights)
    output=manifest({'kind':kind,'seed':seed,'split':split,'response_only':True},[path,PROTOCOL,EXECUTION,DATA/'manifest.json'])
    output.update(rows=rows,metrics={k:aggregate(v) for k,v in rows.items()},runtime_seconds=time.perf_counter()-start)
    save_json(ROOT/split/f'{kind}_{seed}.json',output)


def aggregate(rows):
    result={'exact_accuracy':float(np.mean([r['exact'] for r in rows])),'cases':len(rows)}
    if 'nll' in rows[0]:result['ce']=sum(r['nll'] for r in rows)/sum(r['response_bytes'] for r in rows)
    for key in ('teacher_kl','zero_edge_ce','zero_edge_exact'):
        if key in rows[0]:result[key]=float(np.average([r[key] for r in rows],weights=[r['response_bytes'] if key!='zero_edge_exact' else 1 for r in rows]))
    return result


class NGram:
    def __init__(self,training,smoothing=.1):
        self.counts=[defaultdict(Counter) for _ in range(17)];self.smoothing=smoothing
        for r in training:
            data=(r['prompt']+r['response']).encode();p=len(r['prompt'].encode())
            for i in range(p,len(data)):
                for order in range(min(i,16)+1):self.counts[order][data[i-order:i]][data[i]]+=1
    def distribution(self,prefix,order):
        k=min(order,len(prefix))
        while k and prefix[len(prefix)-k:] not in self.counts[k]:k-=1
        counts=self.counts[k][prefix[len(prefix)-k:] if k else b'']
        values=np.full(256,self.smoothing)
        for token,count in counts.items():values[token]+=count
        return values/values.sum()
    def score(self,row,order):
        prefix=row['prompt'].encode();nll=0
        for token in row['response'].encode():
            nll-=math.log(self.distribution(prefix,order)[token]);prefix+=bytes([token])
        prefix=row['prompt'].encode();answer=bytearray()
        for _ in range(64):
            token=int(self.distribution(prefix,order).argmax());answer.append(token);prefix+=bytes([token])
            if token==10:break
        return nll,int(bytes(answer)==row['response'].encode())


@njit(parallel=True)
def distances(query,training,lengths):
    result=np.empty(len(lengths),dtype=np.int64)
    for i in prange(len(lengths)):
        previous=np.arange(len(query)+1);current=np.empty(len(query)+1,dtype=np.int64)
        for j in range(lengths[i]):
            current[0]=j+1
            for k in range(len(query)):
                current[k+1]=min(current[k]+1,previous[k+1]+1,previous[k]+int(query[k]!=training[i,j]))
            previous,current=current,previous
        result[i]=previous[-1]
    return result


def baselines(split):
    groups=read_data(split);training=sorted(read_data('train'),key=lambda r:r['prompt']);_,execution=config()
    ngram=NGram(training,execution['ngram_smoothing']);encoded=[r['prompt'].encode() for r in training]
    lengths=np.array([len(s) for s in encoded]);matrix=np.zeros((len(encoded),int(lengths.max())),dtype=np.uint8)
    for i,s in enumerate(encoded):matrix[i,:len(s)]=np.frombuffer(s,dtype=np.uint8)
    set_num_threads(4);output={name:{} for name in ['ngram1','ngram4','ngram8','ngram16','retrieval']}
    for category,records in groups.items():
        for key in output:output[key][category]=[]
        for i,r in enumerate(records):
            common={'case_index':i,'subject_attribute':r['subject_attribute'],'subject_relation':r['subject_relation'],'response_bytes':len(r['response'].encode())}
            for order in [1,4,8,16]:
                nll,exact=ngram.score(r,order)
                output[f'ngram{order}'][category].append({**common,'nll':nll,'ce':nll/common['response_bytes'],'exact':exact})
            nearest=int(distances(np.frombuffer(r['prompt'].encode(),dtype=np.uint8),matrix,lengths).argmin())
            output['retrieval'][category].append({**common,'exact':int(training[nearest]['response']==r['response']),'retrieved_prompt':training[nearest]['prompt']})
        print('BASELINES',split,category,{name:aggregate(v[category]) for name,v in output.items()},flush=True)
    for name,rows in output.items():
        record=manifest({'baseline':name,'split':split,'fit':'training responses only; retrieval uses training prompts'},[PROTOCOL,EXECUTION,DATA/'manifest.json',DATA/'train_familiar.jsonl'])
        record.update(rows=rows,metrics={k:aggregate(v) for k,v in rows.items()})
        save_json(ROOT/split/f'{name}.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate','baselines'])
    parser.add_argument('--kind');parser.add_argument('--seed',type=int,default=0);parser.add_argument('--split',default='validation',choices=['validation','qualification','test'])
    args=parser.parse_args()
    if args.mode=='train':train(args.kind,args.seed)
    elif args.mode=='evaluate':evaluate(args.kind,args.seed,args.split)
    else:baselines(args.split)
