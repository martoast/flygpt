import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy import sparse
from scripts import compiler_v3_engine as c
from scripts import g2c_engine as e
from src.provenance import save_json,sha256

def test_fixed_groups_derangement_and_numerical_audit():
    plan=c.schedule(32,8,20,300)
    assert plan==c.schedule(32,8,20,300)
    assert sorted(sum(plan['groups'],[]))==list(range(32))
    for p in plan['derangements']:assert all(i!=j for i,j in enumerate(p))
    tf=torch.randn(32,4,7,generator=torch.Generator().manual_seed(4));mask=torch.ones(32,4,dtype=torch.bool)
    assert c.audit_schedule(plan,tf,mask,1e-6)['passed']
    assert not c.audit_schedule(plan,torch.ones_like(tf),mask,1e-6)['passed']
    with pytest.raises(ValueError):c.schedule(32,2,20,300)

def test_relation_targets_require_index_correspondence():
    tf=torch.randn(8,3,7,generator=torch.Generator().manual_seed(2));mask=torch.ones(8,3,dtype=torch.bool)
    perm=torch.arange(8).roll(1)
    assert c.alignment_loss(tf,tf,mask,torch.nn.Identity(),'relational')==0
    assert c.alignment_loss(tf,tf[perm],mask,torch.nn.Identity(),'relational')>0
    # Demonstrate why the rejected batch-two control is mathematically ineffective.
    assert torch.allclose(c.geometry(tf[:2]),c.geometry(tf[:2])[...,[1,0],:][...,[1,0]],atol=1e-6,rtol=0)

def test_soft_loss_ignores_hard_answer_values():
    z=torch.randn(4,2,8,requires_grad=True);tz=torch.randn_like(z);y=torch.zeros(4,2,dtype=torch.long)
    a=c.objective('B_soft',z,y,None,None,tz,None,None,{'temperature':2.})[0]
    b=c.objective('B_soft',z,y+1,None,None,tz,None,None,{'temperature':2.})[0]
    assert torch.equal(a,b)

@pytest.fixture
def small(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    rows=[{'id':i,'prompt':[5,i%4,6],'response':[(i+1)%4,4]} for i in range(16)]
    data=tmp_path/'data';save_json(data/'validation.json',rows[:4]);save_json(data/'manifest.json',{'counts':{'validation':{'sha256':sha256(data/'validation.json')}}})
    targets=tmp_path/'targets.json';save_json(targets,rows)
    graph=tmp_path/'graph.npz';n=12;sparse.save_npz(graph,sparse.csr_matrix((np.ones(n),(np.arange(n),(np.arange(n)+1)%n)),shape=(n,n)))
    ts=tmp_path/'teacher.json';save_json(ts,{'kind':'teacher','seed':2,'model':{'vocab_size':8,'d_model':8,'n_head':2,'n_layer':1,'max_len':16,'dropout':0.}})
    teacher=e.build(json.loads(ts.read_text())).eval();teacher_ck=tmp_path/'teacher.pt';torch.save({'model':teacher.state_dict()},teacher_ck)
    receipt=tmp_path/'receipt.json';save_json(receipt,{})
    x,y=e.pack(rows,list(range(16)));capture=[];hook=teacher.ln.register_forward_hook(lambda m,a,o:capture.append(o.detach()))
    with torch.no_grad():z=teacher(x)
    hook.remove();cache=tmp_path/'cache.pt';torch.save({'features':capture[0],'logits':z,'mask':y!=-100,'ids':list(range(16))},cache)
    cache_receipt=tmp_path/'cache.json';save_json(cache_receipt,{'cache_sha256':sha256(cache),'hard_answer_generation_seconds':1.,'representation_extraction_and_cache_seconds':2.})
    plan=c.schedule(16,8,4,9);schedule=tmp_path/'schedule.json';save_json(schedule,plan)
    audit=c.audit_schedule(plan,capture[0],y!=-100,1e-6);assert audit['passed']
    audit.update(schedule_sha256=sha256(schedule),cache_sha256=sha256(cache));audit_path=tmp_path/'audit.json';save_json(audit_path,audit)
    monkeypatch.setattr(e,'qualified',lambda p:True)
    monkeypatch.setattr(c,'archive',lambda p,s,step:{'step':step,'path':str(p),'sha256':sha256(p)})
    original=e.data
    def guarded(root,split):
        assert split!='train','Student may not read original training labels'
        return original(root,split)
    monkeypatch.setattr(e,'data',guarded)
    def make(name,method):
        spec={'kind':'graph','seed':9,'projection_seed':77,'method':method,'batch':8,'lr':.001,'temperature':2.,'alignment_weight':1.,
            'model':{'vocab_size':8,'embed_dim':4,'input_fraction':.25,'output_fraction':.25,'inner_steps':2,'backend':'scatter','population_seed':3},
            'budgets':[2,4],'checkpoints':[2,4],'curve_cases':2,'dataset':str(data),'job_dir':str(tmp_path/name),'graph':str(graph),
            'qualification_receipt':str(receipt),'teacher_spec':str(ts),'teacher_checkpoint':str(teacher_ck),
            'training_targets':str(targets),'training_targets_sha256':sha256(targets),'cache':str(cache),'cache_receipt':str(cache_receipt),
            'schedule':str(schedule),'schedule_audit':str(audit_path)}
        p=tmp_path/f'{name}.json';save_json(p,spec);return p
    return make,tmp_path

@pytest.mark.parametrize('method',c.METHODS)
def test_resume_and_no_auxiliary_query_bypass(small,method):
    make,root=small;p=make('continuous',method);q=make('resumed',method)
    c.train(p,4);c.train(q,2);c.train(q,4)
    a=torch.load(root/'continuous/model.pt',weights_only=True);b=torch.load(root/'resumed/model.pt',weights_only=True)
    for k in a['model']:assert torch.equal(a['model'][k],b['model'][k])
    for k,v in a['optimizer']['state'].items():
        for name,value in v.items():assert torch.equal(value,b['optimizer']['state'][k][name])
    if 'auxiliary' in a:
        for k in a['auxiliary']:assert torch.equal(a['auxiliary'][k],b['auxiliary'][k])
    model=e.load(json.loads(q.read_text()),root/'resumed/model.pt');assert not any('auxiliary' in k for k in model.state_dict())
    model.core.edge_w.data.zero_()
    with torch.no_grad():z,_=model(torch.tensor([[5,0,6],[5,1,6]]))
    assert torch.equal(z[0],z[1])
    progress=json.loads((root/'resumed/progress.json').read_text());assert progress['wall_seconds']>=progress['acquisition_seconds']

def test_threshold_is_observed_and_censored():
    p={'evaluations':[{'step':32,'validation':{'metrics':{'accuracy':.5}},'examples_seen':256,'acquisition_seconds':10.,'wall_seconds':12.},
                      {'step':64,'validation':{'metrics':{'accuracy':.875}},'examples_seen':512,'acquisition_seconds':20.,'wall_seconds':24.}],
       'acquisition_seconds':20.,'wall_seconds':24.}
    result=c.learning_summary(p,.85);assert result['crossing_update_interval']==[32,64]
    assert c.learning_summary(p,.95)['right_censored_after_updates']==64
