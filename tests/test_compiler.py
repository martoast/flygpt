import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy import sparse
from scripts import compiler_engine as c
from scripts import g2c_engine as e
from src.provenance import save_json,sha256

@pytest.fixture
def small(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    rows=[{'id':i,'prompt':[5,i,6],'response':[(i+1)%4,4]} for i in range(4)]
    root=tmp_path/'dataset';save_json(root/'train.json',rows);save_json(root/'validation.json',rows)
    save_json(root/'manifest.json',{'counts':{s:{'sha256':sha256(root/f'{s}.json')} for s in ('train','validation')}})
    targets=tmp_path/'targets.json';save_json(targets,rows)
    g=tmp_path/'graph.npz';n=12;src=np.arange(n);dst=(src+1)%n
    sparse.save_npz(g,sparse.csr_matrix((np.ones(n,np.float32),(src,dst)),shape=(n,n)))
    ts=tmp_path/'teacher.json';save_json(ts,{'kind':'teacher','seed':2,'model':{'vocab_size':8,'d_model':8,'n_head':2,'n_layer':1,'max_len':16,'dropout':0.}})
    teacher=e.build(json.loads(ts.read_text()));ck=tmp_path/'teacher.pt';torch.save({'model':teacher.state_dict()},ck)
    receipt=tmp_path/'receipt.json';save_json(receipt,{})
    for module in (c,e):
        monkeypatch.setattr(module,'qualified',lambda path:True)
        monkeypatch.setattr(module,'manifest',lambda config,inputs:{'config':config,'inputs':{str(p):sha256(p) for p in inputs}})
        monkeypatch.setattr(module,'archive',lambda checkpoint,spec,step:{'step':step,'path':str(checkpoint),'sha256':sha256(checkpoint)})
    def spec(name,objective='ce',batch=1,hard=False):
        p=tmp_path/f'{name}.json'
        s={'kind':'graph','seed':12,'sampling_seed':20,'model':{'vocab_size':8,'embed_dim':4,'input_fraction':.25,'output_fraction':.25,'inner_steps':2,'backend':'scatter','population_seed':3},
            'objective':objective,'lr':.001,'batch':batch,'budgets':[2,4],'checkpoints':[2,4],'curve_cases':2,'dataset':str(root),'job_dir':str(tmp_path/name),
            'graph':str(g),'qualification_receipt':str(receipt),'teacher_spec':str(ts),'teacher_checkpoint':str(ck),'temperature':2.,'alpha':.5,'total_updates':4,'alignment_weight':1.}
        if hard:s.update(training_targets=str(targets),training_targets_sha256=sha256(targets))
        save_json(p,s);return p
    return spec,tmp_path,targets


def test_teacher_hard_path_matches_ce_without_reading_original_train(small,monkeypatch):
    make,root,_=small;p=make('ce');q=make('hard',hard=True)
    e.train(p,4)
    original=c.data
    def guarded(root,split):
        assert split!='train','Teacher-only engine must not read original training labels'
        return original(root,split)
    monkeypatch.setattr(c,'data',guarded)
    c.train(q,4)
    a=torch.load(root/'ce/model.pt',weights_only=True);b=torch.load(root/'hard/model.pt',weights_only=True)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert torch.equal(a['rng'],b['rng'])

@pytest.mark.parametrize('objective,batch',[('kd',1),('curriculum',1),('hidden',1),('relational',2)])
def test_objectives_resume_and_keep_original_query_architecture(small,objective,batch):
    make,root,_=small;p=make('continuous',objective,batch,True);q=make('resume',objective,batch,True)
    c.train(p,4);c.train(q,2);c.train(q,4)
    a=torch.load(root/'continuous/model.pt',weights_only=True);b=torch.load(root/'resume/model.pt',weights_only=True)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    for k,values in a['optimizer']['state'].items():
        assert all(torch.equal(v,b['optimizer']['state'][k][name]) for name,v in values.items())
    if objective=='hidden':
        assert all(torch.equal(a['auxiliary'][k],b['auxiliary'][k]) for k in a['auxiliary'])
    model=e.load(json.loads(q.read_text()),root/'resume/model.pt')
    assert not any('auxiliary' in k for k in model.state_dict())
    model.core.edge_w.data.zero_()
    with torch.no_grad():
        z,_=model(torch.tensor([[5,0,6],[5,1,6]]))
    assert torch.equal(z[0],z[1]),'No input-to-output bypass may survive zero-edge recurrence'


def test_offline_checkpoint_migrates_without_changing_reference(tmp_path,monkeypatch):
    from scripts import compiler_storage as storage
    disk=tmp_path/'usb';disk.mkdir();pending=tmp_path/'pending';backup=disk/'backup';state={'connected':False}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(storage,'DISK',disk);monkeypatch.setattr(storage,'BACKUP',backup);monkeypatch.setattr(storage,'PENDING',pending)
    monkeypatch.setattr(Path,'is_mount',lambda self:self==disk and state['connected'])
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'checkpoint bytes')
    result=storage.archive(checkpoint,{'job_dir':'results/compiler_v1/example'},64)
    reference=Path(result['path']);assert reference.read_bytes()==checkpoint.read_bytes()
    assert result['storage']=='local_pending_external_backup' and not reference.is_symlink()
    state['connected']=True;storage.flush_pending()
    assert reference.is_symlink() and reference.read_bytes()==checkpoint.read_bytes()
    assert sha256(reference)==result['sha256']
