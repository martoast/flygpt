import numpy as np
import pytest
import torch
from scipy import sparse
from src.model import SparseGraphRNN
from src.sparse_ops import CSRMultiply
from src.query_flygpt import generate


def test_sparse_gradient_matches_dense():
    a = sparse.csr_matrix(np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]))
    h = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    w = torch.randn(a.nnz, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,y: CSRMultiply.apply(x,y,a.indices,a.indptr),(h,w))
    row = np.repeat(np.arange(3), np.diff(a.indptr))
    dense = torch.zeros(3,3,dtype=torch.double).index_put((torch.tensor(row),torch.tensor(a.indices)),w)
    expected=h@dense.T
    actual=CSRMultiply.apply(h,w,a.indices,a.indptr)
    torch.testing.assert_close(actual,expected)


def test_no_bypass_and_prompt_dependence():
    src=np.array([0,1]); dst=np.array([1,2])
    m=SparseGraphRNN(3,src,dst,1,2,input_fraction=.34,output_fraction=.34,inner_steps=3,backend='scipy')
    with torch.no_grad():
        m.edge_w.fill_(1); m.in_proj.weight.fill_(1); m.out_proj.weight.fill_(1)
    a=torch.tensor([[[1.],[0.],[0.]]]); b=-a
    assert not torch.allclose(m(a)[0],m(b)[0])
    with torch.no_grad(): m.edge_w.zero_()
    torch.testing.assert_close(m(a)[0],m(b)[0])


def test_overlapping_populations_rejected():
    with pytest.raises(ValueError):
        SparseGraphRNN(3,[0],[1],1,2,input_fraction=.8,output_fraction=.8)


def test_generation_consumes_prompt_once():
    class RecordingModel:
        def __init__(self): self.seen=[]
        def __call__(self,x,h=None):
            self.seen.extend(x.flatten().tolist())
            z=torch.full((1,x.shape[1],256),-1e6); z[:,:,120]=0
            return z,torch.ones(1)
    m=RecordingModel()
    assert generate(m,b'ab',max_new=2,top_k=1)==b'abxx'
    assert m.seen==[97,98,120,120]


def test_controls_preserve_degrees():
    from src.graph_controls import make_control
    rng=np.random.default_rng(2)
    keys=rng.choice(20*20,100,replace=False);s=keys//20;d=keys%20
    for condition in ['rewired','configuration','er']:
        x,y,meta=make_control(20,s,d,condition,3,swap_factor=2)
        assert len(x)==len(s)
        if condition!='er':
            np.testing.assert_array_equal(np.bincount(x,minlength=20),np.bincount(s,minlength=20))
            np.testing.assert_array_equal(np.bincount(y,minlength=20),np.bincount(d,minlength=20))
        if condition!='configuration':
            assert len(np.unique(x*20+y))==len(x)
            assert np.sum(x==y)==np.sum(s==d)


def test_preprocessing_aggregates_before_threshold_and_retains_isolates(tmp_path,monkeypatch):
    import pandas as pd
    from src.prepare_malecns import prepare
    monkeypatch.chdir(tmp_path)
    # Avoid depending on a Git checkout in this miniature fixture.
    monkeypatch.setattr('src.prepare_malecns.manifest',lambda *a,**k:{})
    pd.DataFrame({'bodyId':[1,2,3,4],'superclass':['x','x','x',None],
                  'status':['Traced']*4,'somaNeuromere':[None]*4,'type':[None]*4}).to_feather('a.feather')
    pd.DataFrame({'body_pre':[1,1,4],'body_post':[2,2,1],'weight':[1,1,99]}).to_feather('g.feather')
    a=prepare('g.feather','g.npz','a.feather',min_weight=2)
    assert a.shape==(3,3) and a.nnz==1 and a[0,1]==2


def test_rewire_table_rebuild_keeps_exact_rng_trajectory():
    from src.graph_controls import swaps
    rng=np.random.default_rng(45);keys=rng.choice(100*100,600,replace=False)
    s=keys//100;d=keys%100
    old,done_old,attempts_old=swaps(s.copy(),d.copy(),100,6000,777,0)
    rebuilt,done_new,attempts_new=swaps(s.copy(),d.copy(),100,6000,777,600)
    np.testing.assert_array_equal(old,rebuilt)
    assert (done_old,attempts_old)==(done_new,attempts_new)


def test_seeded_initialization_reconstruction_for_weight_audit():
    from src.graph_lm import GraphLanguageModel
    rng=np.random.default_rng(8);keys=rng.choice(100*100,600,replace=False)
    s=keys//100;d=keys%100
    torch.manual_seed(0)
    model=GraphLanguageModel(100,s,d,embed_dim=16,edge_scale=.9,backend='scipy',degree_normalize=True)
    torch.manual_seed(0);torch.randn(256,16)
    initial=torch.randn_like(model.core.edge_w)*.9
    degree=torch.bincount(model.core.dst,minlength=100).clamp_min(1)
    initial/=degree[model.core.dst].sqrt()
    torch.testing.assert_close(initial,model.core.edge_w,rtol=0,atol=0)
