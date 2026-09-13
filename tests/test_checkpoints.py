import json
import numpy as np
import pytest
import torch
from scipy import sparse
from src.graph_lm import GraphLanguageModel
from src.provenance import sha256
from src.query_flygpt import load_model
from src.tinygpt import TinyGPT


def test_compact_checkpoint_roundtrip_and_wrong_graph_rejection(tmp_path):
    a=sparse.csr_matrix(np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]));graph=tmp_path/'g.npz';sparse.save_npz(graph,a)
    coo=a.tocoo();cfg=dict(embed_dim=2,input_fraction=.34,output_fraction=.34,backend='scipy',population_seed=9)
    model=GraphLanguageModel(3,coo.row,coo.col,**cfg)
    ckpt=tmp_path/'m.pt';torch.save({'model':{k:v for k,v in model.state_dict().items() if k not in ('core.src','core.dst')},
                                   'config':cfg,'graph_sha256':sha256(graph),'topology_buffers_external':True},ckpt)
    loaded=load_model(graph,ckpt,'cpu');x=torch.tensor([[97,98,99]])
    torch.testing.assert_close(model(x)[0],loaded(x)[0])
    wrong=tmp_path/'wrong.npz';sparse.save_npz(wrong,a.T)
    with pytest.raises(ValueError,match='hash'):load_model(wrong,ckpt,'cpu')


def test_teacher_checkpoint_and_causal_attention(tmp_path):
    config=dict(d_model=16,n_head=4,n_layer=1,max_len=8)
    model=TinyGPT(**config).eval();path=tmp_path/'t.pt'
    torch.save({'config':config,'model':model.state_dict(),'manifest':json.loads(json.dumps({'torch':torch.__version__,'metric':np.float64(.3)}))},path)
    checkpoint=torch.load(path,weights_only=True);loaded=TinyGPT(**checkpoint['config']).eval();loaded.load_state_dict(checkpoint['model'])
    x=torch.tensor([[1,2,3,4]]);y=torch.tensor([[1,2,8,9]])
    with torch.no_grad():torch.testing.assert_close(loaded(x)[:,:2],loaded(y)[:,:2])
