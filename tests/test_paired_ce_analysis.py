import json
import numpy as np
import pytest
from scripts.paired_ce_analysis import seed_summary,case_summary,learning_summary


def test_seed_statistics_use_paired_differences():
    s=seed_summary([.8,.9,.85],[.7,.5,.6])
    assert s['gap_mean']==pytest.approx(.25)
    assert s['gap_sd']==pytest.approx(.15)
    assert s['variance_ratio_rewired_over_real']==pytest.approx(4.)
    assert s['n']==3 and s['positive_gaps']==3
    assert s['gap_mean_t95'][0]<s['gap_mean']<s['gap_mean_t95'][1]
    assert seed_summary([.5,.5],[.4,.6])['variance_ratio_rewired_over_real'] is None


def test_paired_case_counts_and_alignment():
    a=[{'id':i,'exact':v} for i,v in enumerate([1,1,0,0])]
    b=[{'id':i,'exact':v} for i,v in enumerate([1,0,1,0])]
    r=case_summary(a,b)
    assert all(r[k]==1 for k in ('both_correct','both_wrong','real_only','rewired_only'))
    assert r['gap']==0 and r['real_correct']==r['rewired_correct']==2
    with pytest.raises(ValueError):case_summary(a,b[::-1])


def test_learning_area_uses_shared_observed_domain():
    def progress(values):
        return {'evaluations':[{'step':step,'validation':{'metrics':{'accuracy':acc,'response_ce':ce},'rows':[{'id':1},{'id':2}]}} for step,acc,ce in values]}
    real=progress([(64,0,1),(1024,1,.1)]);rw=progress([(64,.25,1.2),(1024,.75,.4)])
    r=learning_summary(real,rw)
    assert r['steps']==[64,1024]
    assert r['accuracy']['area_gap_real_minus_rewired']==0
    assert r['response_ce']['area_gap_real_minus_rewired']==pytest.approx(-.25)


@pytest.mark.parametrize('edge_format',[False,True])
def test_graph_diagnostics_do_not_change_source(tmp_path,monkeypatch,edge_format):
    from scipy import sparse
    from scripts import paired_graph_diagnostics as g
    from src.provenance import sha256
    p=tmp_path/'cycle.npz'
    if edge_format:np.savez_compressed(p,n=4,src=np.arange(4),dst=(np.arange(4)+1)%4)
    else:sparse.save_npz(p,sparse.csr_matrix((np.ones(4),(np.arange(4),(np.arange(4)+1)%4)),shape=(4,4)))
    before=sha256(p);out=tmp_path/'metrics';monkeypatch.setattr(g,'OUT',out)
    monkeypatch.setattr(g,'manifest',lambda config,inputs:{'inputs':{str(p):sha256(p)},'config':config})
    g.characterize(str(p),'cycle');r=json.loads((out/'cycle.json').read_text())
    assert sha256(p)==before
    assert r['giant_scc_fraction']==1 and r['reciprocal_fraction']==0
    assert r['binary_perron_estimate']==pytest.approx(1.)
    assert r['output_reachable_fraction']==1
