import json
import pytest
import torch
from scripts import g2c_engine as e


def test_algorithms_and_response_mask():
    assert e.transform([0,1,2,3],'substitute')==[1,2,3,0]
    assert e.transform([0,1,2,3],'reverse')==[3,2,1,0]
    assert e.transform([0,1,2,3],'rotate')==[2,3,0,1]
    assert e.transform([0,1,2,3],'checksum')==[2]
    r={'prompt':[5,0,1,6],'response':[1,2,4]}
    x,y=e.pack([r],[0]);assert y.tolist()==[[-100,-100,-100,1,2,4]]
    assert x.tolist()==[[5,0,1,6,1,2]]


def test_instance_splits_are_disjoint(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=e.make_data('substitute');seen=set()
    for path in root.glob('*.json'):
        if path.name=='manifest.json':continue
        rows=json.loads(path.read_text());ids={r['id'] for r in rows}
        assert len(ids)==len(rows) and not seen&ids;seen|=ids
        assert all(r['response']==e.transform(r['input'],'substitute')+[e.EOS] for r in rows)


def test_test_access_is_locked(tmp_path):
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'kind':'teacher'}))
    with pytest.raises(RuntimeError,match='locked'):
        e.evaluate(str(spec),'unused','test_main','unused')


def test_qualification_receipt_detects_changed_input(tmp_path):
    from src.provenance import save_json,sha256
    source=tmp_path/'source';source.write_text('frozen')
    receipt=tmp_path/'receipt.json'
    save_json(receipt,{'passed':True,'threshold':.95,'inputs':{str(source):sha256(source)}})
    assert e.qualified(receipt)['passed']
    source.write_text('modified')
    with pytest.raises(AssertionError):e.qualified(receipt)


def test_frozen_cohort_cannot_be_rewritten(tmp_path):
    from scripts.run_g2c_overnight import frozen
    p=tmp_path/'freeze.json';frozen(p,{'updates':512});frozen(p,{'updates':512})
    with pytest.raises(AssertionError):frozen(p,{'updates':1024})
