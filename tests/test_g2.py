import math
import torch
import json
import pytest
from src.g2 import batch, span_indices, References


def test_full_sentence_padding_and_target_alignment():
    records = [{'text':'the small red dog sees the tiny green cat.\n'},
               {'text':'the huge yellow bird follows the small red fox.\n'}]
    x,y = batch(records,[0,1])
    for i,row in enumerate(records):
        data = row['text'].encode()
        assert bytes(x[i,:len(data)-1].tolist()) == data[:-1]
        assert bytes(y[i,:len(data)-1].tolist()) == data[1:]
        assert (y[i,len(data)-1:]==-100).all()
        selected = span_indices(row['text'],2)
        assert bytes(y[i,selected].tolist()).decode() == row['text'].split()[2]+' '
        selected = span_indices(row['text'],4)
        assert bytes(y[i,selected].tolist()).decode() == row['text'].split()[4]+' '


def test_ngram_reference_handles_unseen_compositions_with_finite_loss():
    reference = References([{'text':'the small red dog sees the cat.\n'}])
    losses = reference.losses('the small blue dog follows the cat.\n',4)
    assert all(math.isfinite(x) and x>0 for x in losses)
    assert len(losses)==len('the small blue dog follows the cat.\n')-1


def test_teacher_training_checkpoint_and_full_sentence_evaluation(tmp_path,monkeypatch):
    # Isolated synthetic fixture, not a G2 experiment or evaluation of G2 test data.
    import src.g2 as g2
    from src.provenance import sha256
    data = tmp_path/'data'; data.mkdir()
    row = {'text':'the small red dog sees the tiny green cat.\n',
           'subject_attribute':['small','red'],'subject_relation':['dog','sees']}
    manifest = {'splits':{}}
    for name in ['train']+[f'validation_{axis}' for axis in g2.AXES]:
        path = data/f'{name}.jsonl'; path.write_text(json.dumps(row)+'\n')
        manifest['splits'][name] = {'records_sha256':sha256(path)}
    (data/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(g2,'DATA',data); monkeypatch.setattr(g2,'ROOT',tmp_path/'results')
    cfg = json.loads(__import__('pathlib').Path('configs/g2_v1.json').read_text())
    cfg.update(teacher_steps=2,teacher_batch=2,validation_sentences_per_axis=1)
    cfg['teacher_model'].update(d_model=16,n_head=2,n_layer=1)
    g2.train('teacher',0,cfg)
    path = g2.ROOT/'teacher_0.pt'
    ck = torch.load(path,weights_only=True)
    assert ck['completed_steps']==2
    assert all(int(s['step'])==2 for s in ck['optimizer']['state'].values())
    result = json.loads(path.with_suffix('.json').read_text())
    assert result['complete'] and result['bytes_seen']==4*(len(row['text'])-1)
    assert result['evaluations'][0]['metrics']['validation_attribute']['sentences']==1
    digest = sha256(path); g2.train('teacher',0,cfg)
    assert sha256(path)==digest  # Completed runs are not silently trained again.


def test_test_split_is_locked_before_the_training_schedule_finishes(tmp_path,monkeypatch):
    import src.g2 as g2
    monkeypatch.setattr(g2,'ROOT',tmp_path)
    cfg = {'teacher_seed':0,'seeds':[0],'conditions':['real_kd']}
    with pytest.raises(RuntimeError,match='locked'):
        g2.final_evaluate('teacher',0,cfg)
