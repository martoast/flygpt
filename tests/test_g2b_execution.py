import json
import numpy as np
import pytest
import torch
from scripts import g2b_experiment as experiment
from scripts.g2b_gate import assess_category
from src.tinygpt import TinyGPT


def test_response_mask_excludes_prompt_and_includes_first_response_byte():
    records=[{'prompt':'swap|abc\n>','response':'xyz\n'},{'prompt':'roles|ab\n>','response':'xy\n'}]
    x,y=experiment.pack(records,[0,1])
    for i,r in enumerate(records):
        p=len(r['prompt']);valid=y[i]!=-100
        assert not valid[:p-1].any()
        assert bytes(y[i,valid].tolist())==r['response'].encode()
        assert x[i,p-1]==ord('>')


def test_qualification_and_test_fail_closed_without_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    for split in ['qualification','test']:
        with pytest.raises(RuntimeError,match='gate'):experiment.read_data(split)
    with pytest.raises(RuntimeError,match='gate'):experiment.train('real_kd',0)


def test_teacher_greedy_has_no_ground_truth_response_input():
    class Recorder(TinyGPT):
        def __init__(self):
            torch.nn.Module.__init__(self);self.inputs=[]
        def forward(self,x):
            self.inputs.append(x.tolist())
            z=torch.zeros(*x.shape,256);z[:,-1,ord('x') if len(self.inputs)==1 else 10]=1
            return z
    model=Recorder()
    assert experiment.generate(model,'Q>')==b'x\n'
    assert model.inputs==[[[81,62]],[[81,62,120]]]


def test_recurrent_generation_prefills_once():
    class Recorder(torch.nn.Module):
        def __init__(self):super().__init__();self.inputs=[]
        def forward(self,x,h=None):
            self.inputs.append((x.tolist(),h))
            z=torch.zeros(*x.shape,256);z[:,-1,ord('x') if h is None else 10]=1
            return z,1
    model=Recorder();assert experiment.generate(model,'Q>')==b'x\n'
    assert model.inputs==[([[81,62]],None),([[120]],1)]


def test_exact_levenshtein_and_lexical_tie_order():
    prompts=[b'abc',b'abd',b'xyz']
    matrix=np.array([list(p) for p in prompts],dtype=np.uint8)
    result=experiment.distances(np.frombuffer(b'abe',dtype=np.uint8),matrix,np.array([3,3,3]))
    assert result.tolist()==[1,1,3]
    assert int(result.argmin())==0


def test_gate_requires_accuracy_and_ce_margins():
    thresholds={'teacher_mean_exact_response_accuracy_min':.8,
                'teacher_mean_exact_response_accuracy_margin_over_best_baseline_min':.1,
                'teacher_mean_response_ce_reduction_vs_best_probabilistic_baseline_min_fraction':.2,
                'paired_seed_and_composition_bootstrap_accuracy_margin_lower_bound_min':0}
    def rows(exact,ce):
        return [{'case_index':i,'subject_attribute':['small',str(i)],'subject_relation':['dog','sees'],
                 'exact':exact,'response_bytes':10,'nll':10*ce} for i in range(8)]
    teachers=[rows(1,.1) for _ in range(5)];baselines={'gru':[rows(0,.5) for _ in range(5)]}
    result=assess_category(teachers,baselines,'attribute',thresholds,50,np.random.default_rng(1))
    assert result['passed']
    baselines['gru']=[rows(1,.05) for _ in range(5)]
    result=assess_category(teachers,baselines,'attribute',thresholds,50,np.random.default_rng(1))
    assert not result['passed'] and not result['checks']['ce_margin'] and not result['checks']['accuracy_margin']


def test_teacher_train_checkpoint_and_evaluate_on_isolated_fixture(tmp_path,monkeypatch):
    from src.provenance import sha256
    data=tmp_path/'data';data.mkdir();root=tmp_path/'results'
    row={'prompt':'swap|a\n>','response':'a\n','subject_attribute':['s','c'],'subject_relation':['a','v'],'operation':'swap'}
    train=data/'train_familiar.jsonl';train.write_text(json.dumps(row)+'\n')
    val=data/'validation_attribute.jsonl';val.write_text(json.dumps(row)+'\n'+json.dumps({**row,'operation':'roles'})+'\n')
    (data/'manifest.json').write_text(json.dumps({'audit':{p.stem:{'sha256':sha256(p)} for p in [train,val]}}))
    p=tmp_path/'protocol.json';e=tmp_path/'execution.json'
    p.write_text(json.dumps({'teacher':{'layers':1,'heads':2,'embedding':16,'context_bytes':128},
                            'updates':2,'batch':1,'learning_rate':.0003,'weight_decay':.01,'gradient_clip':1}))
    e.write_text(json.dumps({'student_conditions':[],'training_sampling_seed_base':10}))
    monkeypatch.setattr(experiment,'DATA',data);monkeypatch.setattr(experiment,'ROOT',root)
    monkeypatch.setattr(experiment,'PROTOCOL',p);monkeypatch.setattr(experiment,'EXECUTION',e)
    experiment.train('teacher',0)
    checkpoint=root/'teacher_0.pt';saved=torch.load(checkpoint,weights_only=True)
    assert saved['step']==2 and all(int(s['step'])==2 for s in saved['optimizer']['state'].values())
    digest=sha256(checkpoint);experiment.train('teacher',0);assert sha256(checkpoint)==digest
    experiment.evaluate('teacher',0,'validation')
    result=json.loads((root/'validation/teacher_0.json').read_text())
    assert result['metrics']['attribute_swap']['cases']==1
