from types import SimpleNamespace
import pytest
from scripts import compiler_efficiency_engine as adapter
from scripts.compiler_efficiency_analysis import analyze
from scripts.run_compiler_efficiency import select_test_ids


def test_only_fresh_domain_is_used():
    ids = select_test_ids(set(range(3712)),256,'fixed')
    assert len(set(ids))==256 and min(ids)>=3712
    assert ids==select_test_ids(set(range(3712)),256,'fixed')


def test_ten_seed_gate():
    with pytest.raises(ValueError):analyze({})


def test_analysis_does_not_promote_equal_accuracy():
    methods=['A_hard','B_soft','C_hidden','D_hidden_shuffled','E_relational','F_relational_shuffled']
    rows={str(s):{m:dict(accuracy=.7,response_ce=.2) for m in methods} for s in range(500,510)}
    result=analyze(rows)
    assert len(result['comparisons'])==10
    assert all(not r['supported_under_preregistered_t_analysis'] for r in result['comparisons'])


def test_linux_rss_adapter_changes_units_not_training_arguments(monkeypatch):
    monkeypatch.setattr(adapter,'sys',SimpleNamespace(platform='linux'))
    monkeypatch.setattr(adapter,'resource',SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda who:SimpleNamespace(ru_maxrss=123)))
    monkeypatch.setattr(adapter.original,'resource',adapter.original.resource)
    monkeypatch.setattr(adapter.original,'archive',adapter.original.archive)
    def fake_train(spec,until):
        assert adapter.original.resource.getrusage(0).ru_maxrss==123*1024
        return spec,until
    monkeypatch.setattr(adapter.original,'train',fake_train)
    assert adapter.train('frozen-spec',256)==('frozen-spec',256)
