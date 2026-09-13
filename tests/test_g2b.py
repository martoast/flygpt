import pytest
from scripts.prepare_g2b import build,audit


def test_g2b_pair_and_answer_leakage_audits():
    data,partitions=build(); result=audit(data,partitions)
    assert result['train_familiar']['records']==8192
    assert all(v['max_sequence_bytes']<=128 for v in result.values())
    assert all(v['answer_overlap_with_other_splits']==0 for v in result.values())
    data['validation_attribute'][0]=data['train_familiar'][0]
    with pytest.raises(AssertionError):audit(data,partitions)
