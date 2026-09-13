import pytest
from scripts.prepare_grammar_v2 import build, audit


def test_compositional_splits_are_disjoint_and_lexically_covered():
    corpora, partitions = build()
    report = audit(corpora, partitions)
    assert len(corpora['train']) == 2048
    assert all(r['verbatim_training_overlap'] == 0 for k, r in report.items() if k != 'train')
    assert all(not r['unseen_words'] for r in report.values())
    assert build() == (corpora, partitions)


def test_audit_rejects_training_sentence_leakage():
    corpora, partitions = build()
    corpora['validation_attribute'][0] = corpora['train'][0]
    with pytest.raises(AssertionError, match='Split overlap'):
        audit(corpora, partitions)
