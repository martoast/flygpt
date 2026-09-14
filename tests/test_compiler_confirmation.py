import pytest
from scripts.compiler_confirmation_analysis import paired, holm, analyze
from scripts.run_compiler_confirmation import select_test_ids


def test_fresh_selection_is_deterministic_and_excludes_all_used():
    used = set(range(3200))
    ids = select_test_ids(used, 512, 'locked')
    assert ids == select_test_ids(used, 512, 'locked')
    assert len(set(ids)) == 512 and not used.intersection(ids)
    assert len(set(range(4096))-used-set(ids)) == 384


def test_insufficient_unused_domain_stops():
    with pytest.raises(ValueError): select_test_ids(range(4000), 512, 'locked')


def test_sign_flip_small_n_floor():
    result = paired([.01, .02, .03, .04, .05])
    assert result['exact_two_sided_sign_flip_p'] == .0625
    assert result['mean_benefit'] == pytest.approx(.03)


def test_no_difference_is_not_evidence():
    result = paired([0]*5)
    assert result['paired_t_p'] == result['exact_two_sided_sign_flip_p'] == 1


def test_holm_adjustment_order_and_monotonicity():
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_analysis_requires_complete_cohort():
    with pytest.raises(ValueError): analyze({})


def test_ce_benefit_direction_and_full_family():
    rows = {}
    for seed in range(400, 405):
        rows[str(seed)] = {m:dict(accuracy=.9, response_ce=.1) for m in ['A_hard','B_soft','C_hidden','D_hidden_shuffled','E_relational','F_relational_shuffled']}
        rows[str(seed)]['B_soft']['response_ce'] -= .01*(seed-399)
    result = analyze(rows)
    assert len(result['comparisons']) == 10
    soft_ce = next(r for r in result['comparisons'] if r['method']=='B_soft' and r['endpoint']=='response_ce')
    assert soft_ce['mean_benefit'] == pytest.approx(.03)
    assert all(not r['supported_under_preregistered_t_analysis'] for r in result['comparisons'] if r['endpoint']=='accuracy')
