import pytest

from amlc.metric import f05, macro_f05


def test_readme_example():
    # README: predict 3, truth 2, both true ones found -> P=2/3, R=1 -> 0.714
    assert f05({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}) == pytest.approx(0.714, abs=1e-3)


def test_singleton_empty_prediction_scores_one():
    assert f05(set(), set()) == 1.0


def test_singleton_any_prediction_scores_zero():
    assert f05({"S2-1"}, set()) == 0.0


def test_missing_all_true_matches_scores_zero():
    assert f05(set(), {"S2-1"}) == 0.0


def test_macro_averages_over_all_truth_entities_including_missing_rows():
    truth = {"a": {"S2-1"}, "b": set()}
    pred = {"a": {"S2-1"}}  # "b" absent -> treated as empty prediction -> 1.0
    assert macro_f05(pred, truth) == 1.0


def test_dev_f05_matches_official_formula_when_nothing_is_sampled():
    from amlc.metric import dev_f05
    truth = {0: {"S2-47", "S3-812"}, 1: set(), 2: {"S2-9"}}
    # S1 0: README example (3 predicted, 2 right) ; S1 1: singleton left empty ; S1 2: missed
    macro, by_c, P, R = dev_f05([0, 0, 0], ["S2-47", "S2-193", "S3-812"], [True] * 3, truth, [0, 1, 2],
                                ["US", "US", "India"], 10.0)
    assert abs(macro - (0.714286 + 1.0 + 0.0) / 3) < 1e-4
    assert by_c == {"US": round((0.714286 + 1) / 2, 4), "India": 0.0}


def test_dev_f05_scales_sampled_false_merges():
    from amlc.metric import dev_f05
    truth = {0: {"a"}}
    m_full, *_ = dev_f05([0, 0], ["a", "x"], [True, True], truth, [0], ["US"], 10.0)
    m_samp, *_ = dev_f05([0, 0], ["a", "x"], [True, False], truth, [0], ["US"], 10.0)
    assert m_samp < m_full   # one sampled false merge stands for ~10 in the full world
