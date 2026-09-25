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
