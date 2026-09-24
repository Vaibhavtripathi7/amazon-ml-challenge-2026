import numpy as np
import pytest

from amlc.metrics import smape


def test_smape_is_zero_for_perfect_predictions():
    assert smape([10, 200, 3000], [10, 200, 3000]) == 0.0


def test_smape_hand_computed_example():
    # |110-100| / ((100+110)/2) = 10/105 -> 9.5238%
    assert smape([100], [110]) == pytest.approx(100 * 10 / 105)


def test_smape_both_zero_counts_as_perfect():
    # 0/0 must not produce NaN and poison the mean
    assert smape([0, 100], [0, 100]) == 0.0


def test_smape_is_capped_at_200():
    # predicting 0 for a nonzero target is the worst possible row
    assert smape([50], [0]) == pytest.approx(200.0)


def test_smape_penalises_over_prediction_less_than_under_by_same_amount():
    # same absolute miss of 50 on a true value of 100
    over = smape([100], [150])   # 50/125 = 40%
    under = smape([100], [50])   # 50/75  = 66.7%
    assert over < under


def test_smape_rejects_length_mismatch():
    with pytest.raises(ValueError):
        smape([1, 2], [1])
