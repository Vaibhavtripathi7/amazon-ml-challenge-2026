import numpy as np
import pandas as pd
import pytest

from amlc.data import make_folds
from amlc.submit import validate_submission


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sample_id": np.arange(1000),
        "price": rng.lognormal(3, 1, 1000),
        "brand": rng.choice(list("abcdefghij"), 1000),
    })


@pytest.mark.parametrize("mode", ["random", "stratified", "group"])
def test_every_row_gets_exactly_one_fold(train, mode):
    f = make_folds(train, n_folds=5, mode=mode, target="price", group_col="brand")
    assert list(f.columns) == ["sample_id", "fold"]
    assert f.sample_id.tolist() == train.sample_id.tolist()
    assert set(f.fold) == {0, 1, 2, 3, 4}


def test_stratified_folds_have_balanced_prices(train):
    f = make_folds(train, n_folds=5, mode="stratified", target="price")
    medians = train.price.groupby(f.fold.values).median()
    assert medians.max() / medians.min() < 1.15


def test_group_folds_never_split_a_group(train):
    f = make_folds(train, n_folds=5, mode="group", group_col="brand")
    assert (train.groupby(f.fold.values).brand.unique().explode().value_counts() == 1).all()


def test_folds_are_reproducible(train):
    a = make_folds(train, n_folds=5, mode="stratified", target="price", seed=42)
    b = make_folds(train, n_folds=5, mode="stratified", target="price", seed=42)
    assert a.equals(b)


@pytest.fixture
def test_ids():
    return pd.Series([10, 11, 12], name="sample_id")


def good_sub():
    return pd.DataFrame({"sample_id": [12, 10, 11], "price": [1.0, 2.0, 3.0]})


def test_valid_submission_passes(test_ids):
    validate_submission(good_sub(), test_ids, columns=["sample_id", "price"])


@pytest.mark.parametrize("breaker, message", [
    (lambda s: s.iloc[:2], "row count"),
    (lambda s: pd.concat([s, s.iloc[:1]]), "row count"),
    (lambda s: s.assign(sample_id=[10, 11, 99]), "ids"),
    (lambda s: s.assign(price=[1.0, np.nan, 3.0]), "NaN"),
    (lambda s: s.assign(price=[1.0, 0.0, 3.0]), "positive"),
    (lambda s: s.assign(price=[1.0, np.inf, 3.0]), "finite"),
    (lambda s: s[["price", "sample_id"]], "columns"),
])
def test_broken_submission_is_rejected(test_ids, breaker, message):
    with pytest.raises(ValueError, match=message):
        validate_submission(breaker(good_sub()), test_ids, columns=["sample_id", "price"])
