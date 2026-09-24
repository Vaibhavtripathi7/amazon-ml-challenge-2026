"""Submission checks: an invalid file scores nothing, so fail loudly before uploading."""
import numpy as np
import pandas as pd

ID = "sample_id"


def validate_submission(sub, test_ids, columns, target="price"):
    """Raise ValueError unless sub has exactly `columns`, one row per test id, finite positive targets."""
    if list(sub.columns) != list(columns):
        raise ValueError(f"columns {list(sub.columns)} != expected {list(columns)}")
    if len(sub) != len(test_ids):
        raise ValueError(f"row count {len(sub)} != test rows {len(test_ids)}")
    if sub[ID].duplicated().any() or set(sub[ID]) != set(test_ids):
        raise ValueError("ids do not match the test set one-to-one")
    y = sub[target].to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError(f"{np.isnan(y).sum()} NaN predictions")
    if not np.isfinite(y).all():
        raise ValueError("non-finite predictions")
    if (y <= 0).any():
        raise ValueError(f"{(y <= 0).sum()} predictions are not positive")


def write_submission(sub, test_ids, path, columns, target="price"):
    """Validate, reorder to the test id order, and write CSV."""
    validate_submission(sub, test_ids, columns, target)
    out = pd.DataFrame({ID: list(test_ids)}).merge(sub, on=ID, how="left")[list(columns)]
    out.to_csv(path, index=False)
    return out
