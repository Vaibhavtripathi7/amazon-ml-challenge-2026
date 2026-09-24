import numpy as np


def smape(y_true, y_pred):
    """Symmetric MAPE in percent, 0 (perfect) to 200 (worst).

    Rows where both values are 0 count as perfect instead of 0/0 = NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true)
    ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    return float(100 * ratio.mean())
