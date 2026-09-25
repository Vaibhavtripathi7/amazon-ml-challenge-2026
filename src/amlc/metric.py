"""Competition metric: F0.5 per Source-1 entity, macro-averaged (README formula)."""


def f05(pred, truth):
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth):
    """pred, truth: {s1_id: set of ids}. Averages over every S1 in truth; a missing pred row = empty list."""
    return sum(f05(pred.get(s1, ()), t) for s1, t in truth.items()) / len(truth)
