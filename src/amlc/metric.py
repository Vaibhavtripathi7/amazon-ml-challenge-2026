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


def dev_f05(s1_idx, rec_ids, full, truth, dev_idx, country, w_other):
    """Macro F0.5 over dev S1 with sampled distractors. Assigned pairs (s1_idx[i], rec_ids[i]); full[i] is False
    when the record came from the distractor sample, so its false merge counts w_other times.
    truth: {s1_idx: set(rec ids)}. Returns (macro, {country: macro}, pair precision, pair recall)."""
    tp = dict.fromkeys(dev_idx, 0)
    fp = dict.fromkeys(dev_idx, 0.0)
    for s, r, f in zip(s1_idx, rec_ids, full):
        if s not in tp:
            continue
        if r in truth[s]:
            tp[s] += 1
        else:
            fp[s] += 1.0 if f else w_other
    scores, by_c = [], {}
    for i in dev_idx:
        T = len(truth[i])
        if T == 0:
            v = 1.0 if fp[i] == 0 else 0.0
        elif tp[i] == 0:
            v = 0.0
        else:
            P, R = tp[i] / (tp[i] + fp[i]), tp[i] / T
            v = 1.25 * P * R / (0.25 * P + R)
        scores.append(v)
        by_c.setdefault(country[i], []).append(v)
    TP, FP, NT = sum(tp.values()), sum(fp.values()), sum(len(truth[i]) for i in dev_idx)
    return (sum(scores) / len(scores), {c: round(sum(v) / len(v), 4) for c, v in by_c.items()},
            TP / max(TP + FP, 1e-9), TP / max(NT, 1))
