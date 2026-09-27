"""Judge v1 features for (S1 row, record row) pairs, built on stage-0 columns (n_*, a_*) + cleaned `nums`.

Similarities only: no raw text, IDs or country as features (nothing to memorise; France is not an unseen category).
Sources: Data Forensics doc (house-number relation, suffix conditional, IDF specificity, containment, margin),
Splink author (IDF weighting, distinguishability), Foursquare solutions (string similarity set).
"""
import collections
import math
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from amlc.keys import KEY_TYPES
from amlc.norm import number_relation

REL = {"equal": 0, "digit_drop": 1, "substitution": 2, "missing": 3}
TF = ["tf_name", "tf_addr", "tf_mix"]
FEATURES = (
    ["nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "fold_tset", "flat_ratio", "alias_tset",
     "nm_idf_shared", "nm_idf_max_shared", "nm_idf_cov_min", "nm_idf_cov_max", "rec_nm_idf", "s1_nm_idf",
     "suffix_state", "rec_is_domain",
     "ad_tset", "ad_partial", "ad_idf_cov_min", "ad_idf_cov_max", "house_rel", "num_jaccard", "num_common",
     "street_jw", "street_eq", "city_jw", "city_eq", "state_state", "rec_has_addr", "s1_has_addr",
     "rec_n_tokens", "s1_n_tokens"]
    + [f"k_{k}" for k in KEY_TYPES] + [f"{t}_score" for t in TF] + [f"{t}_rank" for t in TF]
    + ["n_keys", "n_cands", "nm_tset_gap", "ad_cov_gap", "tf_best_gap", "is_top_tf"]
)


# Sibling-vs-noisy-copy signals (dev wrong merges / test 0.4-0.7 band: a real word swapped, house moved a little).
SIB_FEATURES = ["nm_x_rec", "nm_x_s1", "nm_x_sim", "nm_x_idf", "house_ndiff", "house_lev"]


class Context:
    """Per-split lookups: S1 columns as arrays and per-country IDF of name / address tokens (from S1)."""

    def __init__(self, s1):
        self.s1 = s1
        self.idf = {}
        for kind, col in (("n", "n_core"), ("a", "a_tokens")):
            for c in s1.country.unique():
                cnt = collections.Counter()
                sub = s1.loc[s1.country == c, col]
                for t in sub:
                    cnt.update(set(t.split()))
                N = len(sub)
                self.idf[(c, kind)] = {t: math.log(N / v) for t, v in cnt.items()}
        self.default_idf = {k: math.log(max(len(s1), 2)) for k in ("n", "a")}


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers).astype(np.float32)


def _idf_overlap(xs, ys, countries, ctx, kind):
    """shared IDF sum, max shared IDF, coverage min/max (IDF-weighted containment), IDF totals."""
    n = len(xs)
    out = np.zeros((n, 6), np.float32)
    for i, (x, y, c) in enumerate(zip(xs, ys, countries)):
        idf = ctx.idf.get((c, kind), {})
        d = ctx.default_idf[kind]
        A, B = set(x.split()), set(y.split())
        if not A or not B:
            continue
        wa = sum(idf.get(t, d) for t in A)
        wb = sum(idf.get(t, d) for t in B)
        sh = A & B
        ws = sum(idf.get(t, d) for t in sh)
        ca, cb = ws / wa, ws / wb
        out[i] = (ws, max((idf.get(t, d) for t in sh), default=0.0), min(ca, cb), max(ca, cb), wb, wa)
    return out


def _sibling(sc, rc, sh, rh, countries, ctx):
    """Unmatched name words: counts, how well the worst record word is explained by an S1 word (typo ~0.9,
    swapped word ~0.5), that word's IDF (common word = low); house numeric distance and edit distance."""
    out = np.full((len(sc), 6), -1.0, np.float32)
    for i, (x, y, a, b, c) in enumerate(zip(sc, rc, sh, rh, countries)):
        A, B = set(x.split()), set(y.split())
        ua, ub = A - B, B - A
        out[i, 0], out[i, 1] = len(ub), len(ua)
        if ub:
            idf, d = ctx.idf.get((c, "n"), {}), ctx.default_idf["n"]
            worst, widf = 2.0, -1.0
            for t in sorted(ub):                                  # deterministic tie-break
                sim = max((JaroWinkler.normalized_similarity(t, u) for u in ua), default=0.0)
                if sim < worst:
                    worst, widf = sim, idf.get(t, d)
            out[i, 2], out[i, 3] = worst, widf
        else:
            out[i, 2] = 1.0
        if a and b:
            da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
            if da and db and len(da) < 12 and len(db) < 12:
                out[i, 4] = math.log1p(abs(int(da) - int(db)))
            out[i, 5] = Levenshtein.distance(a, b)
    return out


def pair_features(ctx, recs, s_idx, r_idx, flags, workers=1):
    """flags: DataFrame aligned with the pairs holding k_<key> (0/1) and <tf>_score / <tf>_rank columns."""
    s1 = ctx.s1
    g = lambda df, col, idx: df[col].to_numpy()[idx]
    sc, rc = g(s1, "n_core", s_idx), g(recs, "n_core", r_idx)
    country = g(recs, "country", r_idx)
    f = {
        "nm_ratio": _cp(sc, rc, fuzz.ratio, workers),
        "nm_tset": _cp(sc, rc, fuzz.token_set_ratio, workers),
        "nm_tsort": _cp(sc, rc, fuzz.token_sort_ratio, workers),
        "nm_partial": _cp(sc, rc, fuzz.partial_ratio, workers),
        "nm_jw": _cp(sc, rc, JaroWinkler.normalized_similarity, workers),
        "fold_tset": _cp(g(s1, "n_fold", s_idx), g(recs, "n_fold", r_idx), fuzz.token_set_ratio, workers),
        "flat_ratio": _cp(g(s1, "n_flat", s_idx), g(recs, "n_flat", r_idx), fuzz.ratio, workers),
        "alias_tset": _cp(sc, g(recs, "n_alias", r_idx), fuzz.token_set_ratio, workers),
    }
    nm = _idf_overlap(sc, rc, country, ctx, "n")
    for j, k in enumerate(["nm_idf_shared", "nm_idf_max_shared", "nm_idf_cov_min", "nm_idf_cov_max",
                           "rec_nm_idf", "s1_nm_idf"]):
        f[k] = nm[:, j]
    ssuf, rsuf = g(s1, "n_suffix", s_idx), g(recs, "n_suffix", r_idx)
    # 0 = both absent, 1 = equal, 2 = one side absent, 3 = conflict (both present, different)
    f["suffix_state"] = np.array([0 if not a and not b else 1 if a == b else 2 if not a or not b else 3
                                  for a, b in zip(ssuf, rsuf)], np.int8)
    f["rec_is_domain"] = g(recs, "n_is_domain", r_idx).astype(np.int8)

    sa, ra = g(s1, "a_tokens", s_idx), g(recs, "a_tokens", r_idx)
    f["ad_tset"] = _cp(sa, ra, fuzz.token_set_ratio, workers)
    f["ad_partial"] = _cp(sa, ra, fuzz.partial_ratio, workers)
    ad = _idf_overlap(sa, ra, country, ctx, "a")
    f["ad_idf_cov_min"], f["ad_idf_cov_max"] = ad[:, 2], ad[:, 3]
    f["house_rel"] = np.array([REL[number_relation(set(a.split()), set(b.split()))]
                               for a, b in zip(g(s1, "a_house_alts", s_idx), g(recs, "a_house_alts", r_idx))], np.int8)
    jac, com = np.zeros(len(sc), np.float32), np.zeros(len(sc), np.float32)
    for i, (a, b) in enumerate(zip(g(s1, "nums", s_idx), g(recs, "nums", r_idx))):
        if a and b:
            A, B = set(a.split()), set(b.split())
            com[i] = len(A & B)
            jac[i] = com[i] / len(A | B)
    f["num_jaccard"], f["num_common"] = jac, com
    ss, rs = g(s1, "a_street", s_idx), g(recs, "a_street", r_idx)
    f["street_jw"] = _cp(ss, rs, JaroWinkler.normalized_similarity, workers)
    f["street_eq"] = ((ss == rs) & (ss != "")).astype(np.int8)
    scity, rcity = g(s1, "a_city", s_idx), g(recs, "a_city", r_idx)
    f["city_jw"] = _cp(scity, rcity, JaroWinkler.normalized_similarity, workers)
    f["city_eq"] = ((scity == rcity) & (scity != "")).astype(np.int8)
    sst, rst = g(s1, "a_state", s_idx), g(recs, "a_state", r_idx)
    f["state_state"] = np.where((sst == "") | (rst == ""), 2, np.where(sst == rst, 1, 3)).astype(np.int8)
    f["rec_has_addr"] = g(recs, "a_has_addr", r_idx).astype(np.int8)
    f["s1_has_addr"] = g(s1, "a_has_addr", s_idx).astype(np.int8)
    f["rec_n_tokens"] = np.array([len(x.split()) for x in ra], np.int16)
    f["s1_n_tokens"] = np.array([len(x.split()) for x in sa], np.int16)

    for k in KEY_TYPES:
        f[f"k_{k}"] = flags[f"k_{k}"].to_numpy().astype(np.int8)
    for t in TF:
        f[f"{t}_score"] = flags[f"{t}_score"].to_numpy().astype(np.float32)
        f[f"{t}_rank"] = flags[f"{t}_rank"].to_numpy().astype(np.int8)
    X = pd.DataFrame(f)
    X["n_keys"] = X[[f"k_{k}" for k in KEY_TYPES]].sum(axis=1).astype(np.int8)
    # competition within the record's candidate set (distinguishability)
    grp = pd.Series(r_idx)
    X["n_cands"] = grp.map(grp.value_counts()).to_numpy().astype(np.int16)
    X["nm_tset_gap"] = X.nm_tset - X.nm_tset.groupby(r_idx).transform("max")
    X["ad_cov_gap"] = X.ad_idf_cov_max - X.ad_idf_cov_max.groupby(r_idx).transform("max")
    best_tf = X[[f"{t}_score" for t in TF]].max(axis=1)
    X["tf_best_gap"] = best_tf - best_tf.groupby(r_idx).transform("max")
    X["is_top_tf"] = (X.tf_best_gap == 0).astype(np.int8)
    sib = _sibling(sc, rc, g(s1, "a_house", s_idx), g(recs, "a_house", r_idx), country, ctx)
    for j, k in enumerate(SIB_FEATURES):
        X[k] = sib[:, j]
    return X[FEATURES + SIB_FEATURES]
