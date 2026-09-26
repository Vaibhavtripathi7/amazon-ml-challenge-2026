"""Shared pipeline pieces for judge v2: loading, record streaming, the untrimmed V1 shortlist, and the cheap
stage-1 (ranker) features. judge_v1.py keeps its own copies so upload #2 stays reproducible.

Record sets are fixed, disjoint hash buckets of the record id (crc32 % 100), so no record is ever in two sets:
  rank  : 0-3    stage-1 ranker training      (records owned by dev S1 excluded)
  judge : 4-18   stage-2 judge training       (records owned by dev S1 excluded)
  dev   : records owned by dev S1 (all) + distractors from buckets 19-28 (10%, weighted x10 in scoring)
Dev S1 = crc32(S1 id) % 10 == 0.
"""
import resource
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from amlc.blocking import Blocker
from amlc.keys import KEY_TYPES, filter_s1_keys, join_keys, row_keys, token_df

COLS = ["entity_id", "country", "nums", "n_core", "n_key", "n_fold", "n_flat", "n_suffix", "n_alias", "n_is_domain",
        "a_house", "a_house_alts", "a_street", "a_city", "a_state", "a_tokens", "a_has_addr"]
TF_PASSES = {
    "tf_name": dict(field="n_core", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
    "tf_mix": dict(field="mix", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
}
TF_K = {"tf_name": 10, "tf_mix": 20}
TF = list(TF_PASSES)
RANK_SET, JUDGE_SET, DEV_OTHER = (0, 4), (4, 19), (19, 29)
DEV_OTHER_WEIGHT = 100 / (DEV_OTHER[1] - DEV_OTHER[0])
T0 = time.time()


def log(msg):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"{msg}  [{time.time() - T0:.0f}s, peak {mem:.1f} GB]", flush=True)


def bucket(ids, n):
    return np.fromiter((zlib.crc32(s.encode()) % n for s in ids), np.int64, len(ids))


def load_s1(work, split, sample=None):
    s1 = pd.read_parquet(f"{work}/{split}/source1.parquet", columns=COLS)
    if sample:
        s1 = s1[bucket(s1.entity_id, 1000) < sample * 1000].reset_index(drop=True)
    s1["mix"] = s1.n_core + " " + s1.a_tokens
    return s1


def load_owner(work):
    """record id -> owning S1 id (train)."""
    gt = pd.read_parquet(f"{work}/train/truth.parquet")
    gt = gt[gt.matched_entity_ids != ""]
    ex = gt.assign(r=gt.matched_entity_ids.str.split(",")).explode("r")
    return pd.Series(ex.source1_entity_id.to_numpy(), index=ex.r.to_numpy())


def iter_records(work, split, keep=None, batch=300_000):
    for s in (2, 3):
        for b in pq.ParquetFile(f"{work}/{split}/source{s}.parquet").iter_batches(batch, columns=COLS):
            df = b.to_pandas()
            if keep is not None:
                df = df[keep(df)].reset_index(drop=True)
            if len(df):
                df["mix"] = df.n_core + " " + df.a_tokens
                yield df


def record_set(name, owner, dev_ids, sample_s1_ids=None, sample=None):
    """keep() for a named record set (see module doc). With --sample, restrict to a self-consistent mini world."""
    lo, hi = {"rank": RANK_SET, "judge": JUDGE_SET, "dev": DEV_OTHER}[name]
    rng = np.random.default_rng(0)

    def keep(df):
        ids = df.entity_id.to_numpy()
        o = owner.reindex(ids)
        owned = o.notna().to_numpy()
        b = bucket(ids, 100)
        in_bucket = (b >= lo) & (b < hi)
        dev_owned = o.isin(dev_ids).to_numpy()
        m = (dev_owned | in_bucket) if name == "dev" else (in_bucket & ~dev_owned)
        if sample:
            in_world = (owned & o.isin(sample_s1_ids).to_numpy()) | (~owned & (rng.random(len(df)) < sample))
            m &= in_world
        return m
    return keep


class Shortlister:
    """V1 shortlist, UNTRIMMED: 8 exact keys + tf_name (k=10) + tf_mix (k=20), fitted once on S1."""

    def __init__(self, s1, cap=30, threads=4):
        self.dfreq = token_df(s1)
        self.s1_keys = filter_s1_keys(row_keys(s1, self.dfreq, "s1"), cap)
        self.blockers = [Blocker({p: c}, TF_K[p], threads).fit(s1) for p, c in TF_PASSES.items()]

    def __call__(self, recs):
        """Returns DataFrame [s1_idx, rec_idx, k_<key>..., <tf>_score, <tf>_rank] for every shortlisted pair."""
        kc = join_keys(self.s1_keys, row_keys(recs, self.dfreq, "rec"))
        tcs = [b.query(recs) for b in self.blockers]
        n = len(recs)
        pk = lambda d: d.s1_idx.to_numpy().astype(np.int64) * n + d.rec_idx.to_numpy()
        allk = np.unique(np.concatenate([pk(kc)] + [pk(t) for t in tcs]))
        f = pd.DataFrame({"s1_idx": (allk // n).astype(np.int32), "rec_idx": (allk % n).astype(np.int32)})
        for k in KEY_TYPES:
            f[f"k_{k}"] = np.isin(allk, pk(kc[kc["pass"] == k])).astype(np.int8)
        for t, b in zip(TF, tcs):
            f[f"{t}_score"], f[f"{t}_rank"] = np.float32(0), np.int8(TF_K[t])
            g = b.assign(pk=pk(b)).groupby("pk").agg(score=("score", "max"), rank=("rank", "min"))
            pos = np.searchsorted(allk, g.index.to_numpy())
            f.loc[pos, f"{t}_score"] = g.score.to_numpy().astype(np.float32)
            f.loc[pos, f"{t}_rank"] = g["rank"].to_numpy().astype(np.int8)
        return f


RANK_FEATURES = ([f"k_{k}" for k in KEY_TYPES] + [f"{t}_{s}" for t in TF for s in ("score", "rank")] +
                 ["n_keys", "n_cands", "tf_best", "tf_best_gap", "mix_gap", "nm_tset", "ad_tset", "nm_tset_gap",
                  "ad_tset_gap", "house_eq", "street_eq", "city_eq", "state_state", "rec_has_addr", "s1_has_addr"])


def rank_features(s1, recs, f, threads=4):
    """Cheap, vectorised stage-1 features for every shortlisted pair (f from Shortlister)."""
    si, ri = f.s1_idx.to_numpy(), f.rec_idx.to_numpy()
    g = lambda df, col, idx: df[col].to_numpy()[idx]
    X = f[[c for c in RANK_FEATURES if c in f.columns]].copy()
    X["n_keys"] = f[[f"k_{k}" for k in KEY_TYPES]].sum(axis=1).astype(np.int8)
    X["n_cands"] = pd.Series(ri).map(pd.Series(ri).value_counts()).to_numpy().astype(np.int16)
    X["tf_best"] = f[[f"{t}_score" for t in TF]].max(axis=1).astype(np.float32)
    X["tf_best_gap"] = X.tf_best - X.tf_best.groupby(ri).transform("max")
    X["mix_gap"] = f.tf_mix_score - f.tf_mix_score.groupby(ri).transform("max")
    X["nm_tset"] = process.cpdist(g(s1, "n_core", si), g(recs, "n_core", ri), scorer=fuzz.token_set_ratio,
                                  workers=threads).astype(np.float32)
    X["ad_tset"] = process.cpdist(g(s1, "a_tokens", si), g(recs, "a_tokens", ri), scorer=fuzz.token_set_ratio,
                                  workers=threads).astype(np.float32)
    X["nm_tset_gap"] = X.nm_tset - X.nm_tset.groupby(ri).transform("max")
    X["ad_tset_gap"] = X.ad_tset - X.ad_tset.groupby(ri).transform("max")
    sh, rh = g(s1, "a_house", si), g(recs, "a_house", ri)
    X["house_eq"] = np.where((sh == "") | (rh == ""), 2, (sh == rh).astype(np.int8)).astype(np.int8)
    ss, rs = g(s1, "a_street", si), g(recs, "a_street", ri)
    X["street_eq"] = ((ss == rs) & (ss != "")).astype(np.int8)
    sc, rc = g(s1, "a_city", si), g(recs, "a_city", ri)
    X["city_eq"] = ((sc == rc) & (sc != "")).astype(np.int8)
    sst, rst = g(s1, "a_state", si), g(recs, "a_state", ri)
    X["state_state"] = np.where((sst == "") | (rst == ""), 2, np.where(sst == rst, 1, 3)).astype(np.int8)
    X["rec_has_addr"] = g(recs, "a_has_addr", ri).astype(np.int8)
    X["s1_has_addr"] = g(s1, "a_has_addr", si).astype(np.int8)
    return X[RANK_FEATURES]


def top_n(rec_idx, score, n):
    """Boolean mask keeping the n highest-scoring pairs of every record."""
    order = np.lexsort((-score, rec_idx))
    r_sorted = rec_idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r_sorted)) + 1]
    rank_sorted = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = np.zeros(len(order), bool)
    keep[order[rank_sorted < n]] = True
    rank = np.empty(len(order), np.int16)
    rank[order] = rank_sorted
    return keep, rank
