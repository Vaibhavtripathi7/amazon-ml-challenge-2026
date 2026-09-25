"""Judge v1: V1 shortlist -> pair features -> LightGBM -> best S1 per record -> threshold tuned for macro F0.5.

  python -u scripts/judge_v1.py train --work W --model models/judge_v1.txt
  python -u scripts/judge_v1.py dev   --work W --model models/judge_v1.txt       # honest dev F0.5, picks threshold
  python -u scripts/judge_v1.py test  --work W --model models/judge_v1.txt --out output
Then: python3 <student_resource>/utils/validate_submission.py --matching output/matching_results.tsv ...

Dev = S1 with crc32(id) % 10 == 0. Training never uses records owned by dev S1. Every record is searched
against ALL S1 of its split. Dev scores all dev-owned records plus `--other-frac` of every other record;
false merges coming from that sample are scaled by 1/other-frac (an approximation, checked by the leaderboard).
--sample F: small self-consistent world (F of S1, their records, F of unowned records) for local checks.
"""
import argparse
import json
import resource
import time
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from amlc.blocking import Blocker
from amlc.judge_features import FEATURES, TF, Context, pair_features
from amlc.keys import KEY_TYPES, filter_s1_keys, join_keys, row_keys, token_df
from amlc.metric import dev_f05

COLS = ["entity_id", "country", "nums", "n_core", "n_key", "n_fold", "n_flat", "n_suffix", "n_alias", "n_is_domain",
        "a_house", "a_house_alts", "a_street", "a_city", "a_state", "a_tokens", "a_has_addr"]
TF_PASSES = {
    "tf_name": dict(field="n_core", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
    "tf_addr": dict(field="a_tokens", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
}
MIX_PASS = {"tf_mix": dict(field="mix", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05)}
T0 = time.time()


def log(msg):
    print(f"{msg}  [{time.time() - T0:.0f}s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB]",
          flush=True)


def bucket(ids, n):
    return np.fromiter((zlib.crc32(s.encode()) % n for s in ids), np.int64, len(ids))


def load_s1(a, split):
    s1 = pd.read_parquet(f"{a.work}/{split}/source1.parquet", columns=COLS)
    if a.sample:
        s1 = s1[bucket(s1.entity_id, 1000) < a.sample * 1000].reset_index(drop=True)
    s1["mix"] = s1.n_core + " " + s1.a_tokens
    return s1


def load_owner(a, s1_ids=None):
    gt = pd.read_parquet(f"{a.work}/train/truth.parquet")
    gt = gt[gt.matched_entity_ids != ""]
    if s1_ids is not None:
        gt = gt[gt.source1_entity_id.isin(s1_ids)]
    ex = gt.assign(r=gt.matched_entity_ids.str.split(",")).explode("r")
    return pd.Series(ex.source1_entity_id.to_numpy(), index=ex.r.to_numpy())


def iter_records(a, split, keep=None, batch=300_000):
    for s in (2, 3):
        for b in pq.ParquetFile(f"{a.work}/{split}/source{s}.parquet").iter_batches(batch, columns=COLS):
            df = b.to_pandas()
            if keep is not None:
                df = df[keep(df)].reset_index(drop=True)
            if len(df):
                df["mix"] = df.n_core + " " + df.a_tokens
                yield df


class Shortlister:
    """V1 shortlist: 8 exact keys + TF-IDF name/address (+ optional combined) searches, fitted once on S1."""

    def __init__(self, s1, a):
        self.a = a
        self.dfreq = token_df(s1)
        self.s1_keys = filter_s1_keys(row_keys(s1, self.dfreq, "s1"), a.cap)
        passes = {p: c for p, c in (dict(TF_PASSES) | MIX_PASS).items() if p in a.tf}
        self.blockers = [Blocker({p: c}, a.mix_k if p == "tf_mix" else a.k, a.threads).fit(s1) for p, c in passes.items()]

    def __call__(self, recs):
        kc = join_keys(self.s1_keys, row_keys(recs, self.dfreq, "rec"))
        tcs = [b.query(recs) for b in self.blockers]
        n = len(recs)
        pk = lambda d: d.s1_idx.to_numpy().astype(np.int64) * n + d.rec_idx.to_numpy()
        allk = np.unique(np.concatenate([pk(kc)] + [pk(t) for t in tcs]))
        flags = pd.DataFrame({"pk": allk})
        for k in KEY_TYPES:
            flags[f"k_{k}"] = np.isin(allk, pk(kc[kc["pass"] == k])).astype(np.int8)
        for t in TF:
            flags[f"{t}_score"], flags[f"{t}_rank"] = np.float32(0), np.int8(self.a.k)
        for t in tcs:
            for p_, g in t.groupby("pass", observed=True):
                g = g.assign(pk=pk(g)).groupby("pk").agg(score=("score", "max"), rank=("rank", "min"))
                pos = np.searchsorted(allk, g.index.to_numpy())
                flags.loc[pos, f"{p_}_score"] = g.score.to_numpy()
                flags.loc[pos, f"{p_}_rank"] = g["rank"].to_numpy()
        flags["s1_idx"], flags["rec_idx"] = (allk // n).astype(np.int32), (allk % n).astype(np.int32)
        # model input set: per record the top `max_per_rec` pairs by (keys hit, best search score)
        pri = flags[[f"k_{k}" for k in KEY_TYPES]].sum(axis=1) + flags[[f"{t}_score" for t in TF]].max(axis=1)
        flags = flags.assign(_p=pri.to_numpy()).sort_values(["rec_idx", "_p"], ascending=[True, False])
        flags = flags[flags.groupby("rec_idx").cumcount() < self.a.max_per_rec].reset_index(drop=True)
        return flags.s1_idx.to_numpy(), flags.rec_idx.to_numpy(), flags.drop(columns=["_p", "pk"])


def sample_world(a, s1, owner):
    """keep() for --sample: records owned by the sampled S1 + the same share of unowned records."""
    ids = set(s1.entity_id)
    rng = np.random.default_rng(0)

    def keep(df):
        o = owner.reindex(df.entity_id.to_numpy())
        owned = o.notna().to_numpy()
        return (owned & o.isin(ids).to_numpy()) | (~owned & (rng.random(len(df)) < a.sample))
    return keep if a.sample else (lambda df: np.ones(len(df), bool))


def cmd_train(a):
    s1 = load_s1(a, "train")
    owner = load_owner(a)
    in_world = sample_world(a, s1, owner)
    dev_ids = set(s1.entity_id[bucket(s1.entity_id, 10) == 0])
    rng = np.random.default_rng(1)

    def keep(df):
        o = owner.reindex(df.entity_id.to_numpy())
        return in_world(df) & ~o.isin(dev_ids).to_numpy() & (rng.random(len(df)) < a.train_frac)
    ctx, sl = Context(s1), Shortlister(s1, a)
    log(f"train: S1 {len(s1):,} (dev {len(dev_ids):,} held out); shortlister fitted")
    Xs, ys, grp, lost, owned_n = [], [], [], 0, 0
    s1_ids = s1.entity_id.to_numpy()
    for i, recs in enumerate(iter_records(a, "train", keep)):
        s_idx, r_idx, flags = sl(recs)
        X = pair_features(ctx, recs, s_idx, r_idx, flags, a.threads)
        own = owner.reindex(recs.entity_id.to_numpy()).to_numpy()
        y = (own[r_idx] == s1_ids[s_idx]).astype(np.int8)
        has_owner = pd.notna(own) & pd.Series(own).isin(set(s1_ids)).to_numpy()
        owned_n += has_owner.sum()
        lost += has_owner.sum() - len(np.unique(r_idx[y == 1]))
        Xs.append(X)
        ys.append(y)
        grp.append(bucket(recs.entity_id.to_numpy()[r_idx], 10))
        log(f"  chunk {i}: {len(recs):,} records -> {len(X):,} pairs, {y.mean():.3f} positive")
    X, y, grp = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(grp)
    del Xs, ys
    log(f"train pairs {len(y):,}; owned records whose owner is NOT in the model input: {lost / max(owned_n, 1):.4f}")
    val = grp == 0                                   # validation split by record, so a record's pairs stay together
    params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbose=-1)
    m = lgb.train(params, lgb.Dataset(X[~val], y[~val]), 3000, valid_sets=[lgb.Dataset(X[val], y[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.model, num_iteration=m.best_iteration)
    imp = pd.Series(m.feature_importance("gain"), FEATURES).sort_values(ascending=False)
    log(f"model saved {a.model}: best iter {m.best_iteration}, val logloss {m.best_score['valid_0']['binary_logloss']:.5f}")
    print("top features by gain:", (imp / imp.sum()).head(12).round(3).to_dict())


def scored_pairs(a, split, keep):
    s1 = load_s1(a, split)
    ctx, sl = Context(s1), Shortlister(s1, a)
    model = lgb.Booster(model_file=a.model)
    log(f"{split}: S1 {len(s1):,}; shortlister fitted")
    parts = []
    for i, recs in enumerate(iter_records(a, split, keep(s1) if keep else None)):
        s_idx, r_idx, flags = sl(recs)
        X = pair_features(ctx, recs, s_idx, r_idx, flags, a.threads)
        prob = model.predict(X[FEATURES], num_threads=a.threads).astype(np.float32)
        parts.append(pd.DataFrame({"s1_idx": s_idx.astype(np.int32), "rec_id": recs.entity_id.to_numpy()[r_idx],
                                   "prob": prob}))
        log(f"  chunk {i}: {len(recs):,} records -> {len(X):,} pairs")
    return s1, pd.concat(parts, ignore_index=True)


def cmd_dev(a):
    owner = load_owner(a)
    holder = {}

    def keep(s1):
        in_world = sample_world(a, s1, owner)
        dev_ids = set(s1.entity_id[bucket(s1.entity_id, 10) == 0])
        holder["dev_ids"] = dev_ids
        rng = np.random.default_rng(2)

        def k(df):
            o = owner.reindex(df.entity_id.to_numpy())
            return in_world(df) & (o.isin(dev_ids).to_numpy() | (rng.random(len(df)) < a.other_frac))
        return k
    s1, pairs = scored_pairs(a, "train", keep)
    dev_ids = holder["dev_ids"]
    dev_idx = np.flatnonzero(s1.entity_id.isin(dev_ids).to_numpy())
    pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    own = owner[owner.isin(dev_ids)]
    truth = {i: set() for i in dev_idx}
    for r, s in zip(own.index, pos[own.to_numpy()].to_numpy()):
        truth[s].add(r)
    pairs["full"] = pairs.rec_id.isin(set(own.index))          # dev-owned records are fully present; others sampled
    dev_pairs = pairs[np.isin(pairs.s1_idx, dev_idx)]
    kept = sum(len(truth[s] & set(g)) for s, g in dev_pairs.groupby("s1_idx").rec_id)
    log(f"dev S1 {len(dev_idx):,} | true pairs {len(own):,} | true pairs in model input {kept / max(len(own), 1):.4f}")
    best = pairs.sort_values("prob", ascending=False).drop_duplicates("rec_id")
    best = best[np.isin(best.s1_idx, dev_idx)]
    country = s1.country.to_numpy()
    w_other = 1.0 / a.other_frac
    rows = []
    for t in a.thresholds:
        b = best[best.prob >= t]
        score, per_c, P, R = dev_f05(b.s1_idx.to_numpy(), b.rec_id.to_numpy(), b.full.to_numpy(), truth, dev_idx,
                                     country, 1.0 / a.other_frac)
        rows.append((t, score))
        print(f"  t={t:.2f}  dev F0.5 {score:.4f}  precision {P:.4f}  recall {R:.4f}  by country {per_c}", flush=True)
    t_best = max(rows, key=lambda r: r[1])[0]
    Path(a.model + ".json").write_text(json.dumps({"threshold": t_best}))
    log(f"best threshold {t_best:.2f} saved to {a.model}.json")


def cmd_test(a):
    split = a.split
    s1, pairs = scored_pairs(a, split, None)
    t = a.threshold if a.threshold is not None else json.loads(Path(a.model + ".json").read_text())["threshold"]
    best = pairs.sort_values("prob", ascending=False).drop_duplicates("rec_id")
    best = best[best.prob >= t]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = s1.entity_id.to_numpy()
    for fname, col, df in [("matching_results.tsv", "matched_entity_ids", best),
                           ("candidate_pairs.tsv", "candidate_entity_ids", pairs)]:
        lists = df.drop_duplicates(["s1_idx", "rec_id"]).groupby("s1_idx").rec_id.agg(",".join)
        full = np.full(len(s1), "", dtype=object)
        full[lists.index.to_numpy()] = lists.to_numpy()
        with open(out / fname, "w", encoding="utf-8") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for sid, lst in zip(ids, full):
                f.write(f"{sid}\t{lst}\n")
    log(f"wrote {out}/: {len(best):,} matched records over {best.s1_idx.nunique():,} S1 (threshold {t:.2f})")


p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["train", "dev", "test"])
p.add_argument("--work", default="work")
p.add_argument("--model", default="models/judge_v1.txt")
p.add_argument("--out", default="output")
p.add_argument("--split", default="test", help="test: which split to write (train only for local smoke checks)")
p.add_argument("--threshold", type=float, default=None)
p.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
p.add_argument("--train-frac", type=float, default=0.15)
p.add_argument("--other-frac", type=float, default=0.1)
p.add_argument("--k", type=int, default=10)
p.add_argument("--mix-k", type=int, default=20, help="k for the combined name+address search")
p.add_argument("--tf", nargs="+", default=["tf_name", "tf_mix"],
               help="TF-IDF searches to use (measured 2026-09-26: tf_mix makes tf_addr redundant, +0.0003)")
p.add_argument("--cap", type=int, default=30)
p.add_argument("--max-per-rec", type=int, default=10)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--sample", type=float, default=None)
a = p.parse_args()
{"train": cmd_train, "dev": cmd_dev, "test": cmd_test}[a.cmd](a)
