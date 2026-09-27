"""Judge v2 = untrimmed V1 shortlist -> STAGE 1 ranker (keeps top-N per record) -> STAGE 2 judge -> decision.

Run in order (all on the same --work):
  rank_train : train stage-1 ranker on the 'rank' record set; compares binary vs lambdarank vs the v1 rule
  rank_eval  : on dev, grid of (top-N, min stage-1 prob) -> recall, candidates per S1, F0.5 ceiling; picks the
               smallest candidate set within --tol of the best ceiling; saves it next to the rank model
  train      : stage-2 judge on the 'judge' record set (stage-1 survivors; stage-1 score/rank as features)
  dev        : honest dev F0.5 + loss breakdown + candidates per S1; picks the threshold
  test       : matching_results.tsv + candidate_pairs.tsv (= stage-1 survivors = the judge's input)
Record sets are disjoint (amlc.stage), so stage-1 scores fed to stage 2 never come from its own training data.
--sample F: small self-consistent world for local end-to-end checks.
"""
import argparse
import json
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from amlc.judge_features import FEATURES, SIB_FEATURES, Context, pair_features
from amlc.metric import dev_f05
import collections

from amlc.stage import (TF_CHAR, AMBIG_FEATURES, DEV_OTHER_WEIGHT, RANK_FEATURES, Shortlister, ambiguity_features, bucket,
                        iter_records, load_owner, load_s1, log, make_segmenter, rank_features, record_set, top_n)

J_BASE = FEATURES + ["s1_prob", "s1_rank", "src3"]
J_SETS = {"base": J_BASE, "ambig": J_BASE + AMBIG_FEATURES, "sib": J_BASE + AMBIG_FEATURES + SIB_FEATURES}
if TF_CHAR:   # the char pass's own score/rank also go to the judge
    J_SETS = {k: v + ["tf_char_score", "tf_char_rank"] for k, v in J_SETS.items()}   # --features: A/B switch
V1_RULE = "v1_rule"


def holdout(s1, a):
    """Test-like index: drop a share of NON-dev S1 (separate hash) so their records become orphans, as in test
    (test keeps records of businesses that are not in its S1). Never removes dev S1. Train split only."""
    if not a.holdout_s1:
        return s1
    ids = s1.entity_id.to_numpy()
    h = np.fromiter((zlib.crc32((x + "|holdout").encode()) % 1000 for x in ids), np.int64, len(ids))
    drop = (bucket(ids, 10) != 0) & (h < a.holdout_s1 * 1000)
    return s1[~drop].reset_index(drop=True)


def world(a):
    s1 = holdout(load_s1(a.work, "train", a.sample), a)
    owner = load_owner(a.work)
    dev_ids = set(s1.entity_id[bucket(s1.entity_id, 10) == 0])
    if not dev_ids:
        raise SystemExit("no dev S1 (crc32(id) % 10 == 0) in this S1 set: pick a sample that includes them")
    return s1, owner, dev_ids


def set_keep(a, name, s1, owner, dev_ids):
    return record_set(name, owner, dev_ids, set(s1.entity_id) if a.sample else None, a.sample)


def ranker_score(models, X):
    kind = models["kind"]
    if kind == V1_RULE:
        return (X.n_keys + X.tf_best).to_numpy()
    return models["booster"].predict(X[RANK_FEATURES], num_threads=models["threads"])


def load_ranker(a):
    op = json.loads(Path(a.rank_model + ".json").read_text())
    kind = op["objective"]
    path = a.rank_model if kind == "binary" else a.rank_model.replace(".txt", f"_{kind}.txt")
    return {"kind": kind, "booster": lgb.Booster(model_file=path), "threads": a.threads}, op


# ------------------------------------------------------------------ stage 1
def cmd_rank_train(a):
    s1, owner, dev_ids = world(a)
    sl = Shortlister(s1, threads=a.threads)
    log(f"rank_train: S1 {len(s1):,}; shortlister fitted")
    s1_ids = s1.entity_id.to_numpy()
    Xs, ys, gids, owned_rec = [], [], [], []
    offset = 0
    for i, recs in enumerate(iter_records(a.work, "train", set_keep(a, "rank", s1, owner, dev_ids), seg=make_segmenter(s1))):
        f = sl(recs)
        X = rank_features(s1, recs, f, a.threads)
        own = owner.reindex(recs.entity_id.to_numpy()).to_numpy()
        ri = f.rec_idx.to_numpy()
        ys.append((own[ri] == s1_ids[f.s1_idx.to_numpy()]).astype(np.int8))
        gids.append(ri.astype(np.int64) + offset)
        has = pd.Series(own).isin(set(s1_ids)).to_numpy()
        owned_rec.append(pd.DataFrame({"gid": np.arange(len(recs)) + offset,
                                       "val": bucket(recs.entity_id.to_numpy(), 10) == 0, "owned": has}))
        offset += len(recs)
        Xs.append(X)
        log(f"  chunk {i}: {len(recs):,} records -> {len(X):,} pairs ({len(X) / len(recs):.1f}/record), "
            f"{ys[-1].mean():.3f} positive")
    X, y, gid = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(gids)
    R = pd.concat(owned_rec, ignore_index=True)
    del Xs, ys
    order = np.argsort(gid, kind="stable")                      # lambdarank needs rows grouped by record
    X, y, gid = X.iloc[order].reset_index(drop=True), y[order], gid[order]
    val = np.isin(gid, R.gid[R.val].to_numpy())
    log(f"rank pairs {len(y):,} (val {val.sum():,}); positives {y.sum():,}")

    def groups(mask):
        return np.diff(np.r_[0, np.flatnonzero(np.diff(gid[mask])) + 1, mask.sum()])

    base = dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
                bagging_freq=1, num_threads=a.threads, verbose=-1)
    Path(a.rank_model).parent.mkdir(parents=True, exist_ok=True)
    mb = lgb.train(base | dict(objective="binary"), lgb.Dataset(X[~val], y[~val]), 1000,
                   valid_sets=[lgb.Dataset(X[val], y[val])], callbacks=[lgb.early_stopping(50, verbose=False)])
    mb.save_model(a.rank_model, num_iteration=mb.best_iteration)
    mr = lgb.train(base | dict(objective="lambdarank", metric="ndcg", eval_at=[1, 2, 3], lambdarank_truncation_level=10),
                   lgb.Dataset(X[~val], y[~val], group=groups(~val)), 1000,
                   valid_sets=[lgb.Dataset(X[val], y[val], group=groups(val))],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mr.save_model(a.rank_model.replace(".txt", "_lambdarank.txt"), num_iteration=mr.best_iteration)

    # recall@N on the held-out records: owner within the record's top N (records whose owner the shortlist missed count)
    Xv, yv, gv = X[val], y[val], gid[val]
    n_owned = int(R[R.val].owned.sum())
    print(f"\nstage-1 recall@N on held-out rank records (owned: {n_owned:,}); shortlist ceiling "
          f"{len(np.unique(gv[yv == 1])) / max(n_owned, 1):.4f}")
    print(f"  {'ranker':12s} " + " ".join(f"@{n:<6d}" for n in (1, 2, 3, 5, 10)))
    results = {}
    for kind, sc in [(V1_RULE, (Xv.n_keys + Xv.tf_best).to_numpy()),
                     ("binary", mb.predict(Xv, num_iteration=mb.best_iteration)),
                     ("lambdarank", mr.predict(Xv, num_iteration=mr.best_iteration))]:
        rec = []
        for n in (1, 2, 3, 5, 10):
            keep, _ = top_n(gv, sc, n)
            rec.append(len(np.unique(gv[keep & (yv == 1)])) / max(n_owned, 1))
        results[kind] = rec
        print(f"  {kind:12s} " + " ".join(f"{r:.4f} " for r in rec))
    # default "binary": calibrated probabilities allow a probability cut, which gives the best recall at a SMALL
    # candidate set (5% world: binary top-10 + p>=0.001 -> recall 0.9928 at 4.0 cands/S1, vs lambdarank top-2 0.9895
    # at 6.9). "auto" = the old rule (recall@2, then @1).
    best = (max(["binary", "lambdarank"], key=lambda k: (results[k][1], results[k][0])) if a.rank_objective == "auto"
            else a.rank_objective)
    Path(a.rank_model + ".json").write_text(json.dumps({"objective": best}))
    imp = pd.Series(mb.feature_importance("gain"), RANK_FEATURES).sort_values(ascending=False)
    log(f"chosen objective: {best} (--rank-objective {a.rank_objective}); binary top features "
        f"{(imp / imp.sum()).head(8).round(3).to_dict()}")


def cmd_rank_eval(a):
    s1, owner, dev_ids = world(a)
    models, op = load_ranker(a)
    sl = Shortlister(s1, threads=a.threads)
    dev_idx = np.flatnonzero(s1.entity_id.isin(dev_ids).to_numpy())
    is_dev = np.zeros(len(s1), bool)
    is_dev[dev_idx] = True
    s1_ids = s1.entity_id.to_numpy()
    own_dev = owner[owner.isin(dev_ids)]
    log(f"rank_eval ({models['kind']}): dev S1 {len(dev_idx):,}, true pairs {len(own_dev):,}")
    rows = []
    for i, recs in enumerate(iter_records(a.work, "train", set_keep(a, "dev", s1, owner, dev_ids), seg=make_segmenter(s1))):
        f = sl(recs)
        X = rank_features(s1, recs, f, a.threads)
        sc = ranker_score(models, X)
        keep, rank = top_n(f.rec_idx.to_numpy(), sc, 10)
        si, ri = f.s1_idx.to_numpy()[keep], f.rec_idx.to_numpy()[keep]
        m = is_dev[si]
        rid = recs.entity_id.to_numpy()[ri[m]]
        own = owner.reindex(rid).to_numpy()
        rows.append(pd.DataFrame({"s1_idx": si[m], "rec_id": rid, "prob": sc[keep][m], "rank": rank[keep][m],
                                  "is_true": own == s1_ids[si[m]], "full": pd.Series(own).isin(dev_ids).to_numpy()}))
        log(f"  chunk {i}: {len(recs):,} records, {len(f) / len(recs):.1f} shortlisted/record")
    P = pd.concat(rows, ignore_index=True)
    pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    truth = {i: set() for i in dev_idx}
    for r, s in zip(own_dev.index, pos[own_dev.to_numpy()].to_numpy()):
        truth[s].add(r)
    country = s1.country.to_numpy()
    # min_prob is a probability cut: only meaningful for the binary ranker. LambdaRank scores are unbounded and can be
    # negative, so they get no cut (a ">= 0" cut silently dropped every negative-scored candidate: fixed 2026-09-26).
    mps = [None] if models["kind"] != "binary" else [0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
    grid = []
    print(f"\n{'top-N':>5s} {'min_p':>6s} {'recall':>7s} {'cands/S1':>8s} {'F0.5 ceiling':>12s}")
    for n in (1, 2, 3, 5, 10):
        for mp in mps:
            k = P[(P["rank"] < n) & ((P.prob >= mp) if mp is not None else True)]
            t = k[k.is_true]
            ceiling = dev_f05(t.s1_idx.to_numpy(), t.rec_id.to_numpy(), t.full.to_numpy(), truth, dev_idx, country,
                              DEV_OTHER_WEIGHT)[0]
            cands = (k.full.sum() + (~k.full).sum() * DEV_OTHER_WEIGHT) / len(dev_idx)
            grid.append((n, mp, len(t) / len(own_dev), cands, ceiling))
            print(f"{n:5d} {mp if mp is not None else '-':>6} {grid[-1][2]:7.4f} {cands:8.2f} {ceiling:12.4f}", flush=True)
    # candidate budget first (the organisers rank smaller candidate sets higher), then the smallest set within tol
    pool = [g for g in grid if a.max_cands is None or g[3] <= a.max_cands] or [min(grid, key=lambda g: g[3])]
    best_c = max(g[4] for g in pool)
    ok = [g for g in pool if g[4] >= best_c - a.tol]
    n, mp, r, c, ce = min(ok, key=lambda g: g[3])
    op.update(top_n=n, min_prob=mp if mp is not None else None)
    Path(a.rank_model + ".json").write_text(json.dumps(op))
    log(f"chosen operating point: top-{n}, min_prob {mp} -> recall {r:.4f}, {c:.2f} candidates/S1, "
        f"ceiling {ce:.4f} (best ceiling {best_c:.4f}, tol {a.tol})")


# ------------------------------------------------------------------ stage 2
def stage12(sl, ctx, s1, recs, models, op, threads, name_counts):
    """Shortlist -> stage-1 score -> ambiguity features (before pruning) -> prune -> stage-2 features.
    Returns s_idx, r_idx, X with every candidate feature column (callers pick their feature set)."""
    f = sl(recs)
    X1 = rank_features(s1, recs, f, threads)
    sc = ranker_score(models, X1)
    amb = ambiguity_features(f, sc, recs, s1, name_counts)
    keep, rank = top_n(f.rec_idx.to_numpy(), sc, op["top_n"])
    if op.get("legacy_cut"):                                         # v3 exactly (5d41c82): cut for any ranker
        keep &= sc >= (op.get("min_prob") or 0.0)
    elif models["kind"] == "binary" and op.get("min_prob") is not None:   # probability cut: binary ranker only
        keep &= sc >= op["min_prob"]
    f = f[keep].reset_index(drop=True)
    s_idx, r_idx = f.s1_idx.to_numpy(), f.rec_idx.to_numpy()
    flags = f.assign(tf_addr_score=np.float32(0), tf_addr_rank=np.int8(10))
    X = pair_features(ctx, recs, s_idx, r_idx, flags, threads)
    X["s1_prob"], X["s1_rank"] = sc[keep].astype(np.float32), rank[keep]
    X["src3"] = recs.src3.to_numpy()[r_idx].astype(np.int8)
    if TF_CHAR:
        X["tf_char_score"], X["tf_char_rank"] = f.tf_char_score.to_numpy(), f.tf_char_rank.to_numpy()
    for c in AMBIG_FEATURES:
        X[c] = amb[c].to_numpy()[keep]
    return s_idx, r_idx, X


def cmd_train(a):
    s1, owner, dev_ids = world(a)
    models, op = load_ranker(a)
    ctx, sl = Context(s1), Shortlister(s1, threads=a.threads)
    name_counts = collections.Counter(zip(s1.country, s1.n_key))
    feats = J_SETS[a.features]
    log(f"train (stage 2): S1 {len(s1):,}; ranker {models['kind']} top-{op['top_n']} min_p {op.get('min_prob', 0)}; "
        f"features '{a.features}' ({len(feats)})")
    s1_ids = s1.entity_id.to_numpy()
    kept_ids = np.asarray(s1_ids, dtype=object)
    Xs, ys, grp, distr, pc = [], [], [], [], []
    owned_n = kept_n = 0
    for i, recs in enumerate(iter_records(a.work, "train", set_keep(a, "judge", s1, owner, dev_ids), seg=make_segmenter(s1))):
        s_idx, r_idx, X = stage12(sl, ctx, s1, recs, models, op, a.threads, name_counts)
        X = X[feats]
        own = owner.reindex(recs.entity_id.to_numpy()).to_numpy()
        y = (own[r_idx] == s1_ids[s_idx]).astype(np.int8)
        distr.append(~np.isin(own[r_idx], kept_ids))             # no owner among the kept S1 (distractor or holdout orphan)
        owned_n += pd.Series(own).isin(set(s1_ids)).sum()
        kept_n += len(np.unique(r_idx[y == 1]))
        Xs.append(X)
        ys.append(y)
        pc.append(recs.country.to_numpy()[r_idx])
        grp.append(bucket(recs.entity_id.to_numpy()[r_idx], 10))
        log(f"  chunk {i}: {len(recs):,} records -> {len(X):,} pairs ({len(X) / len(recs):.2f}/record)")
    X, y, grp, distr = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(grp), np.concatenate(distr)
    pc = np.concatenate(pc)
    del Xs, ys
    log(f"judge pairs {len(y):,}; owned records whose owner reached the judge: {kept_n / max(owned_n, 1):.4f}")
    wts = np.where((y == 0) & distr, a.distractor_weight, 1.0)    # branch B: make distractor mistakes cost more
    val = grp == 0
    params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1)
    m = lgb.train(params, lgb.Dataset(X[~val], y[~val], weight=wts[~val]), 3000,
                  valid_sets=[lgb.Dataset(X[val], y[val], weight=wts[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    m.save_model(a.model, num_iteration=m.best_iteration)
    if a.per_country:   # G5: one judge per training country; the pooled model above serves unseen countries (France)
        for c in sorted(set(pc)):
            k = pc == c
            mc = lgb.train(params, lgb.Dataset(X[~val & k], y[~val & k], weight=wts[~val & k]), 3000,
                           valid_sets=[lgb.Dataset(X[val & k], y[val & k], weight=wts[val & k])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            mc.save_model(f"{a.model}.{c}", num_iteration=mc.best_iteration)
            log(f"per-country judge {c}: best iter {mc.best_iteration}, val logloss "
                f"{mc.best_score['valid_0']['binary_logloss']:.5f}")
    imp = pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False)
    log(f"model saved {a.model}: best iter {m.best_iteration}, val logloss {m.best_score['valid_0']['binary_logloss']:.5f}")
    print("top features by gain:", (imp / imp.sum()).head(12).round(3).to_dict())


def scored(a, split, keep_fn):
    s1 = load_s1(a.work, split, a.sample)
    if split == "train":
        s1 = holdout(s1, a)
    models, op = load_ranker(a)
    op["legacy_cut"] = a.legacy_cut
    ctx, sl = Context(s1), Shortlister(s1, threads=a.threads)
    judge = lgb.Booster(model_file=a.model)
    feats = judge.feature_name()                                  # always the features the model was trained on
    per = {c: lgb.Booster(model_file=f"{a.model}.{c}") for c in ("US", "India") if Path(f"{a.model}.{c}").exists()}
    name_counts = collections.Counter(zip(s1.country, s1.n_key))
    log(f"{split}: S1 {len(s1):,}; ranker {models['kind']} top-{op['top_n']}; judge features {len(feats)}")
    parts, st = [], []
    keep = keep_fn(s1) if keep_fn else None
    for i, recs in enumerate(iter_records(a.work, split, keep, seg=make_segmenter(s1))):
        s_idx, r_idx, X = stage12(sl, ctx, s1, recs, models, op, a.threads, name_counts)
        prob = judge.predict(X[feats], num_threads=a.threads).astype(np.float32)
        pcountry = recs.country.to_numpy()[r_idx]
        for c, mc in per.items():                                 # country judges where trained (France: pooled)
            k = pcountry == c
            if k.any():
                prob[k] = mc.predict(X[feats][k], num_threads=a.threads).astype(np.float32)
        parts.append(pd.DataFrame({"s1_idx": s_idx.astype(np.int32), "rec_id": recs.entity_id.to_numpy()[r_idx],
                                   "prob": prob}))
        if a.st_countries:                                        # keep features of the countries to adapt to
            k = np.isin(pcountry, a.st_countries)
            if k.any():
                st.append((i, k, X[feats][k].reset_index(drop=True)))
        log(f"  chunk {i}: {len(recs):,} records -> {len(X):,} pairs ({len(X) / len(recs):.2f}/record)")
    if st:
        self_train(a, judge, parts, st)
    return s1, pd.concat(parts, ignore_index=True)


def self_train(a, judge, parts, st):
    """G4: adapt the judge to a country it never saw in training (France) with its own confident decisions.
    Per record: best candidate p >= st_hi -> 1 and its other candidates -> 0; best p <= st_lo -> all 0.
    A few boosting rounds continue from the trained judge (init_model); then that country is rescored."""
    P = pd.concat([parts[i][k].assign(chunk=i) for i, k, _ in st], ignore_index=True)
    X = pd.concat([x for _, _, x in st], ignore_index=True)
    best = P.groupby("rec_id").prob.transform("max").to_numpy()
    is_best = P.prob.to_numpy() == best
    y = np.where(best >= a.st_hi, is_best.astype(np.int8), np.where(best <= a.st_lo, 0, -1))
    lab = y >= 0
    log(f"self-training on {a.st_countries}: pairs {len(P):,}; pseudo-labelled {lab.sum():,} "
        f"(positives {(y == 1).sum():,}, negatives {(y == 0).sum():,})")
    params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1)
    m = lgb.train(params, lgb.Dataset(X[lab], y[lab]), a.st_rounds, init_model=judge)
    m.save_model(a.model + ".st")
    new = m.predict(X, num_threads=a.threads).astype(np.float32)
    off = 0
    for i, k, x in st:
        prob = parts[i].prob.to_numpy().copy()
        prob[k] = new[off:off + len(x)]
        parts[i]["prob"] = prob
        off += len(x)
    log(f"self-trained judge saved {a.model}.st ({a.st_rounds} extra rounds); {a.st_countries} rescored")


def cmd_dev(a):
    owner = load_owner(a.work)
    holder = {}

    def keep_fn(s1):
        dev_ids = set(s1.entity_id[bucket(s1.entity_id, 10) == 0])
        holder["dev_ids"] = dev_ids
        return set_keep(a, "devfull" if a.dev_full else "dev", s1, owner, dev_ids)
    s1, pairs = scored(a, "train", keep_fn)
    w_other = (1.0 if a.dev_full else DEV_OTHER_WEIGHT) * a.dev_distractor_scale   # branch B: test-like density
    dev_ids = holder["dev_ids"]
    dev_idx = np.flatnonzero(s1.entity_id.isin(dev_ids).to_numpy())
    pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    own = owner[owner.isin(dev_ids)]
    truth = {i: set() for i in dev_idx}
    for r, s in zip(own.index, pos[own.to_numpy()].to_numpy()):
        truth[s].add(r)
    pairs["full"] = pairs.rec_id.isin(set(own.index))
    dev_pairs = pairs[np.isin(pairs.s1_idx, dev_idx)]
    is_true = lambda d: np.fromiter((r in truth.get(s, ()) for s, r in zip(d.s1_idx, d.rec_id)), bool, len(d))
    kept = int(is_true(dev_pairs).sum())
    cands = (dev_pairs.full.sum() + (~dev_pairs.full).sum() * w_other) / len(dev_idx)
    log(f"dev S1 {len(dev_idx):,} | true pairs {len(own):,} | reached the judge {kept / len(own):.4f} | "
        f"candidates per S1 {cands:.2f}")
    best = pairs.sort_values("prob", ascending=False).drop_duplicates("rec_id")
    best = best[np.isin(best.s1_idx, dev_idx)]
    country = s1.country.to_numpy()
    f = lambda d: dev_f05(d.s1_idx.to_numpy(), d.rec_id.to_numpy(), d.full.to_numpy(), truth, dev_idx, country,
                          w_other)
    rows = []
    for t in a.thresholds:
        score, per_c, P, R = f(best[best.prob >= t])
        rows.append((t, score))
        print(f"  t={t:.2f}  dev F0.5 {score:.4f}  precision {P:.4f}  recall {R:.4f}  by country {per_c}", flush=True)
    t_best = max(rows, key=lambda r: r[1])[0]
    per_c = {}
    for c in sorted(set(country[dev_idx])):
        m = country[best.s1_idx.to_numpy()] == c
        grid = [(t, f(best[m & (best.prob >= t).to_numpy()])[1].get(c, 0.0)) for t in a.thresholds]
        per_c[c] = max(grid, key=lambda g: g[1])
    print("best threshold per country: " + ", ".join(f"{c} t={t:.2f} F0.5 {v:.4f}" for c, (t, v) in per_c.items()))
    Path(a.model + ".json").write_text(json.dumps({"threshold": t_best,
                                                   "by_country": {c: t for c, (t, _) in per_c.items()}}))
    oracle = dev_pairs[is_true(dev_pairs)].drop_duplicates("rec_id")
    chosen = best[best.prob >= t_best]
    no_fp = chosen[is_true(chosen)]
    (o, oc, *_), (n_, nc, *_), (s_, sc, *_) = f(oracle), f(no_fp), f(chosen)
    print(f"\nloss breakdown at t={t_best:.2f} (macro F0.5 points out of 1.0):")
    print(f"  best possible with this candidate set (perfect judge): {o:.4f}   lost before the judge : {1 - o:.4f}")
    print(f"  lost by the judge rejecting true matches              : {o - n_:.4f}")
    print(f"  lost by wrong merges (false positives)                 : {n_ - s_:.4f}")
    print(f"  = actual dev F0.5                                      : {s_:.4f}")
    for c in sorted(oc):
        print(f"  {c:8s} ceiling {oc[c]:.4f} | judge misses -{oc[c] - nc[c]:.4f} | wrong merges -{nc[c] - sc[c]:.4f} "
              f"| actual {sc[c]:.4f}")
    out = pairs.assign(s1_id=s1.entity_id.to_numpy()[pairs.s1_idx.to_numpy()], is_true=is_true(pairs),
                       is_dev_s1=np.isin(pairs.s1_idx, dev_idx))
    out[["s1_id", "rec_id", "prob", "full", "is_true", "is_dev_s1"]].to_parquet(a.dev_pairs_out, index=False)
    log(f"threshold {t_best:.2f} saved to {a.model}.json; scored dev pairs -> {a.dev_pairs_out}")


def write_outputs(s1_ids, pairs, t, out, country=None):
    """pairs: [s1_idx, rec_id, prob] (all scored = the judge's input). Each record goes to its best S1 if prob >= t.
    t is a number, or {country: threshold} (then `country` = array of S1 countries aligned with s1_ids)."""
    best = pairs.sort_values("prob", ascending=False).drop_duplicates("rec_id")
    if isinstance(t, dict):
        tt = pd.Series(country[best.s1_idx.to_numpy()]).map(t).fillna(t.get("default", 0.5)).to_numpy()
        best = best[best.prob.to_numpy() >= tt]
    else:
        best = best[best.prob >= t]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for fname, col, df in [("matching_results.tsv", "matched_entity_ids", best),
                           ("candidate_pairs.tsv", "candidate_entity_ids", pairs)]:
        lists = df.drop_duplicates(["s1_idx", "rec_id"]).groupby("s1_idx").rec_id.agg(",".join)
        full = np.full(len(s1_ids), "", dtype=object)
        full[lists.index.to_numpy()] = lists.to_numpy()
        with open(out / fname, "w", encoding="utf-8") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for sid, lst in zip(s1_ids, full):
                fh.write(f"{sid}\t{lst}\n")
    log(f"wrote {out}/: {len(best):,} matched records over {best.s1_idx.nunique():,} S1 (threshold {t}); "
        f"candidates per S1 {len(pairs) / len(s1_ids):.2f}")


def cmd_test(a):
    s1, pairs = scored(a, a.split, None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(Path(a.out) / "test_pairs.parquet", index=False)   # threshold can change later without re-scoring
    if a.threshold is not None:
        t = a.threshold
    else:   # per-country thresholds chosen on the (test-like) dev; unseen countries get the strictest of them
        j = json.loads(Path(a.model + ".json").read_text())
        bc = j.get("by_country") or {}
        t = {**bc, "default": max(bc.values(), default=j["threshold"])} if bc else j["threshold"]
    log(f"test thresholds: {t}")
    write_outputs(s1.entity_id.to_numpy(), pairs, t, a.out, s1.country.to_numpy())


def cmd_rethreshold(a):
    """Rewrite the submission from saved test pairs with a new threshold (minutes, no re-scoring).
    --threshold 0.6, or --by-country India=0.7 US=0.5 France=0.6 default=0.6"""
    s1 = pd.read_parquet(f"{a.work}/{a.split}/source1.parquet", columns=["entity_id", "country"])
    pairs = pd.read_parquet(Path(a.pairs_in or Path(a.out) / "test_pairs.parquet"))
    t = {k: float(v) for k, v in (x.split("=") for x in a.by_country)} if a.by_country else a.threshold
    write_outputs(s1.entity_id.to_numpy(), pairs, t, a.out, s1.country.to_numpy())


p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["rank_train", "rank_eval", "train", "dev", "test", "rethreshold"])
p.add_argument("--work", default="work")
p.add_argument("--rank-model", default="models/rank_v2.txt")
p.add_argument("--model", default="models/judge_v2.txt")
p.add_argument("--out", default="output")
p.add_argument("--dev-pairs-out", default="dev_pairs_v2.parquet")
p.add_argument("--split", default="test")
p.add_argument("--threshold", type=float, default=None)
p.add_argument("--by-country", nargs="+", default=None, help="rethreshold: India=0.7 US=0.5 France=0.6 default=0.6")
p.add_argument("--pairs-in", default=None, help="rethreshold: scored test pairs (default <out>/test_pairs.parquet)")
p.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
p.add_argument("--legacy-cut", action="store_true",
               help="test: reproduce v3's stage-1 cut (score >= min_prob for ANY ranker) to rescore with v3 models")
p.add_argument("--st-countries", nargs="+", default=None,
               help="dev/test: self-train the judge on these countries' confident pseudo-labels, then rescore them")
p.add_argument("--st-hi", type=float, default=0.95)
p.add_argument("--st-lo", type=float, default=0.05)
p.add_argument("--st-rounds", type=int, default=100)
p.add_argument("--per-country", action="store_true", help="train: also one judge per training country (G5)")
p.add_argument("--max-cands", type=float, default=None, help="rank_eval: candidate budget per dev S1")
p.add_argument("--tol", type=float, default=0.001, help="max F0.5-ceiling loss accepted to shrink the candidate set")
p.add_argument("--rank-objective", default="binary", choices=["binary", "lambdarank", "auto"])
p.add_argument("--distractor-weight", type=float, default=1.0, help="train: weight of negatives from no-owner records")
p.add_argument("--dev-distractor-scale", type=float, default=1.0, help="dev: extra weight on sampled wrong merges")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--sample", type=float, default=None)
p.add_argument("--dev-full", action="store_true", help="dev scores ALL records (exact, no distractor scaling; slow)")
p.add_argument("--features", default="ambig", choices=list(J_SETS), help="judge feature set (A/B)")
p.add_argument("--holdout-s1", type=float, default=0.0,
               help="train/dev: drop this share of non-dev S1 so their records become orphans (test-like)")
a = p.parse_args()
{"rank_train": cmd_rank_train, "rank_eval": cmd_rank_eval, "train": cmd_train, "dev": cmd_dev,
 "test": cmd_test, "rethreshold": cmd_rethreshold}[a.cmd](a)
