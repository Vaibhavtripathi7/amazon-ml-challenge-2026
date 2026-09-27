"""Stage 3 (GPU): multilingual cross-encoder re-scoring on top of the LightGBM pipeline (Ditto-style).

Why: the LightGBM judge learns hand-made similarities, so it transfers poorly to a country it never saw (France
leaderboard ~0.81; leave-one-country-out penalty −0.021) and cannot read Indian scripts or French wording.
A multilingual transformer reads both records directly (XLM-R: 100 languages incl. Devanagari/Kannada/Telugu/
Tamil/Bengali/French). Candidates stay ours (small set per S1); only uncertain pairs and unseen countries are
re-scored, then blended with the judge probability.

Commands (each saves its output; re-run a command to resume after a disconnect):
  build  --pairs <dev_pairs.parquet from judge_v2 dev> --data <dataset dir> --out ce/train.parquet
         labelled pairs of DEV S1 only (labels exact), split by S1 into ce-train (80%) / ce-val (20%),
         hard examples kept (judge prob in [lo, hi]) + a sample of easy ones; text = raw + parsed hints.
  train  --in ce/train.parquet --model-dir ce/model [--train-countries US] [--base xlm-roberta-base]
  eval   --in ce/train.parquet --model-dir ce/model --data <dataset dir>
         ce-val: macro F0.5 per country for judge / CE / blend (same weighted dev metric as judge_v2 dev),
         fits the blend (logistic, 2-fold on ce-val) and per-country thresholds -> <model-dir>/blend.json
  apply  --test-pairs <test_pairs.parquet> --data <dataset dir> --model-dir ce/model --out ce/output
         re-scores pairs with judge prob in [lo, hi] and every pair of --unseen countries (France), blends,
         writes matching_results.tsv + candidate_pairs.tsv (same candidate set as the LightGBM run).
"""
import argparse
import csv
import json
import math
import time
import zlib
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # works without `pip install -e .`
from amlc.metric import f05  # noqa: E402
from amlc.norm import parse_address  # noqa: E402

T0 = time.time()
W_OTHER = 10.0   # weight of a wrong merge of a sampled distractor record (judge_v2 dev sampling); 1.0 for --dev-full pairs


def log(msg):
    print(f"{msg}  [{time.time() - T0:.0f}s]", flush=True)


def read_tsv(path, ids=None):
    parts = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, chunksize=500_000):
        parts.append(ch if ids is None else ch[ch.entity_id.isin(ids)])
    return pd.concat(parts, ignore_index=True)


def texts(df):
    """Ditto serialisation + domain-knowledge hints from our parser (house incl. France bis/ter, street, city, state)."""
    out = []
    for n, a, c in zip(df.business_name, df.business_address, df.country):
        p = parse_address(a, c)
        hint = " ".join(f"{k} {v}" for k, v in (("house", p["house"]), ("street", p["street"]), ("city", p["city"]),
                                                ("state", p["state"])) if v)
        out.append(f"{n} | {a} | {hint}")
    return out


def entity_text(data, split, ids):
    ids = set(ids)
    frames = [read_tsv(f"{data}/{split}/{split}_source{s}.tsv", ids) for s in (1, 2, 3)]
    df = pd.concat(frames, ignore_index=True)
    return pd.Series(texts(df), index=df.entity_id), df.set_index("entity_id").country


def h(x, salt, m):
    return zlib.crc32((x + salt).encode()) % m


# ------------------------------------------------------------------ build
def cmd_build(a):
    if Path(a.out).exists() and Path(a.out).with_name("val_truth.parquet").exists():
        log(f"{a.out} already built — skipping")
        return
    P = pd.read_parquet(a.pairs)
    # exact labels for EVERY pair from the ground truth (the saved is_true is exact only for dev S1)
    gt0 = read_tsv(f"{a.data}/train/train_ground_truth.tsv")
    gt0.columns = ["s1", "m"]
    owner = {r: s for s, m in zip(gt0.s1, gt0.m) for r in m.split(",") if r}
    P["is_true"] = [owner.get(r) == s for s, r in zip(P.s1_id, P.rec_id)]
    val = P.is_dev_s1.to_numpy() & np.array([h(s, "|ce", 5) == 0 for s in P.s1_id])
    P["fold"] = np.where(val, "val", "train")
    # train: every negative (decoys) + every hard pair + a share of the easy positives; val: every dev-S1 pair
    easy_pos = P.is_true.to_numpy() & ((P.prob > a.hi) | (P.prob < a.lo)).to_numpy()
    keep = val | ~easy_pos | (np.random.default_rng(0).random(len(P)) < a.easy_frac)
    P = P[keep].reset_index(drop=True)
    tr = P.fold == "train"
    log(f"pairs kept {len(P):,}: train {int(tr.sum()):,} (positives {int(P.is_true[tr].sum()):,}, "
        f"negatives {int((~P.is_true[tr]).sum()):,}) | val {int((~tr).sum()):,} dev-S1 pairs")
    tx, ctry = entity_text(a.data, "train", set(P.s1_id) | set(P.rec_id))
    P["text_s1"], P["text_rec"] = tx.reindex(P.s1_id).to_numpy(), tx.reindex(P.rec_id).to_numpy()
    P["country"] = ctry.reindex(P.s1_id).to_numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(a.out, index=False)
    # truth for the metric: every true record of every dev S1 in ce-val (includes shortlist misses)
    gt = gt0
    val_s1 = set(P.s1_id[P.fold == "val"]) | {s for s in gt.s1 if h(s, "", 10) == 0 and h(s, "|ce", 5) == 0}
    gt = gt[gt.s1.isin(val_s1)]
    gt.assign(country=ctry.reindex(gt.s1).to_numpy()).to_parquet(Path(a.out).with_name("val_truth.parquet"), index=False)
    log(f"wrote {a.out} and val_truth.parquet ({len(gt):,} ce-val S1)")


# ------------------------------------------------------------------ model
def load_model(a, n_train=None):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if a.tiny:                                                       # smoke test: random tiny model, same tokenizer
        cfg = AutoConfig.from_pretrained(a.base, num_labels=1, hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, intermediate_size=64)
        model = AutoModelForSequenceClassification.from_config(cfg)
    elif Path(a.model_dir, "config.json").exists() and n_train is None:
        model = AutoModelForSequenceClassification.from_pretrained(a.model_dir, num_labels=1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1)
    if n_train is None and a.tiny and Path(a.model_dir, "model.pt").exists():
        model.load_state_dict(torch.load(Path(a.model_dir, "model.pt"), map_location="cpu"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(dev), dev


def batches(tok, A, B, bs, max_len, order=None):
    idx = np.arange(len(A)) if order is None else order
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield j, tok([A[k] for k in j], [B[k] for k in j], truncation=True, max_length=max_len, padding=True,
                     return_tensors="pt")


def cmd_train(a):
    import torch
    if Path(a.model_dir, "TRAIN_DONE").exists():
        log(f"{a.model_dir}: already trained (TRAIN_DONE) — skipping")
        return
    P = pd.read_parquet(a.inp)
    P = P[P.fold == "train"]
    if a.train_countries:
        P = P[P.country.isin(a.train_countries)]
    P = P.sample(frac=1.0, random_state=0).head(a.max_pairs).reset_index(drop=True)
    tok, model, dev = load_model(a, n_train=len(P))
    log(f"train pairs {len(P):,} (positives {int(P.is_true.sum()):,}) on {dev}; base {a.base}")
    A, B, y = P.text_s1.tolist(), P.text_rec.tolist(), torch.tensor(P.is_true.to_numpy(), dtype=torch.float32)
    steps = math.ceil(len(P) / a.bs) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps)))
                                              * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.model_dir, "ckpt.pt")
    step = 0
    if ckpt.exists():                                               # resume after a disconnect
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step = st["step"]
        log(f"resumed from checkpoint at step {step}/{steps}")
    model.train()
    per_epoch = math.ceil(len(P) / a.bs)
    lens = np.array([len(x) + len(z) for x, z in zip(A, B)])
    run = []
    for ep in range(a.epochs):
        # length-bucketed batches (less padding, ~1.5x faster): shuffle, sort within blocks of 64 batches by length,
        # cut into batches, shuffle the batch order. Deterministic per epoch, so resume can skip finished batches.
        rng = np.random.default_rng(ep)
        order = rng.permutation(len(P))
        blk = a.bs * 64
        order = np.concatenate([o[np.argsort(lens[o], kind="stable")] for o in np.array_split(order, max(1, math.ceil(len(order) / blk)))])
        bl = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        bl = [bl[i] for i in rng.permutation(len(bl))]
        done = max(0, step - ep * per_epoch)                        # batches already done in this epoch
        if done >= len(bl):
            continue
        for b in bl[done:]:
            j = b
            enc = tok([A[k] for k in j], [B[k] for k in j], truncation=True, max_length=a.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                out = model(**enc).logits.squeeze(-1)
                loss = lossf(out.float(), y[j].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run.append(loss.item())
            if step % 200 == 0 or step == steps:
                log(f"  epoch {ep} step {step}/{steps} loss (mean of last {len(run)} steps) {np.mean(run):.4f}")
                run = []
            if step % a.ckpt_every == 0 and step < steps:
                tmp = ckpt.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step}, tmp)
                tmp.replace(ckpt)                                   # atomic: a half-written checkpoint never wins
                log(f"  checkpoint saved at step {step}")
            if a.stop_after_steps and step >= a.stop_after_steps:  # test hook: simulate a disconnect
                log("stopping early (test hook)")
                return
    if a.tiny:
        torch.save(model.state_dict(), Path(a.model_dir, "model.pt"))
    else:
        model.save_pretrained(a.model_dir)
    tok.save_pretrained(a.model_dir)
    Path(a.model_dir, "TRAIN_DONE").touch()
    ckpt.unlink(missing_ok=True)
    log(f"model saved to {a.model_dir}")


def predict(a, tok, model, dev, A, B):
    import torch
    order = np.argsort([len(x) + len(z) for x, z in zip(A, B)])     # length-sorted batches: little padding
    out = np.zeros(len(A), np.float32)
    model.eval()
    with torch.no_grad():
        for n, (j, enc) in enumerate(batches(tok, A, B, a.bs_pred, a.max_len, order)):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                out[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if n % 500 == 0:
                log(f"  scored {min((n + 1) * a.bs_pred, len(A)):,}/{len(A):,}")
    return out


# ------------------------------------------------------------------ eval (ce-val) + blend
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_prob(w, pj, pc):
    z = w[0] + w[1] * logit(pj) + w[2] * logit(pc)
    return 1 / (1 + np.exp(-z))


def macro(V, truth, prob_col, thr, country):
    """Weighted dev macro F0.5 (as judge_v2 dev): each record -> best S1 if prob >= its country's threshold;
    wrong merges of sampled distractor records (full=False) count x10."""
    b = V.sort_values(prob_col, ascending=False).drop_duplicates("rec_id")
    b = b[b[prob_col].to_numpy() >= b.country.map(thr).to_numpy()]
    tp = b[b.is_true].groupby("s1_id").size()
    fp = b[~b.is_true].assign(w=np.where(b[~b.is_true].full, 1.0, W_OTHER)).groupby("s1_id").w.sum()
    out = {}
    for c in country:
        ss = truth[truth.country == c]
        sc = []
        for s, T in zip(ss.s1, ss.n):
            t_, f_ = tp.get(s, 0), fp.get(s, 0.0)
            if T == 0:
                sc.append(1.0 if f_ == 0 else 0.0)
            elif t_ == 0:
                sc.append(0.0)
            else:
                P_, R_ = t_ / (t_ + f_), t_ / T
                sc.append(1.25 * P_ * R_ / (0.25 * P_ + R_))
        out[c] = float(np.mean(sc))
    return out


def cmd_eval(a):
    from sklearn.linear_model import LogisticRegression
    P = pd.read_parquet(a.inp)
    V = P[P.fold == "val"].reset_index(drop=True)
    truth = pd.read_parquet(Path(a.inp).with_name("val_truth.parquet"))
    truth["n"] = truth.m.map(lambda m: len([x for x in m.split(",") if x]))
    if a.truth_in_pairs:                                           # test slices only: score the S1 the slice covers
        truth = truth[truth.s1.isin(set(V.s1_id))]
    tok, model, dev = load_model(a)
    band = (V.prob >= a.lo) & (V.prob <= a.hi) if not a.all_pairs else np.ones(len(V), bool)
    V["ce"] = V.prob.astype(np.float64)
    V.loc[band, "ce"] = predict(a, tok, model, dev, V.text_s1[band].tolist(), V.text_rec[band].tolist())
    log(f"ce-val pairs {len(V):,}; re-scored {int(band.sum()):,}; ce-val S1 scored {len(truth):,}")
    from sklearn.metrics import roc_auc_score
    yb = V.is_true[band].to_numpy()
    if 0 < yb.sum() < len(yb):   # separation on the re-scored pairs: 0.5 = chance, 1.0 = perfect
        print(f"  AUC on re-scored pairs ({len(yb):,}, {int(yb.sum()):,} true): judge {roc_auc_score(yb, V.prob[band]):.4f}"
              f" | CE {roc_auc_score(yb, V.ce[band]):.4f}", flush=True)
    # blend: logistic on (logit judge, logit ce), 2-fold by S1 on the re-scored band
    fold2 = np.array([h(s, "|blend", 2) for s in V.s1_id])
    V["blend"] = V.prob.astype(np.float64)
    ws = []
    for f in (0, 1):
        tr, te = band & (fold2 != f), band & (fold2 == f)
        X = np.c_[logit(V.prob[tr]), logit(V.ce[tr])]
        lr = LogisticRegression(C=1.0).fit(X, V.is_true[tr])
        w = [float(lr.intercept_[0]), float(lr.coef_[0][0]), float(lr.coef_[0][1])]
        ws.append(w)
        V.loc[te, "blend"] = blend_prob(np.array(w), V.prob[te].to_numpy(), V.ce[te].to_numpy())
    w = np.mean(ws, axis=0)
    countries = sorted(truth.country.dropna().unique())
    res = {}
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for col in ("prob", "ce", "blend"):
        best = {}
        for c in countries:
            sc = {t: macro(V[V.country == c], truth, col, {c: t}, [c])[c] for t in grid}
            best[c] = max(sc, key=sc.get)
            res[(col, c)] = (best[c], sc[best[c]])
        print(f"  {col:6s} " + " | ".join(f"{c}: {res[(col, c)][1]:.4f} (t={res[(col, c)][0]})" for c in countries),
              flush=True)
    thr = {c: res[("blend", c)][0] for c in countries}
    json.dump({"w": list(map(float, w)), "thr": thr, "lo": a.lo, "hi": a.hi, "all_pairs": a.all_pairs,
               "results": {f"{k[0]}|{k[1]}": v for k, v in res.items()}}, open(Path(a.model_dir, f"blend{a.tag}.json"), "w"))
    V[["s1_id", "rec_id", "country", "prob", "ce", "blend", "is_true", "full"]].to_parquet(
        Path(a.model_dir, f"val_scored{a.tag}.parquet"), index=False)
    log(f"blend weights {np.round(w, 3).tolist()}, thresholds {thr} -> {a.model_dir}/blend{a.tag}.json")


def per_s1(V, truth, col, thr):
    """F0.5 per ce-val S1 (Series indexed by S1), same rules as macro()."""
    b = V.sort_values(col, ascending=False).drop_duplicates("rec_id")
    b = b[b[col].to_numpy() >= b.country.map(thr).to_numpy()]
    tp = b[b.is_true].groupby("s1_id").size()
    fp = b[~b.is_true].assign(w=np.where(b[~b.is_true].full, 1.0, W_OTHER)).groupby("s1_id").w.sum()
    t_, f_, n_ = tp.reindex(truth.s1).fillna(0).to_numpy(), fp.reindex(truth.s1).fillna(0).to_numpy(), truth.n.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        P_, R_ = t_ / (t_ + f_), t_ / n_
        f = np.where(n_ == 0, (f_ == 0).astype(float), np.where(t_ == 0, 0.0, 1.25 * P_ * R_ / (0.25 * P_ + R_)))
    return pd.Series(f, index=truth.s1.to_numpy())


def cmd_check(a):
    """Honest estimate: thresholds chosen on half A of the ce-val S1, scored on half B (and vice versa);
    paired bootstrap 95% CI of blend (and ce) minus judge, per country."""
    V = pd.read_parquet(Path(a.model_dir, f"val_scored{a.tag}.parquet"))
    truth = pd.read_parquet(Path(a.inp).with_name("val_truth.parquet"))
    truth["n"] = truth.m.map(lambda m: len([x for x in m.split(",") if x]))
    if a.truth_in_pairs:
        truth = truth[truth.s1.isin(set(V.s1_id))]
    half = np.array([h(s, "|half", 2) for s in truth.s1])
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cols = ["prob", "ce", "blend"]
    if a.bands:   # footprint sweep (needs CE scores on all pairs: eval --all-pairs): re-fit the blend inside each band
        from sklearn.linear_model import LogisticRegression
        fold2 = np.array([h(s, "|blend", 2) for s in V.s1_id])
        for bd in a.bands:
            lo, hi = map(float, bd.split(","))
            band = ((V.prob >= lo) & (V.prob <= hi)).to_numpy()
            col = f"band[{lo},{hi}] {band.mean():.3f}"
            V[col] = V.prob.astype(np.float64)
            for f in (0, 1):
                tr, te = band & (fold2 != f), band & (fold2 == f)
                lr = LogisticRegression(C=1.0).fit(np.c_[logit(V.prob[tr]), logit(V.ce[tr])], V.is_true[tr])
                w = np.array([lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]])
                V.loc[te, col] = blend_prob(w, V.prob[te].to_numpy(), V.ce[te].to_numpy())
            cols.append(col)
    rng = np.random.default_rng(0)
    for c in sorted(truth.country.dropna().unique()):
        out = {}
        for col in cols:
            scores = []
            for f in (0, 1):
                tA, tB = truth[(truth.country == c).to_numpy() & (half == f)], truth[(truth.country == c).to_numpy() & (half != f)]
                t = max(grid, key=lambda t: per_s1(V, tA, col, {c: t}).mean())
                scores.append(per_s1(V, tB, col, {c: t}))
            out[col] = pd.concat(scores)
        line = f"  {c:6s} out-of-fold F0.5: judge {out['prob'].mean():.4f} | CE {out['ce'].mean():.4f} | blend {out['blend'].mean():.4f}"
        for col in cols[1:]:
            d = (out[col] - out["prob"].reindex(out[col].index)).to_numpy()
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            line += f" | {col}-judge {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
        print(line + f"  ({len(out['prob']):,} S1)", flush=True)


# ------------------------------------------------------------------ apply (test)
def cmd_apply(a):
    s1 = read_tsv(f"{a.data}/test/test_source1.tsv")[["entity_id", "country"]]
    T = pd.read_parquet(a.test_pairs)
    T["s1_id"] = s1.entity_id.to_numpy()[T.s1_idx.to_numpy()]
    T["country"] = s1.country.to_numpy()[T.s1_idx.to_numpy()]
    cfg = json.load(open(Path(a.model_dir, f"blend{a.tag}.json")))
    lo, hi = (0.0, 1.0) if cfg.get("all_pairs") else (cfg["lo"], cfg["hi"])   # the band the blend was fitted on
    sel = ((T.prob >= lo) & (T.prob <= hi)) | (T.country.isin(a.unseen or []) & (not a.unseen_band_only))
    log(f"test pairs {len(T):,}; re-scoring {int(sel.sum()):,} (band [{lo}, {hi}] from blend{a.tag}.json + unseen {a.unseen})")
    Path(a.out, "chunks").mkdir(parents=True, exist_ok=True)
    sub = T[sel]
    rows = np.flatnonzero(sel.to_numpy())
    tcache = Path(a.out, "texts.parquet")                           # parsed texts, reused after a disconnect
    if tcache.exists():
        tx = pd.read_parquet(tcache).set_index("id").text
    else:
        tx, _ = entity_text(a.data, "test", set(sub.s1_id) | set(sub.rec_id))
        pd.DataFrame({"id": tx.index, "text": tx.to_numpy()}).to_parquet(tcache, index=False)
    A_all, B_all = tx.reindex(sub.s1_id).tolist(), tx.reindex(sub.rec_id).tolist()
    tok = model = dev = None
    for i in range(0, len(rows), a.chunk):                          # each chunk saved: resume skips finished ones
        f = Path(a.out, "chunks", f"chunk_{i // a.chunk:05d}.parquet")
        if f.exists():
            continue
        if model is None:
            tok, model, dev = load_model(a)
        ce = predict(a, tok, model, dev, A_all[i:i + a.chunk], B_all[i:i + a.chunk])
        pd.DataFrame({"row": rows[i:i + a.chunk], "ce": ce}).to_parquet(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        log(f"chunk {i // a.chunk + 1}/{math.ceil(len(rows) / a.chunk)} saved")
    S = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.out, "chunks").glob("chunk_*.parquet"))],
                  ignore_index=True)
    assert len(S) == len(rows), (len(S), len(rows))
    p = T.prob.to_numpy().astype(np.float64).copy()
    rows_, ce_ = S.row.to_numpy(), S.ce.to_numpy()
    w_seen = np.array(cfg["w"])
    # unseen countries (France): weights fitted in the unseen-country rehearsal (Cell 6: US-only model scored on India),
    # where the transformer transferred poorly (India 0.8909 alone) and the blend had to lean on the judge
    w_unseen = np.array(json.load(open(a.unseen_blend))["w"]) if a.unseen_blend else w_seen
    uns = T.country.to_numpy()[rows_]
    uns = np.isin(uns, a.unseen or [])
    p[rows_[~uns]] = blend_prob(w_seen, p[rows_[~uns]], ce_[~uns])
    p[rows_[uns]] = blend_prob(w_unseen, p[rows_[uns]], ce_[uns])
    log(f"blend weights: seen {np.round(w_seen, 3).tolist()} on {int((~uns).sum()):,} pairs | unseen {a.unseen} "
        f"{np.round(w_unseen, 3).tolist()} on {int(uns.sum()):,} pairs")
    T["prob"] = p
    thr = dict(cfg["thr"])
    thr_unseen = a.unseen_threshold if a.unseen_threshold is not None else max(thr.values())
    for c in a.unseen or []:
        thr.setdefault(c, thr_unseen)
    best = T.sort_values("prob", ascending=False).drop_duplicates("rec_id")
    best = best[best.prob.to_numpy() >= best.country.map(thr).fillna(thr_unseen).to_numpy()]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for fname, col, df in [("matching_results.tsv", "matched_entity_ids", best),
                           ("candidate_pairs.tsv", "candidate_entity_ids", T)]:
        lists = df.drop_duplicates(["s1_idx", "rec_id"]).groupby("s1_idx").rec_id.agg(",".join)
        full = np.full(len(s1), "", dtype=object)
        full[lists.index.to_numpy()] = lists.to_numpy()
        pd.DataFrame({"source1_entity_id": s1.entity_id, col: full}).to_csv(Path(a.out) / fname, sep="\t", index=False)
    T[["s1_idx", "rec_id", "prob"]].to_parquet(Path(a.out, "test_pairs_blend.parquet"), index=False)
    log(f"wrote {a.out}: {len(best):,} matched records over {best.s1_idx.nunique():,} S1; thresholds {thr}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "train", "eval", "apply", "check"])
    p.add_argument("--pairs")
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--in", dest="inp")
    p.add_argument("--model-dir", default="ce/model")
    p.add_argument("--base", default="xlm-roberta-base")
    p.add_argument("--tiny", action="store_true", help="smoke test with a tiny random model")
    p.add_argument("--train-countries", nargs="+", default=None, help="train only on these (unseen-country rehearsal)")
    p.add_argument("--test-pairs")
    p.add_argument("--unseen", nargs="+", default=["France"])
    p.add_argument("--unseen-threshold", type=float, default=None)
    p.add_argument("--lo", type=float, default=0.01, help="re-score pairs with judge prob in [lo, hi]")
    p.add_argument("--hi", type=float, default=0.99)
    p.add_argument("--easy-frac", type=float, default=0.15, help="build: share of easy train pairs kept")
    p.add_argument("--all-pairs", action="store_true", help="eval: re-score every ce-val pair (CE alone, no band)")
    p.add_argument("--max-pairs", type=int, default=2_000_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--bs-pred", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--ckpt-every", type=int, default=1500, help="train: checkpoint to --model-dir every N steps")
    p.add_argument("--stop-after-steps", type=int, default=0, help="test hook: stop training early")
    p.add_argument("--chunk", type=int, default=200_000, help="apply: pairs per saved scoring chunk")
    p.add_argument("--bands", nargs="+", default=None, help='check: footprints to compare, e.g. 0.01,0.99 0.001,0.999 0,1')
    p.add_argument("--unseen-band-only", action="store_true",
                   help="apply: re-score unseen-country pairs only inside the band (not all of them)")
    p.add_argument("--unseen-blend", default=None, help="apply: blend.json fitted in the unseen-country rehearsal")
    p.add_argument("--tag", default="", help="eval/check/apply: suffix for blend/val_scored files (compare variants)")
    p.add_argument("--truth-in-pairs", action="store_true", help="eval on a test slice: only S1 present in the pairs")
    p.add_argument("--w-other", type=float, default=10.0, help="eval: 10 for sampled dev pairs, 1 for --dev-full pairs")
    a = p.parse_args()
    global W_OTHER
    W_OTHER = a.w_other
    {"build": cmd_build, "train": cmd_train, "eval": cmd_eval, "apply": cmd_apply, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
