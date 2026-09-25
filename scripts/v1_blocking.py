"""V1 stage-1 measurement: recall of key blocking + TF-IDF passes on the dev slice, plus hypothesis checks.

python -u scripts/v1_blocking.py --work W [--threads 4] [--cap 30] [--k 10] [--other-frac 0.1]
Local check: nice -n 19 python -u scripts/v1_blocking.py --work <small work dir> --threads 1
Dev = S1 with crc32(id) % 10 == 0, searched against ALL train S1 (realistic namesake competition).
Queries = records owned by dev S1 + `other-frac` of all other records (distractors; scaled back for sizes).
"""
import argparse
import resource
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from amlc.blocking import Blocker
from amlc.keys import KEY_TYPES, key_candidates, row_keys, token_df
from amlc.norm import number_relation

T0 = time.time()
COLS = ["entity_id", "country", "n_core", "n_key", "n_fold", "n_flat", "n_is_domain", "a_house", "a_house_alts",
        "a_street", "a_state", "a_city", "a_tokens", "a_has_addr"]
TF_PASSES = {
    "tf_name": dict(field="n_core", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
    "tf_addr": dict(field="a_tokens", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
}


def log(msg):
    print(f"{msg}  [{time.time() - T0:.0f}s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB]",
          flush=True)


def bucket(ids, n=10):
    return np.fromiter((zlib.crc32(s.encode()) % n for s in ids), np.int64, len(ids))


p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--cap", type=int, default=30)
p.add_argument("--k", type=int, default=10)
p.add_argument("--other-frac", type=float, default=0.1)
a = p.parse_args()
W = f"{a.work}/train"

s1 = pd.read_parquet(f"{W}/source1.parquet", columns=COLS)
is_dev = bucket(s1.entity_id) == 0
gt = pd.read_parquet(f"{W}/truth.parquet")
gt = gt[gt.source1_entity_id.isin(set(s1.entity_id[is_dev])) & (gt.matched_entity_ids != "")]
truth = gt.assign(rec_id=gt.matched_entity_ids.str.split(",")).explode("rec_id")
owned = set(truth.rec_id)
rng = np.random.default_rng(0)
parts = []
for s in (2, 3):
    for b in pq.ParquetFile(f"{W}/source{s}.parquet").iter_batches(400_000, columns=COLS):
        d = b.to_pandas()
        parts.append(d[d.entity_id.isin(owned).to_numpy() | (rng.random(len(d)) < a.other_frac)])
recs = pd.concat(parts, ignore_index=True)
del parts
rec_owned = recs.entity_id.isin(owned).to_numpy()
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
rec_pos = pd.Series(np.arange(len(recs)), index=recs.entity_id)
truth = truth[truth.rec_id.isin(rec_pos.index)]
n = len(recs)
t_key = s1_pos[truth.source1_entity_id].to_numpy().astype(np.int64) * n + rec_pos[truth.rec_id].to_numpy()
log(f"dev S1 {is_dev.sum():,} | true pairs {len(t_key):,} | queries {n:,}")

dfreq = token_df(s1)
kc = key_candidates(row_keys(s1, dfreq, "s1"), row_keys(recs, dfreq, "rec"), cap=a.cap)
kc = kc[is_dev[kc.s1_idx.to_numpy()]]
log(f"key candidates: {len(kc):,} dev pairs")
tc = Blocker(TF_PASSES, a.k, a.threads).fit(s1).query(recs, keep_s1=is_dev)[["s1_idx", "rec_idx", "pass"]]
tc["pass"] = tc["pass"].astype(str)
log(f"tf-idf candidates: {len(tc):,} dev pairs")
cand = pd.concat([kc, tc], ignore_index=True)
cand["ckey"] = cand.s1_idx.to_numpy().astype(np.int64) * n + cand.rec_idx.to_numpy()

passes = KEY_TYPES + list(TF_PASSES)
hit = {p_: np.isin(t_key, cand.ckey[cand["pass"] == p_].to_numpy()) for p_ in passes}
union = np.zeros(len(t_key), bool)
for h in hit.values():
    union |= h
print("\nrecall per pass (share of dev true pairs) | recall lost if this pass is removed:")
for p_ in passes:
    rest = np.zeros(len(t_key), bool)
    for q, h in hit.items():
        if q != p_:
            rest |= h
    print(f"  {p_:13s} {hit[p_].mean():.4f} | {union.mean() - rest.mean():+.4f}")
keys_only = np.zeros(len(t_key), bool)
for p_ in KEY_TYPES:
    keys_only |= hit[p_]
print(f"  {'KEYS ONLY':13s} {keys_only.mean():.4f}")
print(f"  {'UNION':13s} {union.mean():.4f}")

u = cand.drop_duplicates("ckey")
is_true = np.isin(u.ckey.to_numpy(), t_key)
from_owned = rec_owned[u.rec_idx.to_numpy()]
per_s1 = (is_true.sum() + (~is_true & from_owned).sum() + (~is_true & ~from_owned).sum() / a.other_frac) / is_dev.sum()
print(f"\ncandidates per dev S1 (full scale estimate): {per_s1:.1f}")
t_country = s1.country.to_numpy()[(t_key // n)]
print("union recall by country:", {c: round(float(union[t_country == c].mean()), 4) for c in np.unique(t_country)})
log("recall done")

# ---- hypothesis checks on shortlisted pairs: true matches vs hard negatives (same shortlist)
smp = u.sample(min(len(u), 400_000), random_state=0)
lab = np.isin(smp.ckey.to_numpy(), t_key)
si, ri = smp.s1_idx.to_numpy(), smp.rec_idx.to_numpy()
rel = [number_relation(set(x.split()), set(y.split()))
       for x, y in zip(s1.a_house_alts.to_numpy()[si], recs.a_house_alts.to_numpy()[ri])]
h = pd.DataFrame({"true": lab, "country": s1.country.to_numpy()[si], "rel": rel,
                  "city_eq": (s1.a_city.to_numpy()[si] == recs.a_city.to_numpy()[ri]) & (recs.a_city.to_numpy()[ri] != ""),
                  "name_overlap0": [not (set(x.split()) & set(y.split())) for x, y in
                                    zip(s1.n_core.to_numpy()[si], recs.n_core.to_numpy()[ri])],
                  "domain": recs.n_is_domain.to_numpy()[ri] == 1})
print("\n#2 house-number relation, share within class (true match vs hard negative):")
print(h.groupby(["country", "true"]).rel.value_counts(normalize=True).round(3).unstack().fillna(0).to_string())
print("\n#2b P(true match | relation):")
print(h.groupby(["country", "rel"]).true.mean().round(3).unstack().to_string())
print("\n#3 city equal rate:")
print(h.groupby(["country", "true"]).city_eq.mean().round(3).unstack().to_string())
print("\n#8 among TRUE pairs: share with zero shared name tokens (pseudoword-like), share domain-form:")
print(h[h.true].groupby("country")[["name_overlap0", "domain"]].mean().round(3).to_string())
log("done")
