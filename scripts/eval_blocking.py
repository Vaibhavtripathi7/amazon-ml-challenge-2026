"""Measure shortlist recall on a dev slice of train (10% of S1 by id hash), searching against ALL train S1.

python -u scripts/eval_blocking.py --work work [--k 10] [--other-frac 0.1] [--threads 4]
Queries = every record truly matched to a dev S1 + a random `other-frac` of all other records
(those only inflate candidate counts, so they are scaled back up when reporting size).
"""
import argparse
import resource
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from amlc.blocking import PASSES, candidates

COLS = ["entity_id", "country", "name", "addr", "nums"]


def bucket(ids, n=10):
    return np.array([zlib.crc32(s.encode()) % n for s in ids])


def read_filtered(path, keep):
    parts = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=COLS):
        df = b.to_pandas()
        parts.append(df[keep(df.entity_id)])
    return pd.concat(parts, ignore_index=True)


def mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--k", type=int, default=10)
p.add_argument("--other-frac", type=float, default=0.1)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()
t0 = time.time()
W = f"{args.work}/train"

s1 = pd.read_parquet(f"{W}/source1.parquet", columns=COLS)
is_dev = bucket(s1.entity_id) == 0
gt = pd.read_parquet(f"{W}/truth.parquet")
gt = gt[gt.source1_entity_id.isin(set(s1.entity_id[is_dev]))]
truth = gt.assign(rec_id=gt.matched_entity_ids.str.split(",")).explode("rec_id")
truth = truth[truth.rec_id != ""]
matched = set(truth.rec_id)
rng = np.random.default_rng(0)
keep = lambda ids: ids.isin(matched).to_numpy() | (rng.random(len(ids)) < args.other_frac)
recs = pd.concat([read_filtered(f"{W}/source{s}.parquet", keep) for s in (2, 3)], ignore_index=True)
print(f"dev S1 {is_dev.sum():,} | true pairs {len(truth):,} | queries {len(recs):,}  [{time.time() - t0:.0f}s, peak {mem():.1f} GB]", flush=True)

# true pairs as integer keys s1_idx * n_recs + rec_idx
s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
rec_pos = pd.Series(np.arange(len(recs)), index=recs.entity_id)
n = len(recs)
t_s1 = s1_pos[truth.source1_entity_id].to_numpy()
t_key = t_s1 * n + rec_pos[truth.rec_id].to_numpy()
del gt, truth, matched

cand = candidates(s1, recs, k=args.k, n_threads=args.threads, keep_s1=is_dev, verbose=True)
c_key = cand.s1_idx.to_numpy().astype(np.int64) * n + cand.rec_idx.to_numpy()
print(f"candidates done: {len(cand):,} dev pairs  [{time.time() - t0:.0f}s, peak {mem():.1f} GB]", flush=True)

print(f"\nshortlist recall (share of true pairs found), k={args.k}:")
for pname in list(PASSES) + ["UNION"]:
    keys = c_key if pname == "UNION" else c_key[(cand["pass"] == pname).to_numpy()]
    print(f"  {pname:10s} {np.isin(t_key, keys).mean():.4f}")
u = np.unique(c_key)
is_true = np.isin(u, t_key)
per_s1 = (is_true.sum() + (~is_true).sum() / args.other_frac) / is_dev.sum()
print(f"\ncandidates per dev S1 (estimated at full scale): {per_s1:.1f}  (true matches per S1: {len(t_key) / is_dev.sum():.2f})")
found = np.isin(t_key, u)
country = s1.country.to_numpy()[t_s1]
print("union recall by country:", {c: round(found[country == c].mean(), 4) for c in np.unique(country)})
print(f"peak memory {mem():.2f} GB  total {time.time() - t0:.0f}s")
