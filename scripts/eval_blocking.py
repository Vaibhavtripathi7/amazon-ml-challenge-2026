"""Measure shortlist recall on a dev slice of train (10% of S1 by id hash), searching against ALL train S1.

uv run python scripts/eval_blocking.py --work work [--k 10] [--other-frac 0.1]
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


p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--k", type=int, default=10)
p.add_argument("--other-frac", type=float, default=0.1)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
t0 = time.time()
W = f"{args.work}/train"

s1 = pd.read_parquet(f"{W}/source1.parquet", columns=COLS)
dev = set(s1.entity_id[bucket(s1.entity_id) == 0])
gt = pd.read_parquet(f"{W}/truth.parquet")
gt = gt[gt.source1_entity_id.isin(dev)]
truth = gt.assign(rec_id=gt.matched_entity_ids.str.split(",")).explode("rec_id")
truth = truth[truth.rec_id != ""][["source1_entity_id", "rec_id"]].rename(columns={"source1_entity_id": "s1_id"})
matched = set(truth.rec_id)
rng = np.random.default_rng(0)
keep = lambda ids: ids.isin(matched).to_numpy() | (rng.random(len(ids)) < args.other_frac)
recs = pd.concat([read_filtered(f"{W}/source{s}.parquet", keep) for s in (2, 3)], ignore_index=True)
print(f"dev S1 {len(dev):,} | true pairs {len(truth):,} | queries {len(recs):,}  [{time.time() - t0:.0f}s]")

cand = candidates(s1, recs, k=args.k, n_threads=args.threads)
cand = cand[cand.s1_id.isin(dev)]
print(f"candidates done [{time.time() - t0:.0f}s]")

tp = truth.assign(hit=1).set_index(["s1_id", "rec_id"]).hit
key = lambda d: pd.MultiIndex.from_frame(d[["s1_id", "rec_id"]])
print(f"\nshortlist recall (share of true pairs found), k={args.k}:")
for pname in list(PASSES) + ["UNION"]:
    c = cand if pname == "UNION" else cand[cand["pass"] == pname]
    found = tp.index.isin(key(c.drop_duplicates(["s1_id", "rec_id"])))
    print(f"  {pname:10s} {found.mean():.4f}")
u = cand.drop_duplicates(["s1_id", "rec_id"])
is_true = key(u).isin(tp.index)
per_s1 = (is_true.sum() + (~is_true).sum() / args.other_frac) / len(dev)
print(f"\ncandidates per dev S1 (estimated at full scale): {per_s1:.1f}  (true matches per S1: {len(truth) / len(dev):.2f})")
by_c = truth.merge(s1[["entity_id", "country"]], left_on="s1_id", right_on="entity_id")
by_c["found"] = pd.MultiIndex.from_frame(by_c[["s1_id", "rec_id"]]).isin(key(u))
print("union recall by country:", by_c.groupby("country").found.mean().round(4).to_dict())
print(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB  total {time.time() - t0:.0f}s")
