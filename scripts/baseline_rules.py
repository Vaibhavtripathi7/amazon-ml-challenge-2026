"""Rule baseline: same country + same cleaned name -> candidate; + same house number -> match.

uv run python scripts/baseline_rules.py --data <dataset dir> --split test  --out output
uv run python scripts/baseline_rules.py --data <dataset dir> --split train --out output_train   # scores itself
Add --limit N to use only the first N Source-1 rows (Source 2/3 are always read in full).

Writes <out>/matching_results.tsv and <out>/candidate_pairs.tsv (one row per S1, tab-separated).
A record that passes the rule for more than one S1 is matched to none of them (each S2/S3 record
belongs to at most one S1), but stays in the candidates.
"""
import argparse
import csv
import resource
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from amlc.metric import f05
from amlc.normalize import house_number, name_key

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="folder containing train/ and test/")
p.add_argument("--split", choices=["train", "test"], default="test")
p.add_argument("--out", default="output")
p.add_argument("--limit", type=int, default=None)
args = p.parse_args()
t0 = time.time()
D = Path(args.data) / args.split


def read(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, **kw)


# 1. index Source 1 by (country, name key)
s1 = read(D / f"{args.split}_source1.tsv", nrows=args.limit)
index = defaultdict(list)
for sid, name, addr, country in s1.itertuples(index=False):
    index[(country, name_key(name))].append((sid, house_number(addr)))
s1_ids = s1.entity_id.tolist()
del s1
print(f"S1 {len(s1_ids):,} indexed, {len(index):,} keys  [{time.time() - t0:.0f}s]")

# 2. stream Source 2 and 3
cands, matches = defaultdict(list), defaultdict(list)
for src in (2, 3):
    for chunk in read(D / f"{args.split}_source{src}.tsv", chunksize=500_000):
        for rid, name, addr, country in chunk.itertuples(index=False):
            hits = index.get((country, name_key(name)))
            if not hits:
                continue
            hn = house_number(addr)
            ok = []
            for s1_id, s1_hn in hits:
                cands[s1_id].append(rid)
                if hn == s1_hn:
                    ok.append(s1_id)
            if len(ok) == 1:
                matches[ok[0]].append(rid)
    print(f"source {src} done  [{time.time() - t0:.0f}s]")

# 3. write both files, one row per S1
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for fname, col, d in [("matching_results.tsv", "matched_entity_ids", matches),
                      ("candidate_pairs.tsv", "candidate_entity_ids", cands)]:
    with open(out / fname, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{','.join(dict.fromkeys(d.get(sid, ())))}\n")
n_match = sum(len(v) for v in matches.values())
print(f"wrote {out}/: {len(matches):,} S1 with matches ({n_match:,} ids)  [{time.time() - t0:.0f}s]")

# 4. on train: score against ground truth
if args.split == "train":
    gt = read(D / "train_ground_truth.tsv")
    truth = {s: set(m.split(",")) - {""} for s, m in gt.itertuples(index=False)}
    s1c = read(D / "train_source1.tsv", usecols=["entity_id", "country"]).set_index("entity_id").country
    if args.limit:
        keep = set(s1_ids)
        truth = {s: t for s, t in truth.items() if s in keep}
    scores = pd.Series({s: f05(set(matches.get(s, ())), t) for s, t in truth.items()})
    true_pairs = sum(len(t) for t in truth.values())
    found = sum(len(t & set(cands.get(s, ()))) for s, t in truth.items())
    print(f"\nF0.5 macro (train): {scores.mean():.4f}   candidate recall: {found / max(true_pairs, 1):.3f}")
    print("by country:", scores.groupby(s1c.reindex(scores.index)).mean().round(4).to_dict())
print(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
