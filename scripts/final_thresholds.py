"""Write the submission from blended test pairs with explicit per-country thresholds (the final step).

python final_thresholds.py --pairs <out>/test_pairs_blend.parquet --data <dataset dir> --out output \
    --thr India=0.85 US=0.95 France=0.85
Each record goes to its single best S1 if the blended probability passes that S1's country threshold.
candidate_pairs.tsv = every pair the models scored (the matching set is a subset).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--pairs", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--thr", nargs="+", required=True, help="Country=threshold ... (unlisted countries use the maximum)")
a = p.parse_args()
thr = {k: float(v) for k, v in (x.split("=") for x in a.thr)}
s1 = pd.read_csv(f"{a.data}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
B = pd.read_parquet(a.pairs)
B["country"] = s1.country.to_numpy()[B.s1_idx.to_numpy()]
b = B.sort_values("prob", ascending=False).drop_duplicates("rec_id")
b = b[b.prob.to_numpy() >= b.country.map(thr).fillna(max(thr.values())).to_numpy()]
Path(a.out).mkdir(parents=True, exist_ok=True)
for fname, col, df in [("matching_results.tsv", "matched_entity_ids", b), ("candidate_pairs.tsv", "candidate_entity_ids", B)]:
    lists = df.drop_duplicates(["s1_idx", "rec_id"]).groupby("s1_idx").rec_id.agg(",".join)
    full = np.full(len(s1), "", dtype=object)
    full[lists.index.to_numpy()] = lists.to_numpy()
    pd.DataFrame({"source1_entity_id": s1.entity_id, col: full}).to_csv(Path(a.out) / fname, sep="\t", index=False)
print(f"wrote {a.out}: {len(b):,} matched records; per country {b.country.value_counts().to_dict()}; thresholds {thr}")
