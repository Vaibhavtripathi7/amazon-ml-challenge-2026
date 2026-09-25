"""Derive the Indian native-script state table from TRAIN pairs (provided data only).

For India records, every address component containing non-Latin script is transliterated (anyascii) and
mapped to the parsed state of the record's true S1 owner; majority vote per string (min support 5, purity 0.9).
nice -n 19 python scripts/build_native_states.py --work work --pairs 60000
"""
import argparse
import collections
import json
import re

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from anyascii import anyascii

from amlc.norm import parse_address

p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--pairs", type=int, default=60000)
p.add_argument("--out", default="src/amlc/resources/in_native_states.json")
a = p.parse_args()
W, cols = f"{a.work}/train", ["entity_id", "country", "raw_address"]
t = pq.read_table(f"{W}/source1.parquet", columns=cols)
s1 = t.filter(pc.equal(t["country"], "India")).to_pandas().set_index("entity_id")
gt = pd.read_parquet(f"{W}/truth.parquet")
gt = gt[gt.source1_entity_id.isin(s1.index) & (gt.matched_entity_ids != "")]
own = gt.assign(r=gt.matched_entity_ids.str.split(",")).explode("r").set_index("r").source1_entity_id
own = own.sample(min(a.pairs, len(own)), random_state=0)
votes = collections.defaultdict(collections.Counter)
for src in (2, 3):
    for b in pq.ParquetFile(f"{W}/source{src}.parquet").iter_batches(500_000, columns=cols):
        d = b.to_pandas()
        d = d[d.entity_id.isin(own.index) & d.raw_address.str.contains(r"[^\x00-\x7F]")]
        for rid, addr in zip(d.entity_id, d.raw_address):
            st = parse_address(s1.at[own[rid], "raw_address"], "India")["state"]
            if not st:
                continue
            for comp in addr.split(","):
                if re.search(r"[^\x00-\x7F]", comp):
                    key = re.sub(r"[^a-z0-9]+", " ", anyascii(comp).lower()).strip()
                    votes[key][st] += 1
table = {}
for k, c in votes.items():
    st, n = c.most_common(1)[0]
    if n >= 5 and n / sum(c.values()) >= 0.9:
        table[k] = st
json.dump(dict(sorted(table.items())), open(a.out, "w"), indent=0, ensure_ascii=True)
print(f"{len(votes)} native strings seen, {len(table)} kept -> {a.out}")
print(dict(list(sorted(table.items(), key=lambda kv: -sum(votes[kv[0]].values())))[:40]))
