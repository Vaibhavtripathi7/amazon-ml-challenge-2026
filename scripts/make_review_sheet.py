"""Build a plain-text reading sheet of real examples for human review (no interpretation).

nice -n 19 python scripts/make_review_sheet.py --work work --out review/data-explore.md
Sections: random matched groups, singletons with same-name look-alikes, namesake groups,
India local-script records with their business, France (test) samples.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = ["entity_id", "country", "raw_name", "raw_address", "name"]

p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--out", default="review/data-explore.md")
p.add_argument("--data", default="data-2026/student_resource/student_resource/dataset")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
W = f"{a.work}/train"

s1 = pd.read_parquet(f"{W}/source1.parquet", columns=RAW).set_index("entity_id")
gt = pd.read_parquet(f"{W}/truth.parquet").set_index("source1_entity_id").matched_entity_ids
matched_s1 = gt[gt != ""].index.to_numpy()
single_s1 = gt[gt == ""].index.to_numpy()

groups = rng.choice(matched_s1, 40, replace=False)
singles = rng.choice(single_s1, 15, replace=False)
dup_names = s1.name.value_counts()
dup_names = dup_names[(dup_names >= 2) & (dup_names.index.str.len() > 8)]
namesakes = rng.choice(dup_names.index.to_numpy(), 8, replace=False)
namesake_s1 = s1[s1.name.isin(namesakes)].groupby("name").head(3).index.to_numpy()

need = {x for s in np.concatenate([groups, namesake_s1]) for x in gt[s].split(",") if x}
look_names = set(s1.loc[singles, "name"])
rec_parts, look_parts, script_parts = [], [], []
for src in (2, 3):
    for b in pq.ParquetFile(f"{W}/source{src}.parquet").iter_batches(500_000, columns=RAW):
        d = b.to_pandas()
        rec_parts.append(d[d.entity_id.isin(need)])
        look_parts.append(d[d.name.isin(look_names)])
        ind = d[(d.country == "India") & d.raw_name.str.contains(r"[^\x00-\x7F]")]
        script_parts.append(ind.sample(min(len(ind), 3), random_state=a.seed))
recs = pd.concat(rec_parts).set_index("entity_id")
looks = pd.concat(look_parts)
scripts = pd.concat(script_parts).sample(20, random_state=a.seed)
pairs = gt[gt != ""].str.split(",").explode()  # index = S1 id, values = record ids
all_owned = set(pairs.to_numpy())
script_owner = dict(zip(pairs.to_numpy(), pairs.index))
script_owner = {x: script_owner.get(x) for x in scripts.entity_id}

# France exists only in test; read a few raw test rows directly (no cleaned test data needed)
tsv = lambda f, n: pd.read_csv(f"{a.data}/test/{f}", sep="\t", dtype=str, quoting=3, keep_default_na=False, nrows=n).rename(
    columns={"business_name": "raw_name", "business_address": "raw_address"})
fr1 = tsv("test_source1.tsv", 300_000).query("country == 'France'").sample(20, random_state=a.seed)
fr2 = tsv("test_source2.tsv", 300_000).query("country == 'France'").sample(20, random_state=a.seed)


def line(tag, r):
    return f"    {tag}: {r.raw_name}  |  {r.raw_address or '(no address)'}"


out = ["# Data reading sheet (train unless marked test)\n",
       "Read these like a detective. For each block, ask yourself:",
       "1. What exactly tells me these are (or aren't) the same business?",
       "2. What kinds of changes appear between versions? Do they look like a fixed set of rules?",
       "3. Which differences would fool a simple 'how similar are the letters' check?",
       "4. Anything that surprises you?\n",
       "Write your observations under each section.\n",
       "## 1. Random businesses and ALL their true matches\n", "```"]
for s in groups:
    r = s1.loc[s]
    out.append(f"[{r.country}] S1: {r.raw_name}  |  {r.raw_address}")
    for x in gt[s].split(","):
        if x in recs.index:
            out.append(line(x[:2], recs.loc[x]))
    out.append("")
out += ["```", "\nYour notes:\n\n", "## 2. Businesses with NO match, and records with the same cleaned name (look-alikes)\n", "```"]
for s in singles:
    r = s1.loc[s]
    out.append(f"[{r.country}] S1 (no true match): {r.raw_name}  |  {r.raw_address}")
    for x in looks[looks.name == r["name"]].head(4).itertuples():
        out.append(f"    look-alike {x.entity_id[:2]} (belongs to {'another business' if x.entity_id in all_owned else 'NO business'}): "
                   f"{x.raw_name}  |  {x.raw_address or '(no address)'}")
    out.append("")
out += ["```", "\nYour notes:\n\n", "## 3. Namesakes: DIFFERENT businesses with the same cleaned name, and their matches\n", "```"]
for nm in namesakes:
    out.append(f"--- name '{nm}'")
    for s in s1[s1.name == nm].head(3).index:
        r = s1.loc[s]
        out.append(f"[{r.country}] S1: {r.raw_name}  |  {r.raw_address}")
        for x in gt[s].split(",")[:3]:
            if x and x in recs.index:
                out.append(line(x[:2], recs.loc[x]))
    out.append("")
out += ["```", "\nYour notes:\n\n", "## 4. India records written in a local script, next to their true business\n", "```"]
for x in scripts.itertuples():
    s = script_owner.get(x.entity_id)
    own = f"S1: {s1.loc[s].raw_name}  |  {s1.loc[s].raw_address}" if s else "(belongs to NO business)"
    out.append(f"{x.entity_id[:2]}: {x.raw_name}  |  {x.raw_address or '(no address)'}\n    -> {own}\n")
out += ["```", "\nYour notes:\n\n", "## 5. France (TEST, no labels): S1 samples and S2 samples\n", "```"]
out += [f"S1: {r.raw_name}  |  {r.raw_address}" for r in fr1.itertuples()] + [""]
out += [f"S2: {r.raw_name}  |  {r.raw_address or '(no address)'}" for r in fr2.itertuples()]
out += ["```", "\nYour notes:\n"]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(out), encoding="utf-8")
print(f"wrote {a.out}: {len(groups)} groups, {len(singles)} singletons, {len(namesakes)} namesake sets, "
      f"{len(scripts)} script records, {len(fr1) + len(fr2)} France rows")
