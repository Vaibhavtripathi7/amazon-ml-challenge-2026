"""Mistake analysis of a finished dev run (no retraining): where the dev F0.5 is lost, and readable samples.

python -u scripts/mistakes.py --work <train parquet dir root> --pairs <dev_pairs*.parquet> --model-json <judge json>
                              --out mistakes.md [--n 20]
Inputs are the run's own outputs: dev pairs (s1_id, rec_id, prob, full, is_true, is_dev_s1), the cleaned parquet
used by that run (work/train), and the chosen threshold.

Sections: 1) loss by outcome (per dev S1), 2) drivers: tag rates among errors vs among correct matches,
3) samples of wrong merges, judge misses, shortlist misses, singletons given a match.
"""
import argparse
import json
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from amlc.metric import f05
from amlc.norm import number_relation

COLS = ["entity_id", "country", "raw_name", "raw_address", "n_key", "n_core", "n_suffix", "n_is_domain", "a_house",
        "a_house_alts", "a_street", "a_city", "a_state", "a_has_addr"]

p = argparse.ArgumentParser()
p.add_argument("--work", required=True, help="dir containing train/ (the run's cleaned parquet)")
p.add_argument("--pairs", required=True)
p.add_argument("--model-json", required=True)
p.add_argument("--out", default="mistakes.md")
p.add_argument("--n", type=int, default=20, help="samples per country per section")
a = p.parse_args()
W = f"{a.work}/train"
t = json.load(open(a.model_json))["threshold"]

s1 = pd.read_parquet(f"{W}/source1.parquet", columns=COLS).set_index("entity_id")
dev_ids = s1.index[[zlib.crc32(i.encode()) % 10 == 0 for i in s1.index]]
gt = pd.read_parquet(f"{W}/truth.parquet").set_index("source1_entity_id").matched_entity_ids.reindex(dev_ids).fillna("")
truth = {s: set(m.split(",")) - {""} for s, m in gt.items()}
owner = {r: s for s, rs in truth.items() for r in rs}
P = pd.read_parquet(a.pairs)
best = P.sort_values("prob", ascending=False).drop_duplicates("rec_id")
assigned = best[best.prob >= t]
A = assigned[assigned.s1_id.isin(truth.keys())].groupby("s1_id").rec_id.agg(set).to_dict()
C = P[P.s1_id.isin(truth.keys())].groupby("s1_id").rec_id.agg(set).to_dict()
need = set(P.rec_id) | set(owner)
rec = pd.concat([b.to_pandas().pipe(lambda d: d[d.entity_id.isin(need)])
                 for s in (2, 3) for b in pq.ParquetFile(f"{W}/source{s}.parquet").iter_batches(400_000, columns=COLS)])
rec = rec.set_index("entity_id")
native = rec.raw_name.str.contains(r"[ऀ-෿]")

# ---- 1. per dev S1 outcome
rows = []
for s, T in truth.items():
    Aa, Cc = A.get(s, set()), C.get(s, set())
    rows.append({"s1": s, "country": s1.at[s, "country"], "f": f05(Aa, T), "n_true": len(T),
                 "shortlist_miss": len(T - Cc), "judge_miss": len((Cc & T) - Aa), "wrong_merge": len(Aa - T)})
S = pd.DataFrame(rows)
S["outcome"] = np.select([S.f == 1, S.n_true == 0], ["perfect", "singleton given a match"],
                         default=np.where(S.wrong_merge > 0, "has wrong merge",
                                          np.where(S.shortlist_miss > 0, "shortlist miss", "judge miss only")))
out = [f"# Mistake analysis (threshold {t:.2f}, {len(S):,} dev S1, {S.n_true.sum():,} true pairs)\n",
       "## 1. Where the macro F0.5 is lost (per dev S1; distractors are the 10% sample, not scaled here)\n",
       "| outcome | S1 count | share of S1 | F0.5 points lost (of 1.0) |", "| --- | --- | --- | --- |"]
for o, g in S.groupby("outcome"):
    out.append(f"| {o} | {len(g):,} | {len(g) / len(S):.3f} | {(1 - g.f).sum() / len(S):.4f} |")
out.append(f"| **total** | {len(S):,} | 1.000 | **{(1 - S.f).sum() / len(S):.4f}** |")
out.append("\nBy country: " + ", ".join(f"{c} F0.5 {g.f.mean():.4f}" for c, g in S.groupby("country")) + "\n")


# ---- 2. drivers: tags on record-level errors vs correct matches
def tags(r_id, s_id):
    r, s = rec.loc[r_id], s1.loc[s_id]
    rel = number_relation(set(r.a_house_alts.split()), set(s.a_house_alts.split()))
    return {"country": r.country, "source": r_id[:2], "no_address": r.a_has_addr == 0, "domain_name": r.n_is_domain == 1,
            "script_name": bool(native.get(r_id, False)), "same_name_key_as_S1": r.n_key == s.n_key,
            "suffix_conflict": bool(r.n_suffix and s.n_suffix and r.n_suffix != s.n_suffix), "house_" + rel: True}


correct = [(r, s) for s, rs in A.items() for r in rs if r in truth[s]]
wrong = [(r, s) for s, rs in A.items() for r in rs if r not in truth[s]]
jmiss = [(r, s) for s, T in truth.items() for r in (C.get(s, set()) & T) - A.get(s, set())]
smiss = [(r, s) for s, T in truth.items() for r in T - C.get(s, set()) if r in rec.index]
rng = np.random.default_rng(0)
sample = lambda lst, k=3000: [lst[i] for i in rng.choice(len(lst), min(k, len(lst)), replace=False)] if lst else []
tab = {}
for name, lst in [("correct matches", correct), ("wrong merges (vs the S1 we chose)", wrong),
                  ("judge misses", jmiss), ("shortlist misses", smiss)]:
    d = pd.DataFrame([tags(r, s) for r, s in sample(lst)]).fillna(False)
    tab[name] = {c: (d[c] == True).mean() if d[c].dtype == bool or d[c].dtype == object and c not in ("country", "source")
                 else None for c in d.columns}
    tab[name].update({f"country={v}": (d.country == v).mean() for v in ("US", "India")})
    tab[name].update({f"source={v}": (d.source == v).mean() for v in ("S2", "S3")})
keys = sorted({k for v in tab.values() for k, x in v.items() if x is not None and k not in ("country", "source")})
out += ["## 2. What drives the errors: share of each tag (sampled up to 3,000 per column)\n",
        "| tag | " + " | ".join(tab) + " |", "| --- |" + " --- |" * len(tab)]
for k in keys:
    out.append(f"| {k} | " + " | ".join(f"{tab[n].get(k) or 0:.3f}" for n in tab) + " |")
out.append(f"\nCounts: correct {len(correct):,} · wrong merges {len(wrong):,} · judge misses {len(jmiss):,} · "
           f"shortlist misses {len(smiss):,}\n")


# ---- 3. samples
def show_s1(s, label):
    x = s1.loc[s]
    return f"    {label}: {x.raw_name}  |  {x.raw_address}"


def show_rec(r):
    x = rec.loc[r]
    return f"{r[:2]} : {x.raw_name}  |  {x.raw_address or '(no address)'}"


prob_of = P.set_index(["s1_id", "rec_id"]).prob
cands_of = P.sort_values("prob", ascending=False).groupby("rec_id")


def section(title, items, render):
    lines = [f"\n## {title}\n"]
    for c in ("US", "India"):
        sub = [x for x in items if s1.at[x[1], "country"] == c]
        lines += [f"\n### {c} ({len(sub):,} in total)\n", "```"]
        for x in sample(sub, a.n):
            lines += render(*x) + [""]
        lines.append("```")
    return lines


def r_wrong(r, s):
    o = owner.get(r)
    L = [show_rec(r), show_s1(s, f"we matched (p={prob_of.get((s, r), float('nan')):.2f})")]
    L.append(show_s1(o, "true owner" + (f" (p={prob_of[(o, r)]:.2f})" if (o, r) in prob_of else " (NOT a candidate)"))
             if o else "    true owner: none (distractor / non-dev business)")
    return L


def r_jmiss(r, s):
    b = best[best.rec_id == r]
    L = [show_rec(r), show_s1(s, f"true owner (p={prob_of.get((s, r), float('nan')):.2f})")]
    if len(b) and b.s1_id.iat[0] != s:
        L.append(show_s1(b.s1_id.iat[0], f"won instead (p={b.prob.iat[0]:.2f})"))
    return L


def r_smiss(r, s):
    L = [show_rec(r), show_s1(s, "true owner (never a candidate)")]
    if r in cands_of.groups:
        for s2, pr in cands_of.get_group(r)[["s1_id", "prob"]].head(2).itertuples(index=False):
            L.append(show_s1(s2, f"candidate instead (p={pr:.2f})"))
    else:
        L.append("    candidates: none")
    return L


single = [(r, s) for r, s in wrong if not truth[s]]
out += section("3a. Wrong merges (we matched the record to a business that is not its owner)", wrong, r_wrong)
out += section("3b. Singletons given a match (each costs the full 1.0 for that business)", single, r_wrong)
out += section("3c. Judge misses (the owner was a candidate but not chosen)", jmiss, r_jmiss)
out += section("3d. Shortlist misses (the owner never reached the judge)", smiss, r_smiss)
open(a.out, "w").write("\n".join(out))
S.to_csv(a.out.replace(".md", "_per_s1.csv"), index=False)
print(f"wrote {a.out} and {a.out.replace('.md', '_per_s1.csv')}")
print("\n".join(out[:out.index("## 2. What drives the errors: share of each tag (sampled up to 3,000 per column)\n") + 3 + len(keys)]))
