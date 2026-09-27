"""Train vs test, side by side, label-free (facts only; the team reads and decides).

nice -n 19 python scripts/test_analysis.py --pred ~/Downloads/matching_results-v3.tsv
  -> docs/test-analysis.md   numbers: train vs test per country (+ train truth vs our submitted test predictions)
  -> review/test-explore.md  real test examples: our matches, look-alikes we left out, orphan-like records
Pass 1 parses every S1/S2/S3 row once (name key, house, city, state, text flags) and caches it in work/analysis/.
--nrows N reads only the first N rows of each file (smoke test).
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from amlc.norm import _ALIAS, parse_address, parse_name

p = argparse.ArgumentParser()
p.add_argument("--data", default="data-2026/student_resource/student_resource/dataset")
p.add_argument("--pred", required=True, help="our submitted test matching_results.tsv")
p.add_argument("--cache", default="work/analysis")
p.add_argument("--out", default="docs/test-analysis.md")
p.add_argument("--examples", default="review/test-explore.md")
p.add_argument("--nrows", type=int, default=None)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
COUNTRIES = ["US", "India", "France"]
SCRIPT = re.compile(r"[ऀ-෿]")
DOMAIN = re.compile(r"\.(?:com|net|org|in|co|fr|io|biz)\b|^[@#]", re.I)
JUNK = re.compile(r"^[^\wऀ-෿]")


def read(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, **kw)


def num_id(ids):
    """'S2-705547832' -> 2_000_000_000_000 + 705547832 (sources never collide)."""
    s = pd.Series(ids)
    return s.str[1].astype(np.int64) * 10**12 + s.str[3:].astype(np.int64)


def h(strings):
    x = pd.util.hash_array(np.asarray(strings, dtype=object)).astype(np.uint64)
    return x


def parse_rows(df):
    names = [parse_name(n) for n in df.business_name]
    addrs = [parse_address(x, c) for x, c in zip(df.business_address, df.country)]
    key = [n["key"] for n in names]
    house = [x["house"] for x in addrs]
    city = [x["city"] for x in addrs]
    c = df.country.to_numpy()
    out = pd.DataFrame({
        "id": num_id(df.entity_id).to_numpy(),
        "country": c,
        "k": np.where([bool(x) for x in key], h([f"{ci}|{x}" for ci, x in zip(c, key)]), 0).astype(np.uint64),
        "kh": np.where([bool(x) and bool(y) for x, y in zip(key, house)],
                       h([f"{ci}|{x}|{y}" for ci, x, y in zip(c, key, house)]), 0).astype(np.uint64),
        "city": np.where([bool(x) for x in city], h([f"{ci}|{x}" for ci, x in zip(c, city)]), 0).astype(np.uint64),
        "state": [x["state"] or "" for x in addrs],
        "has_addr": [x["has_addr"] == 1 for x in addrs],
        "has_house": [bool(x) for x in house],
        "script": df.business_name.str.contains(SCRIPT).to_numpy(),
        "domain": df.business_name.str.contains(DOMAIN).to_numpy(),
        "alias": df.business_name.str.lower().str.contains(_ALIAS).to_numpy(),
        "paren": df.business_name.str.contains("(", regex=False).to_numpy(),
        "junk": df.business_name.str.contains(JUNK).to_numpy(),
        "allcaps": (df.business_name == df.business_name.str.upper()).to_numpy()
                   & df.business_name.str.contains("[A-Za-z]").to_numpy(),
        "nlen": df.business_name.str.len().to_numpy(np.int32),
        "alen": df.business_address.str.len().to_numpy(np.int32),
    })
    out["state"] = out.state.astype("category")
    out["country"] = out.country.astype("category")
    return out


def load(split, src):
    f = Path(a.cache) / f"{split}_s{src}{'_n' + str(a.nrows) if a.nrows else ''}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    parts = []
    for chunk in read(f"{a.data}/{split}/{split}_source{src}.tsv", chunksize=250_000, nrows=a.nrows):
        parts.append(parse_rows(chunk))
        print(f"  {split} S{src}: {sum(map(len, parts)):,} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["src"] = np.int8(src)
    df["country"] = df.country.astype(str).astype("category")
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f.with_suffix(".tmp"))
    f.with_suffix(".tmp").replace(f)
    return df


def pairs(path):
    """matching_results / ground truth -> (s1 numeric id, record numeric id)."""
    t = read(path)
    t.columns = ["s1", "m"]
    t = t[t.m != ""]
    t = t.assign(m=t.m.str.split(",")).explode("m")
    return pd.DataFrame({"s1": num_id(t.s1).to_numpy(), "rec": num_id(t.m).to_numpy()})


def pct(x):
    return f"{100 * x:.1f}%"


# ------------------------------------------------------------------ pass 1: parse + link
W = {}
for split in ("train", "test"):
    s1 = load(split, 1)
    recs = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    recs["country"] = recs.country.astype(str).astype("category")
    own = pairs(f"{a.data}/train/train_ground_truth.tsv" if split == "train" else a.pred)
    owner = pd.Series(own.s1.to_numpy(), index=own.rec.to_numpy())
    recs["owner"] = owner.reindex(recs.id.to_numpy()).fillna(-1).astype(np.int64).to_numpy()
    s1_per_key = s1[s1.k != 0].groupby("k").size()
    recs["n_s1"] = s1_per_key.reindex(recs.k.to_numpy()).fillna(0).astype(np.int64).to_numpy()
    recs.loc[recs.k == 0, "n_s1"] = 0
    recs["strong"] = np.isin(recs.kh.to_numpy(), s1.kh[s1.kh != 0].unique()) & (recs.kh != 0).to_numpy()
    s1["n_s1"] = s1_per_key.reindex(s1.k.to_numpy()).fillna(0).astype(np.int64).to_numpy()
    recs_per_key = recs[recs.k != 0].groupby("k").size()
    s1["n_rec_same_key"] = recs_per_key.reindex(s1.k.to_numpy()).fillna(0).astype(np.int64).to_numpy()
    s1["group"] = own.groupby("s1").size().reindex(s1.id.to_numpy()).fillna(0).astype(np.int64).to_numpy()
    unique_s1 = s1[s1.n_s1 == 1].set_index("k").id
    recs["unique_s1"] = unique_s1.reindex(recs.k.to_numpy()).fillna(-1).astype(np.int64).to_numpy()
    s1_cities = s1[s1.city != 0].groupby("city").size()
    recs["city_in_s1"] = np.isin(recs.city.to_numpy(), s1_cities.index.to_numpy()) & (recs.city != 0).to_numpy()
    W[split] = (s1, recs, s1_cities)
    print(f"{split}: S1 {len(s1):,}, records {len(recs):,}", flush=True)

# ------------------------------------------------------------------ report
L = ["# Test vs train — side by side",
     "",
     "_Built by `scripts/test_analysis.py` from every row of both splits. Facts only; what they mean is the team's call._",
     "",
     "- **Train** columns use the ground truth. **Test** columns have no labels: where a row needs an owner, it uses "
     f"**our submitted predictions** (`{Path(a.pred).name}`, v2) and says so.",
     "- \"Name key\" = our cleaned, word-sorted business name (`parse_name`), so `Davis & Espinoza LLC` = "
     "`Espinoza Davis L.L.C.`. \"House\" = the parsed house number.",
     "- **Bold** = test differs from train by more than 20% (relative).",
     ""]


def table(title, rows, note=None):
    """rows: list of (label, fn(split, country) -> float, fmt)."""
    L.append(f"## {title}")
    L.append("")
    if note:
        L.append(note)
        L.append("")
    head = "| | " + " | ".join(f"{c} train | {c} test" for c in COUNTRIES) + " |"
    L.append(head)
    L.append("|" + " --- |" * (1 + 2 * len(COUNTRIES)))
    for label, fn, fmt in rows:
        cells = []
        for c in COUNTRIES:
            tr, te = fn("train", c), fn("test", c)
            cells.append(fmt(tr) if tr is not None else "—")
            s = fmt(te) if te is not None else "—"
            if tr is not None and te is not None and tr > 0 and abs(te - tr) / tr > 0.2:
                s = f"**{s}**"
            cells.append(s)
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")


def S(split, c):
    s1 = W[split][0]
    return s1[s1.country == c]


def R(split, c):
    r = W[split][1]
    return r[r.country == c]


def guard(fn):
    def g(split, c):
        s1, r = S(split, c), R(split, c)
        return None if len(s1) == 0 and len(r) == 0 else fn(s1, r)
    return g


n0 = lambda x: f"{x:,.0f}"
n2 = lambda x: f"{x:.2f}"
p1 = pct

table("1. Size", [
    ("S1 businesses", guard(lambda s, r: len(s)), n0),
    ("S2 records", guard(lambda s, r: (r.src == 2).sum()), n0),
    ("S3 records", guard(lambda s, r: (r.src == 3).sum()), n0),
    ("S2+S3 records per S1", guard(lambda s, r: len(r) / max(len(s), 1)), n2),
])

table("2. Group sizes: records attached to each S1", [
    ("mean records per S1", guard(lambda s, r: s.group.mean()), n2),
    ("S1 with 0 (empty list)", guard(lambda s, r: (s.group == 0).mean()), p1),
    ("S1 with 1", guard(lambda s, r: (s.group == 1).mean()), p1),
    ("S1 with 2–3", guard(lambda s, r: s.group.between(2, 3).mean()), p1),
    ("S1 with 4–5", guard(lambda s, r: s.group.between(4, 5).mean()), p1),
    ("S1 with 6+", guard(lambda s, r: (s.group >= 6).mean()), p1),
    ("records attached to no S1", guard(lambda s, r: (r.owner < 0).mean()), p1),
], note="Train = **truth**. Test = **our v2 predictions** (not truth). A gap here is either a real data difference "
        "or a v2 mistake — this table alone can't tell which.")

table("3. What the text looks like (S2+S3 records)", [
    ("no address", guard(lambda s, r: (~r.has_addr).mean()), p1),
    ("address has a house number", guard(lambda s, r: r.has_house.mean()), p1),
    ("name in an Indian script", guard(lambda s, r: r.script.mean()), p1),
    ("name looks like a domain/handle", guard(lambda s, r: r.domain.mean()), p1),
    ("name has formerly / dba / fka", guard(lambda s, r: r.alias.mean()), p1),
    ("name has brackets", guard(lambda s, r: r.paren.mean()), p1),
    ("name starts with junk (<<, ***, --)", guard(lambda s, r: r.junk.mean()), p1),
    ("name ALL CAPS", guard(lambda s, r: r.allcaps.mean()), p1),
    ("median name length", guard(lambda s, r: r.nlen.median()), n0),
    ("median address length", guard(lambda s, r: r.alen.median()), n0),
])

table("4. What the text looks like (S1)", [
    ("no address", guard(lambda s, r: (~s.has_addr).mean()), p1),
    ("address has a house number", guard(lambda s, r: s.has_house.mean()), p1),
    ("name starts with junk", guard(lambda s, r: s.junk.mean()), p1),
    ("name ALL CAPS", guard(lambda s, r: s.allcaps.mean()), p1),
    ("median name length", guard(lambda s, r: s.nlen.median()), n0),
    ("median address length", guard(lambda s, r: s.alen.median()), n0),
])

table("5. Namesakes: how many S1 share a name key", [
    ("S1 whose name key is shared with another S1", guard(lambda s, r: (s.n_s1 >= 2).mean()), p1),
    ("S1 whose name key is shared by 10+ S1", guard(lambda s, r: (s.n_s1 >= 10).mean()), p1),
    ("records whose name key is in NO S1", guard(lambda s, r: (r.n_s1 == 0).mean()), p1),
    ("records whose name key is in exactly 1 S1", guard(lambda s, r: (r.n_s1 == 1).mean()), p1),
    ("records whose name key is in 2–9 S1", guard(lambda s, r: r.n_s1.between(2, 9).mean()), p1),
    ("records whose name key is in 10+ S1", guard(lambda s, r: (r.n_s1 >= 10).mean()), p1),
    ("records with name key AND house equal to some S1", guard(lambda s, r: r.strong.mean()), p1),
])

table("6. Records around one S1 (S1 with a name key no other S1 has)", [
    ("mean records with the same name key", guard(lambda s, r: s[s.n_s1 == 1].n_rec_same_key.mean()), n2),
    ("… 0 records", guard(lambda s, r: (s[s.n_s1 == 1].n_rec_same_key == 0).mean()), p1),
    ("… 1–3 records", guard(lambda s, r: s[s.n_s1 == 1].n_rec_same_key.between(1, 3).mean()), p1),
    ("… 4–6 records", guard(lambda s, r: s[s.n_s1 == 1].n_rec_same_key.between(4, 6).mean()), p1),
    ("… 7+ records", guard(lambda s, r: (s[s.n_s1 == 1].n_rec_same_key >= 7).mean()), p1),
], note="If test kept every record but dropped some S1, a different pattern shows up here and in section 5 "
        "(\"name key in NO S1\").")


def owned_by_unique(s, r):
    x = r[r.n_s1 == 1]
    return (x.owner == x.unique_s1).mean() if len(x) else None


table("7. How reliable is a name-key hit? (train = truth, test = our v2)", [
    ("record with a 1-S1 name key is attached to THAT S1", guard(owned_by_unique), p1),
    ("record with a name key in NO S1 is attached to some S1", guard(lambda s, r: (r[r.n_s1 == 0].owner >= 0).mean()), p1),
    ("record with name key + house hit is attached to some S1", guard(lambda s, r: (r[r.strong].owner >= 0).mean()), p1),
    ("record with a name key in 10+ S1 is attached to some S1", guard(lambda s, r: (r[r.n_s1 >= 10].owner >= 0).mean()), p1),
], note="Train: how often the truth agrees with the simple name-key link. Test: how often **v2** attached such a record. "
        "Large train-vs-test gaps are where v2 behaves differently on test than truth does on train.")

table("8. Places", [
    ("record city appears among S1 cities", guard(lambda s, r: r.city_in_s1.mean()), p1),
    ("S1 per city (mean over cities)", lambda split, c: (lambda s: s[s.city != 0].groupby("city").size().mean()
                                                         if len(s) else None)(S(split, c)), n2),
    ("distinct S1 cities", lambda split, c: (lambda s: s.city[s.city != 0].nunique() if len(s) else None)(S(split, c)), n0),
])

L.append("## 9. Top states (share of S1)")
L.append("")
for c in COUNTRIES:
    tr, te = S("train", c).state.astype(str), S("test", c).state.astype(str)
    top = te[te != ""].value_counts(normalize=True).head(12)
    trs = tr[tr != ""].value_counts(normalize=True) if len(tr) else pd.Series(dtype=float)
    L.append(f"**{c}**: " + " · ".join(f"{k} {pct(v)} (train {pct(trs.get(k, 0)) if len(tr) else '—'})"
                                          for k, v in top.items()))
    L.append("")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(L) + "\n")
print(f"wrote {a.out}", flush=True)

# ------------------------------------------------------------------ pass 2: test examples
rng = np.random.default_rng(a.seed)
s1, recs, _ = W["test"]
picks = {}   # section -> list of (s1 ids, record ids)
want_s1, want_rec = set(), set()
by_owner = recs[recs.owner >= 0].groupby("owner").id.agg(list)
by_key = recs[recs.k != 0].groupby("k").id.agg(list)
s1_by_key = s1[s1.k != 0].groupby("k").id.agg(list)
rec_owner = pd.Series(recs.owner.to_numpy(), index=recs.id.to_numpy())


def take(ids, n):
    ids = np.asarray(ids)
    return ids if len(ids) <= n else rng.choice(ids, n, replace=False)


for c in COUNTRIES:
    sc, rc = s1[s1.country == c], recs[recs.country == c]
    sec = []
    for sid, k in zip(*[x.to_numpy() for x in (lambda d: (d.id, d.k))(sc.loc[take(sc.index, 10)])]):
        m = by_owner.get(sid, [])
        ms = set(m)
        look = [x for x in by_key.get(k, []) if x not in ms][:5] if k else []
        sec.append(("A", [sid], m, look))
    empty = sc[(sc.group == 0) & (sc.n_rec_same_key > 0)]
    for sid, k in zip(*[x.to_numpy() for x in (lambda d: (d.id, d.k))(empty.loc[take(empty.index, 8)])]):
        sec.append(("B", [sid], [], by_key.get(k, [])[:5]))
    strong = rc[(rc.owner < 0) & rc.strong & (rc.n_s1 == 1)]
    for rid, k in zip(*[x.to_numpy() for x in (lambda d: (d.id, d.k))(strong.loc[take(strong.index, 10)])]):
        s = s1_by_key.get(k, [])
        sec.append(("C", s, by_owner.get(s[0], []) if s else [], [rid]))
    orphan = rc[(rc.owner < 0) & (rc.n_s1 == 0)]
    for rid in take(orphan.id.to_numpy(), 8):
        sec.append(("D", [], [], [rid]))
    crowd = sc[sc.n_s1 >= 5].k.unique()
    for k in take(crowd, 4):
        ss = s1_by_key.get(k, [])[:6]
        sec.append(("E", ss, [x for s in ss for x in by_owner.get(s, [])], [x for x in by_key.get(k, [])
                                                                          if rec_owner.get(x, -1) < 0][:5]))
    picks[c] = sec
    for _, ss, m, look in sec:
        want_s1.update(ss)
        want_rec.update(m)
        want_rec.update(look)

raw = {}
for src in (1, 2, 3):
    for chunk in read(f"{a.data}/test/test_source{src}.tsv", chunksize=500_000, nrows=a.nrows):
        ids = num_id(chunk.entity_id).to_numpy()
        keep = np.isin(ids, np.fromiter(want_s1 if src == 1 else want_rec, np.int64))
        for i, row in zip(ids[keep], chunk[keep].itertuples(index=False)):
            raw[i] = row
print(f"collected {len(raw):,} example rows", flush=True)


def line(i, tag=""):
    r = raw.get(i)
    if r is None:
        return f"  {tag}(id {i} not found)"
    return f"  {tag}{r.entity_id} | {r.business_name} | {r.business_address}"


TITLES = {
    "A": "Random S1: what v2 attached (✓) and records with the same name key that v2 left out (?)",
    "B": "S1 that v2 left EMPTY although records with the same name key exist (?)",
    "C": "Records v2 left unattached although ONE S1 has the same name key AND house number",
    "D": "Records v2 left unattached whose name key matches NO S1 (orphan-like)",
    "E": "Crowded names (5+ S1 share the key): S1 with v2 attachments (✓), unattached records (?)",
}
X = ["# Test examples — raw text (no labels; ✓ = our v2 attached it, ? = v2 left it out)",
     "",
     f"_Built by `scripts/test_analysis.py` (seed {a.seed}). Pair with `docs/test-analysis.md` and the train sheet "
     "`review/data-explore.md`._",
     ""]
for c in COUNTRIES:
    X.append(f"## {c}")
    for sect in "ABCDE":
        rows = [x for x in picks[c] if x[0] == sect]
        if not rows:
            continue
        X.append(f"\n### {c} {sect}. {TITLES[sect]}\n")
        X.append("```")
        for _, ss, m, look in rows:
            for s in ss:
                X.append(line(s, "S1 "))
            for r in m:
                X.append(line(r, "✓  "))
            for r in look:
                X.append(line(r, "?  "))
            X.append("")
        X.append("```")
    X.append("")
Path(a.examples).parent.mkdir(parents=True, exist_ok=True)
Path(a.examples).write_text("\n".join(X) + "\n")
print(f"wrote {a.examples}", flush=True)
