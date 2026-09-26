"""Learn Indian-script -> English word dictionaries (names and addresses) from TRAIN pairs of NON-dev S1.

The generator writes English business words in Indian scripts. For every matched India pair where the record
contains native-script words, each transliterated word (anyascii) is paired with the S1's English words; a word
maps to the English word with the highest Dice co-occurrence (min count and Dice thresholds). Dev S1
(crc32(id) % 10 == 0) are excluded so dev evaluation stays honest. Provided data only.

nice -n 19 python scripts/build_xlit_dict.py --work work --out src/amlc/resources/in_xlit.json
"""
import argparse
import collections
import json
import re
import zlib

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from anyascii import anyascii

_W = re.compile(r"[^a-z0-9]+")
_NATIVE = re.compile(r"[\u0900-\u0DFF]")   # Indian-script blocks only (accented Latin is NOT native)


def native_tokens(s):
    """Transliterated tokens of the words that contain non-ASCII (native-script) characters."""
    out = []
    for w in re.split(r"[\s,]+", s):
        if _NATIVE.search(w):
            out += [t for t in _W.sub(" ", anyascii(w).lower()).split() if t]
    return out


def latin_tokens(s):
    return [t for t in _W.sub(" ", anyascii(s).lower()).split() if t and not t.isdigit()]


def learn(pairs, min_n, min_dice):
    co, ct, ce = collections.Counter(), collections.Counter(), collections.Counter()
    for T, E in pairs:
        T, E = set(T), set(E)
        ct.update(T)
        ce.update(E)
        co.update((t, e) for t in T for e in E)
    best = {}
    for (t, e), n in co.items():
        d = 2 * n / (ct[t] + ce[e])
        if n >= min_n and d >= min_dice and d > best.get(t, ("", 0))[1]:
            best[t] = (e, d)
    return {t: e for t, (e, _) in best.items() if t != e}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="work")
    p.add_argument("--out", default="src/amlc/resources/in_xlit.json")
    p.add_argument("--min-n", type=int, default=3)
    p.add_argument("--min-dice", type=float, default=0.5)
    a = p.parse_args()
    W = f"{a.work}/train"
    cols = ["entity_id", "country", "raw_name", "raw_address"]
    t = pq.read_table(f"{W}/source1.parquet", columns=cols)
    s1 = t.filter(pc.equal(t["country"], "India")).to_pandas().set_index("entity_id")
    s1 = s1[[zlib.crc32(i.encode()) % 10 != 0 for i in s1.index]]          # non-dev S1 only
    gt = pd.read_parquet(f"{W}/truth.parquet")
    gt = gt[gt.source1_entity_id.isin(s1.index) & (gt.matched_entity_ids != "")]
    own = gt.assign(r=gt.matched_entity_ids.str.split(",")).explode("r").set_index("r").source1_entity_id
    name_pairs, addr_pairs = [], []
    for s in (2, 3):
        r = pq.read_table(f"{W}/source{s}.parquet", columns=cols)
        r = r.filter(pc.equal(r["country"], "India")).to_pandas()
        r = r[r.entity_id.isin(own.index)]
        for rid, nm, ad in zip(r.entity_id, r.raw_name, r.raw_address):
            o = s1.loc[own[rid]]
            if _NATIVE.search(nm):
                name_pairs.append((native_tokens(nm), latin_tokens(o.raw_name)))
            if _NATIVE.search(ad):
                addr_pairs.append((native_tokens(ad), latin_tokens(o.raw_address)))
    table = {"name": learn(name_pairs, a.min_n, a.min_dice), "addr": learn(addr_pairs, a.min_n, a.min_dice)}
    with open(a.out, "w") as f:
        json.dump(table, f, ensure_ascii=True, sort_keys=True)
    print(f"name pairs {len(name_pairs):,} -> {len(table['name']):,} words; "
          f"address pairs {len(addr_pairs):,} -> {len(table['addr']):,} words -> {a.out}")
