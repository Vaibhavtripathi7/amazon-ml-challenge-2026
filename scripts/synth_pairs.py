"""Synthetic labelled pairs for a country without labels (France), from real S1 records only. See docs/france-synth.md.

Positive: a corrupted copy of S1 record A (operators and rates measured on 73,531 real US/India true pairs), paired with A.
Negative (exact label): the same copy paired with a DIFFERENT real S1 of the same country sharing A's door, street or a
name word (S1 is deduplicated, so different S1 = different businesses) — the sibling-decoy pattern.
Output has the columns of ce_rerank.py train.parquet (fold=train), so `ce_rerank.py train --in` can use it directly.

python scripts/synth_pairs.py --data <dataset dir> --split test --country France --n-s1 40000 --out synth_fr.parquet
"""
import argparse
import csv
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from amlc.norm import FR_DEPT, IN_STATES, parse_address, parse_name  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ce_rerank import texts  # noqa: E402

# country vocabulary (measured: words over-represented in records vs S1; suffix sets from the data)
FILLER = {"France": ["Participations", "Holding", "Groupe", "(France)", "Services"],
          "India": ["Center", "Services", "Service", "Partners", "Enterprises"],
          "US": ["Center", "Services", "Service", "Partners", "Group"]}
SUFFIX = {"France": ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", "SNC"],
          "India": ["Private Limited", "Pvt Ltd", "Pvt. Ltd.", "Limited", "Ltd", "LLP"],
          "US": ["LLC", "Inc", "Inc.", "Corp", "Corporation", "Ltd", "L.L.C."]}
STREET_ABBR = {"France": {"Rue": ["R.", "R", "RUE"], "Avenue": ["Av.", "AV", "Ave"], "Boulevard": ["Bd", "BD", "Blvd"],
                          "Allée": ["All.", "ALL", "Alle"], "Impasse": ["Imp.", "IMP"], "Chemin": ["Ch.", "Chem."],
                          "Place": ["Pl.", "PL"], "Route": ["Rte", "RTE"], "Quai": ["Q."], "Cours": ["Crs"]},
               "India": {"Road": ["Rd", "Rd."], "Street": ["St", "St."], "Nagar": ["Ngr"], "Floor": ["Flr"]},
               "US": {"Street": ["St", "St."], "Avenue": ["Ave"], "Road": ["Rd"], "Drive": ["Dr"], "Lane": ["Ln"]}}
ACCENT = str.maketrans({"a": "à", "e": "é", "i": "î", "o": "ô", "u": "ù", "c": "ç"})
GARBLE = [("ph", "f"), ("f", "ph"), ("c", "k"), ("k", "c"), ("s", "z"), ("ee", "i"), ("oo", "u"), ("w", "v"),
          ("v", "w"), ("ie", "y"), ("y", "ie"), ("ou", "u"), ("tion", "shun"), ("qu", "k"), ("x", "ks")]


def garble(w, rng):
    """Phonetic respelling (measured: 42% of replaced words in true copies are garbles such as knsltimg, phainans)."""
    w2 = w.lower()
    for _ in range(3):
        a, b = GARBLE[rng.integers(len(GARBLE))]
        if a in w2:
            w2 = w2.replace(a, b, 1)
    if w2 == w.lower() and len(w2) > 3:                              # vowel drop / swap
        i = rng.integers(1, len(w2) - 1)
        w2 = w2[:i] + w2[i + 1:]
    return w2.capitalize() if w[:1].isupper() else w2


def typo(w, rng):
    if len(w) < 3:
        return w
    i = rng.integers(0, len(w) - 1)
    op = rng.integers(3)
    return (w[:i] + w[i + 1] + w[i] + w[i + 2:]) if op == 0 else (w[:i] + w[i + 1:]) if op == 1 else \
        (w[:i] + "abcdefghijklmnopqrstuvwxyz"[rng.integers(26)] + w[i + 1:])


def corrupt_name(name, country, rng):
    """Rates from docs/france-synth.md (US/India averages; France uses its own vocabulary)."""
    sufs = SUFFIX[country]
    words = name.split()
    suf_idx = [i for i, w in enumerate(words) if w.strip(".,").upper() in {s.upper() for s in sufs}
               or w.lower().strip(".") in {"pvt", "ltd", "limited", "private", "llc", "inc", "corp"}]
    core = [i for i in range(len(words)) if i not in suf_idx and re.search(r"[A-Za-zÀ-ÿ]", words[i])]
    r = rng.random
    if suf_idx and r() < 0.24:                                       # suffix dropped
        words = [w for i, w in enumerate(words) if i not in suf_idx]; core = list(range(len(words)))
    elif suf_idx and r() < 0.12:                                     # suffix changed
        words[suf_idx[-1]] = sufs[rng.integers(len(sufs))]
    elif not suf_idx and r() < 0.06:                                 # suffix added
        words.append(sufs[rng.integers(len(sufs))])
    core = [i for i in core if i < len(words)]
    if core and r() < 0.17:                                          # core word replaced (filler 53% / garble 42%)
        i = core[rng.integers(len(core))]
        words[i] = FILLER[country][rng.integers(len(FILLER[country]))] if r() < 0.56 else garble(words[i], rng)
    elif core and r() < 0.08:                                        # typo
        i = core[rng.integers(len(core))]; words[i] = typo(words[i], rng)
    if r() < 0.10:                                                   # filler word added
        words.insert(rng.integers(len(words) + 1), FILLER[country][rng.integers(len(FILLER[country]))])
    if len(core) > 1 and r() < 0.05:                                 # core word dropped
        del words[core[rng.integers(len(core))]]
    if len(words) > 1 and r() < 0.03:                                # word order
        rng.shuffle(words)
    out = " ".join(words)
    if r() < 0.06:
        out = out.translate(ACCENT) if r() < 0.3 else "".join(c.translate(ACCENT) if r() < 0.15 else c for c in out)
    if r() < 0.06:
        out = re.sub(r"[^a-z0-9]", "", out.lower()) + (".fr" if country == "France" else ".com")
    if r() < 0.02:
        out = f"{['<<', '***', '--', '...'][rng.integers(4)]} {out}"
    if r() < 0.02:
        out = f"{out} (formerly {FILLER[country][rng.integers(len(FILLER[country]))]})"
    u = r()
    return out.upper() if u < 0.2 else out.lower() if u < 0.35 else out


def corrupt_address(addr, country, region_to_dept, rng):
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    r = rng.random
    if r() < 0.045:
        return ""                                                    # empty address
    if country == "France":
        parts = [rng.choice(region_to_dept.get(p.lower().replace("-", " "), [p])) if r() < 0.6 and
                 p.lower().replace("-", " ") in region_to_dept else p for p in parts]
        parts = [re.sub(r"\b(\d+) (bis|ter)\b", lambda m: m.group(1) + ("B" if m.group(2) == "bis" else "T"), p, flags=re.I)
                 if r() < 0.2 else p for p in parts]
    elif country == "India":
        code = {v: k for k, v in IN_STATES.items()}
        parts = [code.get(p.lower(), p).upper() if r() < 0.3 and p.lower() in code else p for p in parts]
    ab = STREET_ABBR[country]
    for k, opts in ab.items():                                       # street words abbreviated (Rue -> R.)
        parts = [re.sub(rf"\b{k}\b", lambda m, o=opts: o[rng.integers(len(o))], p, flags=re.I) if r() < 0.5 else p
                 for p in parts]
    if len(parts) > 2 and r() < 0.3:                                 # fewer parts
        del parts[rng.integers(1, len(parts))]
    if r() < 0.07:                                                   # first house-number digit changed
        parts = [re.sub(r"\d", lambda m: str(rng.integers(10)), p, count=1) if re.search(r"\d", p) else p for p in parts]
    if r() < 0.06 and parts and re.match(r"\d", parts[0]):
        parts[0] = ("N° " if country == "France" else "No. ") + parts[0]
    if len(parts) > 1 and r() < 0.03:
        rng.shuffle(parts)
    out = ", ".join(parts)
    return out.upper() if r() < (0.4 if country == "France" else 0.3) else out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--country", required=True)
    p.add_argument("--n-s1", type=int, default=40000)
    p.add_argument("--neg-per-pos", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-dev", action="store_true", help="skip dev S1 (crc32 % 10 == 0) — for the India gate")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    s1 = pd.read_csv(f"{a.data}/{a.split}/{a.split}_source1.tsv", sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                     keep_default_na=False)
    s1 = s1[s1.country == a.country].reset_index(drop=True)
    # sibling index over ALL S1 of the country (negatives): door, street+city, rare name word
    pa = [parse_address(x, a.country) for x in s1.business_address]
    pn = [parse_name(x, a.country) for x in s1.business_name]
    keys = {}
    for i, (x, n) in enumerate(zip(pa, pn)):
        ks = []
        if x["house"] and x["street"]:
            ks.append("door|" + x["house"] + "|" + x["street"])
        if x["street"] and x["city"]:
            ks.append("street|" + x["street"] + "|" + x["city"])
        ks += ["word|" + w for w in n["core"].split() if len(w) > 3]
        for k in ks:
            keys.setdefault(k, []).append(i)
    region_to_dept = {}
    for d, r in FR_DEPT.items():
        region_to_dept.setdefault(r, []).append(d.title().replace(" ", "-"))
    pool = np.arange(len(s1))
    if a.exclude_dev:   # never build synthetic pairs from dev S1 (they are the held-out businesses gate 1 scores on)
        pool = pool[[zlib.crc32(x.encode()) % 10 != 0 for x in s1.entity_id]]
    pick = rng.choice(pool, min(a.n_s1, len(pool)), replace=False)
    rows = []
    for i in pick:
        A = s1.iloc[i]
        copy_name = corrupt_name(A.business_name, a.country, rng)
        copy_addr = corrupt_address(A.business_address, a.country, region_to_dept, rng)
        rid = f"SYN-{a.country[:2]}-{i}"
        rows.append((A.entity_id, rid, 1, copy_name, copy_addr))
        # hard negatives: siblings sharing door > street > name word (small groups first = most confusable)
        x, n = pa[i], pn[i]
        cands = []
        for k in (["door|" + x["house"] + "|" + x["street"]] if x["house"] and x["street"] else []) + \
                 (["street|" + x["street"] + "|" + x["city"]] if x["street"] and x["city"] else []) + \
                 ["word|" + w for w in n["core"].split() if len(w) > 3]:
            g = keys.get(k, [])
            if 1 < len(g) <= 200:
                cands += [j for j in g if j != i]
            if len(cands) >= 20:
                break
        for j in (rng.choice(cands, min(a.neg_per_pos, len(set(cands))), replace=False) if cands else []):
            rows.append((s1.entity_id.iloc[j], rid, 0, copy_name, copy_addr))
    R = pd.DataFrame(rows, columns=["s1_id", "rec_id", "is_true", "rec_name", "rec_addr"])
    S = s1.set_index("entity_id")
    left = S.loc[R.s1_id]
    R["text_s1"] = texts(pd.DataFrame({"business_name": left.business_name.to_numpy(),
                                       "business_address": left.business_address.to_numpy(), "country": a.country}))
    R["text_rec"] = texts(pd.DataFrame({"business_name": R.rec_name, "business_address": R.rec_addr, "country": a.country}))
    R = R.assign(is_true=R.is_true.astype(bool), prob=0.5, full=True, is_dev_s1=False, fold="train", country=a.country)
    R = R[["s1_id", "rec_id", "prob", "full", "is_true", "is_dev_s1", "fold", "text_s1", "text_rec", "country"]]
    R.to_parquet(a.out, index=False)
    print(f"{a.country}: {len(R):,} synthetic pairs from {len(pick):,} S1 (positives {int(R.is_true.sum()):,}, "
          f"hard negatives {int((~R.is_true).sum()):,})", flush=True)


if __name__ == "__main__":
    main()
