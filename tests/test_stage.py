import numpy as np
import pandas as pd

from amlc.clean import clean_address
from amlc.norm import parse_address, parse_name
from amlc.stage import RANK_FEATURES, Shortlister, rank_features, record_set, top_n


def frame(rows):
    out = []
    for eid, country, name, addr in rows:
        n, a = parse_name(name), parse_address(addr, country)
        out.append({"entity_id": eid, "country": country, "nums": clean_address(addr)[1],
                    **{"n_" + k: v for k, v in n.items()}, **{"a_" + k: v for k, v in a.items()}})
    df = pd.DataFrame(out)
    df["mix"] = df.n_core + " " + df.a_tokens
    return df


S1 = frame([
    ["S1-a", "US", "Premier Food LLC", "220 Netaji Road, Austin, TX"],
    ["S1-b", "US", "Premier Food LLC", "12 Oak Street, Dallas, TX"],
    ["S1-c", "US", "Premier Food Inc", "99 Pine Road, Houston, TX"],
])


def test_untrimmed_shortlist_keeps_every_candidate_and_ranker_features_separate_namesakes():
    recs = frame([["S2-x", "US", "PREMIER FOOD", "220 NETAJI RD, AUSTIN, TX"]])
    f = Shortlister(S1, threads=1)(recs)
    assert set(f.s1_idx) == {0, 1, 2}                         # nothing trimmed
    X = rank_features(S1, recs, f, threads=1)
    assert list(X.columns) == RANK_FEATURES
    true = X[f.s1_idx.to_numpy() == 0].iloc[0]
    other = X[f.s1_idx.to_numpy() == 1].iloc[0]
    assert true.house_eq == 1 and other.house_eq == 0 and true.ad_tset > other.ad_tset
    assert true.ad_tset_gap == 0 and other.ad_tset_gap < 0


def test_top_n_per_record():
    rec = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.5, 0.3, 0.2])
    keep, rank = top_n(rec, score, 2)
    assert keep.tolist() == [False, True, True, True, True]
    assert rank.tolist() == [2, 0, 1, 0, 1]


def test_record_sets_are_disjoint():
    ids = [f"S2-{i}" for i in range(3000)]
    df = pd.DataFrame({"entity_id": ids})
    owner = pd.Series(["S1-dev"] * 10, index=ids[:10])
    sets = {n: record_set(n, owner, {"S1-dev"})(df) for n in ("rank", "judge", "dev")}
    assert not (sets["rank"] & sets["judge"]).any() and not (sets["rank"] & sets["dev"]).any()
    assert not (sets["judge"] & sets["dev"]).any()
    assert sets["dev"][:10].all() and not sets["rank"][:10].any()   # dev-owned records only in dev
