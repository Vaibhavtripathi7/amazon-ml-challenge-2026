import numpy as np
import pandas as pd

from amlc.blocking import candidates


def frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "country", "name", "addr", "nums"])


S1 = frame([
    ["S1-a", "US", "grace chapel", "harbor drive dundalk md", "2100"],
    ["S1-b", "US", "silver lagoon", "walnut street olney il", "506"],
    ["S1-c", "India", "grace chapel", "mg road pune", "12"],
])


def ids(c, recs):
    return c.assign(s1_id=S1.entity_id.to_numpy()[c.s1_idx], rec_id=recs.entity_id.to_numpy()[c.rec_idx])


def test_typo_record_finds_its_business_by_name_pieces():
    recs = frame([["S2-x", "US", "grace chape1", "harbor drive dundalk md", "2100"]])
    c = ids(candidates(S1, recs, k=1), recs)
    assert set(c.s1_id) == {"S1-a"}


def test_name_change_found_by_address_pass():
    recs = frame([["S3-y", "US", "zzz center", "harbor drive dundalk md", "2100"]])
    c = ids(candidates(S1, recs, k=1), recs)
    assert "S1-a" in set(c[c["pass"] == "addr"].s1_id)


def test_never_crosses_countries():
    recs = frame([["S2-z", "India", "grace chapel", "harbor drive dundalk md", "2100"]])
    c = ids(candidates(S1, recs, k=5), recs)
    assert set(c.s1_id) == {"S1-c"}


def test_keep_mask_drops_other_businesses_but_keeps_ranks():
    recs = frame([["S2-x", "US", "grace chapel", "walnut street olney il", "506"]])
    full = ids(candidates(S1, recs, k=2), recs)
    kept = ids(candidates(S1, recs, k=2, keep_s1=np.array([True, False, False])), recs)
    assert set(kept.s1_id) == {"S1-a"}
    merged = kept.merge(full, on=["s1_id", "rec_id", "pass"])
    assert (merged.rank_x == merged.rank_y).all()
