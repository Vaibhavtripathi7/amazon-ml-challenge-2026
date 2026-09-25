import pandas as pd

from amlc.blocking import candidates


def frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "country", "name", "addr", "nums"])


S1 = frame([
    ["S1-a", "US", "christ chapel", "cameron drive dundalk md", "2100"],
    ["S1-b", "US", "blue hypnosis", "boone street olney il", "506"],
    ["S1-c", "India", "christ chapel", "mg road pune", "12"],
])


def test_typo_record_finds_its_business_by_name_pieces():
    recs = frame([["S2-x", "US", "christ chape1", "cameron drive dundalk md", "2100"]])
    c = candidates(S1, recs, k=1)
    assert set(c.s1_id) == {"S1-a"}


def test_name_change_found_by_address_pass():
    recs = frame([["S3-y", "US", "zzz center", "cameron drive dundalk md", "2100"]])
    c = candidates(S1, recs, k=1)
    assert "S1-a" in set(c[c["pass"] == "addr"].s1_id)


def test_never_crosses_countries():
    recs = frame([["S2-z", "India", "christ chapel", "cameron drive dundalk md", "2100"]])
    c = candidates(S1, recs, k=5)
    assert set(c.s1_id) == {"S1-c"}
