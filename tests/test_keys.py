import pandas as pd

from amlc.keys import key_candidates, row_keys, token_df
from amlc.norm import parse_address, parse_name


def frame(rows):
    out = []
    for eid, country, name, addr in rows:
        n, a = parse_name(name), parse_address(addr, country)
        out.append({"entity_id": eid, "country": country, **{"n_" + k: v for k, v in n.items()},
                    **{"a_" + k: v for k, v in a.items()}})
    return pd.DataFrame(out)


S1 = frame([
    ["S1-a", "US", "Halvorsen Quill Meridian Plutonian LLC", "128 Sablewood Way, Collierville, TN"],
    ["S1-b", "US", "Dermatology Group", "40 Oak Street, Austin, TX"],
    ["S1-c", "US", "Dermatology Group", "900 Pine Road, Dallas, TX"],
])


def cands(recs, cap=30):
    d = token_df(S1)
    c = key_candidates(row_keys(S1, d, "s1"), row_keys(recs, d, "rec"), cap=cap)
    return c.assign(s1_id=S1.entity_id.to_numpy()[c.s1_idx])


def test_pseudoword_name_found_by_address_keys():
    c = cands(frame([["S2-x", "US", "Zorvexilam", "128 SABLEWOOD WAY, COLLIERVILLE, TN"]]))
    assert set(c.s1_id) == {"S1-a"} and "house_street" in set(c["pass"])


def test_namesake_resolved_by_house_street_not_name():
    c = cands(frame([["S2-y", "US", "DERMATOLOGY GROUP", "40 OAK ST, AUSTIN, TX"]]))
    hs = c[c["pass"] == "house_street"]
    assert set(hs.s1_id) == {"S1-b"}


def test_domain_form_found_by_flat_key():
    c = cands(frame([["S3-z", "US", "halvorsenquillmeridianplutonian.com", ""]]))
    assert "S1-a" in set(c[c["pass"] == "flat"].s1_id)


def test_cap_skips_unspecific_keys():
    c = cands(frame([["S2-y", "US", "Dermatology Group", ""]]), cap=1)
    assert "name_key" not in set(c["pass"])
