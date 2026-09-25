import numpy as np
import pandas as pd

from amlc.clean import clean_address
from amlc.judge_features import FEATURES, REL, TF, Context, pair_features
from amlc.keys import KEY_TYPES
from amlc.norm import parse_address, parse_name


def frame(rows):
    out = []
    for eid, country, name, addr in rows:
        n, a = parse_name(name), parse_address(addr, country)
        out.append({"entity_id": eid, "country": country, "nums": clean_address(addr)[1],
                    **{"n_" + k: v for k, v in n.items()}, **{"a_" + k: v for k, v in a.items()}})
    return pd.DataFrame(out)


S1 = frame([
    ["S1-a", "US", "Behavioral Health Partners Inc", "185 Locust View Way, Troy, OH"],
    ["S1-b", "US", "Behavioral Health Partners Inc", "1331 Pleasant View Drive, Des Moines, IA"],
    ["S1-c", "US", "Morganica Mendez Summit Plutonian LLC", "128 Vermilion Way, Collierville, TN"],
])


def feats(recs, s_idx, r_idx):
    flags = pd.DataFrame({f"k_{k}": np.zeros(len(s_idx), np.int8) for k in KEY_TYPES}
                         | {f"{t}_score": np.zeros(len(s_idx)) for t in TF}
                         | {f"{t}_rank": np.full(len(s_idx), 10) for t in TF})
    return pair_features(Context(S1), recs, np.array(s_idx), np.array(r_idx), flags)


def test_true_match_beats_namesake_on_address_features():
    recs = frame([["S2-x", "US", "BEHAVIORAL HEALTH-PARTNERS", "#185 LOCUST VIEW WAY, TROY, OH"]])
    X = feats(recs, [0, 1], [0, 0])
    assert list(X.columns) == FEATURES
    assert X.nm_tset[0] == X.nm_tset[1] == 100          # names identical: name alone can't decide
    assert X.house_rel[0] == REL["equal"] and X.house_rel[1] == REL["substitution"]
    assert X.ad_idf_cov_max[0] > X.ad_idf_cov_max[1] and X.city_eq[0] == 1 and X.state_state[1] == 3
    assert X.ad_cov_gap[0] == 0 and X.ad_cov_gap[1] < 0  # competition: the namesake trails the leader


def test_pseudoword_name_still_has_address_evidence_and_digit_drop():
    recs = frame([["S3-y", "US", "Gildlumxylo", "28 Vermilion Way, Collierville, Tennessee"]])
    X = feats(recs, [2], [0])
    assert X.nm_tset[0] < 50 and X.house_rel[0] == REL["digit_drop"] and X.street_eq[0] == 1


def test_suffix_states_and_domain_and_missing_address():
    recs = frame([["S2-z", "US", "MORGANICAMENDEZSUMMITPLUTONIAN.COM", ""],
                  ["S2-w", "US", "Behavioral Health Partners LLC", ""]])
    X = feats(recs, [2, 0], [0, 1])
    assert X.rec_is_domain[0] == 1 and X.flat_ratio[0] == 100 and X.rec_has_addr[0] == 0
    assert X.suffix_state[1] == 3                       # Inc vs LLC: conflict
    assert X.house_rel[0] == REL["missing"]
