"""Stage 1 key blocking: exact joins on normalised keys (all within country). Uses stage-0 columns (n_*, a_*)."""
import collections

import numpy as np
import pandas as pd

KEY_TYPES = ["name_key", "name_fold", "flat", "house_street", "house_state", "rare_addr2", "rare_addr1", "rare_name1"]


def token_df(s1):
    """Per-country document frequency of S1 address and name tokens."""
    df = collections.defaultdict(collections.Counter)
    for c, a, n in zip(s1.country, s1.a_tokens, s1.n_core):
        df[(c, "a")].update(set(a.split()))
        df[(c, "n")].update(set(n.split()))
    return df


def _rarest(tokens, cnt, k):
    toks = [t for t in set(tokens.split()) if cnt.get(t, 0) > 0]
    return sorted(toks, key=lambda t: (cnt[t], t))[:k]


def row_keys(df, dfreq, kind):
    """Yield (key_type, row_position, key_string) for every key of every row."""
    out = {k: ([], []) for k in KEY_TYPES}

    def add(k, i, v):
        out[k][0].append(i)
        out[k][1].append(v)

    for i, (c, key, fold, flat, house, alts, street, state, toks, core) in enumerate(zip(
            df.country, df.n_key, df.n_fold, df.n_flat, df.a_house, df.a_house_alts, df.a_street, df.a_state,
            df.a_tokens, df.n_core)):
        if key:
            add("name_key", i, f"{c}|{key}")
        if fold:
            add("name_fold", i, f"{c}|{fold}")
        if len(flat) >= 6:
            add("flat", i, f"{c}|{flat}")
        if house and street:
            add("house_street", i, f"{c}|{house}|{street}")
        if state:
            for h in alts.split():
                add("house_state", i, f"{c}|{state}|{h}")
        ra = _rarest(toks, dfreq[(c, "a")], 2)
        if len(ra) == 2:
            add("rare_addr2", i, f"{c}|{ra[0]}|{ra[1]}")
        if ra:
            add("rare_addr1", i, f"{c}|{ra[0]}")
        rn = _rarest(core, dfreq[(c, "n")], 1)
        if rn:
            add("rare_name1", i, f"{c}|{rn[0]}")
    return {k: pd.DataFrame({f"{kind}_idx": np.array(v[0], np.int32), "key": v[1]}) for k, v in out.items()}


def filter_s1_keys(s1_keys, cap=30):
    """Drop S1 key values shared by more than cap S1 rows (too unspecific). Do once, reuse per record chunk."""
    out = {}
    for k, S in s1_keys.items():
        size = S.key.map(S.key.value_counts())
        out[k] = S[size.to_numpy() <= cap]
    return out


def join_keys(s1_keys_filtered, rec_keys):
    parts = []
    for k in KEY_TYPES:
        m = rec_keys[k].merge(s1_keys_filtered[k], on="key")[["s1_idx", "rec_idx"]]
        m["pass"] = k
        parts.append(m.drop_duplicates())
    return pd.concat(parts, ignore_index=True)


def key_candidates(s1_keys, rec_keys, cap=30):
    """Join record keys to S1 keys per key type; S1 key values shared by more than cap S1 rows are skipped."""
    return join_keys(filter_s1_keys(s1_keys, cap), rec_keys)
