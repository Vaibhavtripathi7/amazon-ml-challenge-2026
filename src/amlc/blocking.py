"""Step 2: candidate generation. For each S2/S3 record, the top-k most similar S1 records (same country)
under several TF-IDF views. Vectorisers are fit on the split's own S1 text only (no labels involved).

Output is integer positions (s1_idx into the s1 frame, rec_idx into the recs frame), not ID strings:
tens of millions of pairs as strings would not fit in memory.
"""
import resource
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# max_df: skip tokens present in more than that share of S1 rows. They barely separate businesses but
# make the search explode (every record would be compared with most of S1). Applied only on large inputs.
PASSES = {
    "name_word": dict(field="name", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
    "name_char": dict(field="name", analyzer="char_wb", ngram_range=(3, 3), min_df=1, max_df=0.02),
    "addr": dict(field="addr_full", analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.05),
}
MAX_DF_MIN_ROWS = 10_000


def _addr_full(df):
    nums = df.nums.str.split().map(lambda xs: " ".join("n" + x for x in xs))  # 'n185' so numbers are their own tokens
    return (df.addr + " " + nums).str.strip()


def _topk(A, B, k, n_threads):
    """Rows of A against rows of B (L2-normalised). Returns (a_row, b_col, score, rank) with rank 0 = best."""
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=k, sort=True, n_threads=n_threads).tocsr()
    per_row = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int32), per_row)
    rank = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], per_row)).astype(np.int8)
    return rows, C.indices.astype(np.int32), C.data.astype(np.float32), rank


def candidates(s1, recs, k=10, passes=PASSES, n_threads=4, keep_s1=None, verbose=False):
    """s1, recs: cleaned frames (entity_id, country, name, addr, nums).
    keep_s1: optional boolean mask over s1 rows; pairs for other S1 rows are dropped right away
    (ranks are still computed against ALL of s1, so namesakes compete realistically).
    Returns DataFrame [s1_idx, rec_idx, pass, score, rank]."""
    t0 = time.time()
    s1_addr, rec_addr = _addr_full(s1), _addr_full(recs)
    out = []
    for country in recs.country.unique():
        S_pos = np.flatnonzero((s1.country == country).to_numpy())
        R_pos = np.flatnonzero((recs.country == country).to_numpy())
        if len(S_pos) == 0:
            continue
        for pname, cfg in passes.items():
            cfg = dict(cfg)
            field = cfg.pop("field")
            s_text = (s1_addr if field == "addr_full" else s1[field]).iloc[S_pos]
            r_text = (rec_addr if field == "addr_full" else recs[field]).iloc[R_pos]
            if len(s_text) < MAX_DF_MIN_ROWS:
                cfg["max_df"] = 1.0
            vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **cfg).fit(s_text)
            ri, si, sc, rk = _topk(vec.transform(r_text), vec.transform(s_text), k, n_threads)
            s_glob, r_glob = S_pos[si].astype(np.int32), R_pos[ri].astype(np.int32)
            if keep_s1 is not None:
                m = keep_s1[s_glob]
                s_glob, r_glob, sc, rk = s_glob[m], r_glob[m], sc[m], rk[m]
            out.append(pd.DataFrame({"s1_idx": s_glob, "rec_idx": r_glob, "pass": pname, "score": sc, "rank": rk}))
            if verbose:
                mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                print(f"  {country:8s} {pname:10s} S1 {len(S_pos):>9,} recs {len(R_pos):>9,} "
                      f"kept pairs {len(s_glob):>10,}  [{time.time() - t0:.0f}s, peak {mem:.1f} GB]", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["pass"] = df["pass"].astype(pd.CategoricalDtype(list(passes)))
    return df
