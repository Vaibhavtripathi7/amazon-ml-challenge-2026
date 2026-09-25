"""Step 2: candidate generation. For each S2/S3 record, the top-k most similar S1 records (same country)
under several TF-IDF views. Vectorisers are fit on the split's own S1 text only (no labels involved)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

PASSES = {
    "name_word": dict(field="name", analyzer="word", ngram_range=(1, 1), min_df=1),
    "name_char": dict(field="name", analyzer="char_wb", ngram_range=(3, 3), min_df=1),
    "addr": dict(field="addr_full", analyzer="word", ngram_range=(1, 1), min_df=1),
}


def _addr_full(df):
    nums = df.nums.str.split().map(lambda xs: " ".join("n" + x for x in xs))  # 'n185' so numbers are their own tokens
    return (df.addr + " " + nums).str.strip()


def topk(A, B, k, threshold=0.0, n_threads=4):
    """Rows of A against rows of B (both L2-normalised sparse). Returns (a_idx, b_idx, score) arrays."""
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=threshold, n_threads=n_threads).tocoo()
    return C.row, C.col, C.data


def candidates(s1, recs, k=10, passes=PASSES, n_threads=4):
    """s1, recs: cleaned frames (entity_id, country, name, addr, nums). Returns DataFrame
    [s1_id, rec_id, pass, score, rank] with rank = position among that record's top-k (0 = best)."""
    s1 = s1.assign(addr_full=_addr_full(s1))
    recs = recs.assign(addr_full=_addr_full(recs))
    out = []
    for country in recs.country.unique():
        S, R = s1[s1.country == country], recs[recs.country == country]
        if len(S) == 0 or len(R) == 0:
            continue
        for pname, cfg in passes.items():
            cfg = dict(cfg)
            field = cfg.pop("field")
            vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **cfg).fit(S[field])
            ri, si, sc = topk(vec.transform(R[field]), vec.transform(S[field]), k, n_threads=n_threads)
            df = pd.DataFrame({"s1_id": S.entity_id.to_numpy()[si], "rec_id": R.entity_id.to_numpy()[ri],
                               "pass": pname, "score": sc})
            df["rank"] = df.groupby("rec_id").score.rank(ascending=False, method="first").astype(np.int16) - 1
            out.append(df)
    return pd.concat(out, ignore_index=True)
