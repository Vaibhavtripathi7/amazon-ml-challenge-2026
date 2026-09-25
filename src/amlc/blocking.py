"""Step 2: candidate generation. For each S2/S3 record, the top-k most similar S1 records (same country)
under several TF-IDF views. Vectorisers are fit on the split's own S1 text only (no labels involved).

Blocker.fit(s1) builds the S1 side once per country and pass; Blocker.query(recs) can then be called on
record chunks. Output is integer positions (s1_idx into s1, rec_idx into recs), never ID strings.
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


def _text(df, field):
    return _addr_full(df) if field == "addr_full" else df[field]


def _topk(A, BT, k, n_threads):
    """Rows of A against columns of BT (= B transposed). Returns (a_row, b_col, score, rank), rank 0 = best."""
    C = sp_matmul_topn(A, BT, top_n=k, sort=True, n_threads=n_threads).tocsr()
    per_row = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int32), per_row)
    rank = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], per_row)).astype(np.int8)
    return rows, C.indices.astype(np.int32), C.data.astype(np.float32), rank


class Blocker:
    def __init__(self, passes=PASSES, k=10, n_threads=4):
        self.passes, self.k, self.n_threads = passes, k, n_threads

    def fit(self, s1):
        self.index = {}
        for country in s1.country.unique():
            S_pos = np.flatnonzero((s1.country == country).to_numpy())
            for pname, cfg in self.passes.items():
                cfg = dict(cfg)
                text = _text(s1.iloc[S_pos], cfg.pop("field"))
                if len(text) < MAX_DF_MIN_ROWS:
                    cfg["max_df"] = 1.0
                vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **cfg).fit(text)
                self.index[(country, pname)] = (vec, vec.transform(text).T.tocsr(), S_pos)
        return self

    def query(self, recs, keep_s1=None, verbose=False):
        """Returns DataFrame [s1_idx, rec_idx, pass, score, rank]. keep_s1: optional bool mask over s1 rows."""
        t0, out = time.time(), []
        for country in recs.country.unique():
            R_pos = np.flatnonzero((recs.country == country).to_numpy())
            for pname, cfg in self.passes.items():
                if (country, pname) not in self.index:
                    continue
                vec, ST, S_pos = self.index[(country, pname)]
                A = vec.transform(_text(recs.iloc[R_pos], cfg["field"]))
                ri, si, sc, rk = _topk(A, ST, self.k, self.n_threads)
                s_glob, r_glob = S_pos[si].astype(np.int32), R_pos[ri].astype(np.int32)
                if keep_s1 is not None:
                    m = keep_s1[s_glob]
                    s_glob, r_glob, sc, rk = s_glob[m], r_glob[m], sc[m], rk[m]
                out.append(pd.DataFrame({"s1_idx": s_glob, "rec_idx": r_glob, "pass": pname, "score": sc, "rank": rk}))
                if verbose:
                    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                    print(f"  {country:8s} {pname:10s} S1 {len(S_pos):>9,} recs {len(R_pos):>9,} "
                          f"pairs {len(s_glob):>10,}  [{time.time() - t0:.0f}s, peak {mem:.1f} GB]", flush=True)
        cols = {"s1_idx": np.int32, "rec_idx": np.int32, "score": np.float32, "rank": np.int8}
        df = pd.concat(out, ignore_index=True) if out else pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()} | {"pass": []})
        df["pass"] = df["pass"].astype(pd.CategoricalDtype(list(self.passes)))
        return df


def candidates(s1, recs, k=10, passes=PASSES, n_threads=4, keep_s1=None, verbose=False):
    """One-shot fit + query (kept for eval_blocking and tests)."""
    return Blocker(passes, k, n_threads).fit(s1).query(recs, keep_s1=keep_s1, verbose=verbose)
