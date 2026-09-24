"""Similar-product features: what did the most similar training products cost?

Works on any L2-normalised embedding (TF-IDF+SVD now, text/image model embeddings later).
Train rows only look at neighbours in OTHER folds, so their own price (or a same-fold twin's) never leaks in.
"""
import numpy as np
import pandas as pd


def _topk(query, ref, k, chunk=2048):
    idx = np.empty((len(query), k), dtype=np.int64)
    sim = np.empty((len(query), k), dtype=np.float32)
    for s in range(0, len(query), chunk):
        S = query[s:s + chunk] @ ref.T                      # cosine similarity (rows are unit length)
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        sim[s:s + chunk] = np.take_along_axis(ps, order, axis=1)
    return idx, sim


def _summarise(nb_y, nb_sim, ks):
    out = {"knn1_y": nb_y[:, 0], "knn1_sim": nb_sim[:, 0]}
    for k in ks:
        y, s = nb_y[:, :k], np.clip(nb_sim[:, :k], 1e-6, None)
        out[f"knn{k}_mean"] = y.mean(1)
        out[f"knn{k}_wmean"] = (y * s).sum(1) / s.sum(1)
        out[f"knn{k}_std"] = y.std(1)
        out[f"knn{k}_simmean"] = nb_sim[:, :k].mean(1)
    return pd.DataFrame(out)


def knn_features(train_emb, y, folds, test_emb, k=20, ks=None):
    """y: target on the scale you want averaged (use log1p(price)). Returns (train_df, test_df)."""
    ks = ks or sorted({min(5, k), k})
    train_emb = np.asarray(train_emb, dtype=np.float32)
    test_emb = np.asarray(test_emb, dtype=np.float32)
    nb_y = np.empty((len(train_emb), k))
    nb_sim = np.empty((len(train_emb), k), dtype=np.float32)
    for f in np.unique(folds):
        q, r = np.where(folds == f)[0], np.where(folds != f)[0]
        idx, sim = _topk(train_emb[q], train_emb[r], k)
        nb_y[q], nb_sim[q] = y[r][idx], sim
    idx, sim = _topk(test_emb, train_emb, k)
    return _summarise(nb_y, nb_sim, ks), _summarise(y[idx], sim, ks)
