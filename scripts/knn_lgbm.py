"""Differentiator v1: similar-product prices (kNN) + parsed Value/Unit + Ridge guess -> LightGBM on log1p(price).

Run hour0_baseline.py first (it writes folds.csv and preds/tfidf_ridge_*).
uv run python scripts/knn_lgbm.py --data <dataset dir> [--objective regression|huber|regression_l1]
"""
import argparse
import resource
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from amlc.data import ID, load
from amlc.knn import knn_features
from amlc.metrics import smape
from amlc.submit import write_submission
from amlc.text_features import tfidf_matrices
from amlc.text_parse import parse_catalog

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--text", default="catalog_content")
p.add_argument("--target", default="price")
p.add_argument("--objective", default="regression")
p.add_argument("--k", type=int, default=20)
args = p.parse_args()

t0 = time.time()
train, test = load(args.data)
folds = pd.read_csv("folds.csv")
assert (folds[ID].to_numpy() == train[ID].to_numpy()).all(), "folds.csv does not match train order"
fold = folds.fold.to_numpy()
y = train[args.target].to_numpy(dtype=float)
ylog = np.log1p(y)

# 1. embeddings: TF-IDF squeezed to 256 numbers per product (cached; swap in model embeddings later)
cache = Path("features/cache")
cache.mkdir(parents=True, exist_ok=True)
if (cache / "svd256_train.npy").exists():
    emb_tr, emb_te = np.load(cache / "svd256_train.npy"), np.load(cache / "svd256_test.npy")
else:
    X_tr, X_te = tfidf_matrices(train[args.text], test[args.text])
    svd = TruncatedSVD(256, random_state=0)
    emb_tr = normalize(svd.fit_transform(X_tr)).astype(np.float32)
    emb_te = normalize(svd.transform(X_te)).astype(np.float32)
    np.save(cache / "svd256_train.npy", emb_tr)
    np.save(cache / "svd256_test.npy", emb_te)
    del X_tr, X_te
print(f"embeddings ready {emb_tr.shape} at {time.time() - t0:.0f}s")

# 2. similar-product prices (train rows only see other folds)
knn_tr, knn_te = knn_features(emb_tr, ylog, fold, emb_te, k=args.k)
print(f"kNN features at {time.time() - t0:.0f}s   "
      f"kNN{args.k} weighted mean alone: CV SMAPE {smape(y, np.expm1(knn_tr[f'knn{args.k}_wmean'])):.2f}")

# 3. parsed fields + stacked Ridge guess
par_tr, par_te = parse_catalog(train[args.text]), parse_catalog(test[args.text])
units = pd.CategoricalDtype(sorted(set(par_tr.unit) | set(par_te.unit)))
par_tr["unit"], par_te["unit"] = par_tr.unit.astype(units), par_te.unit.astype(units)
F_tr = pd.concat([knn_tr, par_tr.reset_index(drop=True)], axis=1)
F_te = pd.concat([knn_te, par_te.reset_index(drop=True)], axis=1)
if Path("preds/tfidf_ridge_oof.parquet").exists():
    F_tr["ridge"] = np.log1p(pd.read_parquet("preds/tfidf_ridge_oof.parquet").pred.to_numpy())
    F_te["ridge"] = np.log1p(pd.read_parquet("preds/tfidf_ridge_test.parquet").pred.to_numpy())

# 4. LightGBM, same folds
params = dict(objective=args.objective, learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
oof, test_pred, iters = np.empty(len(train)), np.zeros(len(test)), []
for k in np.unique(fold):
    tr, va = fold != k, fold == k
    m = lgb.train(params, lgb.Dataset(F_tr[tr], ylog[tr]), 3000, valid_sets=[lgb.Dataset(F_tr[va], ylog[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(F_tr[va], num_iteration=m.best_iteration)
    test_pred += m.predict(F_te, num_iteration=m.best_iteration) / len(np.unique(fold))
    iters.append(m.best_iteration)

lo, hi = np.quantile(y, [0.001, 0.999])
oof, test_pred = np.clip(np.expm1(oof), lo, hi), np.clip(np.expm1(test_pred), lo, hi)
name = f"knn_lgbm_{args.objective}"
pd.DataFrame({ID: train[ID], "pred": oof}).to_parquet(f"preds/{name}_oof.parquet")
pd.DataFrame({ID: test[ID], "pred": test_pred}).to_parquet(f"preds/{name}_test.parquet")
write_submission(pd.DataFrame({ID: test[ID], args.target: test_pred}), test[ID],
                 f"submissions/{name}.csv", [ID, args.target], args.target)
imp = pd.Series(m.feature_importance("gain"), F_tr.columns).sort_values(ascending=False)
print("top features (last fold, by gain):", (imp / imp.sum()).head(8).round(3).to_dict())
print(f"{name} CV SMAPE {smape(y, oof):.2f}  (iters {iters})  total {time.time() - t0:.0f}s  "
      f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
