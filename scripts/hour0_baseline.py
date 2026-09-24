"""Hour-0 path: folds -> median submission -> TF-IDF + Ridge baseline (OOF + test preds + submission).

uv run python scripts/hour0_baseline.py --data <dataset dir> [--text catalog_content --target price]
Writes: folds.csv, preds/<model>_{oof,test}.parquet, submissions/<model>.csv
"""
import argparse
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import Ridge

from amlc.data import ID, load, make_folds
from amlc.metrics import smape
from amlc.submit import write_submission

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--text", default="catalog_content")
p.add_argument("--target", default="price")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--mode", default="stratified", choices=["random", "stratified", "group"])
args = p.parse_args()

for d in ["preds", "submissions"]:
    Path(d).mkdir(exist_ok=True)
train, test = load(args.data)
sample_out = next(Path(args.data).glob("sample_test_out.csv"), None)
columns = list(pd.read_csv(sample_out, nrows=1).columns) if sample_out else [ID, args.target]
y = train[args.target].to_numpy(dtype=float)
ylog = np.log1p(y)

# 1. shared fold file: every model uses exactly these folds
folds = make_folds(train, args.folds, args.mode, target=args.target)
folds.to_csv("folds.csv", index=False)
fold = folds.fold.to_numpy()


def save(name, oof, test_pred):
    pd.DataFrame({ID: train[ID], "pred": oof}).to_parquet(f"preds/{name}_oof.parquet")
    pd.DataFrame({ID: test[ID], "pred": test_pred}).to_parquet(f"preds/{name}_test.parquet")
    sub = pd.DataFrame({ID: test[ID], args.target: test_pred})
    write_submission(sub, test[ID], f"submissions/{name}.csv", columns, args.target)
    print(f"{name:12s} CV SMAPE {smape(y, oof):6.2f}  -> submissions/{name}.csv")


# 2. median: proves the format end to end
oof = np.empty(len(train))
for k in range(args.folds):
    oof[fold == k] = np.median(y[fold != k])
save("median", oof, np.full(len(test), np.median(y)))

# 3. TF-IDF words + hashed character pieces -> Ridge on log1p(target)
# Low-memory: hashing keeps no vocabulary, float32 halves storage, and the text matrix is built
# once and sliced per fold (IDF uses no prices, so fitting it on all train text leaks nothing).
t = time.time()
text_tr, text_te = train[args.text].fillna(""), test[args.text].fillna("")
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True, dtype=np.float32)
char_hash = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2**19,
                              alternate_sign=False, norm=None, dtype=np.float32)
char_idf = TfidfTransformer(sublinear_tf=True)
head = lambda s: s.str.slice(0, 600)  # char pieces from the start only (item name + first details): caps memory
Xall = hstack([word.fit_transform(text_tr), char_idf.fit_transform(char_hash.transform(head(text_tr)))]).tocsr()
Xte = hstack([word.transform(text_te), char_idf.transform(char_hash.transform(head(text_te)))]).tocsr()
print(f"text matrix {Xall.shape}, {Xall.data.nbytes / 1e9:.2f} GB, built in {time.time() - t:.0f}s")
oof = np.empty(len(train))
for k in range(args.folds):
    tr, va = fold != k, fold == k
    oof[va] = np.expm1(Ridge(alpha=1.0).fit(Xall[tr], ylog[tr]).predict(Xall[va]))
test_pred = np.expm1(Ridge(alpha=1.0).fit(Xall, ylog).predict(Xte))

lo, hi = np.quantile(y, [0.001, 0.999])
save("tfidf_ridge", np.clip(oof, lo, hi), np.clip(test_pred, lo, hi))
print(f"tfidf_ridge took {time.time() - t:.0f}s")
print(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
