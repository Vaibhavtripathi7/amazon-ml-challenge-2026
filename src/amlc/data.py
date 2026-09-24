"""Loading the competition CSVs and building the shared fold file."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

ID = "sample_id"


def load(data_dir):
    """Read train.csv and test.csv from data_dir, sorted by sample_id."""
    d = Path(data_dir)
    train = pd.read_csv(d / "train.csv").sort_values(ID, ignore_index=True)
    test = pd.read_csv(d / "test.csv").sort_values(ID, ignore_index=True)
    return train, test


def make_folds(df, n_folds=5, mode="stratified", target=None, group_col=None, seed=42):
    """Assign each row a fold number 0..n_folds-1. Returns [sample_id, fold] in df's row order.

    random     - shuffle and deal rows into folds
    stratified - like random, but every fold gets the same mix of cheap and expensive
                 (bins of log1p(target) into 20 quantiles)
    group      - rows sharing a group_col value always land in the same fold
    """
    folds = np.full(len(df), -1)
    if mode == "random":
        splits = KFold(n_folds, shuffle=True, random_state=seed).split(df)
    elif mode == "stratified":
        bins = pd.qcut(np.log1p(df[target]), q=20, labels=False, duplicates="drop")
        splits = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(df, bins)
    elif mode == "group":
        splits = GroupKFold(n_folds, shuffle=True, random_state=seed).split(df, groups=df[group_col])
    else:
        raise ValueError(f"unknown mode {mode!r}")
    for k, (_, val_idx) in enumerate(splits):
        folds[val_idx] = k
    return pd.DataFrame({ID: df[ID].to_numpy(), "fold": folds})
