"""Step 1: clean every source file once and cache as parquet.

uv run python scripts/prepare.py --data <dataset dir> --out work [--split train test] [--limit N]
Writes work/<split>/source{1,2,3}.parquet with columns:
  entity_id, country, raw_name, raw_address, name, addr, nums
(name/addr are cleaned, space-joined words; nums are address numbers). Train also gets work/train/truth.parquet.
"""
import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from amlc.clean import clean_address, clean_name


def _clean_chunk(df):
    df = df.rename(columns={"business_name": "raw_name", "business_address": "raw_address"})
    df["name"] = [clean_name(s) for s in df.raw_name]
    addr = [clean_address(s) for s in df.raw_address]
    df["addr"] = [a for a, _ in addr]
    df["nums"] = [n for _, n in addr]
    return df


def read(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, **kw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="work")
    p.add_argument("--split", nargs="+", default=["train", "test"])
    p.add_argument("--limit", type=int, default=None, help="first N rows per file (smoke test)")
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    t0 = time.time()
    with Pool(args.workers) as pool:
        for split in args.split:
            out = Path(args.out) / split
            out.mkdir(parents=True, exist_ok=True)
            for s in (1, 2, 3):
                chunks = read(Path(args.data) / split / f"{split}_source{s}.tsv", chunksize=200_000, nrows=args.limit)
                df = pd.concat(pool.imap(_clean_chunk, chunks), ignore_index=True)
                df.to_parquet(out / f"source{s}.parquet", index=False)
                print(f"{split} source{s}: {len(df):,} rows  [{time.time() - t0:.0f}s]", flush=True)
                del df
            if split == "train":
                gt = read(Path(args.data) / "train" / "train_ground_truth.tsv")
                gt.to_parquet(out / "truth.parquet", index=False)
