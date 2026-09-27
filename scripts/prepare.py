"""Step 1: clean every source file once and cache as parquet.

uv run python scripts/prepare.py --data <dataset dir> --out work [--split train test] [--limit N]
Writes work/<split>/source{1,2,3}.parquet with columns:
  entity_id, country, raw_name, raw_address, name, addr, nums
(name/addr are cleaned, space-joined words; nums are address numbers). Train also gets work/train/truth.parquet.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from amlc.clean import clean_address, clean_name
from amlc.norm import parse_address, parse_name


def _clean_chunk(df):
    df = df.rename(columns={"business_name": "raw_name", "business_address": "raw_address"})
    df["name"] = [clean_name(s) for s in df.raw_name]
    addr = [clean_address(s, c) for s, c in zip(df.raw_address, df.country)]
    df["addr"] = [a for a, _ in addr]
    df["nums"] = [n for _, n in addr]
    # stage-0 slots (amlc.norm): n_* name fields, a_* address fields
    names = [parse_name(s, c) for s, c in zip(df.raw_name, df.country)]
    for k in ("core", "key", "fold", "flat", "suffix", "alias", "is_domain"):
        df["n_" + k] = [n[k] for n in names]
    addrs = [parse_address(s, c) for s, c in zip(df.raw_address, df.country)]
    for k in ("house", "house_alts", "unit", "street", "street_type", "city", "state", "locality", "tokens", "has_addr"):
        df["a_" + k] = [x[k] for x in addrs]
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
                final = out / f"source{s}.parquet"
                if final.exists():                           # resumable: finished files are kept
                    print(f"{split} source{s}: exists, skipped", flush=True)
                    continue
                tmp = out / f"source{s}.parquet.tmp"
                chunks = read(Path(args.data) / split / f"{split}_source{s}.tsv", chunksize=200_000, nrows=args.limit)
                writer, n = None, 0
                for df in pool.imap(_clean_chunk, chunks):  # write chunk by chunk: memory stays flat
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    writer = writer or pq.ParquetWriter(tmp, table.schema)
                    writer.write_table(table)
                    n += len(df)
                writer.close()
                os.replace(tmp, final)                       # atomic: a half-written file never looks finished
                print(f"{split} source{s}: {n:,} rows  [{time.time() - t0:.0f}s]", flush=True)
            if split == "train":
                gt = read(Path(args.data) / "train" / "train_ground_truth.tsv")
                gt.to_parquet(out / "truth.parquet", index=False)
