"""Download every train+test image, resized. Safe to re-run: finished files are skipped.

uv run python scripts/download_images.py --data <dataset dir> [--out data/images --workers 32 --max-side 512 --limit N]
Writes data/images/<sample_id>.jpg and data/images_manifest.csv (status per id; re-run to retry failures).
"""
import argparse
import time

import pandas as pd

from amlc.data import ID, load
from amlc.images import download_all

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default="data/images")
p.add_argument("--link-col", default="image_link")
p.add_argument("--workers", type=int, default=32)
p.add_argument("--max-side", type=int, default=512)
p.add_argument("--limit", type=int, default=None, help="only the first N rows, for a speed test")
args = p.parse_args()

train, test = load(args.data)
df = pd.concat([train[[ID, args.link_col]], test[[ID, args.link_col]]]).dropna()
if args.limit:
    df = df.head(args.limit)
t = time.time()
res = download_all(dict(zip(df[ID], df[args.link_col])), args.out, workers=args.workers, max_side=args.max_side)
res.to_csv("data/images_manifest.csv", index=False)
dt = time.time() - t
done = (res.status == "ok").sum()
print(res.status.value_counts().to_string())
print(f"{done} downloaded in {dt:.0f}s ({done / max(dt, 1e-9):.1f}/s). Re-run the same command to retry failures.")
