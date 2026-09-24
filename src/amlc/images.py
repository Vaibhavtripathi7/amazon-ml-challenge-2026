"""Parallel, resumable, resizing image downloader. Files are saved as <out_dir>/<sample_id>.jpg."""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def http_fetch(url, timeout=20):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _one(sid, url, out_dir, fetch, max_side, retries, backoff):
    path = out_dir / f"{sid}.jpg"
    if path.exists():
        return sid, "skipped", ""
    err = ""
    for attempt in range(retries):
        try:
            im = Image.open(io.BytesIO(fetch(url))).convert("RGB")
            im.thumbnail((max_side, max_side))
            tmp = path.with_suffix(".part")
            im.save(tmp, format="JPEG", quality=90)
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written .jpg that looks done
            return sid, "ok", ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(backoff * (2 ** attempt))
    return sid, "failed", err[:200]


def download_all(links, out_dir, fetch=http_fetch, max_side=512, workers=32, retries=3, backoff=1.0):
    """links: {sample_id: url}. Returns a DataFrame [sample_id, status, error]; never raises per image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_one, sid, url, out_dir, fetch, max_side, retries, backoff) for sid, url in links.items()]
        rows = [f.result() for f in tqdm(futs, total=len(futs), mininterval=5, disable=len(futs) < 100)]
    return pd.DataFrame(rows, columns=["sample_id", "status", "error"])
