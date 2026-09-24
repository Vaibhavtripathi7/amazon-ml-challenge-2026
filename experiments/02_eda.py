"""Lesson 2 experiment: first look at the 2025 data. Run: uv run python experiments/02_eda.py"""
import re

import numpy as np
import pandas as pd

from amlc.metrics import smape

D = "Prev-year-data-files/68e8d1d70b66d_student_resource/student_resource/dataset/"
tr = pd.read_csv(D + "train.csv")
te = pd.read_csv(D + "test.csv")


def field(s, label):
    return s.str.extract(rf"^{label}: (.*)$", flags=re.M)[0].str.strip()


print("== 1. Shape ==")
print(f"train {tr.shape}, test {te.shape}, missing cells: {tr.isna().sum().sum()}")

print("\n== 2. Price ==")
print(tr.price.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(2).to_string())
bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 3000]
for k, v in pd.cut(tr.price, bins).value_counts().sort_index().items():
    print(f"  ${str(k):14s} {v:6d} " + "#" * int(v / 800))

print("\n== 3. Constant-guess baselines (SMAPE on train) ==")
for name, c in [("mean", tr.price.mean()), ("median", tr.price.median())]:
    print(f"  always ${c:.2f} ({name}) -> {smape(tr.price, np.full(len(tr), c)):.2f}")

print("\n== 4. What's inside catalog_content ==")
for label in ["Item Name", "Bullet Point 1", "Product Description", "Value", "Unit"]:
    print(f"  {label:20s} present in {field(tr.catalog_content, label).notna().mean():6.1%} of rows")
val = pd.to_numeric(field(tr.catalog_content, "Value"), errors="coerce")
unit = field(tr.catalog_content, "Unit").str.lower()
print(f"  distinct Unit spellings: {unit.nunique()}; top: {unit.value_counts().head(6).to_dict()}")

print("\n== 5. Does quantity alone explain price? ==")
lp = np.log1p(tr.price)
for u in ["ounce", "count", "fl oz"]:
    m = (unit == u) & (val > 0)
    r = np.corrcoef(lp[m], np.log(val[m]))[0, 1]
    print(f"  unit={u:6s} rows={m.sum():6d} corr(log price, log value)={r:.2f}")

print("\n== 6. How similar are train and test? ==")
brand = lambda s: field(s, "Item Name").str.split().str[:2].str.join(" ").str.lower()
print(f"  sample_id ranges interleave: train {tr.sample_id.min()}..{tr.sample_id.max()}, test {te.sample_id.min()}..{te.sample_id.max()}")
print(f"  test rows whose brand (first 2 words) appears in train: {brand(te.catalog_content).isin(set(brand(tr.catalog_content))).mean():.1%}")
print(f"  test rows with the exact same image link as a train row: {te.image_link.isin(set(tr.image_link)).mean():.1%}")
print(f"  test rows with identical text to a train row: {te.catalog_content.isin(set(tr.catalog_content)).mean():.1%}")
