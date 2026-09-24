"""Lesson 1 experiment: which prediction does SMAPE reward? Run: uv run python experiments/01_smape_demo.py"""
import numpy as np
from amlc.metrics import smape
rng = np.random.default_rng(0)
# price-like: right-skewed, most items cheap, a few very expensive
y = rng.lognormal(mean=3.0, sigma=1.0, size=50_000)
print(f"prices: median={np.median(y):.1f}  mean={y.mean():.1f}  max={y.max():.0f}\n")
print("A) Predict ONE constant for every row:")
for name, c in [("arithmetic mean (what MSE on raw y learns)", y.mean()),
                ("geometric mean  (what MSE on log y learns)", np.exp(np.log(y).mean())),
                ("median          (what L1 learns)", np.median(y))]:
    print(f"   {name:45s} c={c:6.1f}  SMAPE={smape(y, np.full_like(y, c)):5.1f}")
cs = np.linspace(5, 60, 400); best = cs[np.argmin([smape(y, np.full_like(y, c)) for c in cs])]
print(f"   {'best constant under SMAPE':45s} c={best:6.1f}  SMAPE={smape(y, np.full_like(y, best)):5.1f}")
print("\nB) A decent model with noisy log-predictions, then a global scale s:")
pred = np.exp(np.log(y) + rng.normal(0, 0.5, size=y.size))
for s in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
    print(f"   s={s:.2f}  SMAPE={smape(y, s*pred):.2f}")
