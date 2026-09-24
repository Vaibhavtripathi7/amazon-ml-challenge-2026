# Lesson 1 — The metric (SMAPE)

**Point of the lesson:** the metric decides the target transform, the loss, and the post-processing. Get it wrong and a better model can score worse.

- Code: [`src/amlc/metrics.py`](../../src/amlc/metrics.py)
- Tests: [`tests/test_metrics.py`](../../tests/test_metrics.py) — run `uv run pytest`
- Experiment: [`experiments/01_smape_demo.py`](../../experiments/01_smape_demo.py) — run `uv run python experiments/01_smape_demo.py`

## 1. The formula

For each row, error = |prediction − true| ÷ average of |true| and |prediction|. SMAPE = mean of that × 100.

```
SMAPE = 100/n · Σ |F − A| / ((|A| + |F|) / 2)
```

`A` = actual price, `F` = forecast. Range: **0** (perfect) to **200** (worst).

## 2. Working through examples by hand

| True | Predicted | Row error | Notes |
| --- | --- | --- | --- |
| 100 | 100 | 0% | perfect |
| 100 | 110 | 10 / 105 = 9.5% | the hand-computed test |
| 100 | 150 | 50 / 125 = 40% | over by ₹50 |
| 100 | 50 | 50 / 75 = 66.7% | under by ₹50 — costs more |
| 100 | 200 | 100 / 150 = 66.7% | 2× too high |
| 100 | 50 | 50 / 75 = 66.7% | ½× too low — **same as 2×** |
| 50 | 0 | 50 / 25 = 200% | predicting 0 is the worst possible row |
| 0 | 0 | 0/0 → counted as 0% | edge case: must not become NaN |

Two things to notice:

1. **For the same miss in rupees**, under-predicting costs more than over-predicting (66.7% vs 40%).
2. **For the same miss as a ratio**, it is symmetric: 2× and ½× both cost 66.7%. So SMAPE really measures error in *ratio* terms, which means *log* terms.

That second point is the whole reason for training on `log1p(price)`: in log space, 2× and ½× are the same distance (±0.69) from the truth, which matches how SMAPE scores.

## 3. Walking through the code

```python
denom = (np.abs(y_true) + np.abs(y_pred)) / 2
diff = np.abs(y_pred - y_true)
ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
return float(100 * ratio.mean())
```

- `np.divide(..., where=denom != 0)` only divides where the denominator is nonzero. Everywhere else it keeps the `out` value, which is 0. That is how 0/0 counts as perfect.
- Without that guard, a single `0/0 = NaN` would make the mean NaN, and the whole score would be unusable.
- The shape check raises `ValueError` on mismatched lengths. A misaligned submission should fail loudly, not score silently.

## 4. Why the tests came first

Each test pins down one fact from the table above: perfect = 0, the hand-computed example, 0/0, the 200 cap, the asymmetry, and a length mismatch. They were written before the function and all six failed first. If a test never fails, you can't know it would catch a bug.

When the real metric is announced, write its tests from the rules text first. That is the step that catches a "divide by 2 or not" misreading in minutes instead of on the leaderboard.

## 5. The experiment: what does SMAPE reward?

Synthetic prices drawn from a lognormal distribution (skewed like real prices: median ₹20, mean ₹33, max ₹2,280).

**A. Predict one constant for every row**

| Constant | What learns it | SMAPE |
| --- | --- | --- |
| Arithmetic mean (33.2) | MSE on raw price | 76.7 |
| Geometric mean (20.1) | MSE on log price | 70.1 |
| Median (20.0) | L1 loss | 70.1 |
| Best constant found by search (20.0) | — | 70.1 |

**Lesson:** MSE on raw price chases the mean, which the few expensive items pull upward. That scores clearly worse. Log-MSE and L1 both land on the optimum here.

**B. A decent model (noisy log predictions), multiplied by a global scale `s`**

| s | 0.85 | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 |
| --- | --- | --- | --- | --- | --- | --- |
| SMAPE | 40.17 | 39.10 | 38.49 | **38.28** | 38.43 | 38.90 |

**Lesson:** when predictions are unbiased, the best scale is exactly 1.00. The scale factor doesn't exploit a trick in SMAPE. It **corrects model bias**: real models regress toward the average, predicting cheap items too high and expensive ones too low. Tuning `s` on out-of-fold (OOF) predictions is a cheap check for that bias. Keep it only if it improves CV.

**Correction this lesson caught:** the first version of `decision-review.md` said "SMAPE rewards a slight under-prediction." That is true of **MAPE**, not SMAPE. The doc has been fixed.

## 6. Decisions this lesson locks in

- Train on `log1p(price)`; invert with `expm1`.
- Prefer L1 or Huber loss in log space. They are robust to outliers, and on skewed data they match the median.
- Never train MSE on raw price for a SMAPE-type metric.
- Tune a global scale on OOF predictions; keep it only if CV improves.
- Never output 0 or negative prices: clip in log space.

## 7. Exercises (your turn)

1. **Predict:** a model always predicts 2× the true price. What is its SMAPE? Write the number down first, then add a test to `tests/test_metrics.py` to check it.
2. **Predict:** what about always predicting 10× the true price? What does SMAPE approach as the multiple grows? Why is the cap 200?
3. **Experiment:** in `experiments/01_smape_demo.py`, give the model a bias: `np.log(y) + 0.2 + noise`. Which `s` wins now? Does it match exp(−0.2) ≈ 0.82?
4. **Think:** which rows would dominate SMAPE on the real data, cheap items or expensive ones? Note it for lesson 2, where we check it on the 2025 prices.

## Your notes / ideas

_Add anything you noticed or want to try here._
