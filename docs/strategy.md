# Strategy & Novelty

_As of 2026-09-24 · Vaibhav_

> Revised in [decision-review.md](decision-review.md) — see rows 1–8 there.

The edge over a copy-paste AI baseline comes from four places: target transform, feature extraction, validation strategy, and how models get combined at the end.

## Metric & target transform

SMAPE (the likely price metric) punishes percentage error, not absolute error — a ₹10 miss on a ₹50 item is far worse than a ₹10 miss on a ₹10,000 item. Training on log(target) instead of the raw value aligns the model's loss with how the judge scores it.

- Predict `log1p(target)`, invert with `expm1` before submission
- Applies to any right-skewed numeric target, not just price
- Confirm the eval metric first, then pick the transform to match — this is a one-line change with outsized leaderboard impact

## Feature engineering: structured text + real image signal

`catalog_content` is several fields mashed into one string — pulling them apart beats embedding the whole blob as-is.

- Regex-extract pack quantity, unit of measure, dimensions, brand token, and material keywords before any embedding step
- Normalize units (500ml vs 0.5L vs 500 ml) so numeric features aren't scattered across token variants
- OCR product images for embedded label text (weight, size, ingredients) — a second, independent read on the same facts
- Flag duplicate or near-duplicate images across samples — useful both as a feature and as a leakage check

## Validation strategy: leakage-aware CV

A random K-fold split leaks when near-identical products (same brand or family, different variant) land on both sides — local CV looks great, the private leaderboard doesn't.

- Group CV folds by brand or product-cluster, not by row
- Check train/test overlap on brand, category, and image hash before trusting any local score
- Get this right before ensembling — a bad CV makes every downstream decision unreliable

## Ensembling & post-processing

Different models make different mistakes — blending smooths them out, but only once CV is trustworthy (see above).

- Blend a text-only model, an image-augmented model, and a pure structured-feature model, weighted by validation score
- Add category-level median target as an input feature, computed from training folds only, to avoid leakage
- Add ratio features for any two related numeric fields (e.g. price per unit) — trees don't discover ratios on their own
- Clip final predictions to physically plausible bounds before submitting — a few outlier predictions can wreck a percentage-based metric
