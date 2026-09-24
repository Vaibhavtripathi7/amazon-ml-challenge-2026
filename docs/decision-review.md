# Decision Review

_As of 2026-09-24 · Vaibhav_

The plan is sound; the gains come from four corrections: tune the loss and a post-hoc scale to the metric, match CV to how the test set was actually split, add nearest-neighbour target features, and make OCR conditional instead of default.

## Decision changes

Ordered by expected leaderboard impact. "Keep" means the original call stands but gets sharper.

| # | Original decision | Revised decision | Why |
| --- | --- | --- | --- |
| 1 | Train on `log1p(target)`, invert with `expm1` | Keep, but train with an L1 or Huber loss on the log target, then tune one global scale factor (around 0.90–1.05) on out-of-fold (OOF) predictions | SMAPE is symmetric in ratio terms (2× and ½× the true price both cost 66.7%), so log space is its natural scale; MSE on raw price chases the mean and scores worse. The scale factor is a cheap check for model bias, not a built-in SMAPE preference: on unbiased synthetic predictions the best scale was exactly 1.00. |
| 2 | Group CV folds by brand or product cluster | First run adversarial validation plus a brand/image-hash overlap check between train and test, then pick the split that matches it | If test was drawn at random, group folds are too pessimistic and steer you away from memorisation features that do work on the leaderboard. Group folds only when test brands are unseen. |
| 3 | Frozen embeddings plus a GBM as the main model | Add nearest-neighbour target features (mean log target of the k closest train items in text and image embedding space, computed out of fold) | In catalogue pricing, near-duplicate listings are the strongest signal. kNN features capture them cheaply, and trees can't find them from raw embedding columns. |
| 4 | Ratio features "e.g. price per unit" | Ratios between inputs only (total quantity = pack count × unit size). Also try predicting log(price ÷ total quantity) and adding log quantity back | Price per unit is built from the target, so it leaks. Reframing the target so the model learns unit price is the legitimate version of the same idea. |
| 5 | OCR every product image | Conditional: OCR a 2k-image sample first and run it at full scale only if it recovers quantity or size for at least 10% of rows where text regex fails | OCR over about 75k images takes GPU hours. Catalogue text usually already holds those facts. |
| 6 | Blend three models, weighted by validation score | Every model writes OOF and test predictions using the one shared fold file. Blend in log space with non-negative least squares or hill climbing on OOF | Score-proportional weights ignore correlation between models. Stacking on shared folds is what makes the blend trustworthy. |
| 7 | LightGBM vs CatBoost is an open decision | Not a decision: LightGBM first for speed, CatBoost later as a blend member. Add one fine-tuned text encoder (regression head on the log target) as the high-ceiling model | Two GBMs on the same features add little diversity. A fine-tuned transformer is the model most likely to separate top 10 from top 50. |
| 8 | Clip to plausible bounds | Clip in log space to the train target range from the 0.1th to 99.9th percentile | This is concrete and reproducible, and it avoids hand-set bounds. |
| 9 | First submission in 2–3 hours | First submission within 90 minutes: the train median for every row, just to prove format and scoring. A TF-IDF plus Ridge baseline by hour 3 | A submission with the wrong format costs more than a weak model. It also gives you a calibration point for public LB versus CV. |
| 10 | Not stated | Submission budget: log every submission's CV and public LB score. Choose the final submission by CV, not by public LB | The public LB is usually scored on a slice of test. Chasing it is the most common way teams fall on the private board. |

## Metric playbook

Prepare every row of this table before the drop. Once the metric is announced, choose the matching row within 15 minutes. Code the metric exactly as the rules define it, including edge cases such as 0/0, and check it on a hand-built example.

| Metric | Target / loss | Post-processing | Validation note |
| --- | --- | --- | --- |
| SMAPE (2025 price pattern) | `log1p(y)` with L1 or Huber loss | `expm1`, a global scale tuned on OOF, clip in log space | Stratify folds on target deciles |
| MAPE | `log1p(y)` with L1, or sample weights of 1/y on raw y | Scale tuned slightly low, then clip | Low-value rows dominate the score, so check those rows' error separately |
| RMSE / MAE (raw) | Raw y (Huber for MAE) | Clip only | Log transform hurts here; skip it |
| RMSLE | `log1p(y)` with MSE | `expm1` | The log transform is exactly this metric |
| F1 / exact match (2024 extraction pattern) | Per-attribute extraction: regex and OCR candidates, then a ranker or a small LLM | Normalise units and format strings exactly as the answer key expects | Most of the lost points are format mismatches, so build the formatter first |
| Accuracy / macro-F1 (2023 classification pattern) | Cross-entropy; class weights for macro-F1 | Tune per-class thresholds on OOF | Stratify folds on class |

## Roles and timeline

Split work by role (features vs models), not by modality. Under a text-vs-image split, whoever gets images may own a weak signal for 48 hours, and nobody owns the feature table both models need. Everyone reads and writes the same artifacts: `folds.csv`, `features/*.parquet` keyed by `sample_id`, and `preds/{model}_{oof,test}.parquet`.

The original work split had one conflict: Krithi owns EDA, but hours 0–3 give EDA to teammate 2. The version below gives the modeler the baseline instead.

| Person | Owns | Hours 0–3 |
| --- | --- | --- |
| Vaibhav | Loader, fold file, image download (started in the background at minute 0), embedding cache, submission validator, blend, submission log | Loader, `folds.csv`, median submission, image download running |
| Teammate 2 | Models: GBM on features, fine-tuned text encoder, kNN features | TF-IDF plus Ridge baseline on the log target |
| Krithi | EDA, regex and unit-normaliser feature extraction, error analysis of OOF residuals, writeup | EDA: target distribution, train/test overlap, fields in `catalog_content`, missing images |

| Hours | Milestone | Exit check |
| --- | --- | --- |
| 0–3 | Median and TF-IDF submissions scored; metric row chosen | CV and public LB agree to within about 5% |
| 3–15 | Regex features, text and image embeddings cached, first GBM, kNN features | GBM beats TF-IDF on CV |
| 15–40 | Fine-tuned text encoder, OCR sample test, reformulated unit-price target | Each new model logged with OOF predictions |
| 40–55 | Worst-residual review; fix extraction bugs the residuals point to | Top-100 errors reviewed by category |
| 55–65 | Blend, tune the scale factor, clip | Blend CV beats the best single model |
| 65–72 | Freeze code; final submission chosen by CV; writeup done | No new models after hour 65 |

Plan sleep in shifts so a GPU job always runs overnight. Queue the long fine-tuning runs before each person's sleep block.

## Build before the drop

The biggest lever you still control is running the whole pipeline end to end on last year's price dataset (or a synthetic stand-in) before hour 0. Every item below should run on that data first.

- [ ] `loader.py` and `folds.csv` generator, with random, stratified and group modes behind one flag
- [ ] Parallel, resumable image downloader with retries and a failure manifest; measure images per minute to plan the real run
- [ ] Embedding cache script (one text model, one image model) writing float16 `.npy` files keyed by `sample_id`
- [ ] kNN OOF feature builder on top of that cache (for example FAISS or sklearn NearestNeighbors)
- [ ] Regex library for quantity, units and dimensions, with unit-normalisation tests (500ml = 0.5L = 500 ml)
- [ ] Metric implementations for every row of the playbook, unit-tested
- [ ] Blend script: reads `preds/*_oof.parquet`, fits weights plus the scale factor, writes the submission
- [ ] `submit.py` validator: row count, id set, dtype, no NaN or negative values, column order matching the sample file
- [ ] Fine-tuning script for a text encoder with a regression head, timed on one GPU
- [ ] Submission log sheet: timestamp, model, CV, public LB, notes

## Open questions

- [ ] Pretrained model rules: last year's (2025) rules reportedly capped model size at about 8B parameters and required permissive licences (MIT/Apache). This is from memory and unverified, so check this year's rules before picking the fine-tune model.
- [ ] Compute: which GPUs and how many hours are available across the 72 hours? This decides whether the fine-tuned encoder is a base model or a large one.
- [ ] Submission limits per day, and whether the final pick is one submission or two (if two: one CV-best, one more aggressive blend).
- [ ] What the writeup or finale needs (length, format), so Krithi can scope it from hour 0.
