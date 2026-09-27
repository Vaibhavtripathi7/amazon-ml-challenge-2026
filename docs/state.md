# State — facts only (2026-09-27 12:00). Every line has a source; nothing assumed.

## 1. Problem (official README)
- For every Source-1 (S1) business, list all matching S2/S3 records (0, 1 or many). S1 is deduplicated.
- Columns: entity_id, business_name, business_address, country. Train: US, India. **Test adds France (not in train).**
- **Metric:** F0.5 per S1, macro-averaged. S1 with no true match: 1.0 if we predict nothing, 0.0 otherwise.
- `matching_results.tsv` is scored. `candidate_pairs.tsv` = the exact set the final model scores; matches ⊆ candidates.
- Zip: `<team>_submission.zip` → `output/` (both TSVs), `code/business_entity_resolution/` (`src/`, README, pinned
  requirements), `Documentation_template.md` (filled).
- Rules: no external data or lookups; final model MIT/Apache 2.0 and ≤ 8B parameters. Final rank = private leaderboard.

## 2. Data (measured on all rows)
| | Train | Test |
|---|---|---|
| S1 | 2,206,821 (US 1,323,633 · India 883,188) | 1,732,544 (US 663,106 · India 809,986 · France 259,452) |
| S2+S3 records per S1 | 4.67 | **5.76 US · 5.82 India · 5.53 France** |
| True records per S1 | 3.46 | about 3.3–3.5 (address mixture estimate) |
| Records with no owner | 26% | about 40–43% (same estimate) |
- A record never belongs to two S1; a match never crosses countries.
- IDs, row order and per-source counts carry **no** information (ID distance = random; row-order correlation 0.001).
- Decoys are near-copies: **17% of same-name + same-house record pairs belong to different businesses.**
- Indian-script names: 18% of India records. France: 34% of S1 share a door (house + street) with another S1.

## 3. Leaderboard evidence (public)
| Upload | Score |
|---|---|
| Rules baseline | 0.56 |
| v2 | 0.946 (India emptied 0.572 · France emptied 0.822 · US emptied 0.554) |
| v3 (t = 0.40) | 0.953 |
| v3, France t = 0.7 | 0.957 |
| v3, US + India t = 0.55 | 0.957 |
- From the emptied uploads (exact arithmetic): the public set's singleton share is 0.056; country shares are at least
  US 0.415 / India 0.396 / France 0.131.
- Stricter thresholds help on test in every country (the opposite of train-density dev).

## 4. Where the score is lost
| | Evidence | Size |
|---|---|---|
| **France** | about 0.81 if US/India score their dev values on test (derived from §3) | **about 0.031** of the leaderboard |
| India | full dev 0.9734 | about 0.011 |
| US | full dev 0.9867 | about 0.006 |
| Unseen-country penalty | leave-one-country-out rehearsal (train US, score India): −0.021 | — |
| v3 dev loss split | shortlist misses 0.0095 · judge misses 0.0055 · wrong merges 0.0037 | — |

## 5. Bugs found and fixed (all tested)
1. Stage-1 cut dropped every negative LambdaRank score (v2, v3). Fixed 00b5c9c.
2. France: `5 bis` / `5B` lost the letter, departments not recognised, French street words missing, `St` read as
   street. Fixed with unit tests; train output byte-identical (verified).
3. Holdout orphans not counted as distractors in training weights. Fixed.
4. Random tie-breaks (house numbers; sibling feature) changed results between runs. Fixed; identical under two
   different hash seeds.
5. Rescoring threshold defaulted silently to 0.5 for countries not named. Documented: always pass `default=`.
6. Cross-encoder build: 94% positives (decoys missing). Fixed: every pair labelled from truth; 29–41% negatives.
7. Zip wrote `Documentation.md`; the official name is `Documentation_template.md`. Fixed (this commit).

## 6. Running now
- **Kaggle: v4** = v3 + fixes 1–4 + holdout training + sibling features + candidate budget + per-country
  thresholds + France self-training. Local benchmark: +0.0013 (test-like) / +0.0023 (harsher), significant.
- **Colab: cross-encoder (XLM-R, T4)** on top of our candidates. Tested: metric reproduces v2's known dev score
  (0.9753 vs 0.9754); labels 991/991 correct; resume after a disconnect gives byte-identical output.
  **Not yet known: whether it beats the judge.**

## 7. Decision rules for the cross-encoder (set before seeing results)
| Colab result | Decision |
|---|---|
| Cell 5 AUC: cross-encoder ≤ judge | Stop; no upload |
| Cell 5 blend ≥ judge +0.003 in both countries | Use for US/India |
| Cell 6 US-only model on India within 0.005 of the judge | Transfers → apply to France |
| Both strong | 0.99 becomes realistic (France ≈ 2/3 of the remaining loss) |
