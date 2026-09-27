# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TO FILL]
**Team Members:** Vaibhav Tripathi, Krithi, [TO FILL]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A four-stage pipeline: (0) a normaliser that undoes the data generator's corruptions (Indian-script names, accents,
abbreviations, reordered addresses); (1) retrieval with exact keys and TF-IDF searches, pruned by a LightGBM ranker
to a small candidate set (~6 per Source-1 entity on test); (2) a LightGBM judge on ~60 similarity and ambiguity
features; (3) a multilingual cross-encoder (xlm-roberta-base) that re-scores the pairs the judge is unsure about and is
blended with it. Stage 3 lifted held-out macro F0.5 from 0.9752 → 0.9838 (India) and 0.9873 → 0.9904 (US), both with
95% confidence intervals well above zero.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2.2M S1 (US 60%, India 40%), 10.3M S2/S3 records, 4.67 records per S1; 26% of records belong to no S1;
  5.6% of S1 have no match. A record never belongs to two S1; no match crosses countries.
- Test: 1.73M S1 (India 47%, US 38%, France 15% — France unseen in train), **5.76–5.82 records per S1**: test holds
  about twice as many look-alike records per business as train.
- Corruptions measured on 73,531 real true pairs (India / US): legal suffix dropped 29% / 18%, ALL CAPS 27% / 13%, a
  core name word replaced 16% / 19% (half of them by a generic filler: Center/Services/Partners; ~40% by phonetic
  garbles such as "knsltimg" = consulting), typos 7% / 9%, Indian script 19% (India), address parts dropped 47% / 12%,
  first house-number digit changed 6% / 8%, empty address 4% / 5%.
- Decoys are near-copies of real businesses: 17% of record pairs with identical cleaned name and house number belong
  to different businesses. Decoy name changes use specific real words (enterprises, trading, global), not fillers.
- France: S1 uses regions, records use departments (Nord, Gironde, Loire-Atlantique, Pas-de-Calais); `bis`/`ter` are
  part of the house number; 34% of French S1 share a door (house + street) with another S1.

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned ranker + gradient-boosted classifier + transformer re-scoring (hybrid).
**Core Innovation:** a cheap, high-recall candidate stage and a feature-based judge, with a multilingual cross-encoder
spent only where the judge is uncertain (12% of pairs), blended by weights fitted on held-out businesses; separate,
more conservative blend weights for the unseen country, fitted in a leave-one-country-out rehearsal.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** 8 exact keys (sorted-word name key, homoglyph-folded name, space-stripped name,
  name + house number, house + street, name + city, rare name words), frequency-capped; per-country word TF-IDF on the
  cleaned name (top 20) and on name + address tokens (top 40).
- **Pruning:** a LightGBM ranker scores every (record, S1) candidate on key hits, TF-IDF scores/ranks and quick
  similarities; the top 3 per record are kept (operating point chosen on held-out data by the F0.5 ceiling).
- **Candidate pairs generated:** 10,670,181 on test (6.16 per S1).
- **How you ensured true matches were not lost:** several independent retrievers unioned before pruning; recall
  measured at every stage on held-out records (shortlist ceiling 0.987; 0.971 of true records reach the judge on dev).

---

## 4. Matching Model

**Features used:**
- Name: token-set / sort / partial ratios, Jaro-Winkler, homoglyph-folded and space-stripped ratios, per-country
  IDF-weighted shared-token coverage, legal-form agreement/conflict, alias match, domain-name flag.
- Address: token-set / partial ratios, IDF-weighted coverage, house-number relation (equal / digit dropped /
  substituted / missing), number-set Jaccard, street and city similarity/equality, state agreement.
- Other: which retrievers hit, TF-IDF scores and ranks, stage-1 probability/rank, gap to the best candidate of the same
  record, ambiguity (namesake counts, shortlist size, stage-1 margin), source.

**Model type:** LightGBM judge (binary), then a cross-encoder: `xlm-roberta-base` (MIT, 278M parameters) fine-tuned as
a pair classifier on `name | address | parsed hints (house, street, city, state)` of both records (Ditto-style), trained
on our own held-out dev candidate pairs labelled from the ground truth (29–41% negatives = real decoys), 80/20 split by
S1 business. It re-scores pairs with judge probability in [0.01, 0.99]; final probability = logistic blend of the two
logits, cross-fitted on held-out businesses. France (unseen) uses blend weights fitted in a rehearsal where a
US-only model was scored on India (the transformer alone transfers poorly: 0.8909 vs the judge's 0.9755).

**Threshold selection method:** macro F0.5 per S1 on held-out businesses, per country; thresholds chosen on one half
of the held-out businesses and scored on the other half. Each record goes to its single best S1 if the blended
probability passes that S1's country threshold. Final thresholds: [TO FILL: India / US / France].

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), held-out dev (44,140 businesses, out-of-fold thresholds):**
  India 0.9752 → **0.9838** (+0.0085, 95% CI [+0.0074, +0.0097]); US 0.9873 → **0.9904** (+0.0031 [+0.0025, +0.0037]).
  AUC on the re-scored pairs: judge 0.9354 → cross-encoder 0.9789. Public leaderboard: [TO FILL].
- **Common false positives (wrong merges):** records with no address matched by name alone; legal-form conflicts
  (Pvt Ltd vs LLP); sibling businesses at the same or next door.
- **Common false negatives (missed matches):** records never retrieved (mostly India: Indian-script names, no address,
  heavily garbled names); substituted or missing house numbers.

---

## 6. Conclusion
A feature-based pipeline gives a small candidate set and a strong judge; a multilingual cross-encoder applied only to
the uncertain pairs closes most of the remaining gap on seen countries (India reaches 0.9838 against a 0.9843 ceiling of
its candidate set). The unseen country (France) remains the hardest part: rehearsals showed transformers trained on
other countries transfer poorly, so France relies mainly on the judge.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src_v3/` (exact stage 0–2 code of the submitted run: `run_v2.py` entry point,
`prepare.py`, `judge_v2.py`, `amlc/` library), `src/` (latest code, `ce_rerank.py` for stage 3, `final_thresholds.py`),
`README.md` (exact commands, in order), `requirements.txt` (pinned).

### B. Additional Results (experiments, not in the final run)
Measured on a test-density validation world (5% of train S1, 21% of non-validation S1 removed); paired bootstrap 95%
CI vs the v3-style model. These were built into a later pipeline version whose full run did not finish before the
deadline, so they are **not** part of the submitted file.

| variant | candidates / S1 | test-like | harsh | Δ test-like | Δ harsh |
|---|---|---|---|---|---|
| v3-style | 4.76 | 0.9894 | 0.9873 | — | — |
| + holdout training + candidate budget | 4.06 | 0.9894 | 0.9885 | +0.0000 [−0.0008, +0.0009] | +0.0011 [+0.0003, +0.0020] |
| + sibling features | 4.06 | 0.9906 | 0.9896 | +0.0013 [+0.0004, +0.0022] | +0.0023 [+0.0012, +0.0033] |

Rejected after measurement: graph post-processing over twin records (−0.0005), per-country judges, double decoy weight,
self-training on seen countries (−0.0012), character 3-gram retrieval (no gain after the candidate budget), synthetic
French training pairs (made the transformer alone much better on an unseen country, 0.8909 → 0.9263, but not the blend).
