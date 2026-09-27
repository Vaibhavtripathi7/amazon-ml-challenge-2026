# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TO FILL]
**Team Members:** Vaibhav Tripathi, Krithi, [TO FILL]
**Submission Date:** 2026-09-28

---

## 1. Executive Summary
A three-stage, CPU-only pipeline: (0) a country-aware normaliser that undoes the data generator's corruptions (Indian-script names, accents, reordered addresses, French `bis`/`ter` numbers, departments vs regions); (1) retrieval with exact keys and TF-IDF searches, pruned by a LightGBM ranker to a small candidate set (about 4–6 per Source-1 business); (2) a LightGBM judge on about 70 similarity, ambiguity and "sibling" features, trained in a **test-like world** where part of the reference businesses is held out, so that it learns that a record's owner may be absent.
The key insight: the test set has about twice as many look-alike decoy records per business as train, so a model tuned on train over-merges. We reproduce test density during training and threshold selection.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2.2M S1 (US 60%, India 40%), 10.3M S2/S3 records, 4.67 records per S1, 26% of records belong to no S1, 5.6% of S1 have no match. No record belongs to two S1; no match crosses countries.
- Test: 1.73M S1 (India 47%, US 38%, **France 15%, unseen in train**), **5.76 records per S1**. An address-evidence mixture estimate gives about 3.3–3.5 true records per S1 (as in train), so test has **about 2.4 decoys per S1 vs 1.2 in train**.
- Generator operators seen: word reordering, legal-form changes (Pvt Ltd / LLP / Inc), dropped or added words (`Groupe`, `SNC`, `Center`), OCR-like homoglyphs (0/O, 1/l), accents, junk prefixes, names written in Indian scripts (18% of India records), domain-style names, `formerly`/`dba` aliases, reordered address components, abbreviations (St/Street, R./Rue), state names in local scripts, dropped/altered house numbers, empty addresses (about 3%).
- **Decoys are near-copies of real businesses**: the same name with one word replaced, the legal form changed, or the house number moved (e.g. `Femme Culture, 66 Rue Van Dyck` vs decoy `Femme Sportive, 71 Rue Van Dyck`). 17% of record pairs with identical cleaned name and house number belong to different businesses.
- France: 34% of S1 share their door (house + street) with another S1 (US 15%); S1 uses regions while records use departments; `bis`/`ter` is part of the house number (kept as `bis`/`B` in 99% of matching records).

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned ranker + classifier (hybrid, multi-stage), with a test-like training world.
**Core Innovation:** (1) holdout training: non-validation reference businesses are removed while their records are kept, which reproduces the test's decoy density; (2) "sibling" features that separate a noisy copy of the same business (typo: the differing word is close and rare) from a sibling decoy (a common real word replaced, house moved); (3) a country-aware normaliser with data-derived rules for an unseen country (France).

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** 8 exact keys (sorted-word name key, homoglyph-folded name, space-stripped name, name + house number, house + street, name + city, rare name words), capped by frequency; word TF-IDF on the cleaned name (top 20) and on name + address tokens (top 40), per country; [v4: character 3-gram TF-IDF on the name if kept].
- **Pruning:** a LightGBM ranker (binary) scores every (record, S1) candidate on key hits, TF-IDF scores/ranks and quick similarities; we keep the top N per record above a probability cut. N and the cut are chosen on held-out data under a candidate budget, the smallest set within 0.001 of the best achievable F0.5.
- **Candidate pairs generated:** [v4: TO FILL] (v3: 6.16 per S1 on test).
- **How you ensured true matches were not lost:** multiple independent retrievers (name, address, keys) unioned before pruning; recall measured at every stage on held-out records (shortlist ceiling, recall after pruning); stage-1 operating point chosen by the F0.5 ceiling, not by size alone.

---

## 4. Matching Model

**Features used:**
- Name features: token-set / sort / partial ratios, Jaro-Winkler, homoglyph-folded and space-stripped ratios, IDF-weighted shared tokens and coverage (per country), legal-form agreement/conflict, alias match, domain-name flag; **sibling features**: number of unmatched words on each side, how well the worst unmatched record word is explained by an S1 word (Jaro-Winkler) and that word's IDF.
- Address features: token-set / partial ratios, IDF-weighted address coverage, house-number relation (equal / digit dropped / substituted / missing; France keeps `bis`/`ter`), number-set Jaccard, house numeric and edit distance, street and city similarity/equality, state agreement (departments mapped to regions for France).
- Other: retrieval signals (which keys hit, TF-IDF scores and ranks), stage-1 probability and rank, competition within the record's candidate set (gap to the best candidate), ambiguity (namesake counts, shortlist size, stage-1 margin), source.

**Model type:** LightGBM (gradient-boosted trees), binary objective; one pooled judge (it transfers to France) [v4: per-country judges if kept].
**Threshold selection method:** macro F0.5 per S1 on a held-out, test-density validation set (holdout world), per country; each record goes to its single best S1 if its probability passes the threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [v4: TO FILL]. v3: 0.9814 on train-density validation, 0.953 public leaderboard (0.957 with stricter thresholds).
- **Common false positives (wrong merges):** records with no address matched by name alone (31% of wrong merges vs 2.6% of correct matches); legal-form conflicts (20% vs 7%); sibling businesses at the same or next door with one name word replaced (France).
- **Common false negatives (missed matches):** records never retrieved (India 64%): names in Indian scripts, no address, garbled names; judge rejections with a substituted or missing house number.

---

## 6. Conclusion
A careful normaliser plus a two-stage LightGBM pipeline reaches high precision at a small candidate set. The decisive lesson was that validation must look like test: the generator adds look-alike decoys at twice the train density, and only a test-like validation world (holdout of reference businesses) predicted the leaderboard direction.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/amlc/` (library: `norm.py` stage 0, `keys.py` / `blocking.py` / `stage.py` stage 1, `judge_features.py` stage 2, `metric.py`), `src/prepare.py`, `src/judge_v2.py`, `src/run_v2.py` (entry point: runs every step resumably and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`), `README.md` (exact commands), `requirements.txt` (pinned).

### B. Additional Results
[v4 experiment table: TO FILL from docs/v4-gaps.md]
