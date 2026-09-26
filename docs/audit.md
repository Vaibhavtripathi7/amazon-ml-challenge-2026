# Audit — from data to leaderboard, evidence only

_2026-09-27 (night). Every line is tagged **[measured]** (a number from a log, file or run), **[source]** (research), or **[hypothesis]** (not proven). Nothing untagged._

## 0. Summary

- **[measured]** v2 dev F0.5 = **0.9754**; v2 public LB = **0.946**. Gap **0.029**.
- **[measured]** The dev metric is correct: an independent recomputation from `dev_pairs_v2.parquet` with only the README formula gives 0.9754 at t = 0.40 (0.9777 when sampled wrong merges are not scaled).
- **[measured]** So the gap is a **dev-world vs test-world difference**, not an evaluation error.
- **Not known:** which difference. Two candidates survive (§5); both are tested with the real leaderboard (§6).

## 1. Timeline of runs and scores

| Run | Dev F0.5 | Public LB | Source |
| --- | --- | --- | --- |
| Rule baseline (exact name key + house number) | 0.554 (200k S1) | **0.56** | upload #1, `baseline_rules.py` |
| v1 (LightGBM judge, crude trimming, not uploaded) | 0.9558 | — | Kaggle J3 log |
| v2 (stage-1 ranker + judge) | **0.9754** (t = 0.40) | **0.946** | `v2/SUMMARY.txt`, upload #2 |
| v2 with India emptied | — | 0.572 | diagnostic upload |
| v2 with France emptied | — | 0.822 | diagnostic upload |
| v2 with US emptied | — | 0.554 | diagnostic upload |
| v3 (running) | — | — | Kaggle log (partial, §3.3) |

**[measured]** For the rule baseline, dev ≈ LB (0.554 vs 0.56). For the learned v2 they diverge by 0.029.

## 2. Data facts

| Fact | Train | Test | Tag |
| --- | --- | --- | --- |
| S1 businesses | 2,206,821 (US 60%, India 40%) | 1,732,544 (India 47%, US 38%, France 15%) | [measured] |
| S2+S3 records per S1 | **4.68** | **5.76** (India 5.82, US 5.76, France 5.53) | [measured] |
| Owned records per S1 (truth) | 3.46 | unknown | [measured] / — |
| Share of records with no owner | 26% | unknown | [measured] / — |
| Singleton S1 share | 5.6% | **5.6%** (public LB: 0.946 − sum of the three emptied drops 0.890 = 0.056) | [measured] |
| Record profile (no address, script names, namesakes) | India 3.1% / 17.9% / 20% have ≥10 namesakes | India 2.4% / 18.2% / 18% | [measured] — no India-specific shift |
| Our v2 test output: matched records per S1 | — | 3.37 (France 3.61, US 3.40, India 3.28) | [measured] |
| Test records left unmatched by v2 | — | 41.4% | [measured] |

**[measured]** The public LB subset is **not** a uniform sample of test: bounding each country score at ≤ 1 forces the US share of public ≥ 41.5%, while US is 38.3% of test.

## 3. Stage by stage

### 3.1 Stage 0 (normalisation)
- **[measured]** India script dictionary (134 name words, learned from non-dev pairs): dev S1 name words recovered 0.125 → 0.791; name-key equal 0.012 → 0.652 (20k dev pairs). In v3.
- **[measured]** Domain/handle segmentation: name words recovered 0.029 → 0.825 on 42,343 dev records. In v3.

### 3.2 Stage 1 (shortlist + ranker)
| Measure | v1 | v2 | v3 (partial) | Tag |
| --- | --- | --- | --- | --- |
| Shortlist ceiling (held-out records) | 0.9739 (dev union) | 0.9746 | **0.9867** | [measured] |
| Recall reaching the judge | 0.9238 | 0.9583 | 0.9712 (top-3) | [measured] |
| Candidates per S1 (dev) | ~58 | 4.70 (test 5.91) | 4.81 | [measured] |
| Pairs per record at the judge | 10 | ~0.93 | **1.02** | [measured] |

- **[measured] Bug:** with LambdaRank chosen, a `score ≥ 0` cut meant for probabilities dropped every negative-scored candidate (hence ~1 pair per record despite top-3). 5% world, same model: top-10 recall 0.9898 (bug) → 0.9953 (fixed). Fixed in `00b5c9c`, not yet in any Kaggle run.
- **[measured]** At a small candidate budget, the binary ranker + probability cut beats LambdaRank (5% world: 0.9928 recall at 4.0 candidates per S1 vs 0.9895 at 6.9). Default switched to binary.

### 3.3 Stage 2 (judge) and decision
- **[measured] v2 dev loss breakdown (t = 0.40):** before judge 0.0150 · judge misses 0.0057 · wrong merges 0.0039. India 0.9600, US 0.9857.
- **[measured]** Ambiguity features A/B (5% world, exact dev): 0.9906 → 0.9916, paired bootstrap 95% CI [+0.0003, +0.0017]. In v3.
- **[measured]** Threshold: per-business F0.5 on v2 dev is best at **t ≈ 0.40** both as-is and with sampled wrong merges ×20 (0.9749); higher thresholds lower it. The pair-level (micro) view suggested 0.85, but that is the wrong metric for this leaderboard. The earlier recommendation to raise to 0.85 was **retracted**.
- **[measured]** Probabilities are roughly calibrated once sampled pairs are weighted ×10 (e.g. 0.3–0.4 band: ~0.39 precision).

### 3.4 Dev vs test
- **[measured]** Label-free: v2's test matched pairs have evidence profiles matching true train pairs (US bare house-number substitution 3.9% vs 4.1% in true train pairs; India 12.2% vs 12.3%).
- **[measured]** Test records v2 left unmatched look like train distractors (US: 0.0% "same name key + same house" vs 0.7% for train distractors and 41.7% for owned train records).
- **[measured]** Scaling sampled wrong merges to test density (×20) moves v2 dev only 0.9777 → 0.9749: **distractor density alone does not explain 0.029**.

## 4. Research used

| Source | Used for | Status |
| --- | --- | --- |
| Foursquare Location Matching 1st/7th ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution), [7th](https://future-architect.github.io/articles/20220720a/)) | Multi-retriever shortlist → LightGBM → pruning stage | Applied |
| Sparkly TF-IDF blocking ([VLDB'23](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)) | Top-k TF-IDF retrieval | Applied |
| Splink term frequency / Winkler frequency weights ([Splink](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9336505/)) | Ambiguity features | Applied, A/B positive |
| wordninja DP segmentation ([repo](https://github.com/keredson/wordninja)) | Domain names | Applied |
| Transliteration from parallel pairs ([paper](https://arxiv.org/pdf/1403.7455)) | India dictionary | Applied |
| Collective ER ([Bhattacharya & Getoor](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)) | "Namesake without same-source record" | Tested on data, **rejected** (owner uniquely identified in ~19% of ambiguous cases) |
| Expected-F maximisation ([Dembczynski'11](https://www.weiweicheng.com/research/papers/cheng-nips11.pdf), [Ye'12](https://arxiv.org/pdf/1206.4625)) | Per-S1 decision | Tested on mini world, **not adopted** (0.9946 vs 0.9958) |
| Stacking without leakage ([guide](https://mcpanalytics.ai/articles/stacking-ensemble-practical-guide-for-data-driven-decisions)) | Disjoint record sets for stage 1 / stage 2 / dev | Applied |
| Optimal F threshold ≈ F\*/(1+β²) ([Lipton'14](https://arxiv.org/abs/1402.1892)) | Threshold | **Misapplied** (micro formula; the LB is macro per S1) |

## 5. Hypotheses for the 0.029 gap

| # | Hypothesis | For | Against | Tag |
| --- | --- | --- | --- | --- |
| A | Test S1 have **more true records** (≈4.1 vs 3.46); v2 misses them → test recall ≈ 0.82 | Records per S1 +24%; F0.5 with P ≈ 0.985, R ≈ 0.82 ≈ 0.946; the stage-1 bug drops candidates, and harder or extra records suffer most; the per-business sweep favours lower thresholds | Unmatched test records look like distractors (but hard true records would too, so this is not decisive) | [hypothesis] |
| B | Test has **more distractors** (≈41% vs 26%) → wrong merges | Test unmatched share 41% | Matched pairs look like true pairs; ×20 weighting explains only ~0.003 | [hypothesis], weakened |
| C | **France** scores far lower | France is unseen; France is over-matched (3.61 vs 3.46) | France matched pairs look as clean as US; LB diagnostics cannot isolate countries (public mix unknown) | [hypothesis] |
| D | **Singletons** differ in test | — | Public singleton share measured 5.6% = train | **Rejected** [measured] |
| E | **Dev metric bug** | — | Independent recomputation reproduces 0.9754 | **Rejected** [measured] |
| F | **Stage-1 bug** costs recall everywhere | Measured on dev and 5% world | Size on test unknown | **Confirmed** [measured]; fixed, not yet run |

## 6. Decisive experiments (real test, uploads)

| # | Upload | Proves if the score RISES | Proves if it FALLS |
| --- | --- | --- | --- |
| 1 | v2 + rule matches (exact name + house, unique S1) for records v2 left unmatched | **A:** v2 misses obvious true matches (strong) | Only that those specific rule matches are wrong (weak) |
| 2 | v3 at t = 0.20 vs 0.40 vs 0.60 (`rethreshold`, saved pairs) | More matches help → **A** | Fewer matches help → **B** |
| 3 | Branch run chosen **only** by 1–2 | — | If 1–2 conflict: no branch; upload v3 at its best threshold |

## 7. Errors made (for the record)

- The stage-1 `≥ 0` cut shipped through two Kaggle runs (v2, v3).
- The sampled dev was trusted for thresholds before the exact dev was built.
- The threshold recommendation was made with the pair-level metric, then retracted.
- The v2 test step did not save scored pairs (fixed from v3), so v2 cannot be re-thresholded.
- The diagnostic "strict" file was not ready before the daily upload limit (out of memory, then too slow).
