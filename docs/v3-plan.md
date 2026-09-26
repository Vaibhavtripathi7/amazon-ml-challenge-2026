# V3 plan — closing the gap to 0.99+

_2026-09-26. Built on: v1 dev loss breakdown, v1 missed-pairs sheet, the Data Forensics doc, and new measurements below. v2 (stage-1 ranker) is running on Kaggle._

**Lead:** v2 fixes the biggest self-inflicted loss (trimming). The remaining gaps are **India name scripts**, **name-only records among namesakes**, **domain-form names** and **France**. Each gets a fix below, backed by evidence or a planned measurement. Nothing enters the pipeline without a dev A/B.

## 1. Where the points go (v1 dev, measured)

| Loss | v1 dev | Fix |
| --- | --- | --- |
| Before the judge (shortlist + trimming) | 0.0325 | v2 stage-1 ranker (running); India dictionary; domain segmentation |
| Judge rejecting true matches | 0.0066 | More training data, collective features, source features |
| Wrong merges | 0.0051 | Collective features, margin/abstention, per-country threshold |
| India vs US | 0.929 vs 0.974 | India dictionary (§2A) |

## 2. Problems, research, evidence, decision

### A. India: names written in Indian scripts (about 13 of 30 India misses)

- **Research:** multilingual sentence embeddings (LaBSE, multilingual-e5) map scripts together but need a GPU for about 12M strings ([LaBSE](https://research.google/blog/language-agnostic-bert-sentence-embedding/)). IndicXlit (MIT, about 11M parameters) does neural Indic→Roman transliteration, but it outputs phonetic spellings, not English words ([IndicXlit](https://github.com/AI4Bharat/IndicXlit)). The literature learns transliteration mappings from parallel pairs ([Hindi–English NE transliteration](https://arxiv.org/pdf/1403.7455)).
- **Our evidence:** the generator writes **English words in Indian script**. Learning a word dictionary from train pairs (Dice ≥ 0.5, count ≥ 3) on 575k pairs and testing on 144k held-out pairs:

| | Share of S1 name words present in the record |
| --- | --- |
| anyascii only (today) | **0.397** |
| + learned dictionary | **0.803** |

  11,068 words mapped, covering 71% of token occurrences (`kmstrksn`→construction, `phuds`→foods, `imtrnesnl`→international).
- **Decision: KEEP the learned dictionary** (cheap, deterministic, no GPU, provided data only). **Discard embeddings for now**; revisit only if dev still shows script misses.
- **Leakage guard:** learn the dictionary **only from pairs of non-dev S1**, so dev stays honest. Extend it to address tokens (city names in script) in the same way.

### B. Domain / handle names (a few % of misses in both countries)

- **Research:** unigram-frequency dynamic-programming word segmentation, as in wordninja ([wordninja](https://github.com/keredson/wordninja)).
- **Our evidence:** `producernavimumbai.com` vs S1 "Navi Mumbai Producer". The exact `flat` key fails because the word order is shuffled.
- **Decision: KEEP.** Segment domain and handle names with the DP, using word frequencies **from S1 names** (provided data, not Wikipedia), then treat the segments as a normal name.

### C. Name-only records among namesakes (about 24 of 30 US misses)

- **Research:** collective entity resolution. Iterative relational evidence helps most "in cases with high ambiguity" ([Bhattacharya & Getoor](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)).
- **Our evidence:** 87–88% of S1 have at least one match in each source, and a record has at most one owner. When 3 namesakes compete for a no-address S3 record, the one **without** an S3 match yet is the likelier owner.
- **Decision: KEEP as a second pass.** After the judge scores all pairs, add per-S1 features (how many records per source are already confidently assigned, and the S1's best other score), plus per-record top1 − top2 judge margin, then re-score. Truly ambiguous cases stay unmatched (under F0.5 that's correct).

### D. France (15% of test, no labels)

- **Decision:** derive **department ↔ region** from test data co-occurrence (communes that appear with a region in S1 and a department in S2/S3). **No external gazetteer**, which the rules forbid. Add French normalisation rules (Rue/R./BD/Allée/Impasse/Chemin, N°, SARL/SAS/…). Run a **leave-one-country-out rehearsal** (train US → test India and the reverse) to measure how well the model transfers.
- **French synthetic data (forensics insight 8): POSTPONE.** Do it only if the rehearsal shows a large transfer gap. It's costly, and the gap is unmeasured.

## 3. Forensics-doc ideas: verdicts

| # | Idea | Verdict | Evidence / reason |
| --- | --- | --- | --- |
| 1 | Address is the primary key | **Keep (done)** | Address search 89% vs name 63%; combined name+address search 96.4% (best single) |
| 2 | Suffix as a conditional feature | **Keep (done)** | `suffix_state` is a top-10 judge feature |
| 3 | Abstention: gate + top1−top2 margin | **Keep, extend** | Threshold done; judge-score margin comes in the second pass (§2C) |
| 4 | Name specificity by IDF | **Keep (done)** | IDF shared/max/coverage features |
| 5 | Containment, not Jaccard | **Keep (done)** | IDF-weighted coverage min/max |
| 6 | Collective resolution | **Keep → second pass** | §2C |
| 7 | Structured vs bare house-number change | **Keep (done)** | P(true \| bare substitution) = 0.8% US (full data) |
| 8 | Synthesise French pairs | **Postpone** | Only if the rehearsal shows a gap |
| 9 | OCR edit-cost matrix | **Partial** | Homoglyph folding done; custom-cost Levenshtein as a later feature test |
| 10 | Commune → department → region | **Keep** | From test co-occurrence only (§2D) |
| 11 | City mismatch is weak evidence | **Confirmed** | 21–33% of true pairs differ in city; the model learns the weight |
| 12 | Snap tokens to the S1 lexicon | **Discard for now** | Heavy for 10M records; fuzzy features cover typos. Revisit if misses show it |
| 13 | Generator's bad expansions | **Keep (done)** | Saint→Street, ordinal words |
| 14 | Source-pair features | **Keep** | Add record source (S2/S3) and source × missing-field flags |
| — | Cross-encoder transformer | **Postpone** | GPU-heavy; the doc also warns it transfers badly to France |

## 4. Order of work and measurements

| Step | Build | Measured by (dev) |
| --- | --- | --- |
| 1 | India dictionary (names + address tokens), non-dev pairs only → stage 0 v3 | India shortlist recall, India dev F0.5 |
| 2 | Domain segmentation | Recall on domain-form records |
| 3 | Source features + more judge training data (15% → 40%) | Judge losses |
| 4 | Collective second pass | Judge misses + wrong merges |
| 5 | France: department↔region, French rules, LOCO rehearsal | Rehearsal gap; France review sheet |
| 6 | Candidate-size tuning (operating point) | Candidates per S1 at ≤ 0.001 ceiling loss |

Every step: build → local mini-world run → fresh-clone check → Kaggle run (resumable runner) → dev loss breakdown → keep only if dev improves.

## Sources

- [LaBSE (Google Research)](https://research.google/blog/language-agnostic-bert-sentence-embedding/)
- [AI4Bharat IndicXlit](https://github.com/AI4Bharat/IndicXlit)
- [Hybrid Approach to English–Hindi Name Entity Transliteration](https://arxiv.org/pdf/1403.7455)
- [wordninja (unigram DP word segmentation)](https://github.com/keredson/wordninja)
- [Collective Entity Resolution in Relational Data (Bhattacharya & Getoor)](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)
