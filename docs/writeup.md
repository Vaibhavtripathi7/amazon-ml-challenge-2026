# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Pacman
**Team Members:** Vaibhav Tripathi (leader), Bazaru Lokesh Ram Chand, S J Krithi
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
Our solution has four stages. First we clean and parse every name and address. Then we find a small set of candidate
Source 1 businesses for each Source 2/3 record (6.16 per business on test) and score every candidate with LightGBM.
Finally, a multilingual transformer (xlm-roberta-base) re-checks only the pairs that LightGBM is unsure about, and we
combine the two scores. On held-out validation data the transformer raised macro F0.5 from 0.9752 to 0.9838 for India
and from 0.9873 to 0.9904 for the US. On the public leaderboard it took us from 0.957 to 0.9708, and after tuning the
thresholds our final score is 0.9738.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train has 2,206,821 Source 1 businesses (60% US, 40% India) and 10.3M Source 2/3 records, about 4.7 records per
  business. 26% of the records belong to no business, and 5.6% of the businesses have no match at all. In the ground
  truth a record never belongs to two businesses and a match never crosses countries, so our final decision gives each
  record to at most one business.
- Test has 1,732,544 businesses (47% India, 38% US, 15% France, and France does not appear in train) and 5.75 records per
  business (5.53 to 5.82 by country). From the address evidence we estimate that test has about the same number of
  true records per business as train (3.3 to 3.5), so the extra records are look-alikes: roughly twice as many per
  business as in train.
- To learn how the data was corrupted, we compared 73,531 true pairs from train (India / US). The legal suffix is dropped
  in 29% / 18% of them, the name is written in capitals in 27% / 13%, one name word is replaced in 16% / 19%, there is a
  typo in 7% / 9%, parts of the address are missing in 47% / 12%, and one digit of the house number is changed in
  6% / 8%. In India 19% of the names are written in an Indian script (Devanagari, Kannada, Telugu, Tamil, Bengali).
  When a name word is replaced in a true copy, about half the time the new word is a generic one (Center, Services,
  Service, Partners) and about 40% of the time it is a phonetic misspelling, for example "knsltimg" for consulting.
- The wrong candidates are often very close copies of real businesses. Among records that share their cleaned name and
  house number with another record, about 1 in 6 belongs to a different business. When a wrong candidate has a
  different word in its name, it is usually a specific real word (Enterprises, Trading, Global) and only 6% of the time
  a generic one.
- In France, Source 1 uses regions while the records use departments (Nord, Gironde, Loire-Atlantique, Pas-de-Calais).
  "bis" and "ter" are part of the house number, and 34% of French businesses share their house number and street with
  another business, so an address match alone is weak evidence there.

### 2.2 Solution Strategy
**Approach Type:** Candidate search (blocking) + LightGBM ranker + LightGBM classifier + transformer re-scoring.
**Core Innovation:** LightGBM handles most pairs cheaply. The transformer is used only where LightGBM is unsure (12% of
the validation pairs, 20% of the test pairs), and its score is combined with the LightGBM score using weights learned
on businesses that were held out from training. France has no training data, so we measured how the transformer
behaves on a country it has never seen (trained on the US, tested on India) and used more cautious weights for France.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** 8 exact keys (name words in sorted order, name with look-alike characters merged such as 0/O
  and 1/l, name without spaces, name + house number, house number + street, name + city, and the rarest name words),
  each skipped when the key is too common. We also run TF-IDF searches per country on the cleaned name (top 20) and on
  name + address words (top 40). Indian-script words are first mapped to English with a word list learned from train
  pairs, and website-style names are split into words.
- Pruning: a LightGBM ranker (LambdaRank objective) scores every (record, business) candidate from the key matches,
  the TF-IDF scores and ranks, and a few quick similarity measures. We keep up to 3 candidates per record. The cut-off
  was chosen on held-out data as the smallest candidate set within 0.001 of the best possible F0.5.
- **Candidate pairs generated:** 10,670,181 on test (6.16 per business).
- **How you ensured true matches were not lost:** several independent searches are combined before pruning, and we
  measured recall after each step on held-out data: 98.7% of true records are found by the searches and 97.1% are still
  there after pruning.

---

## 4. Matching Model

**Features used:**
- Name features: fuzzy string ratios (token set, token sort, partial), Jaro-Winkler, the same ratios on the look-alike-merged and
  no-space names, IDF-weighted word overlap per country, legal suffix agreement or conflict, alias match
  (formerly / dba), and whether the name looks like a website.
- Address features: fuzzy ratios, IDF-weighted word overlap, how the house numbers compare (same, one digit missing, different,
  missing), overlap of all numbers, street and city similarity, and whether the states agree.
- Other: which searches found the pair, the TF-IDF scores and ranks, the ranker score and rank, how far this candidate is
  from the best candidate of the same record, how common the name is among businesses and records, how many
  candidates the record has, and the source (S2 or S3).

**Model type:** A LightGBM classifier (about 60 features), followed by a cross-encoder. The cross-encoder is
`xlm-roberta-base` (MIT licence, 278M parameters) fine-tuned to decide whether two records are the same business. Each
record is written as "name | address | house, street, city, state", using our parser for the last part, and the two
records are read together. We trained it on the candidate pairs of our held-out validation run. At first we kept only
pairs of validation businesses, because only those had correct labels in our saved file, but that left just 6% wrong
candidates for training. We fixed this by labelling every pair from the ground truth. Training used up to 600,000
pairs for 2 epochs (learning rate 2e-5, batch size 32, up to 160 tokens, mixed precision) and took about 3 hours on one
T4 GPU. The pairs were split 80/20 by business so that no business was in both parts. The transformer re-scores the
pairs whose LightGBM probability is between 0.01 and 0.99, and the two scores are combined by a logistic regression on
both log-odds, fitted on held-out businesses with 2-fold cross-fitting. For France we used the weights fitted in our
unseen-country test, where the transformer alone scored 0.8909 against 0.9755 for LightGBM, so the combination leans
on LightGBM there.

**Threshold selection method:** Each record goes to the one business with the highest combined score, if that score is
above the threshold for the business's country. We first chose thresholds on held-out businesses by macro F0.5
(India 0.7, US 0.9, France 0.7), checking them on a separate half of those businesses. We then raised them in two steps
on the public leaderboard, which improved the score both times. Final thresholds: India 0.90, US 0.97, France 0.90.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** on held-out validation data (44,140 businesses, thresholds chosen on one half and scored on
  the other), adding the transformer raised India from 0.9752 to 0.9838 (+0.0085, 95% confidence interval +0.0074 to
  +0.0097) and the US from 0.9873 to 0.9904 (+0.0031, interval +0.0025 to +0.0037). On the uncertain pairs the AUC went
  from 0.9354 (LightGBM) to 0.9789 (transformer). Public leaderboard: 0.957 before the transformer, 0.9738 final.
- As a check on test: with the thresholds chosen on validation, the transformer file left 5.6% of the test businesses
  with no match, the same share of businesses without matches that we measured on the leaderboard (section 7). The
  final, stricter file leaves 5.8% empty.
- **Common false positives (wrong merges):** records without an address that are matched on the name alone, records
  whose legal suffix differs (Pvt Ltd vs LLP), and different businesses at the same or a neighbouring address. In
  validation, 31% of the wrong merges had no address, against 2.6% of the correct matches.
- **Common false negatives (missed matches):** records that the searches never find (64% of these are in India: names in
  Indian scripts, missing addresses, badly misspelled names), and records whose house number was changed or removed.

---

## 6. Conclusion
The LightGBM stages give a small candidate set and a strong first score, and the transformer fixes most of the
remaining mistakes on the countries we could train on: India reaches 0.9838, very close to the 0.9843 that its
candidate set allows. France was the hardest part, because a model trained on other countries does not carry over well
to a new one, so for France we rely mostly on LightGBM and on the French address rules. The biggest lesson was that
validation has to look like the test set: once we understood that test has about twice as many look-alike records,
both the model choices and the thresholds moved in the right direction.

---

## 7. How We Validated
- The training data was split by hash into separate parts: 4% of the records for the ranker, 40% for the classifier,
  and 10% of the Source 1 businesses (220,907) for validation, so no model was scored on data it was trained on.
- Our validation scores for the early versions (0.975 to 0.981) were much higher than the leaderboard (0.946 to 0.957).
  To find out why, we made three diagnostic uploads of the v2 file, each with one country's matches removed. Their
  scores (India removed 0.572, US removed 0.554, France removed 0.822, against 0.946 for the full file) showed that 5.6% of the public test businesses have no match, the same as in
  train, and that the public part is not split by country in the same shares as the full test set. Together with the
  record counts, this pointed to the extra look-alike records in test.
- To check ideas under test-like conditions, we built smaller copies of the training data (5% of the businesses) in which
  21% of the businesses, or 35% for a harder copy, were removed while their records were kept. This gives the same
  records per business as test. On these copies the best threshold moved up (0.4 on normal data, 0.6 on test-like, 0.7 to
  0.8 on the harder copy), and the leaderboard later confirmed that direction.
- From v3 onwards, every change was compared with the previous version on the same businesses using a paired bootstrap
  95% confidence interval, and we kept only changes whose interval was above zero.
- France has no labels, so to estimate how a model does on an unseen country we trained on the US only and tested on
  India.

## 8. How the Solution Evolved

| version | main change | validation F0.5 | public leaderboard |
|---|---|---|---|
| rules baseline | exact name key + house number | 0.554 (200k businesses) | 0.56 |
| v1 | LightGBM classifier on similarity features | 0.956 | not uploaded |
| v2 | LightGBM ranker before the classifier, much smaller candidate set | 0.975 | 0.946 |
| v3 | India script word list, website-name splitting, name-frequency features, deeper searches | 0.981 | 0.953 |
| v3, stricter thresholds | France 0.7 (or US and India 0.55) instead of 0.4 | - | 0.957 |
| v3 + transformer | xlm-roberta-base re-scoring the uncertain pairs (India 0.7, US 0.9, France 0.7) | India 0.9838, US 0.9904 | 0.9708 |
| thresholds up | India 0.85, US 0.95, France 0.85 | - | 0.9731 |
| final | India 0.90, US 0.97, France 0.90 | - | 0.9738 |

## 9. Problems We Faced
- Validation vs leaderboard gap. This was explained by the extra look-alike records in test (section 7). Stricter thresholds
  helped on the leaderboard every time, which matched what the test-like copies showed.
- France is not in the training data. We added French address rules (departments mapped to regions, "bis" and "ter"
  kept as part of the house number, French street words), learned from the test data only. A transformer trained
  without French data did not transfer well, so we also tried synthetic French training pairs, made by applying the
  corruption patterns we measured in section 2.1 to real French businesses, with other French businesses at the same
  address or street as the negatives. They made the transformer alone much better on an unseen country (0.8909 to
  0.9263 on India) but did not improve the combined score, so we left them out.
- Bugs we found and fixed. A pruning rule meant for probabilities also removed every candidate with a negative
  LambdaRank score; a French house number such as "5 bis" lost its letter; a random tie-break made results differ
  between runs; and a slow membership check on 1.8M text IDs made our last full run about 60 times slower, which is why
  that version (appendix B) could not finish before the deadline.
- Compute. We developed on a laptop with 5.7 GB of RAM, so full runs (about 6 hours) were done on Kaggle, and the
  transformer was trained on a single T4 GPU in Google Colab. When the Colab GPU quota ran out, we moved the trained
  model to Kaggle and finished scoring there.

## 10. References
- Y. Li, J. Li, Y. Suhara, A. Doan, W.-C. Tan. Deep Entity Matching with Pre-Trained Language Models (Ditto). VLDB 2021.
  We follow its idea of writing both records as text for a transformer and adding hints from our own parser.
- A. Conneau et al. Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). ACL 2020.
- G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.
- Foursquare Location Matching, Kaggle 2022, top solutions: candidate search, LightGBM ranking, then a transformer on the
  remaining candidates. Our pipeline has the same shape.
- D. Paulsen, Y. Govind, A. Doan. Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching. VLDB 2023.
- Splink documentation, term-frequency adjustments. Used for our name-frequency features.
- R. Wang, Y. Li, J. Wang. Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and
  Preparation. ICDE 2023. Used for the synthetic French pairs experiment.

---

## Appendix

### A. Code Artefacts
The library is in `src/amlc/` (cleaning and parsing in `norm.py` and `clean.py`, candidate search in `keys.py`,
`blocking.py` and `stage.py`, judge features in `judge_features.py`, the metric in `metric.py`). The entry points are in
`scripts/`: `run.py` runs stages 0-2 end to end (it calls `prepare.py` and `judge.py`), `ce_rerank.py` is the transformer
stage, and `final_thresholds.py` writes the final files. The README lists every command in order. Our leaderboard file
was produced with an earlier revision of stages 0-2, before the French address rules and the ranker fix were added;
the code here is the final version.

### B. Additional Results
These changes were tested on the test-like copies from section 7 (thresholds chosen on the test-like copy and reused
on the harder one, paired bootstrap 95% intervals against the v3 setup). The full run that included them did not finish
before the deadline, so they are not part of the submitted file.

| variant | candidates per business | test-like | harder copy | change (test-like) | change (harder) |
|---|---|---|---|---|---|
| v3 setup | 4.76 | 0.9894 | 0.9873 | - | - |
| + training with 21% of businesses held out, candidate budget | 4.06 | 0.9894 | 0.9885 | +0.0000 (-0.0008 to +0.0009) | +0.0011 (+0.0003 to +0.0020) |
| + features for a swapped name word and house number distance | 4.06 | 0.9906 | 0.9896 | +0.0013 (+0.0004 to +0.0022) | +0.0023 (+0.0012 to +0.0033) |

Things we tried that did not help: linking records to each other and passing matches along (-0.0005), one model per
country (below one shared model), extra weight on wrong candidates in training, self-training on US and India
(-0.0012), a character n-gram search (it found 0.5% more true records in India at full scale, but gave no gain after
pruning and cost about 2.5 hours), a pretrained multilingual embedding search (it found only 22.5% of the missed India
records in its top 10, against about 50% for character n-grams), and synthetic French training pairs (section 9).
