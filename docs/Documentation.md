# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TO FILL]
**Team Members:** Vaibhav Tripathi (leader), Bazaru Lokesh Ram Chand, S J Krithi
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
Our solution has four stages. First we clean and parse every name and address. Then we find a small set of candidate
Source 1 businesses for each Source 2/3 record (about 6 per business on test) and score them with a LightGBM model.
Finally, a multilingual transformer (xlm-roberta-base) re-checks only the pairs the LightGBM model is unsure about, and
we combine the two scores. On our held-out validation data this last step raised the score from 0.9752 to 0.9838 for
India and from 0.9873 to 0.9904 for the US. Our final public leaderboard score is 0.9738.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train has 2.2M Source 1 businesses (60% US, 40% India) and 10.3M Source 2/3 records, so about 4.7 records per
  business. 26% of records belong to no business, and 5.6% of businesses have no match. A record never belongs to two
  businesses, and a match never crosses countries.
- Test has 1.73M businesses (47% India, 38% US, 15% France, and France does not appear in train). It has 5.8 records
  per business, so there are roughly twice as many look-alike records per business as in train.
- We compared 73,531 true pairs from train to see how records get changed (India / US): the legal suffix is dropped in
  29% / 18% of them, the name is in capitals in 27% / 13%, one name word is replaced in 16% / 19% (usually by a generic
  word like Center, Services or Partners, or by a phonetic misspelling such as "knsltimg" for consulting), there is a
  typo in 7% / 9%, parts of the address are missing in 47% / 12%, and the house number has one digit changed in 6% / 8%.
  In India, 19% of the names are written in an Indian script.
- The wrong candidates are often very close copies of real businesses. 17% of record pairs with the same cleaned name
  and the same house number still belong to different businesses. When a wrong candidate has a different word in its
  name, it is usually a specific real word (Enterprises, Trading, Global) rather than a generic one.
- In France, Source 1 uses regions while the records use departments (Nord, Gironde, Loire-Atlantique,
  Pas-de-Calais). "bis" and "ter" are part of the house number, and 34% of French businesses share the same house
  number and street with another business.

### 2.2 Solution Strategy
**Approach Type:** Candidate search (blocking) + LightGBM ranker + LightGBM classifier + transformer re-scoring.
**Core Innovation:** The cheap LightGBM stages handle most pairs. The transformer is used only on the 12% of pairs where
LightGBM is unsure, and its score is combined with LightGBM using weights learned on businesses held out from training.
France is not in the training data, so we tested how the transformer behaves on a country it has never seen (trained
on US, tested on India) and used more cautious weights for France.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** 8 exact keys (name words in sorted order, name with look-alike characters merged, name
  without spaces, name + house number, house number + street, name + city, rare name words), each limited by
  frequency. We also run TF-IDF searches per country on the cleaned name (top 20) and on name + address (top 40).
- **Pruning:** a LightGBM ranker scores each (record, business) candidate using the key matches, TF-IDF scores and
  quick similarity measures, and we keep the top 3 per record. The cut-off was chosen on held-out data.
- **Candidate pairs generated:** 10,670,181 on test (6.16 per business).
- **How we made sure true matches were not lost:** we combine several independent searches before pruning, and we
  measured recall after every stage on held-out data. 98.7% of true records are found by the searches and 97.1% are
  still there after pruning.

---

## 4. Matching Model

**Features used:**
- Name: fuzzy string ratios (token set, token sort, partial), Jaro-Winkler, ratios on the look-alike-merged and
  no-space names, IDF-weighted word overlap per country, legal suffix agreement or conflict, alias match, and whether
  the name looks like a website.
- Address: fuzzy ratios, IDF-weighted word overlap, how the house numbers compare (same, one digit missing, different,
  missing), overlap of all numbers, street and city similarity, and whether the states agree.
- Other: which searches found the pair, the TF-IDF scores and ranks, the ranker score, how far this candidate is from
  the best candidate of the same record, how common the name is, and the source (S2 or S3).

**Model type:** A LightGBM classifier, followed by a cross-encoder. The cross-encoder is `xlm-roberta-base` (MIT
licence, 278M parameters) fine-tuned to decide whether two records are the same business. Each record is given to it as
"name | address | house, street, city, state". We trained it on candidate pairs from our held-out validation run,
labelled with the ground truth (29-41% of them are wrong candidates), and split 80/20 by business so that no business
is in both parts. It re-scores the pairs where the LightGBM probability is between 0.01 and 0.99, and the two scores
are combined with a logistic regression fitted on held-out businesses. For France we used the weights from our
unseen-country test, because the transformer alone did poorly on a country it had not seen (0.8909 against 0.9755 for
LightGBM).

**Threshold selection method:** We maximised macro F0.5 on held-out businesses, separately for each country, choosing
thresholds on one half of them and checking on the other half. We then moved the thresholds higher step by step using
the public leaderboard (each step was better). Each record is matched to its best business if the combined score is
above that country's threshold. Final thresholds: India 0.90, US 0.97, France 0.90.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** on held-out validation data (44,140 businesses), India went from 0.9752 to 0.9838
  (+0.0085, 95% confidence interval +0.0074 to +0.0097) and the US from 0.9873 to 0.9904 (+0.0031, interval +0.0025 to
  +0.0037) when we added the transformer. Public leaderboard: 0.957 before the transformer, 0.9738 for our final
  submission.
- **Common false positives (wrong merges):** records without an address that are matched on the name alone, records
  whose legal suffix differs (Pvt Ltd vs LLP), and different businesses at the same or a neighbouring address.
- **Common false negatives (missed matches):** records that the searches never find (mostly in India: names in Indian
  scripts, missing addresses, badly misspelled names), and records whose house number was changed or removed.

---

## 6. Conclusion
The LightGBM stages give us a small candidate set and a strong first score, and the transformer fixes most of the
remaining mistakes on the countries we could train on (India reaches 0.9838, very close to the 0.9843 that its
candidate set allows). France was the hardest part, because a model trained on other countries does not carry over
well to a new one, so for France we rely mostly on LightGBM.

---

## Appendix

### A. Code Artefacts
The code is in `code/business_entity_resolution/`. `src_v3/` is the exact code of stages 0-2 used for the submitted
file (entry point `run_v2.py`, with `prepare.py`, `judge_v2.py` and the `amlc/` package). `src/` has our latest code,
including `ce_rerank.py` for the transformer stage and `final_thresholds.py` for the last step. `README.md` lists every
command in order, and `requirements.txt` has the pinned versions.

### B. Additional Results
These changes were tested on a smaller copy of the training data made to look like the test set, but the full run
that included them did not finish before the deadline, so they are not part of the submitted file.

| variant | candidates per business | test-like score | harder copy | change (test-like) | change (harder) |
|---|---|---|---|---|---|
| baseline | 4.76 | 0.9894 | 0.9873 | - | - |
| + holdout training + smaller candidate set | 4.06 | 0.9894 | 0.9885 | +0.0000 | +0.0011 |
| + extra name/house features | 4.06 | 0.9906 | 0.9896 | +0.0013 | +0.0023 |

Things we tried that did not help: linking records to each other and propagating matches (-0.0005), one model per
country, extra weight on wrong candidates in training, self-training on US/India (-0.0012), character n-gram search
(no gain after pruning), and synthetic French training pairs (the transformer alone improved on an unseen country,
0.8909 to 0.9263, but the combined score did not).
