# v4 gap register — every gap: evidence → research → data analysis → options → experiment → result → decision

_Started 2026-09-27 05:25 IST. Status per step is updated after each experiment. Nothing is "done" without a measured result._

**Where we are.** Leaderboard: v2 0.946 → v3 0.953 → v3 stricter France (0.7) 0.957 → v3 stricter US/India (0.55) 0.957.
v3 dev (train-like): 0.9814. The goal is 0.99.

**Arithmetic of 0.99.** Leaderboard ≈ dev − (test-only losses). For 0.99 we need BOTH:
(a) dev losses 0.0186 → about 0.010: recall before the judge 0.0095, judge misses 0.0055, wrong merges 0.0037;
(b) test-only losses about 0.028 → about 0: decoy density, France.

**Test bench (local, labels).** `world5` = 5% of train S1 + all their records + 5% of unowned records (train-like).
`world5_t2` = world5 with 21% (US) / 22% (India) of non-dev S1 removed and their records kept = **test-like density** (5.76 / 5.82 records per S1).
Every arm is scored with the exact dev (`--dev-full`), macro F0.5 per S1, with a paired bootstrap against the baseline arm.

| ID | Gap | Size / evidence | Research | Options | Experiment | Result | Decision |
|---|---|---|---|---|---|---|---|
| G0 | **Dev is optimistic** (dev 0.981 vs leaderboard 0.953) | Dev has train density (1.2 decoys per S1); test has 2.3–2.5; our simulated test world reproduced the drop | Train/test shift; stress testing | Evaluate every arm on world5_t2 (test-like) AND world5_t3 (harsher: 35% of S1 removed); choose by worst case; report both | pending | | |
| G1 | **Sibling decoys on test** (precision) | Leaderboard +0.004 / +0.004 from stricter thresholds; test 0.4–0.7 band: US 36% no-address, India 40% legal-form conflict, France 40% one real word swapped; mistakes: 31% of wrong merges have no address | Hard-negative training; Splink term frequency; Foursquare graph post-processing (0.9778 → 0.9935) | (1) holdout training (test density) (2) sibling features (3) record-cluster consensus features (4) graph post-processing (5) per-country thresholds | (4) twin graph on world5_tl (see below) | (4) purity 0.83, −0.0005 [−0.0008, −0.0002] | (4) **rejected** |
| G2 | **Recall: shortlist misses** | dev 0.0100; India 14.0k vs US 8.0k; 51% no address, 67% no house, 19% script | Foursquare: record-record graph recovers missed links; multi-retriever | (1) stage-1 bug fix (2) candidate budget grid (3) record-record propagation (twin records) (4) extra retrieval pass | pending | | |
| G3 | Judge misses | dev 0.0048; 45% house substitution, 42% house missing | GBDT features on string distances | (1) house edit and numeric distance (2) cluster support | pending | | |
| G4 | **France** (unseen) | est. France about 0.81–0.85 on leaderboard; parse bugs | Leave-one-country-out transfer | (1) normaliser fix ✅ (2) LOCO rehearsal (3) country-specific threshold | normaliser: unit tests + train byte-identical ✅ | | |
| G5 | **Country-wise pipeline** | India dev 0.9734 vs US 0.9867; different decoy signatures per country | Per-segment models vs pooled with interactions | (1) one judge per country (2) pooled + country-interaction features (3) per-country thresholds | pending | | |
| G6 | Language / script / encoding / generator operators | India 18% script names; accents; homoglyphs; `groupe`/`snc` added by the generator | Transliteration from parallel pairs (used) | coverage audit of shortlist misses by operator | pending | | |
| G7 | Candidate-set size (final ranking) | v3 test 6.16 per S1 | — | budget in rank_eval (`--max-cands`) | pending | | |
| G8 | Threshold / calibration under test density | dev best 0.40; the leaderboard prefers stricter | Holdout = test-like training | pick the threshold on world5_t2 per country | pending | | |
| G9 | Engineering correctness | bug #5 (orphans not flagged), random house ties, v3 legacy cut | — | fixes + tests | bug #5 fixed; tests pending | | |
| G10 | Deliverables | zip, README, requirements, documentation template | — | — | pending | | |

## Experiment log

### E1 — twin-record graph post-processing (G1-4, G2-3), 05:45
Twins = records of the same country with equal (name key, house) [+ folded name + house] [+ name + street + city], groups ≤ 20.
On the test-like world (baseline 0.9881, t = 0.5, 11,055 dev S1):
- **Purity 0.83**: 17% of same-name + same-house twins belong to *different* owners. The generator's decoys copy name AND house.
- Recall potential: 60–74 of 752 missed true records (8–10%) have a twin assigned to the owner.
- Rules (fill unassigned twins; move conflicting twins to the anchor S1): −0.0005 [−0.0008, −0.0002]; looser links −0.0013.
- **Decision: rejected.** Foursquare's gain came from groups that really are one place; here the groups mix siblings.

### E2 — test-like benches (G0), 05:50–06:10
`world5_t2`: 21% (US) / 22% (India) of non-dev S1 removed → 5.78 / 5.82 records per S1 (test: 5.76 / 5.82).
`world5_t3`: 35% removed → 6.8 records per S1 (harsher than test).
B0 (v3-style models, no retrain): t2 **0.9894** (best threshold 0.60 for both countries; train-density dev picked 0.40),
t3 **0.9873** (best 0.7–0.8). → The test-like bench reproduces the leaderboard's preference for stricter thresholds (G0/G8 ✅).
Bench limit: only 0.0020 is lost before the judge vs 0.0095 on full dev (5% of S1 → 20× fewer namesakes), so recall (G2)
is judged from the anatomy of real misses plus a full-scale benchmark, not from this bench's score.

### E3 — recall anatomy (G2/G6), 06:00
259 shortlist misses on t2: 74% India; 26% Indian script, 32% no address, 29% share no word with the owner.
A **character 3-gram TF-IDF** on the cleaned name finds the owner in the top 10 for 50% of them (top 20: 56%;
+ address words: 53% / 61%). Research: char n-gram blocking tolerates typos and transliteration
([survey](https://arxiv.org/pdf/1905.06167), [2202.12521](https://arxiv.org/pdf/2202.12521)). → arm A5 + full-scale benchmark.

### E4 — arms (threshold per country chosen on t2, reused on t3; paired bootstrap vs B0, 11,055 dev S1)
| arm | contents | cands/S1 (t2) | t2 | t3 | Δ t2 [95% CI] | Δ t3 [95% CI] |
|---|---|---|---|---|---|---|
| B0 | v3-style (binary ranker top-10, ambig) | 4.76 | 0.9894 | 0.9873 | — | — |
| A1 | + holdout 0.21, bug #5, candidate budget 5.5 | 4.06 | 0.9894 | 0.9885 | +0.0000 [−0.0008, +0.0009] | **+0.0011 [+0.0003, +0.0020]** |
| A2 | A1 + sibling features | 4.06 | **0.9906** | **0.9896** | **+0.0013 [+0.0003, +0.0021]** | **+0.0023 [+0.0012, +0.0034]** |
| A3 | A2 + distractor weight 2 | 4.06 | 0.9901 | 0.9892 | +0.0007 [−0.0003, +0.0016] | +0.0019 [+0.0008, +0.0030] |
Decision: holdout + budget + sibling features **kept** (G1-1, G1-2, G7). Distractor weight 2 **rejected** (below A2 on both
benches; holdout already supplies the decoy pressure).
| A4 | A2 + one judge per country (pooled judge kept for France) | 4.06 | 0.9902 | 0.9891 | +0.0009 [−0.0002, +0.0018] | +0.0018 [+0.0008, +0.0028] |
G5 decision: **one pooled judge + per-country thresholds**. Per-country judges lose (India 0.9866 vs 0.9874 pooled):
the countries share most generator operators, so pooling gives each more evidence.
| A5 | A2 + char 3-gram retrieval pass | 3.97 | 0.9893 | 0.9884 | −0.0000 [−0.0010, +0.0009] | +0.0010 [+0.0000, +0.0021] |
A5 on the 5% bench: worse than A2 (recall reaching the judge 0.9896 vs 0.9909 — extra char candidates compete for the
same budgeted slots; this bench has 20× fewer namesakes, so its recall is already easy). Decision pending the
full-scale benchmark (E8).

### E5 — what A2 still loses on the test-like world (t2, thresholds 0.5/0.5): 0.0094
| error | loss | main pattern |
|---|---|---|
| wrong merges | 0.0030 | 57 of 116 have no address (name only), e.g. `Vanguard Resource` (no address) → S1 `Vanguard Resource Inc` (decoy) |
| judge misses | 0.0035 | 237 of 436 have no address, e.g. `JO JERRY HIGHLAND BIOPHARMACEUTICALS INC` (no address) — true |
| shortlist misses | 0.0029 | garbled names (`Dental Cennerc`, `IHM Priobreits`), script names → char pass (A5) |
No-address records are the largest residual on BOTH sides: with a name only, a true copy and a decoy copy look alike;
the probability threshold already takes the best expected decision there. Recorded as a known limit.

### Engineering (G8, G9)
- G8: the test step now uses the **per-country thresholds chosen on the (holdout = test-like) dev**; unseen countries
  (France) get the strictest of them (leaderboard: stricter France +0.004).
- G9: bug #5 fixed (holdout orphans now count as distractors in training weights); house-number ties deterministic
  (identical output under different hash seeds); `--legacy-cut` reproduces v3's stage 1 exactly when rescoring with v3
  models. 64 unit tests pass.
- G9 (found while testing self-training): the new sibling feature picked among tied words in Python's random set
  order → tiny run-to-run probability differences. Fixed (sorted); the test step now gives **identical pairs and max
  |Δp| = 0.0** under two different PYTHONHASHSEEDs.

### E6 — leave-one-country-out rehearsal (G4), 08:07
Judge + ranker trained on **US only** (holdout, sib), scored on world5_t2 where India plays the unseen country:
India **0.9666** (best t = 0.30) vs **0.9874** when India is in training (A2) → **unseen-country penalty ≈ −0.021**
(India: judge misses 0.0106, wrong merges 0.0125). US unchanged (0.9919). France is in this position on test.
Research: self-training / pseudo-labelling for domain adaptation of matchers (confident predictions on the target
domain used as labels). → E7: `--st-countries` continues the pooled judge on the target country's confident decisions
(best p ≥ 0.95 → 1, its rivals → 0; best p ≤ 0.05 → 0) and rescores that country; tested on this rehearsal.

### E7 — self-training for the unseen country (G4), 08:40–08:55
US-trained judge (L1), India as the unseen country on world5_t2; pseudo-labels from the judge's own confident
decisions on India (134k pairs → 115k labelled), extra rounds continue the judge, India rescored (US untouched).
Paired bootstrap on India dev S1 vs L1:
| threshold | 100 rounds | 300 rounds |
|---|---|---|
| 0.3 | +0.0005 [−0.0014, +0.0026] | −0.0004 [−0.0023, +0.0015] |
| 0.5 | +0.0018 [+0.0000, +0.0035] | **+0.0021 [+0.0005, +0.0037]** |
| 0.7 | +0.0044 [+0.0025, +0.0064] | **+0.0049 [+0.0028, +0.0069]** |
Wrong merges 0.0125 → 0.0093. It makes the unseen country far less sensitive to its (untunable) threshold.
**Decision: adopt for France** (`--st-countries France --st-rounds 300`); France is scored at the strictest per-country
threshold, exactly where the gain is largest.

### E8 — char pass at full scale (G2/G6), 08:57
3,692 owned India records vs the FULL India S1 (883,188): word passes 0.9732 → + char 0.9783 (+0.0051; recovers 19% of
word-pass misses), 3.3 ms/record/core → about +2 to 2.5 h on the full Kaggle run.
**Decision: not in v4** (the full arm A5 was not better on the labelled bench; the retrieval gain is before the ranker's
budget and unproven after it; +2.5 h leaves no room to recover before the deadline). Kept behind `AMLC_TF_CHAR=1` for a
reserve run.

## v4 final configuration (09:00)
`--holdout-s1 0.21 --max-cands 5.5 --features sib --st-countries France --st-rounds 300`, `AMLC_TF_CHAR=0`,
pooled judge, per-country thresholds from the test-like dev (France: strictest), France normaliser, stage-1 fix.
Measured on the test-like bench vs v3-style B0: **+0.0013 [+0.0004, +0.0022] (t2), +0.0023 [+0.0012, +0.0033] (t3)**,
15% fewer candidates; France additionally: normaliser fixes (unit-tested) + self-training (+0.002 to +0.005 on the
unseen-country rehearsal).

## Status of every gap
| ID | status |
|---|---|
| G0 dev optimistic | ✅ test-like + harsh benches built; they reproduce the leaderboard's threshold direction; v4's dev runs in the holdout (test-like) world |
| G1 sibling decoys | ✅ holdout + sibling features kept (significant); graph post-processing rejected (purity 0.83, negative); distractor ×2 rejected |
| G2 recall | ◐ stage-1 fix + budget in; char pass measured (+0.5% India retrieval) but not in v4 (see E8) |
| G3 judge misses | ◐ house distance features in (sib); no-address records remain the main residual (E5) |
| G4 France | ✅ normaliser fixes + self-training; LOCO penalty measured (−0.021 for an unseen country) |
| G5 country-wise | ✅ tested: per-country judges lose; pooled judge + per-country thresholds kept |
| G6 language/script | ◐ xlit dictionary (v3); char pass available; scripts are 26% of remaining shortlist misses |
| G7 candidate size | ✅ budget: 4.06 vs 4.76 per S1 on the bench |
| G8 thresholds | ✅ per-country from test-like dev in the test step; France strictest |
| G9 engineering | ✅ bug #5, deterministic ties (norm + sibling), legacy cut, 64 tests, fresh-clone end-to-end PASS |
| G10 deliverables | ✅ zip builder (tracked code only), README, pinned requirements, documentation draft |
