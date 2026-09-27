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
Decision: holdout + budget + sibling features **kept** (G1-1, G1-2, G7).

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
