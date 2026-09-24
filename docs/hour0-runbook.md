# Hour-0 runbook

**Hackathon:** 72 hours, starting 12:00 on 2026-09-24. **Submission budget: 5 per day, about 15 in total.** Every upload has to earn its place.

Follow these steps in order. Each has a check: if the check fails, stop and fix it before moving on.

## Minute 0–15: read, don't code (everyone)

- [ ] Read the new README / problem statement **out loud together**. Note:
    - [ ] Task type (regression / extraction / classification) and **the exact metric formula**
    - [ ] Column names in train, test and sample output
    - [ ] Model rules (2025: ≤ 8B parameters, MIT/Apache licence) and the external-data rule
    - [ ] How the public leaderboard is scored (2025: 25k of the 75k test rows) and what the write-up needs
- [ ] Pick the matching row of the metric playbook in [`decision-review.md`](decision-review.md#metric-playbook)
- [ ] Unzip the dataset to `dataset/` in the repo root (it's git-ignored, like `data/`)

## Minute 15–30: start the long job, then check the data (Vaibhav)

- [ ] **Start the image download in the background first.** It's the longest job (2025: about 3–4 h for 150k images).
    ```
    nohup uv run python scripts/download_images.py --data dataset --workers 48 > download.log 2>&1 &
    ```
    Check: `tail download.log` shows progress. If it crashes, run the same command again; it resumes.
- [ ] Run the day-1 checks: `uv run python experiments/02_eda.py dataset`
    (it assumes 2025-style columns; adjust the field names if the new schema differs)
    Check: does train/test overlap look random (IDs interleaved, most brands shared)? If yes, use `--mode stratified`. If test brands are mostly unseen, use `--mode group`.

## Minute 30–60: first valid submission

- [ ] If the metric is not SMAPE: add it to `src/amlc/metrics.py` **with tests written from the rules text first**, and switch the script to use it.
- [ ] Run the baseline:
    ```
    uv run python scripts/hour0_baseline.py --data dataset --mode stratified
    ```
    Check: writes `folds.csv`, `preds/*_oof.parquet`, `submissions/median.csv`, `submissions/tfidf_ridge.csv`, and prints a CV score for each.
- [ ] **Upload #1: `submissions/tfidf_ridge.csv`.** The validator has already checked it, so it can't be rejected for format.
- [ ] Log it in the submission log below. Compare CV with the public LB. They should be within a few points; if not, the folds don't match the test, so revisit the fold mode before building anything else.

## After that

Follow the timeline in [`decision-review.md`](decision-review.md#roles-and-timeline). **Rules for the 15 uploads:**

- Upload only when CV improved by a meaningful amount (≥ 0.3 SMAPE points) over the last upload.
- Keep 2 uploads for the last day, for the final blend.
- Choose the final submission by **CV**, not public LB (public = only a third of test).

## Submission log

| # | Time | File | CV | Public LB | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | | | | | |

## 2025 dry-run reference numbers

| What | 2025 result |
| --- | --- |
| Median-only CV SMAPE | about 72.7 |
| TF-IDF + Ridge CV SMAPE | **52.03** (5.5 min on the laptop, peak 2.1 GB RAM) |
| Image download | 300 images in 25 s with 32 workers (about 12/s); about 50 KB each at 512 px |
