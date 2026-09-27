# v4 runbook — Kaggle cells (verified end to end from a fresh clone on a small real dataset)

**Kaggle settings:** CPU notebook (4 cores, ~30 GB RAM), Internet ON (for git clone / pip), dataset `dataset-2026` attached.
**Expected time:** about 6–7 h (v3 took 6.3 h). Start it, then **Save Version → Save & Run All (Commit)**
so everything under `/kaggle/working` survives as the version's Output.

## Cell 1 — setup
```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
%cd /kaggle/working
!git clone -q https://{tok}@github.com/Vaibhavtripathi7/amazon-ml-challenge-2026.git
%cd /kaggle/working/amazon-ml-challenge-2026
!git log --oneline -1
!pip install -q --no-deps -e . && pip install -q anyascii rapidfuzz sparse-dot-topn
```

## Cell 2 — the v4 run (all steps; resumable: re-running skips finished steps)
```python
import os
os.environ["AMLC_TF_CHAR"] = "0"   # char 3-gram retrieval pass: OFF in v4 (E8); "1" only for a reserve run
DATA = "/kaggle/input/datasets/vaibhavtripathi072/dataset-2026/data-2026/student_resource/student_resource/dataset"
!python -u scripts/run_v2.py --data {DATA} --out /kaggle/working/v4 --threads 4 \
    --judge-args="--holdout-s1 0.21 --max-cands 5.5 --features sib --st-countries France --st-rounds 300"
!tail -60 /kaggle/working/v4/SUMMARY.txt
```
What the flags do (each measured, see `docs/v4-gaps.md`):
- `--holdout-s1 0.21`: train/validate in a test-like world (21% of non-dev S1 removed, their records kept as orphans) →
  the judge learns that an owner may be absent; the dev step then picks thresholds at test density.
- `--max-cands 5.5`: candidate budget in rank_eval (smaller candidate set, organisers rank it higher).
- `--features sib`: sibling features (typo vs swapped word, house distance).
- The test step writes the submission with **per-country thresholds from dev**; France gets the strictest.
- `--st-countries France --st-rounds 300`: in the test step the judge adapts to France (never seen in training) with
  its own confident France decisions, then France is rescored (+0.002 to +0.005 on the unseen-country rehearsal, E7).
  US/India scores are untouched. The adapted model is saved as `models/judge_v2.txt.st`.

## Cell 3 — alternative thresholds from the saved pairs (minutes, no re-scoring)
```python
V = "/kaggle/working/v4"
!python scripts/judge_v2.py rethreshold --work {V}/work --pairs-in {V}/output/test_pairs.parquet \
   --out /kaggle/working/v4_strict --by-country US=0.7 India=0.7 France=0.8 default=0.7
!wc -l /kaggle/working/v4*/matching_results.tsv
!cat {V}/models/judge_v2.txt.json
```

## Cell 4 — mistake sheet for the team (dev)
```python
!python -u scripts/mistakes.py --work {V}/work --pairs {V}/dev_pairs_v2.parquet \
   --model-json {V}/models/judge_v2.txt.json --out /kaggle/working/mistakes_v4.md --n 20
```

## Uploads (in order; change one thing at a time)
1. `v4/output/matching_results.tsv` (per-country dev thresholds).
2. If 1 beats v3's 0.957: `v4_strict` (tests whether test wants even stricter than the test-like dev).
3. Keep the best as the final entry; the final zip uses the same output folder.

## Final zip (after choosing the output folder)
```python
!python scripts/make_submission_zip.py --outputs /kaggle/working/v4/output --doc docs/Documentation.md \
   --zip /kaggle/working/submission.zip
```
(Fill the team name / members at the top of `docs/Documentation.md` and the v4 numbers first.)
