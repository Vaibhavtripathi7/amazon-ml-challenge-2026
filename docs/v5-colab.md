# v5-CE on Colab — cross-encoder re-scoring (GPU), step by step

Design and evidence: `scripts/ce_rerank.py` docstring and `docs/v4-gaps.md`. Smoke-tested end to end on real data
(tiny random model, CPU): build → train → eval (blend) → apply → both TSVs written.

**Runtime:** GPU (T4 is enough; L4/A100 faster). Every output goes to Google Drive, so a disconnect only loses the
current step — re-run the same cell to resume.

## What to put in Google Drive first (folder `MyDrive/amlc/`)
- `dev_pairs_v3.parquet` — from the v3 Kaggle notebook output: `v3/dev_pairs_v2.parquet` (rename).
- `test_pairs_v3.parquet` — `v3/output/test_pairs.parquet` (you have it in ~/Downloads/test_pairs.parquet).
- `kaggle.json` — your Kaggle API token (Kaggle → Settings → API → Create New Token), to download the dataset.

## Cell 1 — setup (GitHub token as a Colab secret named GITHUB_TOKEN)
```python
from google.colab import drive, userdata
drive.mount('/content/drive')
tok = userdata.get('GITHUB_TOKEN')
%cd /content
!git clone -q https://{tok}@github.com/Vaibhavtripathi7/amazon-ml-challenge-2026.git
%cd /content/amazon-ml-challenge-2026
!git log --oneline -1
!pip install -q --no-deps -e . && pip install -q anyascii rapidfuzz sparse-dot-topn transformers
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

## Cell 2 — data (Kaggle API → /content/data)
```python
!mkdir -p ~/.kaggle && cp /content/drive/MyDrive/amlc/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d vaibhavtripathi072/dataset-2026 -p /content/data --unzip -q
import glob; DATA = glob.glob("/content/data/**/dataset/test/test_source1.tsv", recursive=True)[0].rsplit("/test/", 1)[0]
print(DATA)
G = "/content/drive/MyDrive/amlc"; CE = f"{G}/ce"
```

## Cell 3 — build training pairs (dev S1 only: exact labels; 80/20 split by S1)
```python
!python -u scripts/ce_rerank.py build --pairs {G}/dev_pairs_v3.parquet --data {DATA} --out {CE}/train.parquet --easy-frac 0.3
```

## Cell 4 — train the cross-encoder (XLM-R base, 2 epochs)
```python
!python -u scripts/ce_rerank.py train --in {CE}/train.parquet --model-dir {CE}/model --epochs 2 --bs 32 --max-len 160
```

## Cell 5 — validate on held-out dev S1 (judge vs CE vs blend, per country) → blend.json
```python
!python -u scripts/ce_rerank.py eval --in {CE}/train.parquet --model-dir {CE}/model --bs-pred 256 --max-len 160
```
**Send me this output.** Decision rule: use v5-CE only if the blend beats the judge on both countries.

## Cell 6 — unseen-country rehearsal (the France question): train on US only, score India
```python
!python -u scripts/ce_rerank.py train --in {CE}/train.parquet --model-dir {CE}/model_us --train-countries US --epochs 2 --bs 32
!python -u scripts/ce_rerank.py eval --in {CE}/train.parquet --model-dir {CE}/model_us --all-pairs --bs-pred 256
```
Compare the **India** row of `ce` (a model that never saw India) with the `prob` row (our judge, which did see India).
If the CE is close, it transfers to unseen countries → it is the France fix.

## Cell 7 — apply to test (re-scores the uncertain band + every France pair), write the submission
```python
!python -u scripts/ce_rerank.py apply --test-pairs {G}/test_pairs_v3.parquet --data {DATA} --model-dir {CE}/model \
    --out {CE}/output --bs-pred 256 --max-len 160
!wc -l {CE}/output/matching_results.tsv && python {DATA}/../utils/validate_submission.py \
    --matching {CE}/output/matching_results.tsv --candidate {CE}/output/candidate_pairs.tsv --test-dir {DATA}/test
```
Upload `MyDrive/amlc/ce/output/matching_results.tsv`.
