"""Build the final submission zip in the official layout.

python scripts/make_submission_zip.py --outputs <dir with matching_results.tsv + candidate_pairs.tsv> \
    --doc docs/Documentation.md --zip submission.zip

Layout (official README):
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/src/   (the amlc package + the pipeline scripts)
  code/business_entity_resolution/README.md, requirements.txt (pinned)
  Documentation_template.md (filled)
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["prepare.py", "judge_v2.py", "run_v2.py", "mistakes.py", "ce_rerank.py", "synth_pairs.py"]
REQUIREMENTS = """anyascii==0.3.3
lightgbm==4.7.0
numpy==2.5.3
pandas==3.0.6
pyarrow==25.0.1
rapidfuzz==3.14.6
scikit-learn==1.9.1
scipy==1.18.1
sparse-dot-topn==1.2.0
tqdm==4.70.1
torch==2.14.0
transformers==5.17.0
"""
README = """# Business Entity Resolution — code

Python 3.12. CPU only; no external data, APIs or pretrained models.

## Install
    pip install -r requirements.txt
    export PYTHONPATH=src            # the `amlc` package lives in src/amlc

## Reproduce output/matching_results.tsv and output/candidate_pairs.tsv
    python -u src/run_v2.py --data <dataset dir containing train/ and test/> --out run --threads 4 \\
        --judge-args="--holdout-s1 0.21 --max-cands 5.5 --features sib"

Steps (resumable; each writes a DONE marker): prepare (clean + parse every row once) -> rank_train
(stage-1 ranker) -> rank_eval (candidate budget) -> train (stage-2 judge) -> dev (held-out validation,
threshold per country) -> test (writes run/output/*.tsv and the scored pairs) -> validate.

Stage 3 (GPU, e.g. one T4): multilingual cross-encoder (xlm-roberta-base, MIT licence) re-scores the uncertain pairs
and is blended with the LightGBM probability (weights fitted on held-out dev businesses):
    python -u src/ce_rerank.py build --pairs run/dev_pairs_v2.parquet --data <dataset dir> --out ce/train.parquet
    python -u src/ce_rerank.py train --in ce/train.parquet --model-dir ce/model --epochs 2 --bs 32 --max-len 160 --max-pairs 600000
    python -u src/ce_rerank.py train --in ce/train.parquet --model-dir ce/model_us --train-countries US --epochs 1 --max-pairs 150000
    python -u src/ce_rerank.py eval  --in ce/train.parquet --model-dir ce/model_us --all-pairs
    python -u src/ce_rerank.py eval  --in ce/train.parquet --model-dir ce/model --lo 0.01 --hi 0.99 --tag _b
    python -u src/ce_rerank.py apply --test-pairs run/output/test_pairs.parquet --data <dataset dir> --model-dir ce/model \
        --tag _b --unseen-blend ce/model_us/blend.json --unseen-band-only --unseen-threshold 0.7 --out output

A threshold can be changed afterwards without re-scoring:
    python -u src/judge_v2.py rethreshold --work run/work --pairs-in run/output/test_pairs.parquet \\
        --out run/output_t --by-country US=0.6 India=0.6 France=0.7 default=0.6

## Layout
- src/amlc/norm.py        stage 0: transliteration, name/address parsing (country-aware; France rules)
- src/amlc/keys.py, blocking.py, stage.py   stage 1: exact keys + TF-IDF retrieval + LightGBM ranker
- src/amlc/judge_features.py                stage 2: pair features for the LightGBM judge
- src/amlc/metric.py      macro F0.5 per Source-1 entity
- src/*.py                pipeline entry points (prepare, judge_v2, run_v2) and error analysis (mistakes)
"""

p = argparse.ArgumentParser()
p.add_argument("--outputs", required=True)
p.add_argument("--doc", default="docs/Documentation.md")
p.add_argument("--zip", default="submission.zip")
a = p.parse_args()
out = Path(a.outputs)
with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(out / f, f"output/{f}")
    base = "code/business_entity_resolution"
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "src/amlc"], capture_output=True, text=True,
                             check=True).stdout.split()          # only committed code, never local scratch files
    for rel in tracked:
        z.write(ROOT / rel, f"{base}/src/amlc/{Path(rel).relative_to('src/amlc')}")
    for s in SCRIPTS:
        z.write(ROOT / "scripts" / s, f"{base}/src/{s}")
    z.writestr(f"{base}/README.md", README)
    z.writestr(f"{base}/requirements.txt", REQUIREMENTS)
    z.write(ROOT / a.doc, "Documentation_template.md")   # official file name
print(f"wrote {a.zip}")
with zipfile.ZipFile(a.zip) as z:
    for i in z.infolist():
        print(f"  {i.file_size:>12,}  {i.filename}")
