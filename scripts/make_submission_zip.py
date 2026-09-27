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
SCRIPTS = ["prepare.py", "judge_v2.py", "run_v2.py", "mistakes.py", "ce_rerank.py", "synth_pairs.py", "final_thresholds.py"]
V3_COMMIT = "5d41c82"   # exact code of the stage 0-2 run behind the submitted file (v3)
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

Python 3.11/3.12. Stages 0–2 run on CPU (~6 h on 4 cores); stage 3 on one GPU (a T4 is enough).
No external data, APIs or lookups. The only pretrained model is xlm-roberta-base (MIT licence, 278M parameters).

## Layout
- `src_v3/`  — the exact stage 0–2 code (commit 5d41c82) that produced the submitted candidate set and judge scores.
- `src/`     — the latest code: same pipeline plus later improvements (France normaliser, holdout training,
               sibling features, candidate budget), and stage 3 (`ce_rerank.py`), `final_thresholds.py`.

## Reproduce output/matching_results.tsv and output/candidate_pairs.tsv
    pip install -r requirements.txt

    # 1) stages 0-2 (normalise -> retrieve + LightGBM ranker -> LightGBM judge), exact submitted run
    cd src_v3 && PYTHONPATH=. python -u run_v2.py --data <dataset dir with train/ and test/> --out ../run --threads 4 && cd ..
    #    -> run/dev_pairs_v2.parquet (held-out dev pairs), run/output/test_pairs.parquet (every scored test pair)

    # 2) stage 3: multilingual cross-encoder, blended with the judge on the uncertain pairs
    export PYTHONPATH=src
    python -u src/ce_rerank.py build --pairs run/dev_pairs_v2.parquet --data <dataset dir> --out ce/train.parquet --easy-frac 0.3
    python -u src/ce_rerank.py train --in ce/train.parquet --model-dir ce/model --epochs 2 --bs 32 --max-len 160 --max-pairs 600000
    python -u src/ce_rerank.py train --in ce/train.parquet --model-dir ce/model_us --train-countries US --epochs 1 --max-pairs 150000
    python -u src/ce_rerank.py eval  --in ce/train.parquet --model-dir ce/model_us --all-pairs
    python -u src/ce_rerank.py eval  --in ce/train.parquet --model-dir ce/model --lo 0.01 --hi 0.99 --tag _b
    python -u src/ce_rerank.py apply --test-pairs run/output/test_pairs.parquet --data <dataset dir> --model-dir ce/model \
        --tag _b --unseen-blend ce/model_us/blend.json --unseen-band-only --unseen-threshold 0.7 --out ce/out

    # 3) final thresholds (see Documentation_template.md for the values used)
    python src/final_thresholds.py --pairs ce/out/test_pairs_blend.parquet --data <dataset dir> --out output \
        --thr India=<t> US=<t> France=<t>
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
    v3_files = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", V3_COMMIT, "src/amlc",
                               "scripts/prepare.py", "scripts/judge_v2.py", "scripts/run_v2.py"],
                              capture_output=True, text=True, check=True).stdout.split()
    for rel in v3_files:                                            # exact v3 code, from git history
        data = subprocess.run(["git", "-C", str(ROOT), "show", f"{V3_COMMIT}:{rel}"], capture_output=True, check=True).stdout
        dest = rel.replace("src/amlc", "amlc").replace("scripts/", "")
        z.writestr(f"{base}/src_v3/{dest}", data)
    z.writestr(f"{base}/README.md", README)
    z.writestr(f"{base}/requirements.txt", REQUIREMENTS)
    z.write(ROOT / a.doc, "Documentation_template.md")   # official file name
print(f"wrote {a.zip}")
with zipfile.ZipFile(a.zip) as z:
    for i in z.infolist():
        print(f"  {i.file_size:>12,}  {i.filename}")
