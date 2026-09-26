"""Resumable end-to-end runner for judge v2. Every step writes into --out and leaves a DONE marker; a restart
skips finished steps. Logs are kept in <out>/logs/<step>.log. Designed for Kaggle "Save & Run All (Commit)",
where everything under /kaggle/working is saved as the notebook's output.

  python -u scripts/run_v2.py --data <dataset dir> --out /kaggle/working/v2 [--work-in <prepared dir>]
         [--stop-after rank_eval] [--threads 4]

Layout of --out: work/ (cleaned parquet, unless --work-in), models/, logs/, output/ (submission files),
dev_pairs_v2.parquet, DONE_<step> markers, SUMMARY.txt (key lines from every step).
"""
import argparse
import subprocess
import sys
from pathlib import Path

STEPS = ["prepare", "rank_train", "rank_eval", "train", "dev", "test", "validate"]

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="dataset dir containing train/ and test/")
p.add_argument("--out", required=True)
p.add_argument("--work-in", default=None, help="already-prepared parquet dir (skips prepare)")
p.add_argument("--stop-after", default=None, choices=STEPS)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--sample", type=float, default=None)
a = p.parse_args()

out = Path(a.out)
for d in ("models", "logs", "output"):
    (out / d).mkdir(parents=True, exist_ok=True)
here = Path(__file__).parent
work = a.work_in or str(out / "work")
common = ["--work", work, "--rank-model", str(out / "models/rank_v2.txt"), "--model", str(out / "models/judge_v2.txt"),
          "--threads", str(a.threads)] + (["--sample", str(a.sample)] if a.sample else [])
validator = Path(a.data).parent / "utils" / "validate_submission.py"
cmds = {
    "prepare": [str(here / "prepare.py"), "--data", a.data, "--out", work, "--workers", str(a.threads)],
    "rank_train": [str(here / "judge_v2.py"), "rank_train"] + common,
    "rank_eval": [str(here / "judge_v2.py"), "rank_eval"] + common,
    "train": [str(here / "judge_v2.py"), "train"] + common,
    "dev": [str(here / "judge_v2.py"), "dev"] + common + ["--dev-pairs-out", str(out / "dev_pairs_v2.parquet")],
    "test": [str(here / "judge_v2.py"), "test"] + common + ["--out", str(out / "output")],
    "validate": [str(validator), "--matching", str(out / "output/matching_results.tsv"),
                 "--candidate", str(out / "output/candidate_pairs.tsv"), "--test-dir", str(Path(a.data) / "test")],
}
for step in STEPS:
    if step == "prepare" and a.work_in:
        continue
    done = out / f"DONE_{step}"
    if done.exists():
        print(f"== {step}: already done, skipping", flush=True)
    else:
        print(f"== {step}: running", flush=True)
        log = open(out / "logs" / f"{step}.log", "w")
        proc = subprocess.Popen([sys.executable, "-u"] + cmds[step], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        keep = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if not line.startswith("  chunk") and "Installing" not in line:   # summary: everything but progress
                keep.append(line)
        rc = proc.wait()
        log.close()
        with open(out / "SUMMARY.txt", "a") as s:
            s.write(f"\n===== {step} (exit code {rc})\n" + "".join(keep))
        if rc != 0:
            sys.exit(f"step {step} failed (exit code {rc}); see {out}/logs/{step}.log. Re-run to resume.")
        done.touch()
    if step == a.stop_after:
        print(f"== stopped after {step} as requested", flush=True)
        break
print(f"== all requested steps done; outputs in {out}", flush=True)
