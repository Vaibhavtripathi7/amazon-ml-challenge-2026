# Docs — Amazon ML Challenge 2026

Team: Vaibhav (lead), Krithi, Teammate 2. Target: top 10 in the 72-hour window.

| File | What it holds |
| --- | --- |
| [toolkit.md](toolkit.md) | **What code is ready, what each piece does, how to run it** |
| [hour0-runbook.md](hour0-runbook.md) | **Hackathon day: step-by-step from minute 0, submission log.** 72 h from 2026-09-24 12:00, 5 uploads/day |
| [decision-review.md](decision-review.md) | Revised decisions, metric playbook, roles + timeline, pre-drop checklist, open questions. **Start here.** |
| [strategy.md](strategy.md) | Original strategy: metric/transform, features, validation, ensembling |
| [work-split.md](work-split.md) | Original roles, phase timeline, open decisions |
| [architecture.md](architecture.md) | Pipeline modules, data flow, what's built before vs. after the drop |

## Lessons (build the prototype one piece at a time)

Each lesson builds one item of the "Build before the drop" checklist in `decision-review.md`. Lesson 12 is the full timed rehearsal.

| # | Lesson | Builds | Needs GPU | Status |
| --- | --- | --- | --- | --- |
| 1 | [The metric (SMAPE)](lessons/01-metric.md) | `metrics.py` (SMAPE) | No | Done |
| 2 | [Looking at the 2025 data](lessons/02-data.md) | `02_eda.py` (reusable day-1 checks) | No | Done |
| 3 | Loader and fold file | `loader.py`, `folds.csv` (random / stratified / group) | No | Next |
| 4 | First submission | `submit.py` validator + median submission | No | — |
| 5 | First real model | TF-IDF + Ridge baseline; the `preds/` OOF format every model follows | No | — |
| 6 | Reading the text properly | Value/Unit parser, unit normaliser, regex features | No | — |
| 7 | Getting the photos | Resumable, resizing image downloader | No | — |
| 8 | Turning text and photos into numbers | Embedding cache (one text model, one image model) | Helpful | — |
| 9 | Tree model + "similar products" prices | LightGBM on features + kNN price features | No | — |
| 10 | The big model | Fine-tuned text encoder with a price head | **Yes** | — |
| 11 | Mixing models | Blend + scale factor + clipping → final submission | No | — |
| 12 | Hour-0 rehearsal | Full pipeline end to end, timed, plus a 1-page write-up draft | Yes | — |

Where `decision-review.md` conflicts with the other files, it wins.

Constraint: no external data or scraping — provided dataset only.
