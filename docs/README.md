# Docs — Amazon ML Challenge 2026

Team: Vaibhav (lead), Krithi, Teammate 2. Target: top 10 in the 72-hour window.

| File | What it holds |
| --- | --- |
| [decision-review.md](decision-review.md) | Revised decisions, metric playbook, roles + timeline, pre-drop checklist, open questions. **Start here.** |
| [strategy.md](strategy.md) | Original strategy: metric/transform, features, validation, ensembling |
| [work-split.md](work-split.md) | Original roles, phase timeline, open decisions |
| [architecture.md](architecture.md) | Pipeline modules, data flow, what's built before vs. after the drop |

## Lessons (build the prototype one piece at a time)

| # | Lesson | Status |
| --- | --- | --- |
| 1 | [The metric (SMAPE)](lessons/01-metric.md) | Done |
| 2 | [Looking at the 2025 data](lessons/02-data.md) | Done |
| 3 | Loader and fold file | Next |
| 4 | Median submission + validator | — |
| 5 | TF-IDF + Ridge baseline | — |
| 6 | Features, embeddings, nearest-neighbour prices, blend | — |

Where `decision-review.md` conflicts with the other files, it wins.

Constraint: no external data or scraping — provided dataset only.
