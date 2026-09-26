# Amazon ML Challenge 2026 — Business Entity Resolution

Team: Vaibhav (lead), Krithi, Teammate 2. Hackathon: 72 h from 2026-09-25 12:00. Uploads: 5 per day.

Docs: [v3-plan.md](v3-plan.md) (**current plan to 0.99+**) · [approach.md](approach.md) (**research + chosen architecture, for review**) · [lessons/01-problem-and-data.md](lessons/01-problem-and-data.md) (**start here**, plain-words intro) · [analysis.md](analysis.md) (what the data shows) · [plan.md](plan.md) (pipeline, timeline, roles, open questions, submission log)

Official files: `data-2026/student_resource/student_resource/` (README, `utils/validate_submission.py`, `Documentation_template.md`, `dataset/`). The data is not in git.

## The task (from the official README)

- **3 sources** of business records, with columns `entity_id`, `business_name`, `business_address`, `country`. The source is given by the ID prefix: `S1-`, `S2-` or `S3-`.
- **Source 1** is the deduplicated reference. For **every Source 1 entity**, find all Source 2 and Source 3 records that are the same real-world business. That can be zero, one or many.
- **Noise:** name abbreviations, legal suffixes, DBA names, `&` vs "and", word order, typos, transliterations (Hindi script seen). Addresses: Rd/Road, missing PIN or state, landmark references ("Near SBI ATM"), reordered parts.
- **Countries:** train has `US` and `India`. **Test also has `France`, which isn't in train.** Don't hard-code the country list.

## Data size

| Split | Source 1 | Source 2 | Source 3 | Labels |
| --- | --- | --- | --- | --- |
| Train | 2.2M | 5.0M | 5.3M | `train_ground_truth.tsv`: one row per S1 entity, comma-separated matches (empty = no match) |
| Test | 1.7M | 4.9M | 5.1M | — |

All files are **tab-separated** (`sep="\t"`). About 2.4 GB in total.

## Metric

**F0.5, macro-averaged per Source 1 entity** (precision counts 2× recall):

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed per S1 entity, then averaged over all S1 entities.
- An S1 entity with no true matches scores **1.0** for an empty prediction and **0.0** for any prediction.
- Public LB uses a subset of test; **final ranking uses the private remainder**.

## What we submit

| File | Scored | Content |
| --- | --- | --- |
| `output/matching_results.tsv` | **Yes** (upload) | Columns `source1_entity_id`, `matched_entity_ids`: one row per test S1 entity |
| `output/candidate_pairs.tsv` | No (required in zip) | Columns `source1_entity_id`, `candidate_entity_ids`: the exact candidate set the matching model scored. Final matches must be a subset |
| Final zip | Top teams reviewed | `output/` + `code/business_entity_resolution/` (`src/`, `README.md`, pinned `requirements.txt`) + filled `Documentation_template.md` |

Check before every upload (run from `student_resource/`):

```
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Organiser update (2026-09-26): candidate set size counts

- `candidate_pairs.tsv` and **the code that produces it** are reviewed for the **final ranking**, "alongside" the score.
- **"The approach that generates a smaller candidate set per Source 1 entity will be ranked higher"**, beyond the public and private leaderboards. The weighting is not stated.
- Our target: a top score **and** a small candidate set (a few per S1; true matches average about 3.5). Measure recall vs candidates per S1.

## Rules

- Models: at most 8B parameters, MIT or Apache 2.0 licence.
- **No external lookups:** no entity-resolution APIs, no business registries, no geocoding APIs, no internet data.

## Our setup

- Python 3.12 via uv: `uv sync`, then `uv run ...`. This laptop has 5.7 GB of RAM and no GPU.
- `src/amlc/`: our package (empty, fresh start).
- `docs/archive-2025/`: the 2025 price-challenge prep (lessons, plans, experiments), kept for reference. The code from that prep is in git tag `prep-2025-price` (`git checkout prep-2025-price -- <path>` restores any file).
