# Work Split

_As of 2026-09-24_

> Revised roles and timeline are in [decision-review.md](decision-review.md#roles-and-timeline).

## Roles

| Person | Role |
| --- | --- |
| Vaibhav (lead) | Pipeline & infra (loader, image-fetch cache, submission validator), final architecture calls, ensembling |
| Teammate 2 | Modeling end-to-end — gets a baseline scored and iterating the moment features exist |
| Krithi | EDA first pass + writeup, updated incrementally as decisions get made |

## Phase timeline

- Hour 0–3: Vaibhav builds loader + submission validator; teammate 2 runs first EDA; Krithi starts the writeup skeleton
- Hour 3–15: split feature/modeling work between Vaibhav and teammate 2, once EDA shows which modality (text vs image) matters more
- Hour 15–50: iterate — feature engineering, model tuning, validation checks
- Hour 50–65: ensembling, post-processing, final CV sanity checks
- Hour 65–72: lock submission, finish writeup, buffer for last-minute issues

## Open decisions

- Text vs. image split between Vaibhav and teammate 2 — decide once EDA shows which modality carries more signal
  - **Resolved:** split by role (features vs models), not modality — see decision-review.md.
- Tie-breaker on architecture disagreements under time pressure: Vaibhav has final call as team lead
- Which modeling library to standardize on (LightGBM vs CatBoost) — decide after seeing the target type
  - **Resolved:** both; LightGBM first, CatBoost as a blend member — see decision-review.md row 7.
