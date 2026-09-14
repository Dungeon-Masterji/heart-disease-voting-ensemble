# Project Status

Full requirements live in the PRD (`Multi-Model Diagnosis Support Tool —
PRD`, Polaris Product Company Standard, 12-week Advanced Data Science
track). This file tracks what has actually been **built and verified** so
far against that plan, so the README never has to overstate progress.

## Done (Weeks 1–3)

| Week | PRD Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Research grounding (JAIR 1999 ensemble paper) + dataset study | Done | `docs/DATA_DICTIONARY.md` |
| 2 | Requirements docs, repo scaffold | Done | this `docs/` tree, `docs/adr/` |
| 3 | Data ingestion | Done | `training/src/data/ingestion.py`, `training/tests/test_data.py` |
| 3 | Data validation (missingness, duplicates, dtypes, ranges) | Done | `training/src/data/validation.py`, `training/artifacts/validation_report.json` |
| 3 | EDA (distributions, correlations, target analysis) | Done | `training/src/eda/run_eda.py`, `training/artifacts/eda/` |

All of the above run end-to-end with real output committed under
`training/artifacts/` (generated from the actual 303-row dataset — see
`docs/DATA_DICTIONARY.md`), and `training/tests/test_data.py` passes
against that data.

## Not started yet (Weeks 4–12)

Feature engineering, baseline model, Voting Classifier (hard/soft voting),
hyperparameter tuning, stratified 5-fold CV, threshold selection, FastAPI
backend, React frontend, model card, and deployment are all **future work**
per the PRD roadmap (§17) and are not represented by any code in this
repository yet. They will be added incrementally, week by week, with their
own tests and documentation — not described in the README until they exist.
