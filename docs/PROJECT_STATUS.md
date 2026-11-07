# Project Status

Tracks what is built and how it was verified, mapped to the 12-week PRD plan.
Update the "Verified" column only after you have actually run the check.

| Week | Deliverable | Where | Verified by |
|---|---|---|---|
| 1-2 | Research, requirements, ADRs | `docs/` | Not re-run in this implementation session |
| 3 | Ingestion, validation, EDA | `training/src/data`, `training/src/eda` | `source venv/bin/activate && python -m pytest training/tests -q` (23 passed) |
| 4 | Leakage-safe preprocessing | `training/src/features/preprocessing.py` | Included in the training test run above |
| 5 | Baseline (dummy, logistic regression) | `training/src/models/baseline.py`, training artifacts | Included in quick/full training runs |
| 6 | Voting Classifier (hard/soft) | `training/src/models/voting_ensemble.py`, training artifacts | `source venv/bin/activate && python -m training.src.models.train --quick` |
| 7 | Nested-CV tuning | `training/src/models/tuning.py` | Quick and full training commands completed |
| 8 | Threshold, evaluation, error analysis, model artifact | `training/src/models/`, `training/artifacts/models/` | Quick/full training completed; full run reported F1 0.871 |
| 9 | FastAPI service | `backend/app` | Backend tests, `uvicorn`, smoke test, and Docker smoke were run in earlier project verification |
| 10 | React UI | `frontend/` | Typecheck, 15 tests, and production build were run in earlier project verification |
| 11 | Integration, CI, Docker | `.github/workflows/ci.yml`, `backend/Dockerfile` | Quality gates and Docker image/container smoke were run in earlier project verification |
| 12 | Model card, README, demo | `scripts/generate_model_card.py`, `docs/MODEL_CARD.md` | `python scripts/generate_model_card.py` (regenerated in this chunk) |

## Known limitations (be upfront in the viva)
- Tuned ensemble ~ logistic regression within noise on 302 patients.
- 1980s single-site data; descriptive subgroup analysis exists, but small samples limit conclusions; no external validation.
- No drift/calibration monitoring, no authentication, no deployment to a live
  host, no `deploy.yml` workflow. Rate limiter is per-process.
- Batch scoring is not implemented.
