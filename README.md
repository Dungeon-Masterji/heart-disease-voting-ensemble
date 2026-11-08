# HeartGuard AI — Multi-Model Diagnosis Support Tool

A 12-week student project building a voting-ensemble clinical
decision-support prototype on the UCI Heart Disease (Cleveland) dataset.
The repository contains the training pipeline, FastAPI backend, React frontend,
Docker runtime, tests, analysis artifacts, and model card. See
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for commands that were
actually run and features that remain unverified.

## ⚠️ Disclaimer

Educational prototype only. Not a medical device, not validated for
clinical use, and (once a model exists) any prediction will be an advisory
risk signal, not a diagnosis, requiring review by a qualified medical
professional.

## Problem statement

Clinical teams need a repeatable, data-driven signal to help prioritize
patients with elevated estimated heart-disease risk, for human-in-the-loop
diagnostic workflows — not to replace clinical judgment.

## Implemented

- **Data ingestion** (`training/src/data/ingestion.py`) — loads the raw CSV
  and enforces the documented column schema.
- **Data validation** (`training/src/data/validation.py`) — checks
  missingness, duplicates, dtypes, and out-of-documented-range values;
  writes a JSON report.
- **EDA** (`training/src/eda/run_eda.py`) — target distribution,
  feature distributions, correlation heatmap, missingness heatmap,
  target-grouped boxplots, and a written findings summary.
- **Tests** (`training/tests/test_data.py`) — 10 passing tests covering
  ingestion and validation against the real 303-row dataset.

Real findings from this dataset (not placeholders) are documented in
[`training/artifacts/eda/eda_summary.md`](training/artifacts/eda/eda_summary.md)
and [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — for example: a
1.2:1 class balance, 1 duplicate row, and 7 rows with out-of-documented-range
`ca`/`thal` codes likely corresponding to the original dataset's missing
values.

## Known gaps

Authentication, live deployment, deploy workflow, drift monitoring, distributed
rate limiting, and batch scoring are not implemented. These are documented,
not implied by this README.

## Repository structure

```
heart-disease-voting-ensemble/
├── docs/
│   ├── DATA_DICTIONARY.md      # dataset provenance, schema, known issues
│   ├── PROJECT_STATUS.md       # done vs. planned, mapped to the PRD
│   └── adr/                    # architecture decision records made so far
├── data/
│   ├── raw/                    # fetched via scripts/download_data.py (gitignored)
│   └── processed/              # reserved for Week 4 (gitignored)
├── training/
│   ├── src/
│   │   ├── data/                # schema.py, ingestion.py, validation.py
│   │   └── eda/                 # run_eda.py
│   ├── tests/                   # test_data.py
│   ├── artifacts/eda/           # generated plots + eda_summary.md
│   └── requirements.txt
├── scripts/
│   └── download_data.py
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites
- Python 3.11+

### 1. Set up the environment
```bash
python -m venv venv && source venv/bin/activate
pip install -r training/requirements.txt
```

### 2. Get the dataset
```bash
python scripts/download_data.py
```
This fetches a 303-row, 14-column recoded CSV of the UCI Cleveland Heart
Disease data. See `docs/DATA_DICTIONARY.md` for exactly what "recoded"
means and why.

### 3. Run ingestion, validation, and EDA
```bash
python -m training.src.data.ingestion
python -m training.src.data.validation
python -m training.src.eda.run_eda
```
Validation prints a summary and writes `training/artifacts/validation_report.json`.
EDA writes plots and `eda_summary.md` to `training/artifacts/eda/`.

### 4. Run the tests
```bash
pytest training/tests/ -v
```

## Tech stack

| Layer | Technologies |
|---|---|
| Data / EDA / training | Python 3.13, pandas, NumPy, SciPy, scikit-learn |
| API | FastAPI, Python 3.13 |
| Frontend | React, TypeScript, Vite |
| Testing / quality | pytest, Vitest, Ruff, Black, mypy, Bandit |

## Roadmap

| Phase | Weeks | Status |
|---|---|---|
| Research and requirements | 1–2 | Implemented/documented |
| Data and EDA | 3 | Implemented and tested |
| Features | 4 | Implemented and tested |
| Baseline | 5 | Implemented |
| Ensemble and tuning | 6–7 | Implemented and trained |
| Evaluation | 8 | Implemented and trained |
| API (FastAPI) | 9 | Implemented |
| Frontend (React) | 10 | Implemented |
| Integration and Docker | 11 | Implemented |
| Finalization and model card | 12 | Implemented |

## License

MIT — see [LICENSE](LICENSE).
