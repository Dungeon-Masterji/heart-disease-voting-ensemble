# HeartGuard AI — Multi-Model Diagnosis Support Tool

A 12-week student project building a voting-ensemble clinical
decision-support prototype on the UCI Heart Disease (Cleveland) dataset.
**This repository currently reflects Weeks 1–3 of that plan**: research,
requirements, data ingestion, data validation, and exploratory data
analysis. The Voting Classifier, FastAPI backend, and React frontend
described in the full PRD are future work and are not in this repo yet —
see [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for exactly what's
done vs. planned.

## ⚠️ Disclaimer

Educational prototype only. Not a medical device, not validated for
clinical use, and (once a model exists) any prediction will be an advisory
risk signal, not a diagnosis, requiring review by a qualified medical
professional.

## Problem statement

Clinical teams need a repeatable, data-driven signal to help prioritize
patients with elevated estimated heart-disease risk, for human-in-the-loop
diagnostic workflows — not to replace clinical judgment.

## What's actually implemented (Weeks 1–3)

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

## Not yet built

Feature engineering, baseline model, Voting Classifier (hard/soft voting),
hyperparameter tuning, stratified 5-fold CV, threshold selection, FastAPI
API, React UI, and model card are Week 4+ deliverables per the roadmap
below. No API, no server, and no trained model exist in this snapshot.

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

## Tech stack (current scope)

| Layer | Technologies |
|---|---|
| Data / EDA | Python 3.11, pandas, NumPy, matplotlib, seaborn |
| Testing | pytest |

FastAPI, scikit-learn (Voting Classifier), React, and Tailwind will be added
to this table as they're actually implemented in later weeks.

## Roadmap

| Phase | Weeks | Status |
|---|---|---|
| Research | 1 | Done |
| Requirements | 2 | Done |
| Data (ingestion, validation, EDA) | 3 | **Done — this repo** |
| Features | 4 | Not started |
| Baseline | 5 | Not started |
| Ensemble (Voting Classifier) | 6–7 | Not started |
| Evaluation | 8 | Not started |
| API (FastAPI) | 9 | Not started |
| Frontend (React) | 10 | Not started |
| Integration | 11 | Not started |
| Finalization | 12 | Not started |

## License

MIT — see [LICENSE](LICENSE).
