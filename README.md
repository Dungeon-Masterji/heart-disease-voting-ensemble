# HeartGuard AI — Multi-Model Diagnosis Support Tool

> A voting-ensemble clinical decision-support tool that flags high-risk heart-disease cases, built with scikit-learn, FastAPI, and React.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ⚠️ Disclaimer

**This is an educational clinical decision-support prototype — NOT a medical device.**
All predictions are advisory risk signals, not diagnoses. Every output requires review by a qualified medical professional. This project has no FDA/CE or any other regulatory approval and must not be used for real clinical decisions.

---

## 📌 Problem Statement

Clinical teams need a repeatable, data-driven signal to prioritize patients with elevated estimated heart-disease risk. Manual screening is inconsistent and time-consuming. This tool provides a model-driven **risk flag** for human-in-the-loop diagnostic workflows — helping physicians focus attention on high-risk cases without replacing clinical judgment.

## 💡 Solution

A **Voting Classifier ensemble** (hard & soft voting) trained on the UCI Heart Disease dataset, evaluated with **stratified 5-fold cross-validation**, and served through a FastAPI backend with a React clinician UI.

Grounded in *"Popular Ensemble Methods: An Empirical Study"* (Journal of Artificial Intelligence Research, 1999) — combining multiple diverse learners improves generalization over any single model.

## ✨ Features

- **EDA** — distributions, correlations, missingness, and target analysis with visualizations
- **Leakage-safe preprocessing** — all transformations fitted within CV training folds only
- **Baseline model** — Logistic Regression / Dummy classifier for a performance floor
- **Voting Classifier** — hard & soft voting strategies with diverse base estimators (LR, RF, GBM, SVM)
- **Stratified 5-fold CV** — honest, leakage-free model selection and hyperparameter tuning
- **Clinical metrics** — Sensitivity, Specificity, F1-score at the chosen operating threshold
- **Risk flagging** — High Risk / Low Risk tiers with documented clinical rationale
- **FastAPI REST API** — Pydantic validation, health checks, model versioning
- **React clinician UI** — scoring form, risk result display, performance dashboard
- **Model card** — capabilities, limitations, and ethical considerations
- **Experiment tracking** — seeds, configs, metrics, and artifact versioning

## 🏗️ Architecture

```
┌─────────────┐   HTTPS    ┌──────────────┐   ┌──────────────────────┐
│  React +    │ ─────────> │   FastAPI    │──>│  Preprocessing Pipe  │
│ Tailwind UI │   Axios    │  (Uvicorn)   │   │  (fitted at train)   │
└─────────────┘            └──────────────┘   └──────────┬───────────┘
                                                         │
                              ┌──────────────────────────▼───────────┐
                              │   Voting Classifier (hard/soft)      │
                              └──────────────────────────┬───────────┘
                                                         │
                              ┌──────────────────────────▼───────────┐
                              │   Risk-Flag Mapper (threshold)       │
                              └──────────────────────────┬───────────┘
                                                         │
                              JSON: risk_probability, risk_flag,
                              model_version, disclaimer
```

**Training pipeline:** UCI Dataset → Ingestion & Validation → EDA → Feature Engineering → Baseline → Voting Classifier → Hyperparameter Tuning (within CV) → Stratified 5-Fold CV → Threshold Selection → Versioned Artifact Export

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Core ML (Python) | scikit-learn, pandas, NumPy, matplotlib, joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend (minimal JS) | React, Tailwind CSS, Axios |
| Testing | pytest, HTTPX, React Testing Library |
| Deployment | Docker, Uvicorn/Gunicorn |

## 🗂️ Repository Structure

```
heartguard-ai/
├── docs/                  # BRD, PRD, ADRs, architecture diagrams
├── data/                  # Raw & processed data (gitignored)
├── training/
│   ├── src/
│   │   ├── data/          # Ingestion, validation, splits
│   │   ├── eda/           # Distributions, correlations, target analysis
│   │   ├── features/      # Leakage-safe preprocessing & engineering
│   │   ├── models/        # Baseline, voting ensemble, tuning, evaluation
│   │   └── experiments/   # Experiment tracking
│   ├── notebooks/         # EDA & experiment notebooks
│   ├── configs/           # Hyperparameter search spaces
│   └── artifacts/         # Versioned model artifacts (gitignored / DVC)
├── backend/
│   ├── app/
│   │   ├── api/           # health.py, score.py, model.py, model_card.py
│   │   ├── core/          # config, predictor, pipeline
│   │   ├── models/        # Pydantic schemas
│   │   └── utils/
│   ├── tests/             # Unit, integration, API tests
│   └── Dockerfile
├── frontend/
│   └── src/
│       ├── components/    # RiskForm, RiskResult, MetricsDashboard, ModelCard
│       ├── pages/         # ScorePage, DashboardPage, ModelCardPage
│       ├── services/      # api.ts (Axios)
│       └── hooks/         # usePrediction.ts
├── scripts/               # setup.sh, train.sh, download_data.py
└── .github/workflows/     # lint-test.yml, deploy.yml
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Clone & set up environment
```bash
git clone https://github.com/<your-username>/heartguard-ai.git
cd heartguard-ai
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Download the dataset
```bash
python scripts/download_data.py
```

### 3. Train the model
```bash
cd training
python -m src.models.voting_ensemble
```

### 4. Start the API
```bash
cd ../backend
uvicorn app.main:app --reload
```

### 5. Start the frontend
```bash
cd ../frontend
npm install && npm run dev
```

## 🔌 API Reference

Base URL: `/api/v1` · Content-Type: `application/json`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status, `model_loaded`, model version |
| `/score` | POST | Submit patient features → risk probability, risk flag, model version, disclaimer |
| `/model` | GET | Model metadata: version, voting strategy, threshold, metrics, CV summary |
| `/model-card` | GET | Model card as JSON |

**Example — POST /score**

```json
// Request
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}

// Response (200)
{
  "risk_probability": 0.87,
  "risk_flag": "High Risk",
  "model_version": "m1.0.0",
  "voting_strategy": "soft",
  "threshold": 0.5,
  "timestamp": "2026-09-09T00:00:00Z",
  "disclaimer": "Clinical decision-support signal, not a medical diagnosis."
}
```

**Error codes:** `400` malformed JSON · `422` validation error · `429` rate limit · `503` model unavailable · `500` server error

## 🧪 Testing

```bash
cd backend && pytest          # API, unit & integration tests
cd training && pytest         # Data science tests (CV reproducibility, leakage)
cd frontend && npm test       # UI tests
```

## 🔒 Security & Responsible Use

- Pydantic input validation with clinical range constraints
- No PHI in logs or the repository — synthetic data only for demos
- SHA-256 checksums verify model artifact integrity
- Population bias note: the UCI dataset reflects 1980s Cleveland clinic data and may not represent diverse populations
- Human oversight: every risk flag is advisory and requires clinician confirmation

## 📄 Model Card

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — intended use, out-of-scope uses, training data, validation method, evaluation results, ethical considerations, and monitoring plan.

## 🗺️ Roadmap (12 Weeks)

| Phase | Weeks | Deliverable |
|---|---|---|
| Research | 1 | JAIR 1999 paper + dataset study |
| Requirements | 2 | BRD, PRD, UX wireframes |
| Data | 3 | Ingestion, validation, EDA |
| Features | 4 | Leakage-safe preprocessing pipeline |
| Baseline | 5 | Baseline metrics |
| Ensemble | 6–7 | Voting Classifier + hyperparameter tuning |
| Evaluation | 8 | Threshold selection, Sensitivity/Specificity/F1 |
| API | 9 | FastAPI service |
| Frontend | 10 | React UI |
| Integration | 11 | End-to-end tests & deployment |
| Finalization | 12 | Model card, README, demo & viva |

## 📜 License

MIT — see [LICENSE](LICENSE).

---

**Educational decision-support prototype. Not for real clinical use without appropriate validation, regulatory approval, security hardening, clinical governance, and human oversight.**
# heart-disease-voting-ensemble
