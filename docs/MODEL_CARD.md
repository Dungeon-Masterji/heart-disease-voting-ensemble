# Model Card - Voting Classifier Heart Disease Risk Flag

> Educational prototype. **Not a medical device.** Outputs are advisory risk signals, not diagnoses.

## 1. Model details

| Field | Value |
|---|---|
| Version | m1.0.0 |
| Type | Supervised binary classification ensemble |
| Implementation | scikit-learn `VotingClassifier` inside a `Pipeline` with leakage-safe preprocessing |
| Voting strategy | soft (chosen because threshold selection needs probabilities) |
| Base estimators | Logistic Regression, Random Forest, Gradient Boosting, SVM (`probability=True`) |
| Training date | 2026-11-07 |
| Artifact SHA-256 | `de982ba3a628db4dd83230b4d3d0e9bf3c71396d7e13796c57c2401e2afc511a` |
| Hyperparameters | `{"clf__weights": [2, 2, 1, 1], "clf__voting": "soft", "clf__svm__gamma": 0.01, "clf__svm__C": 0.1, "clf__rf__min_samples_leaf": 1, "clf__rf__max_depth": 3, "clf__lr__C": 0.1, "clf__gb__n_estimators": 50, "clf__gb__max_depth": 2, "clf__gb__learning_rate": 0.03}` |

## 2. Intended use

- Educational demonstration of the full ML lifecycle (data, validation, evaluation, serving).
- A model-driven risk flag to help *prioritise* patients for further workup, always with clinician review.
- Research into ensemble methods (grounded in *Popular Ensemble Methods: An Empirical Study*, JAIR 1999).

## 3. Out of scope

- Autonomous diagnosis or any real clinical decision.
- Any regulatory claim: no FDA, CE or other approval exists.
- Populations, geographies or time periods different from the training data.
- Emergency or real-time critical-care decisions (latency and reliability not validated).

## 4. Training data

- Source: UCI Heart Disease (Cleveland subset), via a recoded 303-row CSV variant. Provenance and the
  encoding differences from the UCI documentation are in `docs/DATA_DICTIONARY.md`.
- Rows used for training: **302** (1 exact duplicate removed). Class balance (0 / 1): 138 / 164.
- 7 values outside the documented coding range (`ca`=4, `thal`=0) were treated as missing and imputed inside the pipeline, using statistics from training folds only.

## 5. Validation

- Stratified 5-fold cross-validation (seed 42); every preprocessing step is re-fitted on each fold's training split.
- Hyperparameters tuned with **nested** CV (inner 3-fold randomized search, selection metric F1) so the reported estimate is not inflated by tuning.

## 6. Evaluation results

Operating threshold: **0.5000575873252479**. Rule: Maximise specificity subject to sensitivity >= 0.9 on out-of-fold predictions.

| Metric (out-of-fold predictions) | Value |
|---|---|
| sensitivity | 0.902 |
| specificity | 0.797 |
| precision | 0.841 |
| f1 | 0.871 |
| ROC AUC | 0.908 |

Honest check (threshold re-chosen inside each fold, never on the fold being scored): sensitivity 0.902, specificity 0.797.

| | Predicted Low | Predicted High |
|---|---|---|
| **Actually no disease** | TN = 110 | FP = 28 |
| **Actually disease** | FN = 16 | TP = 148 |

Nested CV at threshold 0.5, mean (std) across 5 folds:

| Metric | Mean | Std |
|---|---|---|
| sensitivity | 0.903 | 0.049 |
| specificity | 0.796 | 0.146 |
| f1 | 0.872 | 0.050 |
| roc_auc | 0.906 | 0.059 |

**Baseline comparison.** Plain logistic regression under the same CV scored F1 0.881, sensitivity n/a, ROC AUC n/a. The tuned ensemble's F1 differs by -0.009.

## 10. Additional descriptive analysis

- Soft-voting out-of-fold calibration Brier score: 0.121. Lower is better.
- Continuous-feature variance inflation factors and target-association tests are recorded under `training/artifacts/`. Association is descriptive, not causal.
- Subgroup metrics at the chosen threshold are reported by sex and age band in `training/artifacts/subgroup_metrics.json`; counts and small-sample caveats are included. These results are not evidence of causal or generalizable subgroup differences.
That difference is smaller than the fold-to-fold standard deviation, so on this dataset the ensemble is **not demonstrably better** than the simple baseline.

## 7. Ethical considerations and limitations

- **Population bias:** single hospital, 1980s data, ~300 patients; subgroup coverage is limited and descriptive subgroup estimates can be unstable in small groups.
- **Historical bias:** labels reflect the diagnostic practice of the era.
- **Small sample:** fold-level metrics vary a lot (about 60 patients per fold); treat differences of a few points as noise.
- **Prevalence:** this dataset is roughly half positive, far above real screening prevalence; precision in practice would be much lower than reported here.
- **Proxy variables:** `sex` and `age` may encode societal bias.
- **Human oversight:** every output is advisory and requires confirmation by a qualified clinician.

## 8. Monitoring (planned - NOT implemented)

No drift, calibration or outcome monitoring exists yet. A production system would need input-drift checks (KS/PSI), prediction-rate tracking, calibration checks, and outcome-based recomputation of sensitivity/specificity once diagnoses are confirmed.

## 9. Versioning and reproducibility

Seed 42; the model artifact is checksum-verified when the API starts; every API response includes the model version and threshold. Training steps are recorded in `training/artifacts/`.
