# Model Card - Voting Classifier Heart Disease Risk Flag

> Educational prototype. **Not a medical device.** Outputs are advisory risk signals, not diagnoses.

## 1. Model details

| Field | Value |
|---|---|
| Version | m2.0.0 |
| Label | 1 = heart disease present (UCI num > 0), 0 = absent (UCI num = 0) |
| Type | Supervised binary classification ensemble |
| Implementation | scikit-learn `VotingClassifier` inside a `Pipeline` with leakage-safe preprocessing |
| Voting strategy | soft (chosen because threshold selection needs probabilities) |
| Base estimators | Logistic Regression, Random Forest, Gradient Boosting, SVM (`probability=True`) |
| Training date | 2026-11-22 |
| Artifact SHA-256 | `50bd7eddf7efe61d12ffc0c084916d411d7ece234b35ea88cb31debb0d61615b` |
| Hyperparameters | `{"clf__weights": null, "clf__voting": "soft", "clf__svm__gamma": 0.01, "clf__svm__C": 0.1, "clf__rf__min_samples_leaf": 5, "clf__rf__max_depth": null, "clf__lr__C": 1, "clf__gb__n_estimators": 50, "clf__gb__max_depth": 3, "clf__gb__learning_rate": 0.03}` |

## 2. Intended use

- Educational demonstration of the full ML lifecycle (data, validation, evaluation, serving).
- A model-driven risk flag to help *prioritise* patients for further workup, always with clinician review.
- Research into ensemble methods (grounded in *Popular Ensemble Methods: An Empirical Study*, JAIR 1999).

## 3. Out of scope

- Autonomous diagnosis or any real clinical decision.
- Any regulatory claim: no FDA, CE or other approval exists.
- Populations, geographies or time periods different from the training data.
- Model m1.0.0, which was trained on an inverted label and must not be used (`docs/adr/ADR-006-target-label-inversion.md`). The API refuses to load it.
- Emergency or real-time critical-care decisions (latency and reliability not validated).

## 4. Training data

- Source: UCI Heart Disease (Cleveland subset), via a recoded 303-row CSV variant. Provenance and the
  encoding differences from the UCI documentation are in `docs/DATA_DICTIONARY.md`.
- Rows used for training: **302** (1 exact duplicate removed). Class balance (0 / 1): 164 / 138.
- 7 values outside the documented coding range (`ca`=4, `thal`=0) were treated as missing and imputed inside the pipeline, using statistics from training folds only.

## 5. Validation

- Stratified 5-fold cross-validation (seed 42); every preprocessing step is re-fitted on each fold's training split.
- Hyperparameters tuned with **nested** CV (inner 3-fold randomized search, selection metric F1) so the reported estimate is not inflated by tuning.

## 6. Evaluation results

Operating threshold: **0.26681850441553856**. Rule: Maximise specificity subject to sensitivity >= 0.9 on out-of-fold predictions.

| Metric (out-of-fold predictions) | Value |
|---|---|
| sensitivity | 0.906 |
| specificity | 0.646 |
| precision | 0.683 |
| f1 | 0.779 |
| ROC AUC | 0.908 |

**Honest check** (threshold chosen without the fold being scored; the figure to expect on new patients from this population): sensitivity 0.855, specificity 0.713, F1 0.779. The table above is optimistic because its threshold was chosen on the same predictions.

| | Predicted Low | Predicted High |
|---|---|---|
| **Actually no disease** | TN = 106 | FP = 58 |
| **Actually disease** | FN = 13 | TP = 125 |

Nested CV at the operating threshold, mean (std) across 5 folds:

| Metric | Mean | Std |
|---|---|---|
| sensitivity | 0.906 | 0.076 |
| specificity | 0.647 | 0.095 |
| f1 | 0.779 | 0.059 |
| roc_auc | 0.908 | 0.059 |

**Baseline comparison** (same folds, same threshold rule). Logistic regression: ROC AUC 0.914, sensitivity 0.884, specificity 0.732, F1 0.803. Voting ensemble: ROC AUC 0.908, sensitivity 0.855, specificity 0.713, F1 0.779. The ensemble is **not demonstrably better** than the baseline; it is kept as a project requirement.

### External validation (hospitals never used for training)

| Site | n (disease / healthy) | ROC AUC (95% CI) | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| Hungary (Budapest) | 294 (106 / 188) | 0.882 (0.838-0.921) | 0.840 | 0.745 | 0.131 |
| VA Long Beach | 200 (149 / 51) | 0.707 (0.621-0.790) | 0.812 | 0.529 | 0.254 |
| Switzerland (Zurich/Basel) | 123 (115 / 8) | 0.767 (0.581-0.927) | 0.870 | 0.375 | 0.225 |

Published model and threshold, nothing refitted. `ca` and `thal` are missing for most of these patients and are imputed, so this tests the model with degraded inputs. Switzerland has too few healthy patients to be conclusive. The sensitivity target of 0.90 is not met at these sites.

### Additional descriptive analysis

- Soft-voting out-of-fold calibration Brier score: 0.121. Lower is better.
- Continuous-feature variance inflation factors and target-association tests are recorded under `training/artifacts/`. Association is descriptive, not causal.
- Subgroup metrics at the chosen threshold are reported by sex and age band in `training/artifacts/subgroup_metrics.json`; counts and small-sample caveats are included. These results are not evidence of causal or generalizable subgroup differences.

## 7. Ethical considerations and limitations

- **Population bias:** trained on a single hospital, 1980s data, ~300 patients; subgroup coverage is limited and descriptive subgroup estimates can be unstable in small groups.
- **Historical bias:** labels reflect the diagnostic practice of the era.
- **Small sample:** fold-level metrics vary a lot (about 60 patients per fold); treat differences of a few points as noise.
- **Prevalence:** this dataset is roughly half positive, far above real screening prevalence; precision in practice would be much lower than reported here.
- **Proxy variables:** `sex` and `age` may encode societal bias.
- **Human oversight:** every output is advisory and requires confirmation by a qualified clinician.

## 8. Monitoring (planned - NOT implemented)

No drift, calibration or outcome monitoring exists yet. A production system would need input-drift checks (KS/PSI), prediction-rate tracking, calibration checks, and outcome-based recomputation of sensitivity/specificity once diagnoses are confirmed.

## 9. Versioning and reproducibility

Seed 42; the model artifact is checksum-verified when the API starts; every API response includes the model version and threshold. Training steps are recorded in `training/artifacts/`.

## 10. Second model: patients without `ca` and/or `thal`

Version **m2.0.0-noscan**, threshold **0.25881272038388564**, SHA-256 `99e2e4089c8c43d03c92a3c90d23a46c9a5c164e24b3a398d4ca02ab5a49fe29`. Same voting ensemble, search space, patients and threshold rule, trained without the two scan features. The API uses it whenever either scan is missing and ignores the other one. Rationale: `docs/adr/ADR-007-no-scan-model.md`.

| Cleveland, held-out folds | ROC AUC | Sensitivity | Specificity |
|---|---|---|---|
| This model (honest check) | 0.872 | 0.891 | 0.634 |
| This model (out-of-fold, chosen threshold) | 0.872 | 0.906 | 0.616 |
| Full model, `ca` blanked and imputed | 0.881 | 0.833 | 0.756 |
| Full model, `thal` blanked and imputed | 0.893 | 0.862 | 0.695 |
| Full model, both scans blanked and imputed | 0.857 | 0.710 | 0.817 |
| Logistic regression without scans (honest check) | 0.879 | 0.862 | 0.665 |

| External site | ROC AUC (95% CI) | Sensitivity | Specificity | Full model AUC / sens / spec |
|---|---|---|---|---|
| Hungary (Budapest) | 0.886 (0.843-0.923) | 0.915 | 0.628 | 0.882 / 0.840 / 0.745 |
| VA Long Beach | 0.720 (0.637-0.797) | 0.906 | 0.333 | 0.707 / 0.812 / 0.529 |
| Switzerland (Zurich/Basel) | 0.766 (0.612-0.908) | 0.948 | 0.375 | 0.767 / 0.870 / 0.375 |

Higher sensitivity comes with lower specificity: ranking (AUC) improves only slightly. Other missing values at these sites are still imputed, whereas the API rejects them. VA and Hungary results were seen during development, so treat them as reference, not a blind test.
