"""Modest descriptive and model-calibration analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def variance_inflation_factors(X: pd.DataFrame) -> list[dict]:
    values = X[list(CONTINUOUS_FEATURES)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return []
    centered = values - values.mean(axis=0)
    centered = centered.to_numpy(dtype=float)
    results = []
    for index, feature in enumerate(CONTINUOUS_FEATURES):
        predictors = np.delete(centered, index, axis=1)
        predictors = np.column_stack((np.ones(len(values)), predictors))
        coefficient, _, _, _ = np.linalg.lstsq(
            predictors, centered[:, index], rcond=None
        )
        fitted = predictors @ coefficient
        residual_sum = float(np.sum((centered[:, index] - fitted) ** 2))
        total_sum = float(np.sum(centered[:, index] ** 2))
        r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
        vif = float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
        results.append({"feature": feature, "vif": vif})
    return results


def feature_target_tests(df: pd.DataFrame) -> list[dict]:
    rows = []
    target = df["target"]
    for feature in CONTINUOUS_FEATURES:
        low = df.loc[target == 0, feature].dropna()
        high = df.loc[target == 1, feature].dropna()
        result = mannwhitneyu(low, high, alternative="two-sided")
        rows.append(
            {
                "feature": feature,
                "test": "Mann-Whitney U",
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    for feature in CATEGORICAL_FEATURES:
        contingency = pd.crosstab(df[feature], target)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        statistic, p_value, _, _ = chi2_contingency(contingency, correction=False)
        rows.append(
            {
                "feature": feature,
                "test": "chi-square",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return rows


def calibration_summary(y_true, probabilities, n_bins: int = 10) -> dict:
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=n_bins, strategy="quantile"
    )
    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "curve": [
            {"mean_predicted_probability": float(p), "observed_fraction": float(o)}
            for p, o in zip(predicted, observed)
        ],
    }


def subgroup_metrics(
    X: pd.DataFrame, y_true, probabilities, threshold: float, metrics_at
) -> dict:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= threshold
    groups = {
        "sex=0": X["sex"].to_numpy() == 0,
        "sex=1": X["sex"].to_numpy() == 1,
        "age=<50": X["age"].to_numpy() < 50,
        "age=50-59": (X["age"].to_numpy() >= 50) & (X["age"].to_numpy() < 60),
        "age=60+": X["age"].to_numpy() >= 60,
    }
    results = {}
    for name, mask in groups.items():
        count = int(mask.sum())
        metric = metrics_at(y_true[mask], predictions[mask], threshold) if count else {}
        results[name] = {
            "count": count,
            "metrics": metric,
            "caveat": (
                "Small sample; descriptive only; not suitable for subgroup performance claims."
                if count < 30
                else "Descriptive only; association is not evidence of causal differences."
            ),
        }
    return results
