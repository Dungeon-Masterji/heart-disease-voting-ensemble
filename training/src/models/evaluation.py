import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from training.src.models.threshold import select_threshold


def metrics_at(y_true, y_pred, threshold=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold) if threshold is not None else None,
        "sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(specificity),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def sweep_thresholds(y_true, probabilities, thresholds=None):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if thresholds is None:
        thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    rows = []
    for threshold in thresholds:
        rows.append(metrics_at(y_true, probabilities >= threshold, threshold))
    return pd.DataFrame(rows)


def honest_threshold_check(fold_models, X, y, cv, target_sensitivity=0.90, seed=42):
    """Score each outer fold with a threshold chosen without that fold.

    For every outer fold the threshold comes from inner out-of-fold predictions
    on the training part only, using that fold's model settings, and is applied
    to the held-out part only. Nothing about the held-out fold is used.
    """
    y_values = np.asarray(y)
    probabilities = np.zeros(len(y_values), dtype=float)
    predictions = np.zeros(len(y_values), dtype=bool)
    thresholds = []
    for (train_idx, test_idx), fold_model in zip(cv.split(X, y), fold_models):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        inner_probabilities = cross_val_predict(
            clone(fold_model), X_train, y_train, cv=inner, method="predict_proba"
        )[:, 1]
        choice = select_threshold(
            sweep_thresholds(y_train, inner_probabilities), target_sensitivity
        )
        threshold = float(choice["threshold"]) if choice else 0.0
        thresholds.append(threshold)
        model = clone(fold_model).fit(X_train, y_train)
        probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        predictions[test_idx] = probabilities[test_idx] >= threshold
    result = metrics_at(y_values, predictions)
    result.pop("threshold")
    result["roc_auc"] = float(roc_auc_score(y_values, probabilities))
    result["thresholds_by_fold"] = thresholds
    return result


def held_out_permutation_importance(fold_models, X, y, cv, seed=42):
    """Mean drop in ROC AUC when a feature is shuffled, on held-out folds only.

    `fold_models` must be fitted on the matching training folds.
    """
    drops = [
        permutation_importance(
            model,
            X.iloc[test_idx],
            y.iloc[test_idx],
            scoring="roc_auc",
            n_repeats=10,
            random_state=seed,
        ).importances_mean
        for (_, test_idx), model in zip(cv.split(X, y), fold_models)
    ]
    return {str(c): float(v) for c, v in zip(X.columns, np.mean(drops, axis=0))}
