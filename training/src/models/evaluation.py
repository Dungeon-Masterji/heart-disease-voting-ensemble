from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate


def metrics_at(y_true, y_pred, threshold=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold) if threshold is not None else None,
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
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


def honest_leave_one_fold_out_threshold(model_factory, X, y, seed=42, target_sensitivity=0.90,
                                         inner_splits=5):
    probabilities = np.zeros(len(y), dtype=float)
    thresholds = []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(X, y):
        folds = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
        # Threshold selection uses only OOF predictions from the training fold.
        train_probabilities = np.zeros(len(train_idx), dtype=float)
        for inner_train, inner_test in folds.split(X.iloc[train_idx], y.iloc[train_idx]):
            inner_model = model_factory()
            inner_model.fit(X.iloc[train_idx[inner_train]], y.iloc[train_idx[inner_train]])
            train_probabilities[inner_test] = inner_model.predict_proba(X.iloc[train_idx[inner_test]])[:, 1]
        fold_sweep = sweep_thresholds(y.iloc[train_idx], train_probabilities)
        from training.src.models.threshold import select_threshold
        choice = select_threshold(fold_sweep, target_sensitivity)
        threshold = float(choice["threshold"]) if choice else 0.0
        thresholds.append(threshold)

        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    overall_threshold = float(np.mean(thresholds))
    predictions = probabilities >= overall_threshold
    return {
        "sensitivity": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "specificity": float((~predictions[y.to_numpy() == 0]).mean()),
        "thresholds_by_fold": thresholds,
        "threshold": overall_threshold,
    }
