import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from training.src.data.ingestion import load_raw_dataset
from training.src.features.preprocessing import load_xy
from training.src.models.baseline import make_baseline
from training.src.models.evaluation import metrics_at, sweep_thresholds
from training.src.models.export import export_model
from training.src.models.threshold import select_threshold
from training.src.models.tuning import final_soft_search, nested_soft_cv


SEED = 42
TARGET_SENS = 0.90
REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    df = load_raw_dataset(REPO_ROOT / "data" / "raw" / "heart_disease.csv")
    X, y = load_xy(df)
    nested_rows = nested_soft_cv(X, y, seed=SEED, n_iter=20, quick=args.quick)
    final_search = final_soft_search(X, y, seed=SEED, n_iter=40, quick=args.quick)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    cv_folds = []
    for (train_idx, test_idx), row in zip(cv.split(X, y), nested_rows):
        oof[test_idx] = row["model"].predict_proba(X.iloc[test_idx])[:, 1]
        cv_folds.append({"fold": int(row["fold"]), "mean_test_f1": float(row["mean_test_f1"])})

    sweep = sweep_thresholds(y, oof)
    choice = select_threshold(sweep, TARGET_SENS)
    if choice is None:
        raise RuntimeError("No threshold satisfies the required sensitivity")
    threshold = float(choice["threshold"])

    final_model = final_search.best_estimator_.fit(X, y)
    predictions = oof >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    importance_result = permutation_importance(
        final_model, X, y, n_repeats=10, random_state=SEED, scoring="f1"
    )
    importance = {
        str(column): float(value)
        for column, value in zip(X.columns, importance_result.importances_mean)
    }

    metrics = metrics_at(y, predictions, threshold)
    baseline_scores = cross_val_score(make_baseline(SEED), X, y, cv=cv, scoring="f1", n_jobs=-1)

    metadata = {
        "model_name": "Voting Classifier Heart Disease Risk Flag",
        "model_version": "m1.0.0",
        "voting_strategy": "soft",
        "threshold": threshold,
        "threshold_rationale": f"Maximise specificity subject to sensitivity >= {TARGET_SENS} on out-of-fold predictions.",
        "training_date": datetime.date.today().isoformat(),
        "n_training_rows": int(len(y)),
        "hyperparameters": final_search.best_params_,
        "metrics_at_threshold_oof": {
            "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
            "precision": metrics["precision"], "f1": metrics["f1"],
        },
        "roc_auc_oof": float(roc_auc_score(y, oof)),
        "metrics_honest_check": {"sensitivity": sensitivity, "specificity": specificity},
        "cv_folds": cv_folds,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance": importance,
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "baseline_logreg_cv": {"f1": float(baseline_scores.mean())},
        "seed": SEED,
    }
    artifacts = REPO_ROOT / "training" / "artifacts"
    model_path, metadata_path, metadata = export_model(final_model, metadata, artifacts / "models")
    sweep.to_csv(artifacts / "week8_threshold_sweep.csv", index=False)
    pd.DataFrame({"feature": list(importance), "importance": list(importance.values())}).to_csv(
        artifacts / "week8_permutation_importance.csv", index=False
    )
    print(json.dumps({
        "model": str(model_path.relative_to(REPO_ROOT)),
        "metadata": str(metadata_path.relative_to(REPO_ROOT)),
        "roc_auc_oof": metadata["roc_auc_oof"],
        "threshold": threshold,
        "metrics": metadata["metrics_at_threshold_oof"],
        "baseline_logreg_f1": metadata["baseline_logreg_cv"]["f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
