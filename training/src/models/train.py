import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from training.src.data.ingestion import load_raw_dataset
from training.src.data.labels import LABEL_DEFINITION
from training.src.eda.analysis import (
    calibration_summary,
    feature_target_tests,
    subgroup_metrics,
    variance_inflation_factors,
)
from training.src.experiments.tracking import (
    dataset_record,
    git_commit_hash,
    git_dirty,
    split_fold_indices,
    write_run,
)
from training.src.features.preprocessing import SCAN_FEATURES, load_xy
from training.src.models.baseline import make_baseline
from training.src.models.evaluation import (
    held_out_permutation_importance,
    honest_threshold_check,
    metrics_at,
    sweep_thresholds,
)
from training.src.models.export import export_model
from training.src.models.threshold import select_threshold
from training.src.models.tuning import final_soft_search, nested_soft_cv

MODEL_VERSION = "m2.0.0"
NOSCAN_VERSION = f"{MODEL_VERSION}-noscan"
SEED = 42
TARGET_SENS = 0.90
REPO_ROOT = Path(__file__).resolve().parents[3]


def blanked_tests_comparison(full_fold_models, X, y, cv, full_threshold, features):
    """What the full model does on held-out patients whose scan results are missing.

    Its imputer fills them with the most common (healthy) values.
    """
    blanked = X.copy()
    blanked[features] = np.nan
    proba = np.zeros(len(y), dtype=float)
    for (_, test_idx), model in zip(cv.split(X, y), full_fold_models):
        proba[test_idx] = model.predict_proba(blanked.iloc[test_idx])[:, 1]
    result = metrics_at(y, proba >= full_threshold, full_threshold)
    result["roc_auc"] = float(roc_auc_score(y, proba))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="2-iteration smoke run; writes to training/artifacts/quick and never "
        "touches the published model",
    )
    args = parser.parse_args()

    df = load_raw_dataset(REPO_ROOT / "data" / "raw" / "heart_disease.csv")
    X, y = load_xy(df)
    artifacts = REPO_ROOT / "training" / "artifacts"
    if args.quick:
        artifacts = artifacts / "quick"
    full_folds, full_threshold = train_variant(
        df, X, y, args.quick, MODEL_VERSION, artifacts, artifacts
    )
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    comparisons = {
        name: blanked_tests_comparison(
            full_folds, X, y, outer, full_threshold, features
        )
        for name, features in {
            "ca": ["ca"],
            "thal": ["thal"],
            "both": SCAN_FEATURES,
        }.items()
    }
    train_variant(
        df,
        X.drop(columns=SCAN_FEATURES),
        y,
        args.quick,
        NOSCAN_VERSION,
        artifacts,
        artifacts / "noscan",
        exclude=tuple(SCAN_FEATURES),
        extra={"full_model_with_tests_blanked": comparisons},
    )


def train_variant(
    df, X, y, quick, version, artifacts, report_dir, exclude=(), extra=None
):
    """Tune, evaluate and export one ensemble; return its fold models and threshold."""
    report_dir.mkdir(parents=True, exist_ok=True)
    search_args = {"seed": SEED, "quick": quick, "exclude": exclude}
    nested_rows = nested_soft_cv(X, y, n_iter=20, **search_args)
    final_search = final_soft_search(X, y, n_iter=40, **search_args)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_indices = split_fold_indices(cv, X, y)
    oof = np.zeros(len(y), dtype=float)
    per_fold_metrics = []
    for (train_idx, test_idx), row in zip(cv.split(X, y), nested_rows):
        oof[test_idx] = row["model"].predict_proba(X.iloc[test_idx])[:, 1]
        fold_predictions = oof[test_idx] >= 0.5
        per_fold_metrics.append(
            {
                "fold": int(row["fold"]),
                "roc_auc": float(roc_auc_score(y.iloc[test_idx], oof[test_idx])),
                "metrics_at_default_probability": metrics_at(
                    y.iloc[test_idx], fold_predictions, 0.5
                ),
            }
        )

    sweep = sweep_thresholds(y, oof)
    choice = select_threshold(sweep, TARGET_SENS)
    if choice is None:
        raise RuntimeError("No threshold satisfies the required sensitivity")
    threshold = float(choice["threshold"])
    roc_fpr, roc_tpr, _ = roc_curve(y, oof)
    pr_precision, pr_recall, _ = precision_recall_curve(y, oof)

    def downsample(values, limit=101):
        values = np.asarray(values, dtype=float)
        if len(values) <= limit:
            return values.tolist()
        indices = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(int))
        return values[indices].tolist()

    roc_points = [
        {"fpr": float(fpr), "tpr": float(tpr)}
        for fpr, tpr in zip(downsample(roc_fpr), downsample(roc_tpr))
    ]
    pr_points = [
        {"precision": float(precision), "recall": float(recall)}
        for precision, recall in zip(downsample(pr_precision), downsample(pr_recall))
    ]
    cv_folds = []
    for (_, test_idx), row in zip(cv.split(X, y), nested_rows):
        fold_predictions = oof[test_idx] >= threshold
        fold_metrics = metrics_at(y.iloc[test_idx], fold_predictions, threshold)
        cv_folds.append(
            {
                "fold": int(row["fold"]),
                "sensitivity": fold_metrics["sensitivity"],
                "specificity": fold_metrics["specificity"],
                "f1": fold_metrics["f1"],
                "roc_auc": float(roc_auc_score(y.iloc[test_idx], oof[test_idx])),
                "mean_test_f1": float(row["mean_test_f1"]),
            }
        )

    final_model = final_search.best_estimator_  # refit on all rows by the search
    predictions = oof >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    fold_models = [row["model"] for row in nested_rows]
    importance = held_out_permutation_importance(fold_models, X, y, cv, SEED)

    metrics = metrics_at(y, predictions, threshold)
    honest = honest_threshold_check(fold_models, X, y, cv, TARGET_SENS, SEED)
    # Same folds and same threshold rule as the ensemble, so it is a fair comparison.
    baseline = honest_threshold_check(
        [make_baseline(SEED, exclude)] * cv.get_n_splits(), X, y, cv, TARGET_SENS, SEED
    )
    descriptive_analysis = {
        "vif_continuous": variance_inflation_factors(X),
        "feature_target_tests": [
            row
            for row in feature_target_tests(df.drop_duplicates())
            if row["feature"] not in exclude
        ],
        "calibration": calibration_summary(y, oof),
        "subgroups": subgroup_metrics(X, y, oof, threshold, metrics_at),
        "disclaimer": (
            "Descriptive association tests are not causal; subgroup results have "
            "limited precision in small samples."
        ),
    }

    metadata = {
        "model_name": "Voting Classifier Heart Disease Risk Flag",
        "model_version": version,
        "features": list(X.columns),
        "label_definition": LABEL_DEFINITION,
        "voting_strategy": "soft",
        "threshold": threshold,
        "threshold_rationale": f"Maximise specificity subject to sensitivity >= {TARGET_SENS} on out-of-fold predictions.",
        "training_date": datetime.date.today().isoformat(),
        "n_training_rows": len(y),
        "hyperparameters": final_search.best_params_,
        "metrics_at_threshold_oof": {
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
        },
        "roc_auc_oof": float(roc_auc_score(y, oof)),
        "metrics_honest_check": honest,
        "cv_folds": cv_folds,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "feature_importance": importance,
        "class_balance": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "baseline_logreg_cv": baseline,
        "seed": SEED,
        "curves": {
            "roc": roc_points,
            "precision_recall": pr_points,
            "threshold_marker": {"threshold": threshold},
        },
        "analysis": descriptive_analysis,
        **(extra or {}),
    }
    model_path, metadata_path, metadata = export_model(
        final_model, metadata, artifacts / "models"
    )
    np.save(report_dir / "oof_proba_soft.npy", oof)
    np.save(report_dir / "y_true.npy", y.to_numpy())
    sweep.to_csv(report_dir / "week8_threshold_sweep.csv", index=False)
    pd.DataFrame(
        {"feature": list(importance), "importance": list(importance.values())}
    ).to_csv(report_dir / "week8_permutation_importance.csv", index=False)
    pd.DataFrame(descriptive_analysis["feature_target_tests"]).to_csv(
        report_dir / "feature_target_tests.csv", index=False
    )
    pd.DataFrame(descriptive_analysis["vif_continuous"]).to_csv(
        report_dir / "continuous_vif.csv", index=False
    )
    pd.DataFrame(descriptive_analysis["subgroups"]).T.to_json(
        report_dir / "subgroup_metrics.json", orient="columns", indent=2
    )
    pd.DataFrame(descriptive_analysis["calibration"]["curve"]).to_csv(
        report_dir / "calibration_curve.csv", index=False
    )
    _, run_record = write_run(
        {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "git_commit": git_commit_hash(REPO_ROOT),
            "git_dirty": git_dirty(REPO_ROOT),
            "dataset": dataset_record(
                REPO_ROOT / "data" / "raw" / "heart_disease.csv",
                len(df),
                df.columns,
                after_cleaning=False,
            ),
            "seed": SEED,
            "fold_indices": fold_indices,
            "preprocessing": (
                "median numeric and most-frequent categorical imputation; standard scaling; "
                "unknown-category-tolerant one-hot encoding; leakage-safe row-wise feature engineering"
            ),
            "model_type": "soft VotingClassifier",
            "hyperparameters": final_search.best_params_,
            "cv_strategy": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "selection_metric": "f1",
            "per_fold_metrics": per_fold_metrics,
            "aggregated_metrics": metrics,
            "threshold": threshold,
            "threshold_rationale": metadata["threshold_rationale"],
            "artifact_path": str(model_path.relative_to(REPO_ROOT)),
            "artifact_sha256": metadata["artifact_sha256"],
        },
        artifacts,
    )
    print(
        f"run record: {artifacts.relative_to(REPO_ROOT)}/runs/{run_record['run_id']}.json"
    )
    print(
        json.dumps(
            {
                "model": str(model_path.relative_to(REPO_ROOT)),
                "metadata": str(metadata_path.relative_to(REPO_ROOT)),
                "roc_auc_oof": metadata["roc_auc_oof"],
                "threshold": threshold,
                "metrics": metadata["metrics_at_threshold_oof"],
                "honest_check": metadata["metrics_honest_check"],
                "baseline_logreg": metadata["baseline_logreg_cv"],
            },
            indent=2,
        )
    )
    return fold_models, threshold


if __name__ == "__main__":
    main()
