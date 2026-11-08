"""Builds a tiny, throw-away model artifact for tests, so the test-suite never
depends on the real trained model (and still exercises the real preprocessing)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.predictor import sha256_of
from training.src.features.preprocessing import build_preprocessor, load_xy

THRESHOLD = 0.4


def make_training_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(
        age=rng.integers(30, 77, n), sex=rng.integers(0, 2, n), cp=rng.integers(0, 4, n),
        trestbps=rng.integers(94, 200, n), chol=rng.integers(126, 564, n), fbs=rng.integers(0, 2, n),
        restecg=rng.integers(0, 3, n), thalach=rng.integers(71, 202, n), exang=rng.integers(0, 2, n),
        oldpeak=rng.random(n) * 4, slope=rng.integers(0, 3, n), ca=rng.integers(0, 4, n),
        thal=rng.integers(1, 4, n)))
    df["target"] = ((df.thalach / 200 - df.exang * 0.3 - df.oldpeak / 8
                     + rng.normal(0, 0.15, n)) > 0.3).astype(int)
    df.loc[5, "ca"] = 4      # reproduce the out-of-range codes seen in the real data
    df.loc[6, "thal"] = 0
    return df


def build_artifact(directory: Path, tamper_meta: bool = False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    X, y = load_xy(make_training_frame())
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=0))]).fit(X, y)
    model_path = directory / "model_test.joblib"
    meta_path = directory / "model_test.json"
    joblib.dump(pipe, model_path)
    folds = [{"fold": i + 1, "sensitivity": 0.9 - 0.01 * i, "specificity": 0.8 + 0.01 * i,
              "f1": 0.85, "roc_auc": 0.9} for i in range(5)]
    meta = {
        "model_version": "m-test", "voting_strategy": "soft", "threshold": THRESHOLD,
        "threshold_rationale": "test rationale", "training_date": "2026-01-01",
        "n_training_rows": int(len(y)),
        "metrics_at_threshold_oof": {"sensitivity": 0.9, "specificity": 0.8, "precision": 0.85, "f1": 0.87},
        "roc_auc_oof": 0.91, "metrics_honest_check": {"sensitivity": 0.89, "specificity": 0.79},
        "cv_folds": folds, "confusion_matrix": {"tn": 100, "fp": 30, "fn": 10, "tp": 100},
        "feature_importance": {"thalach": 0.1, "age": 0.01},
        "artifact_sha256": "0" * 64 if tamper_meta else sha256_of(model_path),
    }
    meta_path.write_text(json.dumps(meta))
    return model_path, meta_path
