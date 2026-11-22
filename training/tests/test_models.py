import hashlib
import json

import numpy as np
import pandas as pd

from training.src.models.evaluation import honest_threshold_check, metrics_at
from training.src.models.export import export_model
from training.src.models.threshold import select_threshold
from training.src.models.voting_ensemble import build_voting


def test_threshold_rule_maximises_specificity_at_required_sensitivity():
    sweep = pd.DataFrame([
        {"threshold": 0.2, "sensitivity": 0.95, "specificity": 0.3},
        {"threshold": 0.4, "sensitivity": 0.91, "specificity": 0.7},
        {"threshold": 0.5, "sensitivity": 0.89, "specificity": 0.9},
        {"threshold": 0.6, "sensitivity": 0.90, "specificity": 0.7},
    ])
    chosen = select_threshold(sweep, 0.90)
    assert chosen["threshold"] == 0.6


def test_metrics_at_uses_prediction_at_exact_threshold():
    y = np.array([0, 1, 1, 0])
    probabilities = np.array([0.49, 0.50, 0.7, 0.1])
    metrics = metrics_at(y, probabilities >= 0.50, 0.50)
    assert metrics["threshold"] == 0.50
    assert metrics["sensitivity"] == 1.0


def test_export_writes_matching_sha256(tmp_path):
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
    model_path, meta_path, metadata = export_model(
        model, {"model_version": "test", "threshold": 0.5}, tmp_path
    )
    assert model_path.name == "model_test.joblib"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert json.loads(meta_path.read_text())["artifact_sha256"] == metadata["artifact_sha256"]


def test_hard_voting_can_be_built_without_predict_proba():
    model = build_voting("hard")
    assert model.voting == "hard"
    assert not hasattr(model, "predict_proba")


def test_honest_check_never_uses_held_out_labels():
    """Changing only a held-out fold's labels must not change its threshold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import PredefinedSplit

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] + rng.normal(size=120) > 0).astype(int))
    cv = PredefinedSplit([0] * 60 + [1] * 60)
    models = [LogisticRegression()] * 2
    flipped = y.copy()
    flipped.iloc[:60] = 1 - flipped.iloc[:60]
    before = honest_threshold_check(models, X, y, cv)["thresholds_by_fold"]
    after = honest_threshold_check(models, X, flipped, cv)["thresholds_by_fold"]
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_noscan_pipeline_never_sees_the_scan_features():
    from training.src.features.preprocessing import SCAN_FEATURES
    from training.src.models.baseline import make_baseline
    from training.tests.test_preprocessing import make_df

    df = make_df()
    X, y = df.drop(columns=["target", *SCAN_FEATURES]), df["target"]
    model = make_baseline(exclude=tuple(SCAN_FEATURES)).fit(X, y)
    names = model.named_steps["prep"].named_steps["columns"].feature_names_in_
    assert not set(SCAN_FEATURES) & set(names)
    changed = X.assign(ca=3, thal=3)
    assert np.array_equal(model.predict_proba(X), model.predict_proba(changed))
