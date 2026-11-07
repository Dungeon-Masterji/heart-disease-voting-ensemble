import hashlib
import json

import numpy as np
import pandas as pd

from training.src.models.evaluation import metrics_at
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
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert json.loads(meta_path.read_text())["artifact_sha256"] == metadata["artifact_sha256"]


def test_hard_voting_can_be_built_without_predict_proba():
    model = build_voting("hard")
    assert model.voting == "hard"
    assert not hasattr(model, "predict_proba")
