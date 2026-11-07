import numpy as np
import pandas as pd

from training.src.eda.analysis import calibration_summary, subgroup_metrics
from training.src.models.evaluation import metrics_at


def test_calibration_summary_includes_brier_and_curve():
    result = calibration_summary(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.2, 0.9])
    )
    assert 0.0 <= result["brier_score"] <= 1.0
    assert result["curve"]


def test_subgroup_metrics_include_counts_and_caveats():
    X = pd.DataFrame({"sex": [0, 1, 1], "age": [45, 55, 65]})
    result = subgroup_metrics(
        X, np.array([0, 1, 1]), np.array([0.2, 0.8, 0.7]), 0.5, metrics_at
    )
    assert sum(group["count"] for group in result.values()) >= len(X)
    assert all(group["caveat"] for group in result.values())
