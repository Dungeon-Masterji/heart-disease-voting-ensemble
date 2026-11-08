import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from training.src.experiments.tracking import split_fold_indices, write_run


def frame():
    return pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })


def test_same_seed_produces_same_fold_indices_and_metrics(tmp_path):
    X = frame()[["x"]]
    y = frame()["target"]
    cv1 = StratifiedKFold(5, shuffle=True, random_state=42)
    cv2 = StratifiedKFold(5, shuffle=True, random_state=42)
    folds1 = split_fold_indices(cv1, X, y)
    folds2 = split_fold_indices(cv2, X, y)
    assert folds1 == folds2

    metrics = {"sensitivity": 0.9, "specificity": 0.8, "f1": 0.85}
    first, _ = write_run({"run_id": "test-run-1", "seed": 42, "fold_indices": folds1,
                          "aggregated_metrics": metrics}, tmp_path / "artifacts")
    second, _ = write_run({"run_id": "test-run-2", "seed": 42, "fold_indices": folds2,
                           "aggregated_metrics": metrics}, tmp_path / "artifacts")
    first_record = json.loads(first.read_text())
    second_record = json.loads(second.read_text())
    assert first_record["fold_indices"] == second_record["fold_indices"]
    assert first_record["aggregated_metrics"] == second_record["aggregated_metrics"]
