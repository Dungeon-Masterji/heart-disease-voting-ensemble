import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "training" / "artifacts" / "models" / "model_m2.0.0.json"


BACKEND_KEYS = {
    "model_version", "voting_strategy", "threshold", "threshold_rationale",
    "training_date", "n_training_rows", "metrics_at_threshold_oof",
    "roc_auc_oof", "metrics_honest_check", "cv_folds",
    "confusion_matrix", "feature_importance", "class_balance",
    "baseline_logreg_cv", "seed", "artifact_sha256", "label_definition",
}


MODEL_CARD_KEYS = BACKEND_KEYS


def test_real_metadata_contract():
    metadata = json.loads(METADATA_PATH.read_text())
    assert BACKEND_KEYS <= metadata.keys()
    assert MODEL_CARD_KEYS <= metadata.keys()
    assert metadata["voting_strategy"] == "soft"


def test_honest_check_is_not_a_copy_of_the_headline_metrics():
    metadata = json.loads(METADATA_PATH.read_text())
    honest = metadata["metrics_honest_check"]
    assert len(honest["thresholds_by_fold"]) == 5
    headline = metadata["metrics_at_threshold_oof"]
    assert (honest["sensitivity"], honest["specificity"]) != (
        headline["sensitivity"], headline["specificity"])
