"""Model loading and prediction. Deliberately free of FastAPI imports so it can
be unit-tested on its own (and reused by scripts).

Responsibilities (PRD TR-010, TR-011, security design):
  * verify the model artifact's SHA-256 against the metadata before loading it
  * load the already-fitted pipeline (no fitting ever happens at serving time)
  * map probability -> High Risk / Low Risk using the documented threshold
"""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import pandas as pd

FEATURE_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
DISCLAIMER = "Clinical decision-support signal, not a medical diagnosis."
HIGH_RISK = "High Risk"
LOW_RISK = "Low Risk"
_FOLD_METRICS = ("sensitivity", "specificity", "f1", "roc_auc")


class ModelUnavailable(Exception):
    """Raised when the artifact is missing, tampered with, or cannot be loaded."""


def risk_flag(probability: float, threshold: float) -> str:
    """Probability exactly equal to the threshold counts as High Risk.
    (Documented rule: flag if probability >= threshold, matching training.)"""
    return HIGH_RISK if probability >= threshold else LOW_RISK


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_repo_root_on_path() -> None:
    """The pickled pipeline references training.src.features.preprocessing, so
    that package must be importable. Layout: <root>/backend/app/core/predictor.py"""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)


def clean_json(obj: Any) -> Any:
    """Replace NaN/inf with None (they are not valid JSON and would crash the response)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    return obj


def summarise_folds(folds: Optional[list]) -> Optional[dict]:
    """Mean and sample std (ddof=1, same as pandas) of each metric across CV folds."""
    if not folds:
        return None
    mean, std = {}, {}
    for m in _FOLD_METRICS:
        vals = [f[m] for f in folds
                if isinstance(f.get(m), (int, float)) and not math.isnan(f[m])]
        if vals:
            mean[m] = round(statistics.fmean(vals), 4)
            std[m] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    return {"folds": folds, "mean": mean, "std": std}


class Predictor:
    def __init__(self, model: Any, meta: dict):
        self._model = model
        self.meta = meta
        self.version: str = meta["model_version"]
        self.threshold: float = float(meta["threshold"])
        self.voting_strategy: str = meta.get("voting_strategy", "soft")

    @classmethod
    def load(cls, model_path: Path, meta_path: Path, verify_checksum: bool = True) -> "Predictor":
        model_path, meta_path = Path(model_path), Path(meta_path)
        if not model_path.exists():
            raise ModelUnavailable(f"Model artifact not found: {model_path}")
        if not meta_path.exists():
            raise ModelUnavailable(f"Model metadata not found: {meta_path}")
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            raise ModelUnavailable(f"Model metadata is not valid JSON: {exc}") from exc
        for key in ("model_version", "threshold"):
            if key not in meta:
                raise ModelUnavailable(f"Model metadata is missing '{key}'")
        if verify_checksum:
            expected = meta.get("artifact_sha256")
            if not expected:
                raise ModelUnavailable("Model metadata has no artifact_sha256 to verify against")
            if sha256_of(model_path) != expected:
                raise ModelUnavailable("Model artifact checksum mismatch (file changed or corrupted)")
        _ensure_repo_root_on_path()
        try:
            model = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - any unpickling failure means 'unavailable'
            raise ModelUnavailable(f"Model artifact could not be loaded: {exc}") from exc
        return cls(model, meta)

    def predict(self, features: Mapping[str, float]) -> dict:
        missing = [f for f in FEATURE_ORDER if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        row = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        probability = float(self._model.predict_proba(row)[0, 1])
        return {
            "risk_probability": round(probability, 4),
            "risk_flag": risk_flag(probability, self.threshold),
            "model_version": self.version,
            "voting_strategy": self.voting_strategy,
            "threshold": self.threshold,
        }

    def public_metadata(self) -> dict:
        """Non-sensitive metadata for GET /model (PRD 8.3)."""
        m = self.meta
        metrics = dict(m.get("metrics_at_threshold_oof", {}))
        if "roc_auc_oof" in m:
            metrics["roc_auc"] = m["roc_auc_oof"]
        return clean_json({
            "model_version": self.version,
            "voting_strategy": self.voting_strategy,
            "threshold": self.threshold,
            "threshold_rationale": m.get("threshold_rationale", ""),
            "training_date": m.get("training_date", ""),
            "n_training_rows": m.get("n_training_rows"),
            "class_balance": m.get("class_balance"),
            "metrics": metrics,
            "honest_check": m.get("metrics_honest_check"),
            "cv_summary": summarise_folds(m.get("cv_folds")),
            "confusion_matrix": m.get("confusion_matrix"),
            "feature_importance": m.get("feature_importance"),
            "baseline_logreg_cv": m.get("baseline_logreg_cv"),
            "curves": clean_json(m.get("curves")),
        })
