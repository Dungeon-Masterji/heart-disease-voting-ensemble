"""Unit tests for loading, prediction, thresholding and artifact integrity."""
import math

import pytest

from app.core.predictor import (ModelUnavailable, Predictor, clean_json, risk_flag,
                                summarise_folds)
from app.core.ratelimit import SlidingWindowLimiter
from tests.artifact_factory import THRESHOLD, build_artifact

PATIENT = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
           "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}


# TEST-005: probability exactly at the threshold
def test_flag_boundary_is_high_risk():
    assert risk_flag(0.5, 0.5) == "High Risk"
    assert risk_flag(0.4999, 0.5) == "Low Risk"
    assert risk_flag(0.0, 0.5) == "Low Risk"
    assert risk_flag(1.0, 0.5) == "High Risk"


def test_predict_returns_probability_in_range(tmp_path):
    p = Predictor.load(*build_artifact(tmp_path))
    out = p.predict(PATIENT)
    assert 0.0 <= out["risk_probability"] <= 1.0
    assert out["risk_flag"] in ("High Risk", "Low Risk")
    assert out["model_version"] == "m-test"
    assert out["threshold"] == THRESHOLD


def test_flag_is_consistent_with_threshold(tmp_path):
    p = Predictor.load(*build_artifact(tmp_path))
    out = p.predict(PATIENT)
    assert (out["risk_flag"] == "High Risk") == (out["risk_probability"] >= THRESHOLD - 1e-4)


# TEST-006: determinism
def test_same_input_same_output(tmp_path):
    p = Predictor.load(*build_artifact(tmp_path))
    assert p.predict(PATIENT) == p.predict(PATIENT)


def test_missing_feature_raises(tmp_path):
    p = Predictor.load(*build_artifact(tmp_path))
    bad = dict(PATIENT); bad.pop("age")
    with pytest.raises(ValueError):
        p.predict(bad)


# TEST-004: artifact missing / tampered
def test_missing_artifact_raises(tmp_path):
    with pytest.raises(ModelUnavailable):
        Predictor.load(tmp_path / "nope.joblib", tmp_path / "nope.json")


def test_checksum_mismatch_raises(tmp_path):
    model_path, meta_path = build_artifact(tmp_path, tamper_meta=True)
    with pytest.raises(ModelUnavailable):
        Predictor.load(model_path, meta_path)


def test_public_metadata_has_cv_summary_and_no_nan(tmp_path):
    meta = Predictor.load(*build_artifact(tmp_path)).public_metadata()
    assert meta["model_version"] == "m-test"
    assert meta["metrics"]["roc_auc"] == 0.91
    assert set(meta["cv_summary"]["mean"]) == {"sensitivity", "specificity", "f1", "roc_auc"}
    assert len(meta["cv_summary"]["folds"]) == 5


def test_clean_json_replaces_nan():
    assert clean_json({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}


def test_summarise_folds_mean_and_std():
    s = summarise_folds([{"f1": 0.8}, {"f1": 0.9}, {"f1": float("nan")}])
    assert math.isclose(s["mean"]["f1"], 0.85) and s["std"]["f1"] > 0


# rate limiter (pure logic, fake clock)
def test_rate_limiter_blocks_then_recovers():
    t = [0.0]
    lim = SlidingWindowLimiter(2, 60.0, clock=lambda: t[0])
    assert lim.allow("a") and lim.allow("a") and not lim.allow("a")
    assert lim.allow("b")            # other clients unaffected
    t[0] = 61.0
    assert lim.allow("a")            # window passed
    assert SlidingWindowLimiter(0).allow("x")   # 0 disables limiting
