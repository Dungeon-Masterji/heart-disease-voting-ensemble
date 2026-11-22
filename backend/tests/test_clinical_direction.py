"""The published model must score sicker patients higher. Uses the real artifact:
v1 passed every other test while ranking patients backwards."""
import pytest

from app.config import DEFAULT_MODEL_DIR, Settings
from app.core.predictor import FEATURE_ORDER, ModelUnavailable, Predictor

BASE = {"age": 55, "sex": 1, "trestbps": 130, "chol": 240, "fbs": 0, "restecg": 1}
WELL = {**BASE, "cp": 2, "thalach": 175, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2}
SICK = {**BASE, "cp": 0, "thalach": 110, "exang": 1, "oldpeak": 3.0, "slope": 1, "ca": 3, "thal": 3}


@pytest.fixture(scope="module")
def predictor():
    settings = Settings()
    return Predictor.load(settings.model_path, settings.meta_path)


@pytest.fixture(scope="module")
def noscan():
    settings = Settings()
    return Predictor.load(settings.noscan_model_path, settings.noscan_meta_path)


def without_scans(patient):
    return {k: v for k, v in patient.items() if k not in ("ca", "thal")}


def test_noscan_model_flags_severe_not_clean_profile(noscan):
    assert noscan.features == [f for f in FEATURE_ORDER if f not in ("ca", "thal")]
    assert noscan.predict(without_scans(SICK))["risk_flag"] == "High Risk"
    assert noscan.predict(without_scans(WELL))["risk_flag"] == "Low Risk"


@pytest.mark.parametrize("change", [{"oldpeak": 4.0}, {"exang": 1}, {"thalach": 110}, {"cp": 0}])
def test_noscan_each_worsening_finding_raises_risk(noscan, change):
    before = noscan.predict(without_scans(WELL))["risk_probability"]
    assert noscan.predict({**without_scans(WELL), **change})["risk_probability"] > before


def test_severe_profile_is_high_risk_and_clean_profile_is_low_risk(predictor):
    assert predictor.predict(SICK)["risk_flag"] == "High Risk"
    assert predictor.predict(WELL)["risk_flag"] == "Low Risk"


@pytest.mark.parametrize("change", [
    {"ca": 3}, {"oldpeak": 4.0}, {"exang": 1}, {"thal": 3}, {"thalach": 110}, {"cp": 0},
])
def test_each_worsening_finding_raises_risk(predictor, change):
    before = predictor.predict(WELL)["risk_probability"]
    assert predictor.predict({**WELL, **change})["risk_probability"] > before


def test_known_uci_patients_rank_correctly(predictor):
    # UCI processed.cleveland.data rows 1 (num=0) and 3 (num=1)
    healthy = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
               "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
    diseased = {"age": 67, "sex": 1, "cp": 0, "trestbps": 120, "chol": 229, "fbs": 0, "restecg": 0,
                "thalach": 129, "exang": 1, "oldpeak": 2.6, "slope": 1, "ca": 2, "thal": 3}
    assert (predictor.predict(diseased)["risk_probability"]
            > predictor.predict(healthy)["risk_probability"])


def test_inverted_v1_artifact_is_refused():
    with pytest.raises(ModelUnavailable, match="label"):
        Predictor.load(DEFAULT_MODEL_DIR / "model_m1.0.0.joblib",
                       DEFAULT_MODEL_DIR / "model_m1.0.0.json")
