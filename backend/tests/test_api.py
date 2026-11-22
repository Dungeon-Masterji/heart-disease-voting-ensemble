"""API tests (PRD 12.1). Needs: pip install -r requirements-dev.txt"""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.artifact_factory import build_artifact

PATIENT = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
           "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
BASE = "/api/v1"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return build_artifact(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="module")
def noscan_artifact(tmp_path_factory):
    return build_artifact(tmp_path_factory.mktemp("noscan"), exclude=("ca", "thal"))


@pytest.fixture()
def client(artifact, noscan_artifact, tmp_path):
    model_path, meta_path = artifact
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# Test card")
    settings = Settings(model_path=model_path, meta_path=meta_path, model_card_path=card,
                        noscan_model_path=noscan_artifact[0], noscan_meta_path=noscan_artifact[1],
                        rate_limit_per_minute=1000)
    with TestClient(create_app(settings)) as c:     # `with` runs the startup (model load)
        yield c


def test_health_ok(client):
    r = client.get(f"{BASE}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True and body["model_version"] == "m-test"


# TEST-001
def test_score_valid(client):
    r = client.post(f"{BASE}/score", json=PATIENT)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["risk_probability"] <= 1
    assert b["risk_flag"] in ("High Risk", "Low Risk")
    assert b["model_version"] == "m-test" and b["disclaimer"] and b["timestamp"]


# TEST-006
def test_score_is_deterministic(client):
    a = client.post(f"{BASE}/score", json=PATIENT).json()
    b = client.post(f"{BASE}/score", json=PATIENT).json()
    assert a["risk_probability"] == b["risk_probability"] and a["risk_flag"] == b["risk_flag"]


# TEST-002
def test_missing_field_422(client):
    bad = dict(PATIENT); bad.pop("chol")
    r = client.post(f"{BASE}/score", json=bad)
    assert r.status_code == 422
    assert any(e["field"] == "chol" for e in r.json()["errors"])


# TEST-003
def test_wrong_type_422(client):
    r = client.post(f"{BASE}/score", json={**PATIENT, "age": "abc"})
    assert r.status_code == 422


def test_numeric_string_is_not_silently_coerced(client):
    assert client.post(f"{BASE}/score", json={**PATIENT, "age": "63"}).status_code == 422


def test_out_of_range_422(client):
    assert client.post(f"{BASE}/score", json={**PATIENT, "age": -5}).status_code == 422
    assert client.post(f"{BASE}/score", json={**PATIENT, "ca": 4}).status_code == 422


def test_unknown_field_422(client):
    assert client.post(f"{BASE}/score", json={**PATIENT, "name": "x"}).status_code == 422


def test_malformed_json_400(client):
    r = client.post(f"{BASE}/score", content="{bad json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error_code"] == "MALFORMED_JSON"


def test_unknown_route_404(client):
    assert client.get(f"{BASE}/nope").status_code == 404


# TEST-010
def test_model_endpoint(client):
    b = client.get(f"{BASE}/model").json()
    assert b["model_version"] == "m-test" and "threshold" in b
    assert b["metrics"]["sensitivity"] == 0.9
    assert b["cv_summary"]["mean"]["f1"] > 0


def test_model_card_endpoint(client):
    r = client.get(f"{BASE}/model-card")
    assert r.status_code == 200 and r.json()["content"].startswith("# Test card")


# TEST-004
def test_model_unavailable_503(tmp_path):
    settings = Settings(model_path=tmp_path / "missing.joblib", meta_path=tmp_path / "missing.json",
                        model_card_path=tmp_path / "x.md", rate_limit_per_minute=1000)
    with TestClient(create_app(settings)) as c:
        h = c.get(f"{BASE}/health")
        assert h.status_code == 503 and h.json()["model_loaded"] is False
        r = c.post(f"{BASE}/score", json=PATIENT)
        assert r.status_code == 503 and r.json()["error_code"] == "MODEL_UNAVAILABLE"


def test_rate_limit_429(artifact, tmp_path):
    model_path, meta_path = artifact
    settings = Settings(model_path=model_path, meta_path=meta_path,
                        model_card_path=tmp_path / "x.md", rate_limit_per_minute=2)
    with TestClient(create_app(settings)) as c:
        codes = [c.post(f"{BASE}/score", json=PATIENT).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert c.get(f"{BASE}/health").status_code == 200   # health is never rate-limited


def test_complete_patient_uses_the_full_model(client):
    b = client.post(f"{BASE}/score", json=PATIENT).json()
    assert b["model_used"] == "full" and b["missing_tests"] == [] and b["notice"] is None


@pytest.mark.parametrize("missing", [["ca"], ["thal"], ["ca", "thal"]])
@pytest.mark.parametrize("as_null", [True, False])
def test_any_missing_scan_uses_the_noscan_model(client, missing, as_null):
    body = {k: v for k, v in PATIENT.items() if k not in missing}
    if as_null:
        body.update(dict.fromkeys(missing))
    r = client.post(f"{BASE}/score", json=body)
    assert r.status_code == 200
    b = r.json()
    assert b["model_used"] == "no_scan" and b["model_version"] == "m-test-noscan"
    assert b["missing_tests"] == missing and "trained without" in b["notice"]


def test_other_fields_are_still_required(client):
    for field in set(PATIENT) - {"ca", "thal"}:
        assert client.post(f"{BASE}/score", json={**PATIENT, field: None}).status_code == 422


def test_noscan_model_missing_only_affects_patients_missing_a_scan(artifact, tmp_path):
    settings = Settings(model_path=artifact[0], meta_path=artifact[1],
                        noscan_model_path=tmp_path / "x.joblib", noscan_meta_path=tmp_path / "x.json",
                        model_card_path=tmp_path / "x.md", rate_limit_per_minute=1000)
    with TestClient(create_app(settings)) as c:
        assert c.get(f"{BASE}/health").json()["noscan_model_loaded"] is False
        assert c.post(f"{BASE}/score", json=PATIENT).status_code == 200
        for missing in ({"ca": None}, {"thal": None}, {"ca": None, "thal": None}):
            r = c.post(f"{BASE}/score", json={**PATIENT, **missing})
            assert r.status_code == 503 and r.json()["error_code"] == "MODEL_UNAVAILABLE"


def test_model_endpoint_describes_both_models(client):
    b = client.get(f"{BASE}/model").json()
    assert "ca" in b["features"] and "ca" not in b["noscan_model"]["features"]
