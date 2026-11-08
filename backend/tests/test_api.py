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


@pytest.fixture()
def client(artifact, tmp_path):
    model_path, meta_path = artifact
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# Test card")
    settings = Settings(model_path=model_path, meta_path=meta_path, model_card_path=card,
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
