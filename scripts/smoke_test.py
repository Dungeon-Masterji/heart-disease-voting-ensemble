"""Post-deploy smoke test (PRD 13.4): /health + one sample prediction.

Usage:  python scripts/smoke_test.py [base_url]     (default http://localhost:8000)
Exit code 0 = healthy, 1 = failure.
"""

import json
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip(
    "/"
) + "/api/v1"
SAMPLE = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST" if data else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    status, health = call("/health")
    print("GET /health ->", status, health)
    if status != 200 or not health.get("model_loaded"):
        print("FAIL: service unhealthy")
        return 1
    status, score = call("/score", SAMPLE)
    print(
        "POST /score ->",
        status,
        {k: score.get(k) for k in ("risk_flag", "model_version", "threshold")},
    )
    if status != 200 or score.get("risk_flag") not in ("High Risk", "Low Risk"):
        print("FAIL: scoring failed")
        return 1
    print("OK: smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
