"""Score the published model, at its published threshold, on unseen hospitals.

Nothing is fitted or tuned here.

Run with: python -m training.src.models.external_validation
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from training.src.data.external import SITES, load_site
from training.src.models.evaluation import metrics_at
from training.src.models.export import sha256
from training.src.models.train import MODEL_VERSION, NOSCAN_VERSION

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def bootstrap_auc_ci(y, probabilities, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores: list[float] = []
    while len(scores) < n_boot:
        index = rng.integers(0, len(y), len(y))
        if 0 < y[index].sum() < len(index):
            scores.append(float(roc_auc_score(y[index], probabilities[index])))
    low, high = np.percentile(scores, [2.5, 97.5])
    return [float(low), float(high)]


def evaluate_site(model, threshold, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    missing = X.isna().mean()
    result = metrics_at(y, probabilities >= threshold, threshold)
    result.update(
        n=len(y),
        n_disease=int(y.sum()),
        n_healthy=int((y == 0).sum()),
        roc_auc=float(roc_auc_score(y, probabilities)),
        roc_auc_ci95=bootstrap_auc_ci(y, probabilities),
        brier_score=float(brier_score_loss(y, probabilities)),
        missing_fraction={c: round(float(v), 3) for c, v in missing.items() if v > 0},
    )
    return result


def validate(version, out_path):
    model_path = ARTIFACTS / "models" / f"model_{version}.joblib"
    metadata = json.loads(model_path.with_name(f"model_{version}.json").read_text())
    if sha256(model_path) != metadata["artifact_sha256"]:
        raise RuntimeError("Model file does not match its metadata checksum")
    model = joblib.load(model_path)
    report = {
        "model_version": version,
        "artifact_sha256": metadata["artifact_sha256"],
        "threshold": metadata["threshold"],
        "sites": {},
    }
    print(version)
    for site, name in SITES.items():
        X, y = load_site(site)
        X = X[metadata["features"]]
        result = evaluate_site(model, metadata["threshold"], X, y)
        report["sites"][site] = {"name": name, **result}
        print(
            f"  {name}: n={result['n']} AUC={result['roc_auc']:.3f} "
            f"CI={[round(v, 3) for v in result['roc_auc_ci95']]} "
            f"sens={result['sensitivity']:.3f} spec={result['specificity']:.3f} "
            f"brier={result['brier_score']:.3f}"
        )
    out_path.write_text(json.dumps(report, indent=2))


def main():
    validate(MODEL_VERSION, ARTIFACTS / "external_validation.json")
    validate(NOSCAN_VERSION, ARTIFACTS / "noscan" / "external_validation.json")


if __name__ == "__main__":
    main()
