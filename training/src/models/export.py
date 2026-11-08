import hashlib
import json
from pathlib import Path

import joblib


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_model(model, metadata, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_m1.0.0.joblib"
    metadata_path = model_dir / "model_m1.0.0.json"
    joblib.dump(model, model_path)
    metadata = dict(metadata)
    metadata["artifact_sha256"] = sha256(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    return model_path, metadata_path, metadata
