import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(repo_root):
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def git_dirty(repo_root):
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def split_fold_indices(cv, X, y):
    return [
        {
            "fold": fold,
            "train": [int(index) for index in train_idx],
            "validation": [int(index) for index in validation_idx],
        }
        for fold, (train_idx, validation_idx) in enumerate(cv.split(X, y), 1)
    ]


def dataset_record(dataset_path, row_count, columns, after_cleaning=False):
    dataset_path = Path(dataset_path)
    return {
        "path": str(dataset_path),
        "sha256": sha256(dataset_path),
        "row_count": int(row_count),
        "column_schema": {
            "columns": [str(column) for column in columns],
            "after_cleaning": bool(after_cleaning),
        },
    }


def write_run(run, artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = run.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    record = dict(run)
    record["run_id"] = run_id
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    def default(value):
        if isinstance(value, numpy.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot JSON serialize {type(value).__name__}")

    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=default, sort_keys=True))
    return path, record
