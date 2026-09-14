"""
Downloads the raw heart-disease CSV used by this project into data/raw/.

This project uses a widely-circulated, pre-cleaned, zero-indexed recoding of
the UCI Cleveland Heart Disease data (303 patients, 14 columns, target
already binarized to {0,1}) rather than the original `?`-delimited
processed.cleveland.data file. See docs/DATA_DICTIONARY.md for exactly why,
and for the encoding differences versus the UCI documentation page
(archive.ics.uci.edu/dataset/45/heart+disease).

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# A commonly-used, publicly-hosted mirror of the same 303-row / 14-column
# recoded dataset. If this specific mirror ever goes down, any host serving
# the identical "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,
# slope,ca,thal,target" 303-row CSV is an equivalent, valid substitute --
# just verify row count (303) and column order after downloading.
DATA_URL = "https://huggingface.co/datasets/prhbrt/machine_learning_courses/raw/main/heart.csv"

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart_disease.csv"


def main() -> None:
    if OUT_PATH.exists():
        print(f"Dataset already present at {OUT_PATH} — skipping download.")
        print("Delete the file first if you want to re-fetch it.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n  {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, OUT_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "If this mirror is unreachable, fetch the same dataset manually "
            "(see docs/DATA_DICTIONARY.md for provenance and an equivalent "
            "source) and save it to data/raw/heart_disease.csv with columns:\n"
            "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,"
            "slope,ca,thal,target",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
