"""
Data ingestion (BR-001 / TR-001).

Loads the raw heart-disease CSV from disk, applies the documented column
schema, and returns a pandas DataFrame. Does NOT impute, scale, encode, or
drop anything — ingestion is intentionally "dumb": it only parses. All
cleaning decisions belong to validation.py (reporting) and, from Week 4
onward, the leakage-safe preprocessing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .schema import ALL_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "heart_disease.csv"


def load_raw_dataset(path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the raw heart-disease CSV and verify it matches the expected schema.

    Parameters
    ----------
    path : str | Path
        Location of the raw CSV. Defaults to data/raw/heart_disease.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with columns in the documented order (see schema.py).

    Raises
    ------
    FileNotFoundError
        If the raw file is missing (run `python scripts/download_data.py` first).
    ValueError
        If the file's columns don't match the documented schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python scripts/download_data.py` "
            "first (see README.md — 'Local Setup')."
        )

    df = pd.read_csv(path)

    missing_cols = set(ALL_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(ALL_COLUMNS)
    if missing_cols or extra_cols:
        raise ValueError(
            f"Schema mismatch. Missing columns: {sorted(missing_cols)}. "
            f"Unexpected columns: {sorted(extra_cols)}."
        )

    df = df[ALL_COLUMNS]
    logger.info("Loaded raw dataset: %s rows, %s columns from %s", df.shape[0], df.shape[1], path)
    return df


if __name__ == "__main__":
    frame = load_raw_dataset()
    print(frame.head())
    print(f"\nShape: {frame.shape}")
