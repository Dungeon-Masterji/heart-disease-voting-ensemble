"""UCI hospitals never used for training, recoded to this project's encoding.

The recoding was derived by matching every row of the training CSV to UCI
`processed.cleveland.data`; each raw code maps to exactly one project code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import FEATURE_COLUMNS

SITES = {
    "hungarian": "Hungary (Budapest)",
    "va": "VA Long Beach",
    "switzerland": "Switzerland (Zurich/Basel)",
}
EXTERNAL_DIR = Path(__file__).resolve().parents[3] / "data" / "external"

UCI_TO_PROJECT = {
    "cp": {1: 3, 2: 1, 3: 2, 4: 0},
    "restecg": {0: 1, 1: 2, 2: 0},
    "slope": {1: 2, 2: 1, 3: 0},
    "thal": {3: 2, 6: 1, 7: 3},
}


def recode_uci(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features in project encoding, disease label 1 = num > 0)."""
    disease = (raw["num"] > 0).astype(int)
    X = raw[FEATURE_COLUMNS].astype(float)
    for column, mapping in UCI_TO_PROJECT.items():
        X[column] = X[column].map(mapping)
    # A recorded 0 is impossible for these two measurements: it marks missing.
    for column in ("chol", "trestbps"):
        X[column] = X[column].mask(X[column] == 0)
    return X, disease


def load_site(
    site: str, directory: Path = EXTERNAL_DIR
) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(directory) / f"processed.{site}.data"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_external_data.py` first."
        )
    raw = pd.read_csv(path, names=[*FEATURE_COLUMNS, "num"], na_values="?")
    return recode_uci(raw)
