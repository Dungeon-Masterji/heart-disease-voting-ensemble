"""Target-label orientation (see docs/adr/ADR-006-target-label-inversion.md).

In the raw CSV `target` is inverted relative to the UCI diagnosis: 1 means
`num == 0` (NO disease) and 0 means `num > 0` (disease). All 303 rows were
matched against UCI `processed.cleveland.data` to confirm this. The rest of the
project uses 1 = heart disease present, so ingestion flips the column once.

The flip is only correct while the file stays inverted, so it is guarded by
patients whose UCI diagnosis is known. A mirror with corrected labels fails
loudly instead of being silently inverted again.
"""

from __future__ import annotations

import pandas as pd

LABEL_DEFINITION = "1 = heart disease present (UCI num > 0), 0 = absent (UCI num = 0)"

# (age, sex, trestbps, chol, thalach, exang, oldpeak) -> UCI disease (num > 0)
_ANCHOR_COLUMNS = ["age", "sex", "trestbps", "chol", "thalach", "exang", "oldpeak"]
UCI_ANCHORS: list[tuple[tuple[float, ...], int]] = [
    ((63, 1, 145, 233, 150, 0, 2.3), 0),
    ((67, 1, 160, 286, 108, 1, 1.5), 1),
    ((67, 1, 120, 229, 129, 1, 2.6), 1),
    ((37, 1, 130, 250, 187, 0, 3.5), 0),
    ((41, 0, 130, 204, 172, 0, 1.4), 0),
    ((62, 0, 140, 268, 160, 0, 3.6), 1),
]


class LabelOrientationError(ValueError):
    """The label does not have the orientation this project relies on."""


def check_disease_labels(df: pd.DataFrame) -> None:
    """Raise unless `target` agrees with the known UCI diagnoses."""
    for values, disease in UCI_ANCHORS:
        mask = pd.Series(True, index=df.index)
        for column, value in zip(_ANCHOR_COLUMNS, values):
            mask &= (df[column] - value).abs() < 1e-6
        found = df.loc[mask, "target"]
        if found.empty:
            raise LabelOrientationError(
                f"Reference patient {values} not found; this is not the expected dataset."
            )
        if not (found == disease).all():
            raise LabelOrientationError(
                f"Reference patient {values} should have disease={disease} per UCI, "
                f"got {found.tolist()}. The raw label encoding has changed."
            )


def to_disease_label(raw: pd.DataFrame) -> pd.DataFrame:
    """Flip the raw label to 1 = disease and verify the result."""
    out = raw.copy()
    out["target"] = 1 - out["target"]
    check_disease_labels(out)
    return out
