"""Guards against the inverted-target bug of m1.0.0 (ADR-006)."""

import pandas as pd
import pytest

from training.src.data.external import recode_uci
from training.src.data.ingestion import DEFAULT_RAW_PATH, load_raw_dataset
from training.src.data.labels import (
    LabelOrientationError,
    check_disease_labels,
    to_disease_label,
)
from training.src.data.schema import FEATURE_COLUMNS

# UCI processed.cleveland.data, rows 1-3 (raw UCI codes, last value is num).
UCI_ROWS = [
    [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
    [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
    [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
]


def uci_frame():
    return pd.DataFrame(UCI_ROWS, columns=[*FEATURE_COLUMNS, "num"])


def test_loaded_dataset_has_disease_labels():
    check_disease_labels(load_raw_dataset())


def test_raw_file_is_inverted_and_is_rejected_unflipped():
    with pytest.raises(LabelOrientationError):
        check_disease_labels(pd.read_csv(DEFAULT_RAW_PATH))


def test_an_already_corrected_file_is_not_flipped_again():
    with pytest.raises(LabelOrientationError):
        to_disease_label(load_raw_dataset())


def test_unrelated_dataset_is_rejected():
    with pytest.raises(LabelOrientationError):
        to_disease_label(load_raw_dataset().iloc[:50])


def test_disease_goes_with_worse_clinical_findings():
    means = load_raw_dataset().groupby("target").mean(numeric_only=True)
    for feature in ("oldpeak", "ca", "exang"):
        assert means.loc[1, feature] > means.loc[0, feature]
    assert means.loc[1, "thalach"] < means.loc[0, "thalach"]


def test_uci_recoding_reproduces_the_training_rows():
    X, disease = recode_uci(uci_frame())
    project = load_raw_dataset()
    for i in range(len(X)):
        match = project[(project[FEATURE_COLUMNS] == X.loc[i]).all(axis=1)]
        assert len(match) == 1
        assert match["target"].iloc[0] == disease[i]


def test_recoding_treats_impossible_zeros_as_missing():
    raw = uci_frame()
    raw.loc[0, ["chol", "trestbps"]] = 0
    X, _ = recode_uci(raw)
    assert X.loc[0, ["chol", "trestbps"]].isna().all()
