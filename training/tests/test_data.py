"""
Data science tests for the Week 1-3 scope: ingestion + validation only.
Maps to TEST-006/007/008 style checks from the PRD, scoped to what has
actually been built so far (no CV/leakage tests yet — those arrive with
the preprocessing pipeline in Week 4).

Run with: pytest training/tests/test_data.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from training.src.data.ingestion import load_raw_dataset
from training.src.data.schema import ALL_COLUMNS, TARGET_COLUMN
from training.src.data.validation import validate_dataset


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw_dataset()


def test_ingestion_loads_expected_shape(raw_df):
    assert raw_df.shape[0] == 303
    assert raw_df.shape[1] == 14


def test_ingestion_columns_match_schema(raw_df):
    assert list(raw_df.columns) == ALL_COLUMNS


def test_ingestion_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(tmp_path / "does_not_exist.csv")


def test_ingestion_raises_on_schema_mismatch(tmp_path):
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        load_raw_dataset(bad_file)


def test_validation_detects_known_duplicate(raw_df):
    report = validate_dataset(raw_df)
    assert report.n_duplicate_rows == 2
    assert set(report.duplicate_row_indices) == {163, 164}


def test_validation_detects_no_nan_missingness(raw_df):
    report = validate_dataset(raw_df)
    assert report.total_missing_cells == 0


def test_validation_flags_out_of_range_ca_and_thal(raw_df):
    report = validate_dataset(raw_df)
    assert "ca" in report.out_of_range
    assert "thal" in report.out_of_range
    assert report.out_of_range["ca"]["n_out_of_range"] == 5
    assert report.out_of_range["thal"]["n_out_of_range"] == 2


def test_validation_class_balance_matches_known_counts(raw_df):
    report = validate_dataset(raw_df)
    assert report.class_balance == {"0": 138, "1": 165}
    assert 1.0 < report.imbalance_ratio < 1.5


def test_target_is_binary(raw_df):
    assert set(raw_df[TARGET_COLUMN].unique()) == {0, 1}


def test_validation_is_deterministic(raw_df):
    """Running validation twice on the same frame must give identical results
    (no hidden randomness / mutation)."""
    r1 = validate_dataset(raw_df)
    r2 = validate_dataset(raw_df)
    assert r1.to_dict() == r2.to_dict()
