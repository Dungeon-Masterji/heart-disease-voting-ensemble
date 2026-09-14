"""
Data validation (BR-002 / TR-002).

Runs data-quality checks against the ingested dataframe and produces a
structured report: missingness, duplicates, dtypes, and out-of-documented-
range values. This module never modifies or drops rows — it only reports,
so the findings can inform the leakage-safe preprocessing pipeline built in
Week 4, and so every run is reproducible from the untouched raw file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schema import FEATURE_SCHEMA


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    missing_by_column: dict[str, int]
    total_missing_cells: int
    n_duplicate_rows: int
    duplicate_row_indices: list[int]
    dtype_by_column: dict[str, str]
    out_of_range: dict[str, dict] = field(default_factory=dict)
    class_balance: dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_by_column": self.missing_by_column,
            "total_missing_cells": self.total_missing_cells,
            "n_duplicate_rows": self.n_duplicate_rows,
            "duplicate_row_indices": self.duplicate_row_indices,
            "dtype_by_column": self.dtype_by_column,
            "out_of_range": self.out_of_range,
            "class_balance": self.class_balance,
            "imbalance_ratio": self.imbalance_ratio,
        }


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Run all data-quality checks and return a ValidationReport.

    Checks performed (mirrors BR-002 acceptance criteria):
      - missingness per column and in total
      - exact duplicate rows
      - dtypes per column
      - values outside the documented clinical/coding range per feature
      - target class balance and imbalance ratio
    """
    missing_by_column = df.isna().sum().to_dict()
    missing_by_column = {k: int(v) for k, v in missing_by_column.items()}
    total_missing = int(df.isna().sum().sum())

    dup_mask = df.duplicated(keep=False)
    duplicate_indices = df.index[dup_mask].tolist()

    dtype_by_column = {c: str(t) for c, t in df.dtypes.items()}

    out_of_range: dict[str, dict] = {}
    for col, spec in FEATURE_SCHEMA.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        bad = series[(series < spec.min_value) | (series > spec.max_value)]
        if len(bad) > 0:
            out_of_range[col] = {
                "documented_range": [spec.min_value, spec.max_value],
                "n_out_of_range": int(len(bad)),
                "observed_bad_values": sorted(bad.unique().tolist()),
                "row_indices": bad.index.tolist(),
            }

    class_balance = {}
    imbalance_ratio = 0.0
    if "target" in df.columns:
        counts = df["target"].value_counts().sort_index()
        class_balance = {str(k): int(v) for k, v in counts.items()}
        if len(counts) == 2:
            imbalance_ratio = round(float(counts.max() / counts.min()), 3)

    return ValidationReport(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        missing_by_column=missing_by_column,
        total_missing_cells=total_missing,
        n_duplicate_rows=int(dup_mask.sum()),
        duplicate_row_indices=duplicate_indices,
        dtype_by_column=dtype_by_column,
        out_of_range=out_of_range,
        class_balance=class_balance,
        imbalance_ratio=imbalance_ratio,
    )


def save_report(report: ValidationReport, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_report.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2))
    return out_path


def print_summary(report: ValidationReport) -> None:
    print(f"Rows x Cols: {report.n_rows} x {report.n_cols}")
    print(f"Total missing cells: {report.total_missing_cells}")
    print(f"Duplicate rows: {report.n_duplicate_rows} (indices: {report.duplicate_row_indices})")
    print(f"Target class balance: {report.class_balance} (imbalance ratio: {report.imbalance_ratio})")
    if report.out_of_range:
        print("Out-of-documented-range values found:")
        for col, info in report.out_of_range.items():
            print(
                f"  - {col}: {info['n_out_of_range']} value(s) outside "
                f"{info['documented_range']} -> observed {info['observed_bad_values']}"
            )
    else:
        print("No out-of-documented-range values found.")


if __name__ == "__main__":
    from .ingestion import load_raw_dataset

    frame = load_raw_dataset()
    rep = validate_dataset(frame)
    print_summary(rep)
    saved_to = save_report(rep, Path(__file__).resolve().parents[3] / "training" / "artifacts")
    print(f"\nFull report written to {saved_to}")
