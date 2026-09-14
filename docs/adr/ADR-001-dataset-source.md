# ADR-001: Use a recoded/binarized CSV mirror of the UCI Cleveland dataset

**Status:** Accepted (Week 3)

## Context
The PRD requires the UCI Heart Disease (Cleveland) dataset. The canonical
`processed.cleveland.data` file uses 1-indexed categorical codes and `?` for
missing values. Standard practice (and the PRD itself, §1.6) is to binarize
the multi-class target before modeling.

## Decision
Use a pre-cleaned, zero-indexed, pre-binarized 303-row/14-column CSV variant
of the same Cleveland patients (see `docs/DATA_DICTIONARY.md` for full
provenance and the exact encoding differences from the UCI documentation
page).

## Consequences
- Schema and encoding differences from the raw UCI file are documented
  explicitly in `docs/DATA_DICTIONARY.md` and `training/src/data/schema.py`,
  not silently assumed.
- Validation (`training/src/data/validation.py`) treats the 7 out-of-range
  `ca`/`thal` values as a flagged, unresolved data-quality finding rather
  than assuming they're missing values and imputing them at ingestion time.
- If a future phase needs the strictly-original UCI encoding (e.g. to match
  published baselines exactly), re-deriving it from `processed.cleveland.data`
  is a documented, isolated change to `scripts/download_data.py` and
  `training/src/data/schema.py` only.
