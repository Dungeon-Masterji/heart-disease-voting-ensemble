# ADR-006: Correct the inverted target label

**Status:** Accepted. Supersedes every m1.0.0 result.

## Context
m1.0.0 treated `target = 1` in the raw CSV as heart disease. Matching all 303 rows
to UCI `processed.cleveland.data` showed the opposite: every `target = 1` patient
has `num = 0` (healthy) and every `target = 0` patient has `num > 0`. m1.0.0
therefore reported the probability of being healthy as "risk". Against the UCI
diagnosis its operating point had sensitivity 0.20 and specificity 0.10.

The inversion is symmetric, so cross-validation, AUC and the test suite all looked
healthy. The notebook's error analysis did show it: "caught" patients had milder
findings than "missed" ones.

## Decision
- Keep the raw file untouched; flip once in `load_raw_dataset()`.
- Verify against patients with known UCI diagnoses on every load. A mirror that
  ships corrected labels raises an error rather than being flipped again.
- Nothing else reads the CSV directly, including the notebook.
- Metadata carries `label_definition`; the API refuses a model without it, which
  blocks m1.0.0.
- Tests assert clinical direction on the served model.
- Only `train.py` exports a model. The notebook checks it agrees with it.

## Consequences
- Disease is now the minority class, so F1 and specificity are lower than the
  m1.0.0 figures. Those figures were never valid for disease.
- The same encoding makes external UCI hospitals usable as unseen test sets.
