# ADR-007: Second ensemble for patients without `ca` and/or `thal`

**Status:** Accepted

## Context
`ca` (fluoroscopy) and `thal` (thallium scan) are specialist tests many patients have
not had. The full model imputes a missing value with the most common one (`ca` = 0,
`thal` = normal), so "not done" was treated as "normal". On held-out Cleveland
patients with both scans blanked, its sensitivity fell to 0.710 (AUC 0.857).

## Decision
- Train the same voting ensemble on the other 11 features (`m2.0.0-noscan`), with its
  own tuning, threshold and honest check (sensitivity 0.891, specificity 0.634, AUC 0.872).
- `POST /score` accepts `ca` and `thal` as null or omitted; all other fields stay required.
  If either is missing the no-scan model scores the patient and the response says so.
  The full model refuses any patient with a missing scan.
- The form needs an explicit "Test not done" choice, so a blank field is still an error.
- If the no-scan model cannot be loaded, patients missing a scan get a 503.

## Consequences
- The no-scan model is less accurate than the full model with its tests available.
- When only one scan is missing, the other is ignored. Using the full model there was tried
  and rejected. On held-out folds it ranked slightly better (AUC 0.881 without `ca`, 0.893
  without `thal`, against 0.872; differences not statistically clear) but its sensitivity was
  0.833 and 0.862, against 0.906. At equal sensitivity the two were about the same.
- Its higher sensitivity is partly bought with specificity.
