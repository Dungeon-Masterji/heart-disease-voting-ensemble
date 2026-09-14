# Data Dictionary — Heart Disease Dataset

## Source and provenance

- **Origin dataset:** UCI Machine Learning Repository, *Heart Disease*
  (Cleveland Clinic Foundation subset), donated 1988.
  https://archive.ics.uci.edu/dataset/45/heart+disease
- **File actually used in this repo:** `data/raw/heart_disease.csv`, fetched
  by `scripts/download_data.py`. This is a **widely-circulated, pre-cleaned,
  zero-indexed recoding** of the same 303 Cleveland patients (commonly
  distributed as `heart.csv` on Kaggle/HuggingFace/GitHub mirrors), **not**
  the original `processed.cleveland.data` file with `?`-marked missing
  values.
- **Row count:** 303 patients, matching the UCI-documented instance count
  for the Cleveland subset.
- **Columns:** 13 features + 1 target = 14 columns.

## Why the recoded variant, not the raw UCI file

The raw `processed.cleveland.data` file uses 1-indexed categorical codes and
a `?` sentinel for missing values, and its multi-class `num` target
(0–4) is standard practice to binarize before modeling anyway (PRD §10.2,
§1.6). The CSV used here already reflects those two conventional
transformations. This is documented explicitly — rather than silently
assumed — because the PRD (§1.6) requires the schema to be taken from
actual documentation, not fabricated. Section "Encoding discrepancy" below
spells out exactly what changed.

## Column reference

| Column | Type | Documented range (this variant) | Clinical meaning |
|---|---|---|---|
| `age` | int | 0–120 (observed: 29–77) | Age in years |
| `sex` | int | {0,1} | 1 = male, 0 = female |
| `cp` | int | {0,1,2,3} | Chest pain type (0=typical angina ... 3=asymptomatic) |
| `trestbps` | int | 60–260 (observed: 94–200) | Resting blood pressure, mm Hg |
| `chol` | int | 50–700 (observed: 126–564) | Serum cholesterol, mg/dl |
| `fbs` | int | {0,1} | Fasting blood sugar > 120 mg/dl |
| `restecg` | int | {0,1,2} | Resting ECG result |
| `thalach` | int | 60–250 (observed: 71–202) | Max heart rate achieved |
| `exang` | int | {0,1} | Exercise-induced angina |
| `oldpeak` | float | 0.0–7.0 (observed: 0.0–6.2) | ST depression vs. rest |
| `slope` | int | {0,1,2} | Slope of peak exercise ST segment |
| `ca` | int | {0,1,2,3} documented; **5 rows observed at 4** | Major vessels colored by fluoroscopy |
| `thal` | int | {1,2,3} documented; **2 rows observed at 0** | Thalassemia code |
| `target` | int | {0,1} | 1 = heart disease present |

Full clinical range bounds used for programmatic validation live in
`training/src/data/schema.py` (single source of truth — this table mirrors
it for human reading, so if they ever drift, the code is authoritative).

## Encoding discrepancy vs. the original UCI documentation

| Field | UCI-documented (raw file) | This CSV | Note |
|---|---|---|---|
| `cp` | 1–4 | 0–3 | Shifted down by 1 |
| `slope` | 1–3 | 0–2 | Shifted down by 1 |
| `thal` | {3, 6, 7} | {0,1,2,3} observed | Recoded, not a simple shift |
| `num`/`target` | 0–4 (multi-class) | {0,1} (binary) | Pre-binarized |

## Known data quality issues (confirmed by `training/src/data/validation.py`)

1. **7 out-of-documented-range values.** 5 rows have `ca == 4` (documented
   max is 3); 2 rows have `thal == 0` (documented minimum is 1). These are
   very likely where the original UCI file's 6 `?` (missing) values landed
   after this variant's cleaning process coerced them to an integer instead
   of dropping them — 6 is suspiciously close to 7, and the original file is
   documented to have exactly 6 missing values across `ca` and `thal`. This
   is a hypothesis stated openly, not a fact — the exact provenance can't be
   confirmed without the original preprocessing script. **No value is
   dropped or imputed at this stage** — that decision is deferred to Week 4
   leakage-safe feature engineering, per PRD §10.4.
2. **1 duplicate row pair** (index 163 and 164, identical across all 14
   columns) — a known artifact of this widely-circulated CSV. Flagged, not
   silently removed.
3. **No `NaN` values** in this file (0 missing cells) — the missingness that
   exists takes the form of the out-of-range codes above, not empty cells.

## Applicability to the "clinical decision-support" business context (PRD §1.6)

This is 1980s single-institution (Cleveland Clinic) data, 303 patients.
Per the PRD's own risk register (§2.6) and Model Card ethical
considerations (§21.7), it is **not** representative of a modern, diverse
patient population, and any model trained on it is a portfolio/educational
artifact, not a validated clinical tool. This assessment carries forward
unchanged into later weeks' Model Card.
