# EDA Summary — Week 3

Dataset: 303 rows x 14 columns (see docs/DATA_DICTIONARY.md for full column provenance).

## Target distribution
- Class 0 (absence): 138
- Class 1 (presence): 165
- Imbalance ratio: 1.2:1 — mild imbalance, not severe. Stratified k-fold CV (planned for Week 6+) is still warranted.

## Missingness
- Total missing cells (NaN): 0.
- No NaNs are present in this CSV variant, but validation.py flags 5 rows where `ca`=4 and 2 rows where `thal`=0 — both outside the documented coding range for this dataset variant. These almost certainly correspond to the 6 originally-missing values in the raw UCI Cleveland file (marked `?`) that were coerced to an out-of-range integer by whichever preprocessing produced this CSV, rather than true clinical measurements. This is flagged, not silently dropped — the imputation/exclusion decision belongs to Week 4 feature engineering, not to EDA.

## Duplicates
- 1 exact duplicate row pair detected (patients at index 163/164 — identical across all 14 columns). Documented as a known artifact of this widely-circulated CSV variant; to be resolved during Week 4 preprocessing (drop or keep-with-justification), not silently here.

## Correlation with target (top 3 by |r|)
- `exang`: r = -0.437
- `cp`: r = 0.434
- `oldpeak`: r = -0.431

*Correlation does not imply causality — this is descriptive EDA only, consistent with the PRD's caution in section 10.3.*

## Outlier counts (IQR method, continuous features)
- `age`: 0 potential outlier(s)
- `trestbps`: 9 potential outlier(s)
- `chol`: 5 potential outlier(s)
- `thalach`: 1 potential outlier(s)
- `oldpeak`: 5 potential outlier(s)

## Figures
- `target_distribution.png`
- `continuous_distributions.png`
- `categorical_distributions.png`
- `correlation_heatmap.png`
- `missingness_heatmap.png`
- `target_boxplots.png`

## Carried forward to Week 4
- Decide imputation/exclusion for the 7 out-of-documented-range `ca`/`thal` rows (fit on train folds only).
- Decide whether to drop the 1 duplicate row before or within CV (duplicate must not span train/validation split in the same fold).
- `oldpeak` and `chol` show the most outliers by IQR — review clinical plausibility before deciding on winsorization vs. keep.