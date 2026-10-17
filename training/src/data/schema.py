

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureRange:
    dtype: str
    min_value: float
    max_value: float
    description: str


# Clinical plausibility ranges used for range validation (BR-002 / TR-002).
# Bounds are deliberately generous (physiological plausibility, not the
# narrow observed range in this 303-row sample) so validation catches
# genuinely bad values without rejecting legitimate edge-case patients.
FEATURE_SCHEMA: dict[str, FeatureRange] = {
    "age": FeatureRange("int", 0, 120, "Age in years"),
    "sex": FeatureRange("int", 0, 1, "1 = male, 0 = female"),
    "cp": FeatureRange("int", 0, 3, "Chest pain type (0-3, dataset-observed encoding)"),
    "trestbps": FeatureRange("int", 60, 260, "Resting blood pressure (mm Hg)"),
    "chol": FeatureRange("int", 50, 700, "Serum cholesterol (mg/dl)"),
    "fbs": FeatureRange("int", 0, 1, "Fasting blood sugar > 120 mg/dl (1=true)"),
    "restecg": FeatureRange("int", 0, 2, "Resting ECG result (0-2)"),
    "thalach": FeatureRange("int", 60, 250, "Max heart rate achieved"),
    "exang": FeatureRange("int", 0, 1, "Exercise-induced angina (1=yes)"),
    "oldpeak": FeatureRange("float", 0.0, 7.0, "ST depression induced by exercise"),
    "slope": FeatureRange("int", 0, 2, "Slope of peak exercise ST segment (0-2)"),
    "ca": FeatureRange("int", 0, 3, "Number of major vessels colored by fluoroscopy (0-3 documented)"),
    "thal": FeatureRange("int", 1, 3, "Thalassemia code (1-3 documented in this recoding)"),
    "target": FeatureRange("int", 0, 1, "1 = heart disease present, 0 = absent"),
}

FEATURE_COLUMNS = [c for c in FEATURE_SCHEMA if c != "target"]
TARGET_COLUMN = "target"
ALL_COLUMNS = list(FEATURE_SCHEMA.keys())
