"""Pydantic request/response schemas. Ranges mirror training/src/data/schema.py
(clinical plausibility + documented codes for this dataset variant)."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


class PatientFeatures(BaseModel):
    # strict: "63" (a string) is rejected instead of silently coerced; unknown keys rejected.
    model_config = ConfigDict(strict=True, extra="forbid",
                              json_schema_extra={"examples": [SAMPLE_PATIENT]})

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (dataset code 0-3)")
    trestbps: int = Field(..., ge=60, le=260, description="Resting blood pressure, mm Hg")
    chol: int = Field(..., ge=50, le=700, description="Serum cholesterol, mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(..., ge=60, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0.0, le=7.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3)")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia code (1-3)")


class ScoreResponse(BaseModel):
    risk_probability: float
    risk_flag: Literal["High Risk", "Low Risk"]
    model_version: str
    voting_strategy: str
    threshold: float
    timestamp: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    timestamp: str


class ModelCardResponse(BaseModel):
    format: str
    content: str


class FieldError(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
    errors: Optional[List[FieldError]] = None
