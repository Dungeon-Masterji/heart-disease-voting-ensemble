from datetime import UTC, datetime

from fastapi import APIRouter, Request

from app.core.errors import ApiError
from app.core.predictor import DISCLAIMER, NOSCAN_NOTICE, SCAN_FEATURES
from app.models.schemas import PatientFeatures, ScoreResponse

router = APIRouter(tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def score(features: PatientFeatures, request: Request):
    values = features.model_dump()
    # The full model would treat a missing scan as "normal" and miss sick patients,
    # so if either is missing a model trained without them is used (ADR-007).
    missing = [f for f in SCAN_FEATURES if values[f] is None]
    state = request.app.state
    predictor = state.noscan_predictor if missing else state.predictor
    if predictor is None:
        raise ApiError(
            503,
            "MODEL_UNAVAILABLE",
            "The scoring model is currently unavailable. Please try again later.",
        )
    result = predictor.predict(values)
    return ScoreResponse(
        **result,
        model_used="no_scan" if missing else "full",
        missing_tests=missing,
        notice=NOSCAN_NOTICE if missing else None,
        timestamp=datetime.now(UTC).isoformat(),
        disclaimer=DISCLAIMER,
    )
