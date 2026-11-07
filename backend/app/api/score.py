from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.core.errors import ApiError
from app.core.predictor import DISCLAIMER
from app.models.schemas import PatientFeatures, ScoreResponse

router = APIRouter(tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def score(features: PatientFeatures, request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise ApiError(503, "MODEL_UNAVAILABLE",
                       "The scoring model is currently unavailable. Please try again later.")
    result = predictor.predict(features.model_dump())
    return ScoreResponse(
        **result,
        timestamp=datetime.now(timezone.utc).isoformat(),
        disclaimer=DISCLAIMER,
    )
