from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    """Fails with 503 when the model artifact is not loaded (PRD 6.4)."""
    predictor = request.app.state.predictor
    loaded = predictor is not None
    body = HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_version=predictor.version if loaded else None,
        timestamp=datetime.now(UTC).isoformat(),
    )
    return JSONResponse(
        status_code=200 if loaded else 503, content=body.model_dump(mode="json")
    )
