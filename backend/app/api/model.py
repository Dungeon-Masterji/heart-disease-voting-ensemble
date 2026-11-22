from fastapi import APIRouter, Request

from app.core.errors import ApiError

router = APIRouter(tags=["model"])


@router.get("/model")
def model_info(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise ApiError(503, "MODEL_UNAVAILABLE", "The model is currently unavailable.")
    noscan = request.app.state.noscan_predictor
    return {
        **predictor.public_metadata(),
        "noscan_model": noscan.public_metadata() if noscan else None,
    }
