from fastapi import APIRouter, Request

from app.core.errors import ApiError
from app.models.schemas import ModelCardResponse

router = APIRouter(tags=["model"])


@router.get("/model-card", response_model=ModelCardResponse)
def model_card(request: Request):
    path = request.app.state.settings.model_card_path
    if not path.exists():
        raise ApiError(404, "MODEL_CARD_NOT_FOUND",
                       "Model card not generated yet. Run: python scripts/generate_model_card.py")
    return ModelCardResponse(format="markdown", content=path.read_text(encoding="utf-8"))
