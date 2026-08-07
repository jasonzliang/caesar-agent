"""GET /models — OpenAI models offered as synthesis-model overrides.

Sourced from rome's LLMHandler pricing table (single source of truth) so the
public-mode model dropdown never drifts from what the agent actually supports.
"""
from fastapi import APIRouter

from ..config import synthesis_model_choices
from ..schemas import ModelOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
async def list_models() -> list[ModelOut]:
    return [ModelOut(**m) for m in synthesis_model_choices()]
