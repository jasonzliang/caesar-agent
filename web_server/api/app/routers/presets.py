"""GET /presets — exposes the UI-facing preset definitions."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import PRESETS
from ..schemas import PresetOut

router = APIRouter(prefix="/presets", tags=["presets"])


@router.get("", response_model=list[PresetOut])
async def list_presets() -> list[PresetOut]:
    return [PresetOut(**p) for p in PRESETS]
