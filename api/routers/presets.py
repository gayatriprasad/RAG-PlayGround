"""Presets router — GET /presets (Skill 37E: one-click playground presets)."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["presets"])

_PRESETS_DIR = Path(__file__).resolve().parents[2] / "rag-lab" / "presets"


@router.get("/presets")
async def list_presets():
    """List all available one-click config presets."""
    if not _PRESETS_DIR.exists():
        return {"presets": []}

    presets = []
    for path in sorted(_PRESETS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as e:
            logger.warning(f"Failed to load preset {path.name}: {e}")
            continue
        presets.append(
            {
                "id": path.stem,
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
            }
        )
    return {"presets": presets}


@router.get("/presets/{preset_id}")
async def get_preset(preset_id: str):
    """Get the full config fragment for a single preset."""
    path = _PRESETS_DIR / f"{preset_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    data = yaml.safe_load(path.read_text()) or {}
    return data
