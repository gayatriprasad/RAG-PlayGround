"""Cost router — GET /cost/summary endpoint (Skill 27)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cost"])

_RAG_LAB_ROOT = Path(__file__).resolve().parents[2] / "rag-lab"
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"


@router.get("/cost/summary")
async def cost_summary(
    experiment: str = Query(..., description="Experiment name to load cost summary for"),
):
    """Load the per-experiment cost summary JSON written by run_experiment.py."""
    summary_path = _OUT_DIR / experiment / f"{experiment}_cost_summary.json"

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No cost summary found for experiment '{experiment}' at {summary_path}. "
                "Run the experiment first with cost tracking enabled (cfg.cost.track=True)."
            ),
        )

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    return {"experiment": experiment, **summary}
