"""Export & share router — Skill 35."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from api.routers._shared import _OUT_DIR, find_experiment_config, load_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])

_MEDIA_TYPES = {
    "markdown": "text/markdown",
    "csv": "text/csv",
    "html": "text/html",
    "json": "application/json",
}


@router.get("/export/run/{run_id}")
async def export_run(
    run_id: str,
    format: Literal["markdown", "csv", "html", "json"] = Query("markdown"),
):
    """Export an experiment's results ("run") in the requested format.

    `run_id` is the experiment name (one experiment = one results CSV).
    """
    import pandas as pd

    from raglab.utils.exporter import RunExporter

    config_path = find_experiment_config(run_id)
    cfg = load_config(config_path)
    csv_path = _OUT_DIR / cfg.experiment.name / f"{cfg.experiment.name}_results.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for '{run_id}'. Run the experiment first.",
        )

    df = pd.read_csv(csv_path)
    exporter = RunExporter()

    if format == "markdown":
        content = exporter.to_markdown(cfg.experiment.name, df, cfg)
    elif format == "csv":
        content = exporter.to_csv(df)
    elif format == "html":
        content = exporter.to_html(cfg.experiment.name, df, cfg)
    else:
        content = exporter.to_json(cfg.experiment.name, df, cfg)

    return PlainTextResponse(content=content, media_type=_MEDIA_TYPES[format])


@router.get("/share/config")
async def share_config(experiment: Optional[str] = None):
    """Return a shareable URL encoding the current (non-secret) config."""
    from raglab.utils.exporter import encode_config

    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    token = encode_config(cfg)
    return {"token": token, "url": f"/load?c={token}"}


@router.get("/load")
async def load_shared_config(c: str):
    """Decode a shared config token and return the Config for the frontend to apply."""
    from raglab.utils.exporter import decode_config

    try:
        cfg = decode_config(c)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid share token: {e}")
    return cfg.model_dump()
