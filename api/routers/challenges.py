"""Challenges router — guided learning mode (Skill 34).

Progress is tracked client-side (localStorage) for the OSS tier; this API
only lists challenge definitions and checks a single challenge against an
experiment's already-computed results CSV.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.routers._shared import _OUT_DIR, find_experiment_config, load_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/challenges", tags=["challenges"])

_CHALLENGES_PATH = None  # resolved lazily via config.challenge.challenges_path


def _resolve_challenges_path() -> str:
    from api.routers._shared import _RAG_LAB_ROOT

    config_path = find_experiment_config(None)
    cfg = load_config(config_path)
    path = cfg.challenge.challenges_path
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute():
        p = _RAG_LAB_ROOT / path
    return str(p)


@router.get("")
async def list_challenges():
    """List all challenge definitions."""
    from raglab.challenges import load_challenges

    path = _resolve_challenges_path()
    try:
        challenges = load_challenges(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"challenges": [c.model_dump() for c in challenges]}


@router.post("/{challenge_id}/check")
async def check_challenge(challenge_id: str, experiment: Optional[str] = None):
    """Check a challenge's goal against `experiment`'s saved results CSV."""
    import pandas as pd

    from raglab.challenges import ChallengeRunner

    path = _resolve_challenges_path()
    runner = ChallengeRunner()
    try:
        challenges = runner.load_challenges(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    challenge = next((c for c in challenges if c.id == challenge_id), None)
    if challenge is None:
        raise HTTPException(status_code=404, detail=f"Unknown challenge: '{challenge_id}'")

    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    csv_path = _OUT_DIR / cfg.experiment.name / f"{cfg.experiment.name}_results.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment '{cfg.experiment.name}'. Run the experiment first.",
        )

    df = pd.read_csv(csv_path)
    result = runner.evaluate_dataframe(challenge, df)
    return result.model_dump()
