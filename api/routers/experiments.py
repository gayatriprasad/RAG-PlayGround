"""Experiments router — GET /experiments endpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter, HTTPException

from api.models import ExperimentListResponse, ExperimentSummary

logger = logging.getLogger(__name__)

router = APIRouter(tags=["experiments"])

_RAG_LAB_ROOT = Path(__file__).resolve().parents[2] / "rag-lab"
_EXPERIMENTS_DIR = _RAG_LAB_ROOT / "experiments"
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments():
    """List all experiments with their config and result status."""
    experiments = []

    if not _EXPERIMENTS_DIR.exists():
        return ExperimentListResponse(experiments=[])

    for exp_dir in sorted(_EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            continue

        # Load config
        try:
            with open(config_path) as f:
                config_data: Dict[str, Any] = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config for {exp_dir.name}: {e}")
            config_data = {}

        # Check for results
        exp_name = exp_dir.name
        result_csv = _OUT_DIR / exp_name / f"{exp_name}_results.csv"
        has_results = result_csv.exists()

        result_count = None
        mean_score = None
        if has_results:
            try:
                import pandas as pd

                df = pd.read_csv(result_csv)
                result_count = len(df)
                if "overall_score" in df.columns:
                    mean_score = round(df["overall_score"].mean(), 4)
            except Exception as e:
                logger.warning(f"Failed to read results for {exp_name}: {e}")

        experiments.append(
            ExperimentSummary(
                name=exp_name,
                config=config_data,
                has_results=has_results,
                result_count=result_count,
                mean_score=mean_score,
            )
        )

    return ExperimentListResponse(experiments=experiments)


@router.get("/experiments/{name}/config")
async def get_experiment_config(name: str):
    """Get the raw YAML config for a specific experiment."""
    config_path = _EXPERIMENTS_DIR / name / "config.yaml"

    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{name}' not found or has no config.yaml",
        )

    try:
        raw_yaml = config_path.read_text(encoding="utf-8")
        config_data = yaml.safe_load(raw_yaml) or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")

    return {
        "experiment": name,
        "config": config_data,
        "raw_yaml": raw_yaml,
    }
