"""Shared helpers for API routers: experiment config discovery + loading."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

_RAG_LAB_ROOT = Path(__file__).resolve().parents[2] / "rag-lab"
_RAG_LAB_SRC = _RAG_LAB_ROOT / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))

_EXPERIMENTS_DIR = _RAG_LAB_ROOT / "experiments"
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"


def find_experiment_config(experiment: Optional[str] = None) -> Path:
    """Find the config.yaml for an experiment. If None, use the latest with results."""
    if experiment:
        config_path = _EXPERIMENTS_DIR / experiment / "config.yaml"
        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment}' config not found at {config_path}",
            )
        return config_path

    experiment_dirs = sorted(
        [d for d in _EXPERIMENTS_DIR.iterdir() if d.is_dir() and (d / "config.yaml").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not experiment_dirs:
        raise HTTPException(
            status_code=404,
            detail="No experiments found. Run an experiment first.",
        )
    for d in experiment_dirs:
        result_csv = _OUT_DIR / d.name / f"{d.name}_results.csv"
        if result_csv.exists():
            return d / "config.yaml"
    return experiment_dirs[0] / "config.yaml"


def load_config(config_path: Path):
    """Load a Config from a YAML file."""
    import yaml
    from raglab.config import Config

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
