"""Health & readiness endpoints — Skill 32.

`/health` is a pure liveness check (always 200 if the process is up).
`/ready` checks the three things the playground actually needs: DB, vector
index, and at least one LLM provider — returns 503 if any is not ready.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter

from api.routers._shared import _RAG_LAB_ROOT, find_experiment_config, load_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness check — always 200 if the process can respond."""
    return {"status": "alive"}


def _check_db() -> bool:
    try:
        from raglab.db.connection import get_backend, get_pool
        from raglab.db.queries import _run

        pool = get_pool()
        backend = get_backend()
        _run(pool, backend, "SELECT 1 AS ok", {})
        return True
    except Exception as e:
        logger.warning(f"/ready: DB check failed: {e}")
        return False


def _check_vector_index() -> bool:
    original_cwd = os.getcwd()
    try:
        config_path = find_experiment_config(None)
        cfg = load_config(config_path)
        os.chdir(str(_RAG_LAB_ROOT))
        from raglab.index import get_index

        index = get_index(cfg.index, cfg.embed)
        if hasattr(index, "is_built"):
            return bool(index.is_built(cfg.experiment.name))
        return True
    except Exception as e:
        logger.warning(f"/ready: vector index check failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def _check_llm() -> bool:
    original_cwd = os.getcwd()
    try:
        config_path = find_experiment_config(None)
        cfg = load_config(config_path)
        os.chdir(str(_RAG_LAB_ROOT))
        from raglab.models import get_llm

        client = get_llm(cfg.llm)
        # Cheap ping — a 1-token completion, not a real query.
        client.complete(
            [{"role": "user", "content": "ping"}], max_tokens=1
        )
        return True
    except Exception as e:
        logger.warning(f"/ready: LLM check failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)


@router.get("/ready")
async def ready():
    """Readiness check — DB reachable, vector index built, LLM responsive."""
    db_ok = _check_db()
    vector_ok = _check_vector_index()
    llm_ok = _check_llm()
    ready_state = db_ok and vector_ok and llm_ok

    payload = {"db": db_ok, "vector": vector_ok, "llm": llm_ok, "ready": ready_state}
    if not ready_state:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=payload)
    return payload
