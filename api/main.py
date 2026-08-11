"""
FastAPI backend for RAG-PlayGround.

Run: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# Ensure raglab is importable
_RAG_LAB_SRC = Path(__file__).resolve().parent.parent / "rag-lab" / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))

from raglab.net.rate_limit import limiter

from api.routers import (
    analytics,
    annotate,
    arena,
    benchmark,
    challenges,
    corpus,
    cost,
    experiments,
    export,
    health,
    improve,
    presets,
    prompt_lab,
    query,
    upload,
    viz,
)

app = FastAPI(
    title="RAG-PlayGround API",
    description="Backend API for the RAG research playground. "
    "Exposes query, experiment management, and benchmark result endpoints.",
    version="0.1.0",
)

# Networking resilience (Skill 31) — rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow Next.js dev server and common local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(experiments.router)
app.include_router(benchmark.router)
app.include_router(cost.router)
app.include_router(arena.router)
app.include_router(viz.router)
app.include_router(analytics.router)
app.include_router(upload.router)
app.include_router(challenges.router)
app.include_router(export.router)
app.include_router(presets.router)
app.include_router(corpus.router)
app.include_router(improve.router)
app.include_router(annotate.router)
app.include_router(prompt_lab.router)


@app.on_event("shutdown")
async def _shutdown():
    from raglab.net.http_client import aclose

    await aclose()

