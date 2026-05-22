"""
FastAPI backend for RAG-PlayGround.

Run: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure raglab is importable
_RAG_LAB_SRC = Path(__file__).resolve().parent.parent / "rag-lab" / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))

from api.routers import benchmark, experiments, query

app = FastAPI(
    title="RAG-PlayGround API",
    description="Backend API for the RAG research playground. "
    "Exposes query, experiment management, and benchmark result endpoints.",
    version="0.1.0",
)

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
app.include_router(query.router)
app.include_router(experiments.router)
app.include_router(benchmark.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "rag-playground-api"}
