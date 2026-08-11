"""Query router — POST /query endpoint."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.models import (
    IntentResponse,
    QueryRequest,
)

# Add rag-lab/src to path so raglab is importable
_RAG_LAB_SRC = Path(__file__).resolve().parents[2] / "rag-lab" / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))

from raglab.classifiers import get_classifier
from raglab.config import (
    Config,
    IndexCfg,
    IntentCfg,
    LLMCfg,
    NetworkCfg,
    RetrieveCfg,
    PRESET_FIELD_MAP,
    apply_preset,
)
from raglab.index import get_index
from raglab.net.rate_limit import limiter
from raglab.pipelines import AgenticRAGPipeline, NaiveRAGPipeline
from raglab.rerankers import get_reranker
from raglab.types import Question
from raglab.utils.memory import ConversationMemory

_QUERY_RATE_LIMIT = f"{NetworkCfg().rate_limit_per_minute}/minute"

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

# ─── Session Memory Store ──────────────────────────────────────────────────────

# Global session store: session_id → ConversationMemory
# In production, this should be Redis or similar persistent storage
SESSION_MEMORIES: Dict[str, ConversationMemory] = {}

def get_or_create_memory(session_id: Optional[str]) -> Optional[ConversationMemory]:
    """
    Get or create conversation memory for a session.
    
    Args:
        session_id: Session identifier (optional)
        
    Returns:
        ConversationMemory instance, or None if no session_id
    """
    if not session_id:
        return None
    
    if session_id not in SESSION_MEMORIES:
        SESSION_MEMORIES[session_id] = ConversationMemory(max_turns=5)
        logger.info(f"📝 Created new conversation memory for session: {session_id}")
    
    return SESSION_MEMORIES[session_id]

# ─── Helpers ───────────────────────────────────────────────────────────────────

_RAG_LAB_ROOT = Path(__file__).resolve().parents[2] / "rag-lab"
_EXPERIMENTS_DIR = _RAG_LAB_ROOT / "experiments"


def _find_experiment_config(experiment: Optional[str] = None) -> Path:
    """Find the config.yaml for an experiment. If None, use the latest."""
    if experiment:
        config_path = _EXPERIMENTS_DIR / experiment / "config.yaml"
        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment}' config not found at {config_path}",
            )
        return config_path

    # Find experiments — prefer ones that have results (built index)
    _out_dir = _RAG_LAB_ROOT / "out" / "raglab_out"
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
    # Prefer experiment with results
    for d in experiment_dirs:
        result_csv = _out_dir / d.name / f"{d.name}_results.csv"
        if result_csv.exists():
            return d / "config.yaml"
    return experiment_dirs[0] / "config.yaml"


def _load_config(config_path: Path) -> Config:
    """Load config from YAML file."""
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _apply_query_overrides(cfg: Config, req: QueryRequest) -> Config:
    """Apply query-time overrides using the same field map presets use.

    `chunk_strategy` is intentionally excluded: chunking happens at build time,
    so changing it on a live query against an already-built index is a no-op.
    """
    live_overrides = {
        key: getattr(req, key, None)
        for key in PRESET_FIELD_MAP
        if key != "chunk_strategy" and getattr(req, key, None) not in (None, "none")
    }
    if live_overrides:
        cfg = apply_preset(cfg, live_overrides)

    # Standalone rerank toggle remains explicit (not a mapped preset key).
    if req.rerank:
        cfg.retrieve.rerank = True

    return cfg


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/query")
@limiter.limit(_QUERY_RATE_LIMIT)
async def query(request: Request, req: QueryRequest):
    """
    Run a single query through the RAG pipeline.

    Classifies intent, routes to naive/agentic pipeline, retrieves chunks,
    generates an answer, and returns the full result.
    """
    import os

    start = time.perf_counter()

    # Load experiment config
    config_path = _find_experiment_config(req.experiment)
    cfg = _load_config(config_path)

    cfg = _apply_query_overrides(cfg, req)

    # Disable caching for live API queries to avoid stale results
    cfg.retrieve.cache_mode = "none"

    # Change to rag-lab dir so relative paths in config resolve correctly
    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))

    try:
        # Initialize components
        index = get_index(cfg.index, cfg.embed)
        experiment_name = cfg.experiment.name
        if hasattr(index, "is_built") and not index.is_built(experiment_name):
            raise HTTPException(
                status_code=400,
                detail=f"Index not built for experiment '{experiment_name}'. Run the experiment first.",
            )

        classifier = get_classifier(cfg.intent, cfg.llm)
        reranker = get_reranker(cfg.retrieve)

        # If session_id provided, augment query with conversation context
        memory = get_or_create_memory(req.session_id)
        query_text = req.question

        if memory:
            augmented_query = memory.augment_query(req.question)
            if augmented_query != req.question:
                logger.info(f"Augmented query with {len(memory.turns)} previous turns")
                query_text = augmented_query

        # Classify intent (or use override) — classify on the original question
        if req.pipeline_override:
            intent_label = "simple" if req.pipeline_override == "naive" else "complex"
            intent_response = IntentResponse(
                label=intent_label, confidence=1.0, method="override"
            )
        else:
            intent_result = classifier.classify(req.question)
            intent_response = IntentResponse(
                label=intent_result.label,
                confidence=intent_result.confidence,
                method=intent_result.method,
            )

        # Create a Question object for the pipeline (using the memory-augmented text)
        question = Question(
            id="api_query",
            text=query_text,
            ground_truth="",  # No ground truth for live queries
            source_type=req.source_type or "all",
            category="api_query",
        )

        # Route to pipeline
        if intent_response.label == "simple":
            pipeline = NaiveRAGPipeline(index, reranker, cfg)
        else:
            pipeline = AgenticRAGPipeline(index, reranker, cfg)

        # Streaming path (Skill 32): only the naive pipeline can stream
        # token-by-token (it makes a single LLM call). The agentic pipeline
        # makes multiple internal LLM calls, so streaming requests against it
        # fall back to running the full pipeline and emitting the finished
        # answer as one SSE event — the frontend contract stays identical.
        if req.stream:
            if intent_response.label == "simple":
                retrieved_chunks = pipeline._retrieve_chunks(question)
                messages = pipeline._build_prompt(question, retrieved_chunks)
                token_source = pipeline.llm_client.stream(messages)
            else:
                full_result = pipeline.run(question)
                token_source = iter([full_result.predicted_answer])
                retrieved_chunks = full_result.retrieved_chunks

            chunks_meta = [
                {
                    "chunk_id": rc.chunk.id,
                    "doc_id": rc.chunk.doc_id,
                    "source_type": rc.chunk.source_type,
                    "score": rc.score,
                }
                for rc in retrieved_chunks
            ]

            def _generate():
                from raglab.net.streaming import format_sse_event

                yield format_sse_event(
                    {
                        "meta": {
                            "pipeline": intent_response.label,
                            "intent_confidence": intent_response.confidence,
                            "retrieved_chunks": chunks_meta,
                        }
                    }
                )
                answer_parts: List[str] = []
                try:
                    for token in token_source:
                        answer_parts.append(token)
                        yield format_sse_event({"token": token})
                except Exception as e:
                    logger.error(f"Streaming generation failed: {e}")
                    yield format_sse_event({"error": str(e)})
                finally:
                    if memory:
                        memory.add(
                            question=req.question,
                            answer="".join(answer_parts),
                            chunks=retrieved_chunks,
                        )
                    yield "data: [DONE]\n\n"

            from raglab.net.streaming import SSE_HEADERS

            return StreamingResponse(
                _generate(), media_type="text/event-stream", headers=SSE_HEADERS
            )

        result = pipeline.run(question)

        # Add this turn to conversation memory (if session_id provided)
        if memory:
            memory.add(
                question=req.question,  # Store original question, not augmented
                answer=result.predicted_answer,
                chunks=result.retrieved_chunks,
            )
            logger.info(f"Stored turn in memory (total turns: {len(memory.turns)})")
    finally:
        os.chdir(original_cwd)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Build response
    chunks_response = [
        {
            "chunk_id": rc.chunk.id,
            "doc_id": rc.chunk.doc_id,
            "content": rc.chunk.content,
            "source_type": rc.chunk.source_type,
            "score": rc.score,
            "reasoning_path": rc.reasoning_path,
            "chunk": {
                "content": rc.chunk.content,
                "source_type": rc.chunk.source_type,
            },
        }
        for rc in result.retrieved_chunks
    ]

    return {
        "answer": result.predicted_answer,
        "pipeline_used": result.pipeline,
        "pipeline": result.pipeline,
        "intent": {
            "label": intent_response.label,
            "confidence": intent_response.confidence,
            "method": intent_response.method,
        },
        "intent_label": intent_response.label,
        "intent_confidence": intent_response.confidence,
        "retrieved_chunks": chunks_response,
        "latency_ms": round(elapsed_ms, 1),
    }
