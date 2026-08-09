"""Langfuse Tracer — Production observability for RAG pipelines.

Wraps every pipeline step with Langfuse spans for comprehensive tracing.
Falls back to JSONL tracer if Langfuse not configured.

Setup:
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # or self-hosted

Usage:
    from raglab.observability import get_tracer
    
    tracer = get_tracer(cfg)
    trace_id = tracer.start_trace(experiment_name, question.text)
    
    tracer.start_span("retrieval", query=query)
    # ... retrieval logic
    tracer.end_span({"num_chunks": len(chunks)})
    
    tracer.end_trace({"overall_score": result.overall_score})
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Langfuse Availability Check ───────────────────────────────────────────────

LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse SDK available")
except ImportError:
    logger.info("ℹ️  Langfuse not installed (optional). Using JSONL tracer.")
    Langfuse = None  # type: ignore


# ─── Base Tracer Interface ─────────────────────────────────────────────────────

class BaseTracer:
    """Base interface for pipeline tracers."""
    
    def start_trace(self, experiment: str, input_text: str) -> str:
        """Start a new trace. Returns trace_id."""
        raise NotImplementedError
    
    def start_span(self, name: str, **metadata) -> str:
        """Start a span within current trace. Returns span_id."""
        raise NotImplementedError
    
    def end_span(self, output: Optional[Dict] = None, **metadata) -> None:
        """End the current span."""
        raise NotImplementedError
    
    def add_score(self, name: str, value: float, **metadata) -> None:
        """Add a score to the current trace."""
        raise NotImplementedError
    
    def end_trace(self, output: Optional[Dict] = None) -> None:
        """End the current trace."""
        raise NotImplementedError


# ─── Langfuse Tracer ───────────────────────────────────────────────────────────

class LangfuseTracer(BaseTracer):
    """
    Production-grade tracer using Langfuse.
    
    Wraps every pipeline step with spans for full observability.
    """
    
    def __init__(self):
        """Initialize Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            raise ImportError("Langfuse not installed. Run: pip install langfuse")
        
        # Check environment variables
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not secret_key or not public_key:
            raise ValueError(
                "LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY must be set. "
                "Get your keys at: https://cloud.langfuse.com"
            )
        
        self.client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host
        )
        
        self.current_trace = None
        self.current_span = None
        self.span_stack: List[Any] = []
        
        logger.info(f"✅ Langfuse tracer initialized (host: {host})")
    
    def start_trace(self, experiment: str, input_text: str) -> str:
        """Start a new trace."""
        self.current_trace = self.client.trace(
            name=f"rag-{experiment}",
            input={"question": input_text}
        )
        self.span_stack = []
        
        trace_id = self.current_trace.id
        logger.info(f"📊 Started Langfuse trace: {trace_id}")
        return trace_id
    
    def start_span(self, name: str, **metadata) -> str:
        """Start a span within current trace."""
        if not self.current_trace:
            raise RuntimeError("No active trace. Call start_trace() first.")
        
        span = self.current_trace.span(
            name=name,
            input=metadata.get("input"),
            metadata={k: v for k, v in metadata.items() if k != "input"}
        )
        
        self.span_stack.append(span)
        self.current_span = span
        
        return span.id
    
    def end_span(self, output: Optional[Dict] = None, **metadata) -> None:
        """End the current span."""
        if not self.span_stack:
            logger.warning("⚠️  No active span to end")
            return
        
        span = self.span_stack.pop()
        
        # Update span with output and metadata
        if output:
            span.end(output=output)
        
        # Update current span to parent (if any)
        self.current_span = self.span_stack[-1] if self.span_stack else None
    
    def add_score(self, name: str, value: float, **metadata) -> None:
        """Add a score to the current trace."""
        if not self.current_trace:
            logger.warning("⚠️  No active trace to score")
            return
        
        self.current_trace.score(
            name=name,
            value=value,
            comment=metadata.get("comment")
        )
    
    def end_trace(self, output: Optional[Dict] = None) -> None:
        """End the current trace."""
        if not self.current_trace:
            logger.warning("⚠️  No active trace to end")
            return
        
        if output:
            self.current_trace.update(output=output)
        
        # Flush to Langfuse
        self.client.flush()
        
        logger.info(f"✅ Ended Langfuse trace: {self.current_trace.id}")
        self.current_trace = None
        self.span_stack = []


# ─── JSONL Fallback Tracer ────────────────────────────────────────────────────

class JSONLTracer(BaseTracer):
    """
    Fallback tracer that logs to JSONL files.
    
    Used when Langfuse is not configured (free tier fallback).
    """
    
    def __init__(self, output_dir: str = "./out/raglab_out/traces"):
        """Initialize JSONL tracer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_trace: Optional[Dict] = None
        self.current_span: Optional[Dict] = None
        self.span_stack: List[Dict] = []
        
        logger.info(f"✅ JSONL tracer initialized (dir: {output_dir})")
    
    def start_trace(self, experiment: str, input_text: str) -> str:
        """Start a new trace."""
        import uuid
        trace_id = str(uuid.uuid4())
        
        self.current_trace = {
            "trace_id": trace_id,
            "experiment": experiment,
            "input": input_text,
            "spans": [],
            "scores": {},
            "start_time": time.time()
        }
        
        logger.info(f"📊 Started JSONL trace: {trace_id}")
        return trace_id
    
    def start_span(self, name: str, **metadata) -> str:
        """Start a span within current trace."""
        if not self.current_trace:
            raise RuntimeError("No active trace. Call start_trace() first.")
        
        import uuid
        span_id = str(uuid.uuid4())
        
        span = {
            "span_id": span_id,
            "name": name,
            "metadata": metadata,
            "start_time": time.time()
        }
        
        self.span_stack.append(span)
        self.current_span = span
        
        return span_id
    
    def end_span(self, output: Optional[Dict] = None, **metadata) -> None:
        """End the current span."""
        if not self.span_stack:
            logger.warning("⚠️  No active span to end")
            return
        
        span = self.span_stack.pop()
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
        
        if output:
            span["output"] = output
        
        span["metadata"].update(metadata)
        
        # Add to trace
        if self.current_trace:
            self.current_trace["spans"].append(span)
        
        # Update current span to parent (if any)
        self.current_span = self.span_stack[-1] if self.span_stack else None
    
    def add_score(self, name: str, value: float, **metadata) -> None:
        """Add a score to the current trace."""
        if not self.current_trace:
            logger.warning("⚠️  No active trace to score")
            return
        
        self.current_trace["scores"][name] = {
            "value": value,
            "metadata": metadata
        }
    
    def end_trace(self, output: Optional[Dict] = None) -> None:
        """End the current trace."""
        if not self.current_trace:
            logger.warning("⚠️  No active trace to end")
            return
        
        self.current_trace["end_time"] = time.time()
        self.current_trace["duration_ms"] = (
            self.current_trace["end_time"] - self.current_trace["start_time"]
        ) * 1000
        
        if output:
            self.current_trace["output"] = output
        
        # Write to JSONL file
        trace_file = self.output_dir / f"{self.current_trace['experiment']}_traces.jsonl"
        with open(trace_file, "a") as f:
            f.write(json.dumps(self.current_trace) + "\n")
        
        logger.info(f"✅ Wrote trace to: {trace_file}")
        self.current_trace = None
        self.span_stack = []


# ─── Tracer Factory ────────────────────────────────────────────────────────────

def get_tracer(use_langfuse: bool = True, backend: Optional[str] = None, **backend_kwargs) -> BaseTracer:
    """
    Get appropriate tracer based on configuration.

    Args:
        use_langfuse: Legacy flag (Skill 32) — try Langfuse if available and
            configured. Ignored if `backend` is explicitly given.
        backend: Skill 47C — one of "jsonl" | "langfuse" | "phoenix" |
            "openllmetry". Always falls back to JSONLTracer if the requested
            backend isn't installed/configured, never raises.
        **backend_kwargs: forwarded to the backend tracer's constructor
            (e.g. project_name for phoenix, app_name for openllmetry).

    Returns:
        BaseTracer instance.
    """
    if backend is not None:
        if backend == "jsonl":
            return JSONLTracer()
        if backend == "langfuse":
            if LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
                try:
                    return LangfuseTracer()
                except Exception as e:
                    logger.warning(f"Failed to initialize Langfuse: {e}. Falling back to JSONL tracer.")
            else:
                logger.warning("Langfuse backend requested but not available/configured. Falling back to JSONL tracer.")
            return JSONLTracer()
        if backend == "phoenix":
            from raglab.observability.phoenix_tracer import PHOENIX_AVAILABLE, PhoenixTracer

            if PHOENIX_AVAILABLE:
                try:
                    return PhoenixTracer(**backend_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to initialize Phoenix: {e}. Falling back to JSONL tracer.")
            else:
                logger.warning("Phoenix backend requested but arize-phoenix not installed. Falling back to JSONL tracer.")
            return JSONLTracer()
        if backend == "openllmetry":
            from raglab.observability.openllmetry_tracer import OPENLLMETRY_AVAILABLE, OpenLLMetryTracer

            if OPENLLMETRY_AVAILABLE:
                try:
                    return OpenLLMetryTracer(**backend_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenLLMetry: {e}. Falling back to JSONL tracer.")
            else:
                logger.warning("OpenLLMetry backend requested but traceloop-sdk not installed. Falling back to JSONL tracer.")
            return JSONLTracer()
        raise ValueError(f"Unknown observability backend: '{backend}'. Valid: jsonl, langfuse, phoenix, openllmetry")

    if use_langfuse and LANGFUSE_AVAILABLE:
        # Check if Langfuse is configured
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        
        if secret_key and public_key:
            try:
                return LangfuseTracer()
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Langfuse: {e}")
                logger.info("   Falling back to JSONL tracer")

    # Fallback to JSONL tracer
    return JSONLTracer()


# ─── Convenience Wrapper ───────────────────────────────────────────────────────

class PipelineTracer:
    """
    High-level wrapper for tracing RAG pipelines.
    
    Usage:
        with PipelineTracer(experiment_name, question.text) as tracer:
            with tracer.span("retrieval", query=query):
                chunks = index.retrieve(query, top_k)
            
            with tracer.span("generation"):
                answer = llm_generate(chunks, question)
            
            tracer.score("overall_score", result.overall_score)
    """
    
    def __init__(self, experiment: str, input_text: str, use_langfuse: bool = True):
        """Initialize pipeline tracer."""
        self.tracer = get_tracer(use_langfuse)
        self.experiment = experiment
        self.input_text = input_text
        self.trace_id = None
    
    def __enter__(self):
        """Start trace."""
        self.trace_id = self.tracer.start_trace(self.experiment, self.input_text)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End trace."""
        if exc_type:
            self.tracer.add_score("error", 1.0, error=str(exc_val))
        self.tracer.end_trace()
    
    def span(self, name: str, **metadata):
        """Context manager for spans."""
        return SpanContext(self.tracer, name, metadata)
    
    def score(self, name: str, value: float, **metadata):
        """Add score to trace."""
        self.tracer.add_score(name, value, **metadata)


class SpanContext:
    """Context manager for spans."""
    
    def __init__(self, tracer: BaseTracer, name: str, metadata: Dict):
        self.tracer = tracer
        self.name = name
        self.metadata = metadata
        self.span_id = None
    
    def __enter__(self):
        self.span_id = self.tracer.start_span(self.name, **self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        output = {}
        if exc_type:
            output["error"] = str(exc_val)
        self.tracer.end_span(output)
