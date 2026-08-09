"""Arize Phoenix Tracer — Skill 47C.

Uses OpenTelemetry spans emitted through Phoenix's OTel exporter
(`phoenix.otel.register`). Phoenix runs locally (`phoenix.launch_app()`) or
points at a hosted collector via `PHOENIX_COLLECTOR_ENDPOINT`.

Falls back to JSONLTracer (via get_tracer()) if arize-phoenix isn't
installed or the OTel exporter fails to initialize — same fallback
contract as LangfuseTracer.

Setup:
    pip install arize-phoenix
    export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"  # optional
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from raglab.observability.langfuse_tracer import BaseTracer

logger = logging.getLogger(__name__)

PHOENIX_AVAILABLE = False
try:
    from phoenix.otel import register as _phoenix_register

    PHOENIX_AVAILABLE = True
    logger.info("Arize Phoenix SDK available")
except ImportError:
    logger.info("Arize Phoenix not installed (optional). Using JSONL tracer.")
    _phoenix_register = None  # type: ignore


class PhoenixTracer(BaseTracer):
    """
    Tracer backed by Arize Phoenix's OpenTelemetry integration.

    One OTel span per start_span()/end_span() pair, nested under one root
    span per trace. Scores are recorded as span attributes since Phoenix
    doesn't have a first-class "trace score" concept like Langfuse.
    """

    def __init__(self, project_name: str = "neuralbench"):
        if not PHOENIX_AVAILABLE:
            raise ImportError("arize-phoenix not installed. Run: pip install arize-phoenix")

        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        tracer_provider = _phoenix_register(project_name=project_name, endpoint=endpoint, batch=False)
        self._otel_tracer = tracer_provider.get_tracer(__name__)

        self._root_span = None
        self._root_ctx = None
        self.span_stack: List[Any] = []
        self.span_ctx_stack: List[Any] = []

        logger.info(f"Phoenix tracer initialized (project={project_name})")

    def start_trace(self, experiment: str, input_text: str) -> str:
        self._root_span = self._otel_tracer.start_span(f"rag-{experiment}")
        self._root_span.set_attribute("input.value", input_text)
        self.span_stack = []
        trace_id = format(self._root_span.get_span_context().trace_id, "032x")
        logger.info(f"Started Phoenix trace: {trace_id}")
        return trace_id

    def start_span(self, name: str, **metadata) -> str:
        if self._root_span is None:
            raise RuntimeError("No active trace. Call start_trace() first.")

        span = self._otel_tracer.start_span(name)
        for k, v in metadata.items():
            span.set_attribute(f"metadata.{k}", str(v))
        self.span_stack.append(span)
        span_id = format(span.get_span_context().span_id, "016x")
        return span_id

    def end_span(self, output: Optional[Dict] = None, **metadata) -> None:
        if not self.span_stack:
            logger.warning("No active span to end")
            return
        span = self.span_stack.pop()
        if output:
            span.set_attribute("output.value", str(output))
        for k, v in metadata.items():
            span.set_attribute(f"metadata.{k}", str(v))
        span.end()

    def add_score(self, name: str, value: float, **metadata) -> None:
        if self._root_span is None:
            logger.warning("No active trace to score")
            return
        self._root_span.set_attribute(f"score.{name}", value)

    def end_trace(self, output: Optional[Dict] = None) -> None:
        if self._root_span is None:
            logger.warning("No active trace to end")
            return
        if output:
            self._root_span.set_attribute("output.value", str(output))
        self._root_span.end()
        logger.info("Ended Phoenix trace")
        self._root_span = None
        self.span_stack = []
