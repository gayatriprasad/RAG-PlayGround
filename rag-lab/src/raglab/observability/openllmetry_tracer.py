"""OpenLLMetry Tracer — Skill 47C.

Uses `traceloop-sdk` (OpenLLMetry), which wraps OpenTelemetry with
LLM-specific span conventions and works with any OTel-compatible backend
(local console exporter by default, or a hosted collector).

Falls back to JSONLTracer (via get_tracer()) if traceloop-sdk isn't
installed or init fails — same fallback contract as LangfuseTracer.

Setup:
    pip install traceloop-sdk
    export TRACELOOP_BASE_URL="https://api.traceloop.com"  # optional, hosted
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from raglab.observability.langfuse_tracer import BaseTracer

logger = logging.getLogger(__name__)

OPENLLMETRY_AVAILABLE = False
try:
    from traceloop.sdk import Traceloop

    OPENLLMETRY_AVAILABLE = True
    logger.info("OpenLLMetry (traceloop-sdk) available")
except ImportError:
    logger.info("traceloop-sdk not installed (optional). Using JSONL tracer.")
    Traceloop = None  # type: ignore


class OpenLLMetryTracer(BaseTracer):
    """
    Tracer backed by traceloop-sdk's OpenTelemetry integration.

    Mirrors PhoenixTracer's span-stack approach so both alternative
    backends satisfy the same BaseTracer contract as LangfuseTracer.
    """

    def __init__(self, app_name: str = "neuralbench"):
        if not OPENLLMETRY_AVAILABLE:
            raise ImportError("traceloop-sdk not installed. Run: pip install traceloop-sdk")

        Traceloop.init(app_name=app_name, disable_batch=True)

        from opentelemetry import trace as otel_trace

        self._otel_tracer = otel_trace.get_tracer(__name__)
        self._root_span = None
        self.span_stack: List[Any] = []

        logger.info(f"OpenLLMetry tracer initialized (app={app_name})")

    def start_trace(self, experiment: str, input_text: str) -> str:
        self._root_span = self._otel_tracer.start_span(f"rag-{experiment}")
        self._root_span.set_attribute("input.value", input_text)
        self.span_stack = []
        trace_id = format(self._root_span.get_span_context().trace_id, "032x")
        logger.info(f"Started OpenLLMetry trace: {trace_id}")
        return trace_id

    def start_span(self, name: str, **metadata) -> str:
        if self._root_span is None:
            raise RuntimeError("No active trace. Call start_trace() first.")

        span = self._otel_tracer.start_span(name)
        for k, v in metadata.items():
            span.set_attribute(f"metadata.{k}", str(v))
        self.span_stack.append(span)
        return format(span.get_span_context().span_id, "016x")

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
        logger.info("Ended OpenLLMetry trace")
        self._root_span = None
        self.span_stack = []
