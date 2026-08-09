"""
Unit tests for the Phoenix/OpenLLMetry tracer backends + get_tracer()
dispatch (Skill 47C). Real arize-phoenix/traceloop-sdk packages are not
required — tests stub the SDK modules via sys.modules and/or drive the
graceful-fallback path when they're genuinely absent.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.observability import JSONLTracer, get_tracer


def test_get_tracer_jsonl_backend_returns_jsonl_tracer(tmp_path):
    tracer = get_tracer(backend="jsonl")
    assert isinstance(tracer, JSONLTracer)


def test_get_tracer_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown observability backend"):
        get_tracer(backend="not-a-real-backend")


def test_get_tracer_phoenix_falls_back_when_not_installed(monkeypatch):
    import raglab.observability.phoenix_tracer as phoenix_module

    monkeypatch.setattr(phoenix_module, "PHOENIX_AVAILABLE", False)
    tracer = get_tracer(backend="phoenix")
    assert isinstance(tracer, JSONLTracer)


def test_get_tracer_openllmetry_falls_back_when_not_installed(monkeypatch):
    import raglab.observability.openllmetry_tracer as openllmetry_module

    monkeypatch.setattr(openllmetry_module, "OPENLLMETRY_AVAILABLE", False)
    tracer = get_tracer(backend="openllmetry")
    assert isinstance(tracer, JSONLTracer)


def test_get_tracer_langfuse_falls_back_when_unconfigured(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracer = get_tracer(backend="langfuse")
    assert isinstance(tracer, JSONLTracer)


def _make_fake_span():
    span = MagicMock()
    span.get_span_context.return_value = MagicMock(trace_id=0xABCDEF, span_id=0x123)
    return span


def test_phoenix_tracer_full_lifecycle(monkeypatch):
    import raglab.observability.phoenix_tracer as phoenix_module

    fake_tracer_provider = MagicMock()
    root_span = _make_fake_span()
    child_span = _make_fake_span()
    fake_otel_tracer = MagicMock()
    fake_otel_tracer.start_span.side_effect = [root_span, child_span]
    fake_tracer_provider.get_tracer.return_value = fake_otel_tracer

    monkeypatch.setattr(phoenix_module, "PHOENIX_AVAILABLE", True)
    monkeypatch.setattr(phoenix_module, "_phoenix_register", lambda **kwargs: fake_tracer_provider)

    tracer = phoenix_module.PhoenixTracer(project_name="test-proj")
    trace_id = tracer.start_trace("exp1", "hello?")
    assert trace_id == format(0xABCDEF, "032x")

    span_id = tracer.start_span("retrieval", query="hello?")
    assert span_id == format(0x123, "016x")

    tracer.end_span(output={"num_chunks": 3})
    child_span.end.assert_called_once()

    tracer.add_score("overall_score", 0.9)
    root_span.set_attribute.assert_any_call("score.overall_score", 0.9)

    tracer.end_trace(output={"answer": "yo"})
    root_span.end.assert_called_once()


def test_openllmetry_tracer_full_lifecycle(monkeypatch):
    import raglab.observability.openllmetry_tracer as openllmetry_module

    fake_traceloop = MagicMock()
    root_span = _make_fake_span()
    fake_otel_tracer = MagicMock()
    fake_otel_tracer.start_span.return_value = root_span

    monkeypatch.setattr(openllmetry_module, "OPENLLMETRY_AVAILABLE", True)
    monkeypatch.setattr(openllmetry_module, "Traceloop", fake_traceloop)

    fake_otel_module = MagicMock()
    fake_otel_module.get_tracer.return_value = fake_otel_tracer
    monkeypatch.setitem(sys.modules, "opentelemetry", MagicMock(trace=fake_otel_module))

    tracer = openllmetry_module.OpenLLMetryTracer(app_name="test-app")
    fake_traceloop.init.assert_called_once()

    trace_id = tracer.start_trace("exp1", "hello?")
    assert trace_id == format(0xABCDEF, "032x")

    tracer.end_trace(output={"answer": "yo"})
    root_span.end.assert_called_once()
