"""
Tests for Skill 54 — RLM pipeline (sandboxed code execution over raw docs).

RestrictedPython is not installed in this sandboxed dev environment (pip
installs are network-blocked here), so `_execute_safe` is tested against
both: (a) the honest ImportError path when RestrictedPython is absent
(confirming there is NO unsafe exec() fallback), and (b) a lightweight fake
RestrictedPython module injected via sys.modules that exercises the real
sandboxing call sequence (compile_restricted / safe_globals / guards) so the
surrounding pipeline logic (code generation, rewrite loop, sub-model
delegation, aggregation) is verified end-to-end.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.config import Config, ExperimentCfg, GoldenCfg, RLMCfg
from raglab.pipelines.rlm import RLMPipeline
from raglab.types import Document, Question


def _make_docs():
    return [
        Document(id="d1", content="Postgres uses MVCC for concurrency control.", source_type="confluence"),
        Document(id="d2", content="ChromaDB is a local embedded vector store.", source_type="confluence"),
    ]


def _make_cfg():
    return Config(
        experiment=ExperimentCfg(name="test_rlm", corpus_glob=["*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/questions.jsonl"),
    )


def _install_fake_restrictedpython(monkeypatch, result_value):
    """Install a minimal fake RestrictedPython package into sys.modules that
    mimics the real API surface used by RLMPipeline._execute_safe, so we can
    test the sandboxing call sequence without the real (network-installed)
    dependency."""

    def fake_compile_restricted(code, filename, mode):
        return compile(code, filename, mode)

    guards_module = types.ModuleType("RestrictedPython.Guards")
    guards_module.guarded_iter_unpack_sequence = lambda it, spec: it
    guards_module.safer_getattr = getattr

    rp_module = types.ModuleType("RestrictedPython")
    rp_module.compile_restricted = fake_compile_restricted
    rp_module.safe_globals = {"__builtins__": {"len": len, "str": str, "list": list, "range": range}}
    rp_module.Guards = guards_module

    monkeypatch.setitem(sys.modules, "RestrictedPython", rp_module)
    monkeypatch.setitem(sys.modules, "RestrictedPython.Guards", guards_module)


def test_execute_safe_raises_import_error_without_restrictedpython(monkeypatch):
    monkeypatch.setitem(sys.modules, "RestrictedPython", None)
    with patch("raglab.models.get_llm", return_value=MagicMock()):
        pipeline = RLMPipeline(_make_docs(), _make_cfg())

    with pytest.raises(ImportError, match="RestrictedPython"):
        pipeline._execute_safe("result = ['x']", _make_docs())


def test_execute_safe_runs_generated_code_in_sandbox(monkeypatch):
    _install_fake_restrictedpython(monkeypatch, None)
    with patch("raglab.models.get_llm", return_value=MagicMock()):
        pipeline = RLMPipeline(_make_docs(), _make_cfg())

    code = "result = [d.content for d in documents if 'Postgres' in d.content]"
    slices = pipeline._execute_safe(code, _make_docs())
    assert slices == ["Postgres uses MVCC for concurrency control."]


def test_execute_safe_rejects_non_list_result(monkeypatch):
    _install_fake_restrictedpython(monkeypatch, None)
    with patch("raglab.models.get_llm", return_value=MagicMock()):
        pipeline = RLMPipeline(_make_docs(), _make_cfg())

    with pytest.raises(ValueError, match="must be a list"):
        pipeline._execute_safe("result = 'not a list'", _make_docs())


def test_extract_code_parses_fenced_python_block():
    with patch("raglab.models.get_llm", return_value=MagicMock()):
        pipeline = RLMPipeline(_make_docs(), _make_cfg())

    response = "Here is the code:\n```python\nresult = ['a', 'b']\n```\nDone."
    code = pipeline._extract_code(response)
    assert code.strip() == "result = ['a', 'b']"


def test_serialize_corpus_truncates_to_preview_chars():
    cfg = _make_cfg()
    cfg.rlm.corpus_preview_chars = 5
    with patch("raglab.models.get_llm", return_value=MagicMock()):
        pipeline = RLMPipeline(_make_docs(), cfg)

    preview = pipeline._serialize_corpus()
    assert "preview='Postg'" in preview


def test_run_end_to_end_with_sandboxed_code_and_sub_model(monkeypatch):
    _install_fake_restrictedpython(monkeypatch, None)

    root_client = MagicMock()
    root_client.complete.side_effect = [
        "```python\nresult = [d.content for d in documents if 'Postgres' in d.content]\n```",
        "Postgres uses MVCC.",
    ]
    sub_client = MagicMock()
    sub_client.complete.return_value = "MVCC is a concurrency control method."

    with patch("raglab.models.get_llm", side_effect=[root_client, sub_client]):
        pipeline = RLMPipeline(_make_docs(), _make_cfg())

    question = Question(
        id="q1", text="What concurrency model does Postgres use?",
        ground_truth="MVCC", source_type="confluence", category="single_doc",
    )
    result = pipeline.run(question)

    assert result.pipeline == "rlm"
    assert result.index_backend == "none"
    assert result.predicted_answer == "Postgres uses MVCC."
    assert result.metadata["n_slices"] == 1


def test_rlm_cfg_defaults():
    cfg = RLMCfg()
    assert cfg.max_iterations == 5
    assert cfg.sub_provider == "ollama"
    assert cfg.max_code_rewrites == 2
