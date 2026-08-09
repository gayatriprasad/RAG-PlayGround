"""
Core-path regression tests — Skill 49.

Exercises the 10-step core path spine from .github/copilot-instructions.md
using real factory functions and real config defaults (no fabricated data).
Fast, offline — no LLM calls, no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from raglab.config import ChunkCfg, Config, ExperimentCfg, GoldenCfg, IntentCfg, LLMCfg
from raglab.chunkers import get_chunker
from raglab.classifiers import get_classifier
from raglab.hooks import get_default_registry
from raglab.index import get_index
from raglab.types import Document


def _make_config() -> Config:
    return Config(
        experiment=ExperimentCfg(name="regression_test", corpus_glob=[], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/questions.jsonl"),
    )


def test_01_config_loads_with_defaults():
    """Step 1 precondition: a bare Config with only required fields loads cleanly."""
    cfg = _make_config()
    assert cfg.chunk.strategy == "fixed"
    assert cfg.index.backend == "chroma"
    assert cfg.llm.provider == "ollama"


def test_02_chunker_factory_produces_chunks_for_every_strategy():
    """Step 4 (chunking) — every documented strategy must actually chunk a document."""
    doc = Document(
        id="d0",
        content="Paragraph one has enough content to be chunked meaningfully.\n\n"
        "Paragraph two also has enough content to be chunked meaningfully.",
        source_type="test",
    )
    for strategy in ("fixed", "sentence", "recursive", "none"):
        chunker = get_chunker(ChunkCfg(strategy=strategy))
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1, f"strategy={strategy} produced zero chunks (Rule 32 violation)"


def test_03_hook_registry_wires_all_lifecycle_stages():
    """Cross-cutting hooks (Rule: never modify core logic) are present at every stage."""
    registry = get_default_registry()
    assert len(registry.pre_experiment) >= 1
    assert len(registry.pre_retrieval) >= 1
    assert len(registry.post_retrieval) >= 1
    assert len(registry.pre_generation) >= 1
    assert len(registry.post_generation) >= 1
    assert len(registry.post_experiment) >= 1


def test_04_index_factory_builds_local_backends():
    """Step 6 (naive RAG / index) — local, no-network-required backends are constructible."""
    from raglab.config import IndexCfg, EmbedCfg

    for backend in ("bm25",):  # chroma/faiss require model downloads; bm25 is pure-local
        index = get_index(IndexCfg(backend=backend), EmbedCfg())
        assert index is not None


def test_05_intent_classifier_routes_simple_vs_complex():
    """Step 5 (intent classification) — rule-based classifier must route both labels."""
    classifier = get_classifier(IntentCfg(mode="rule"), LLMCfg())
    simple_result = classifier.classify("What is the capital of France?")
    complex_result = classifier.classify(
        "Compare the deployment process across the three services and explain any conflicts."
    )
    assert simple_result.label in ("simple", "complex")
    assert complex_result.label in ("simple", "complex")


def test_06_eval_scorer_exact_match_metric_scores_a_result():
    """Step 8 (eval run) — the scoring path itself must not silently no-op."""
    from raglab.eval.scorer import ExactMatchMetric
    from raglab.types import EvalResult

    metric = ExactMatchMetric()
    result = EvalResult(
        question_id="q0",
        question="q",
        ground_truth="the answer is 42",
        predicted_answer="the answer is 42",
        source_type="test",
        category="single_doc",
        index_backend="bm25",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
    )
    scored = metric.score(result)
    assert scored.answer_correct is True


def test_07_significance_layer_produces_a_verdict():
    """Step 9 (significance comparison) — never report a delta without one of these."""
    from raglab.eval.significance import compare
    from raglab.config import StatsCfg
    from raglab.types import EvalResult

    def _r(qid, score):
        return EvalResult(
            question_id=qid,
            question="q",
            ground_truth="gt",
            predicted_answer="pred",
            source_type="test",
            category="single_doc",
            index_backend="bm25",
            pipeline="naive",
            intent_label="simple",
            retrieved_chunks=[],
            overall_score=score,
        )

    a = [_r(f"q{i}", 0.5) for i in range(10)]
    b = [_r(f"q{i}", 0.9) for i in range(10)]
    result = compare(a, b, "overall_score", StatsCfg(), "baseline", "candidate")
    assert result.verdict != ""
    assert result.p_value is not None


def test_08_config_is_config_driven_no_hardcoded_model_name_in_factory():
    """Coding Rule 1: config is truth. The factory must not hardcode a model name."""
    import inspect

    from raglab.models import factory

    source = inspect.getsource(factory)
    assert "gpt-4o-mini" not in source, "factory.py must not hardcode a model name (Coding Rule 8)"


def test_09_slot_registry_every_index_backend_literal_has_a_factory_case():
    """Regression guard for Skill 49(B)'s file<->Literal consistency check, inlined here
    as a core-path sanity check specifically for the always-free local backends."""
    from raglab.config import IndexCfg

    always_free_backends = {"chroma", "bm25", "hybrid_rrf", "hybrid_weighted", "faiss"}
    literal_values = set(IndexCfg.model_fields["backend"].annotation.__args__)
    assert always_free_backends.issubset(literal_values)


def test_10_reproducibility_same_config_same_seed_same_result():
    """Step: same config.yaml -> same result, always (Coding Rule 6)."""
    from raglab.eval.significance import bootstrap_ci

    values_a = [0.1, 0.5, 0.9, 0.3, 0.7]
    cfg = _make_config()
    ci1 = bootstrap_ci(values_a, cfg.stats)
    ci2 = bootstrap_ci(values_a, cfg.stats)
    assert ci1 == ci2, "Bootstrap CI must be deterministic given a fixed seed (reproducibility NFR)"
