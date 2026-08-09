"""
REAL, non-mocked integration tests against a locally running Ollama server.

Unlike the rest of the test suite (which mocks every LLM/HTTP call for
speed and determinism), these tests make genuine network calls to a local
Ollama instance and assert on the real behavior of the pipelines: real BM25
retrieval, real prompt construction, and real model-generated text.

Automatically skipped (not failed) when Ollama isn't reachable at
OLLAMA_BASE_URL, so the standard offline/CI test run is unaffected:

    pytest rag-lab/tests/ -k "not test_extended_combinations and not test_integration_e2e"

Run explicitly with a local Ollama server running:

    pytest rag-lab/tests/test_real_ollama_integration.py -v

Model defaults to the smallest local model (llama3.2:1b) for speed. Override
with OLLAMA_TEST_MODEL. These tests make real network calls, so when running
them via a sandboxed shell you must allow network access to localhost.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_TEST_MODEL", "llama3.2:1b")


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(), reason=f"Ollama not reachable at {OLLAMA_BASE_URL} — skipping real-model tests"
)

from raglab.config import (  # noqa: E402
    AgenticCfg,
    Config,
    ExperimentCfg,
    GoldenCfg,
    IndexCfg,
    IntentCfg,
    ModelRegistryCfg,
    RetrieveCfg,
)
from raglab.index.bm25_index import BM25Index  # noqa: E402
from raglab.pipelines.agentic_rag import AgenticRAGPipeline  # noqa: E402
from raglab.pipelines.naive_rag import NaiveRAGPipeline, build_llm_client  # noqa: E402
from raglab.types import Chunk, Question  # noqa: E402


def _llm_cfg() -> ModelRegistryCfg:
    return ModelRegistryCfg(
        provider="ollama",
        model=MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
        temperature=0.0,
        max_tokens=200,
    )


def _make_config(tmp_path: Path, **overrides) -> Config:
    cfg = Config(
        experiment=ExperimentCfg(name="real_ollama_test", corpus_glob=["*.txt"], representations=["bm25"]),
        golden=GoldenCfg(path="./golden/questions.jsonl"),
        index=IndexCfg(backend="bm25", persist_dir=str(tmp_path)),
        retrieve=RetrieveCfg(top_k=3, confidence_threshold=0.0, use_cache=False, cache_mode="none"),
        llm=_llm_cfg(),
        **overrides,
    )
    return cfg


_CORPUS_CHUNKS = [
    Chunk(
        id="c1",
        doc_id="d1",
        content=(
            "The NeuralBench core path has exactly 10 steps: setup, dev server, "
            "readiness check, playground query, intent classification, naive RAG, "
            "agentic RAG, eval run, significance comparison, and full benchmark."
        ),
        source_type="confluence",
        chunk_index=0,
    ),
    Chunk(
        id="c2",
        doc_id="d2",
        content=(
            "Ollama is used as the free local LLM provider in this project, "
            "reachable at http://localhost:11434 with an OpenAI-compatible API."
        ),
        source_type="confluence",
        chunk_index=0,
    ),
    Chunk(
        id="c3",
        doc_id="d3",
        content=(
            "ChromaDB is the default local vector store, requiring no external "
            "infrastructure or API keys for the free tier."
        ),
        source_type="confluence",
        chunk_index=0,
    ),
]


@pytest.fixture
def bm25_index(tmp_path):
    idx = BM25Index(IndexCfg(backend="bm25", persist_dir=str(tmp_path)))
    idx.build(_CORPUS_CHUNKS, experiment_name="real_ollama_test")
    return idx


def test_ollama_reachable_and_returns_real_completion():
    """A direct, unmocked call to the real Ollama server via OllamaClient."""
    client = build_llm_client(_llm_cfg())
    answer = client.complete(
        [{"role": "user", "content": "Reply with a single word: the capital of France is"}],
        temperature=0.0,
        max_tokens=20,
    )
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


def test_bm25_retrieval_finds_real_keyword_match(bm25_index):
    """Real BM25 retrieval (no embeddings, no network) over the fixture corpus."""
    results = bm25_index.retrieve("How many steps does the core path have?", top_k=3, experiment_name="real_ollama_test")
    assert len(results) > 0
    assert "core path" in results[0].chunk.content.lower() or "10 steps" in results[0].chunk.content.lower()


def test_naive_rag_pipeline_end_to_end_real_ollama(tmp_path, bm25_index):
    """Full naive RAG pipeline: real BM25 retrieval -> real Ollama generation.

    No LLM client or index is mocked here — this exercises actual retrieval
    and actual model inference end to end.
    """
    cfg = _make_config(tmp_path)
    pipeline = NaiveRAGPipeline(index=bm25_index, reranker=None, cfg=cfg)

    question = Question(
        id="q1",
        text="How many steps are in the NeuralBench core path?",
        ground_truth="10",
        source_type="confluence",
        category="single_doc",
    )
    result = pipeline.run(question)

    assert result.predicted_answer
    assert len(result.predicted_answer.strip()) > 0
    assert result.retrieved_chunks
    assert result.retrieved_chunks[0].chunk.id == "c1"


def test_agentic_rag_decompose_end_to_end_real_ollama(tmp_path, bm25_index):
    """Agentic 'decompose' strategy with real sub-question generation and
    real synthesis — validates the multi-hop LLM calls actually work against
    a real model's (sometimes imperfect) output, not a canned mock response."""
    cfg = _make_config(tmp_path, agentic=AgenticCfg(strategy="decompose"))
    pipeline = AgenticRAGPipeline(index=bm25_index, reranker=None, cfg=cfg)

    question = Question(
        id="q2",
        text="What LLM provider and what vector store does this project use?",
        ground_truth="Ollama and ChromaDB",
        source_type="confluence",
        category="multi_doc",
    )
    result = pipeline.run(question)

    assert result.predicted_answer
    assert len(result.predicted_answer.strip()) > 0
    assert result.pipeline == "agentic"


def test_llm_classifier_real_ollama_returns_valid_schema():
    """LLM-based intent classification against a real model — validates the
    JSON-parsing + fallback logic holds up against genuine (imperfect) model
    output, not a hand-crafted mock response."""
    from raglab.classifiers.llm_classifier import LLMClassifier

    classifier = LLMClassifier(cfg=IntentCfg(llm_model=MODEL), llm_cfg=_llm_cfg())
    result = classifier.classify("What is the capital of France?")

    assert result.label in ("simple", "complex")
    assert 0.0 <= result.confidence <= 1.0
    assert result.method == "llm"


def test_zero_shot_prompt_real_generation(bm25_index):
    """Builds real prompt messages via ZeroShotPrompt, then sends them to the
    real Ollama model and checks a real, non-empty answer comes back."""
    from raglab.prompts.zero_shot import ZeroShotPrompt
    from raglab.config import GenerationCfg
    from raglab.types import RetrievedChunk

    retrieved = bm25_index.retrieve("What vector store is used?", top_k=2, experiment_name="real_ollama_test")
    prompt = ZeroShotPrompt(GenerationCfg())
    messages = prompt.build_messages("What vector store is used?", retrieved)

    client = build_llm_client(_llm_cfg())
    answer = client.complete(messages, temperature=0.0, max_tokens=100)

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
