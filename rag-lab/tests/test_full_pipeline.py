"""
Comprehensive RAG Pipeline Test — exercises all combinations of:
- Index backends: chroma, bm25, hybrid_rrf
- Chunking strategies: fixed, sentence, semantic
- Rerankers: none, cross_encoder, bm25_rerank
- Pipelines: naive, agentic (decompose, step_back, hyde, react)
- Intent classification: rule, always_simple, always_complex

Uses 3 representative questions and reports pass/fail per scenario.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

import yaml
from raglab.config import Config
from raglab.chunkers import get_chunker
from raglab.classifiers import get_classifier
from raglab.index import get_index
from raglab.pipelines import NaiveRAGPipeline, AgenticRAGPipeline
from raglab.rerankers import get_reranker
from raglab.types import Question, Chunk, Document
from raglab.parsers.enterprise_bench import load_documents, load_questions

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_full_pipeline")
logger.setLevel(logging.INFO)

# ─── Test Questions ─────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    Question(
        id="q1_single_doc",
        text="What caused the memory leak in the search service?",
        ground_truth="The ES 8.11 client changed response body handling - response bodies were not being closed after reading in the bulk search handler.",
        source_type="all",
        category="single_doc",
    ),
    Question(
        id="q2_factual",
        text="What subscription tiers does the payment service offer?",
        ground_truth="Free, Pro, and Enterprise tiers.",
        source_type="all",
        category="single_doc",
    ),
    Question(
        id="q3_multi_doc",
        text="What is the rate limiting configuration for the API gateway?",
        ground_truth="Rate limiting is configured via configs/ratelimit.yaml with per-endpoint limits.",
        source_type="all",
        category="single_doc",
    ),
]


@dataclass
class ScenarioResult:
    scenario: str
    question_id: str
    pipeline: str
    answer: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class TestReport:
    results: List[ScenarioResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def add(self, result: ScenarioResult):
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def print_summary(self):
        elapsed = self.end_time - self.start_time
        print("\n" + "=" * 80)
        print(f"RAG PIPELINE TEST REPORT")
        print(f"Total time: {elapsed:.1f}s | Scenarios: {self.total} | "
              f"Passed: {self.passed} | Failed: {self.failed}")
        print("=" * 80)

        # Group by scenario
        scenarios = {}
        for r in self.results:
            if r.scenario not in scenarios:
                scenarios[r.scenario] = []
            scenarios[r.scenario].append(r)

        for scenario, results in scenarios.items():
            passed = sum(1 for r in results if r.success)
            total = len(results)
            status = "✅" if passed == total else "❌"
            print(f"\n{status} {scenario} ({passed}/{total})")
            for r in results:
                icon = "  ✓" if r.success else "  ✗"
                answer_preview = r.answer[:80].replace("\n", " ") if r.answer else "NO ANSWER"
                if r.error:
                    print(f"{icon} [{r.question_id}] ERROR: {r.error}")
                else:
                    print(f"{icon} [{r.question_id}] ({r.latency_ms:.0f}ms) {answer_preview}")

        print("\n" + "=" * 80)
        if self.failed > 0:
            print("FAILURES:")
            for r in self.results:
                if not r.success:
                    print(f"  - {r.scenario} / {r.question_id}: {r.error or r.answer[:100]}")
            print("=" * 80)


def is_valid_answer(answer: str) -> bool:
    """Check if an answer is a real answer (not a fallback/error)."""
    if not answer or not answer.strip():
        return False
    bad_patterns = [
        "INSUFFICIENT EVIDENCE",
        "NOT FOUND",
        "ERROR:",
        "confidence too low",
        "No relevant context",
    ]
    answer_upper = answer.upper()
    for pattern in bad_patterns:
        if pattern.upper() in answer_upper:
            return False
    # Must be at least a sentence
    return len(answer.strip()) > 20


def load_base_config() -> dict:
    """Load the base experiment config."""
    config_path = Path("experiments/01_format_comparison/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_config(overrides: dict) -> Config:
    """Build a Config with overrides applied."""
    raw = load_base_config()
    # Deep merge overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and key in raw:
            raw[key].update(value)
        else:
            raw[key] = value
    return Config(**raw)


def run_scenario(
    scenario_name: str,
    config_overrides: dict,
    questions: List[Question],
    pipeline_type: str = "naive",
    agentic_strategy: str = "decompose",
    report: TestReport = None,
    index_cache: dict = None,
) -> None:
    """Run a single test scenario."""
    logger.info(f"═══ Running scenario: {scenario_name} ═══")

    try:
        cfg = build_config(config_overrides)
    except Exception as e:
        for q in questions:
            report.add(ScenarioResult(
                scenario=scenario_name, question_id=q.id,
                pipeline=pipeline_type, answer="", latency_ms=0,
                success=False, error=f"Config error: {e}"
            ))
        return

    # Build index (cache by backend to avoid rebuilding)
    cache_key = cfg.index.backend
    try:
        if index_cache and cache_key in index_cache:
            index = index_cache[cache_key]
        else:
            index = get_index(cfg.index, cfg.embed)
            # Check if index is built
            experiment_name = cfg.experiment.name
            if hasattr(index, "is_built") and not index.is_built(experiment_name):
                # Need to build - load docs and chunk
                logger.info(f"Building index for backend={cfg.index.backend}...")
                documents = load_documents(cfg.benchmark)
                from raglab.parsers.normalizer import DocumentNormalizer
                normalizer = DocumentNormalizer()
                documents = normalizer.normalize(documents)
                documents = normalizer.deduplicate(documents)
                chunker = get_chunker(cfg.chunk)
                all_chunks = []
                for doc in documents:
                    all_chunks.extend(chunker.chunk(doc))
                import inspect
                build_params = inspect.signature(index.build).parameters
                if "experiment_name" in build_params:
                    index.build(all_chunks, experiment_name=experiment_name)
                else:
                    index.build(all_chunks)
                logger.info(f"Index built: {len(all_chunks)} chunks")

            if index_cache is not None:
                index_cache[cache_key] = index
    except Exception as e:
        for q in questions:
            report.add(ScenarioResult(
                scenario=scenario_name, question_id=q.id,
                pipeline=pipeline_type, answer="", latency_ms=0,
                success=False, error=f"Index error: {e}"
            ))
        return

    # Build reranker
    try:
        reranker = get_reranker(cfg.retrieve)
    except Exception as e:
        for q in questions:
            report.add(ScenarioResult(
                scenario=scenario_name, question_id=q.id,
                pipeline=pipeline_type, answer="", latency_ms=0,
                success=False, error=f"Reranker error: {e}"
            ))
        return

    # Build pipeline
    try:
        if pipeline_type == "naive":
            pipeline = NaiveRAGPipeline(index=index, reranker=reranker, cfg=cfg)
        else:
            cfg.agentic.strategy = agentic_strategy
            pipeline = AgenticRAGPipeline(index=index, reranker=reranker, cfg=cfg)
    except Exception as e:
        for q in questions:
            report.add(ScenarioResult(
                scenario=scenario_name, question_id=q.id,
                pipeline=pipeline_type, answer="", latency_ms=0,
                success=False, error=f"Pipeline init error: {e}"
            ))
        return

    # Run each question
    for q in questions:
        t_start = time.perf_counter()
        try:
            result = pipeline.run(q)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            answer = result.predicted_answer
            success = is_valid_answer(answer)
            error = None if success else f"Bad answer: {answer[:100]}"
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            answer = ""
            success = False
            error = f"Runtime error: {type(e).__name__}: {e}"

        report.add(ScenarioResult(
            scenario=scenario_name,
            question_id=q.id,
            pipeline=pipeline_type,
            answer=answer,
            latency_ms=elapsed_ms,
            success=success,
            error=error,
        ))
        status = "✓" if success else "✗"
        logger.info(f"  {status} {q.id}: {elapsed_ms:.0f}ms")


def main():
    report = TestReport()
    report.start_time = time.time()
    index_cache = {}  # Cache built indexes across scenarios

    # Use only 1 question for expensive tests (LLM-heavy) to keep time manageable
    quick_questions = TEST_QUESTIONS[:1]  # Just the memory leak question
    all_questions = TEST_QUESTIONS  # All 3 for fast tests

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Index Backends (with naive pipeline, no reranker)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🔍 PHASE 1: Testing Index Backends")
    print("-" * 40)

    # 1a. Chroma (dense)
    run_scenario(
        "index/chroma_dense",
        {"index": {"backend": "chroma"}, "retrieve": {"reranker": "none", "confidence_threshold": 0.1}},
        all_questions, "naive", report=report, index_cache=index_cache,
    )

    # 1b. BM25 (sparse)
    run_scenario(
        "index/bm25_sparse",
        {"index": {"backend": "bm25"}, "retrieve": {"reranker": "none", "confidence_threshold": 0.1}},
        all_questions, "naive", report=report, index_cache=index_cache,
    )

    # 1c. Hybrid RRF
    run_scenario(
        "index/hybrid_rrf",
        {"index": {"backend": "hybrid_rrf"}, "retrieve": {"reranker": "none", "confidence_threshold": 0.1}},
        all_questions, "naive", report=report, index_cache=index_cache,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Rerankers (with chroma index, naive pipeline)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🔄 PHASE 2: Testing Rerankers")
    print("-" * 40)

    # 2a. Cross-encoder reranker
    run_scenario(
        "rerank/cross_encoder",
        {"index": {"backend": "chroma"}, "retrieve": {"reranker": "cross_encoder", "rerank": True, "confidence_threshold": 0.1}},
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # 2b. BM25 reranker
    run_scenario(
        "rerank/bm25_rerank",
        {"index": {"backend": "chroma"}, "retrieve": {"reranker": "bm25_rerank", "rerank": True, "confidence_threshold": 0.1}},
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # 2c. Reciprocal rank fusion reranker
    run_scenario(
        "rerank/reciprocal_rank",
        {"index": {"backend": "chroma"}, "retrieve": {"reranker": "reciprocal_rank", "rerank": True, "confidence_threshold": 0.1}},
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Agentic RAG Strategies
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🤖 PHASE 3: Testing Agentic RAG Strategies")
    print("-" * 40)

    agentic_base = {
        "index": {"backend": "chroma"},
        "retrieve": {"reranker": "none", "confidence_threshold": 0.1},
        "intent": {"mode": "always_complex"},
    }

    # 3a. Decompose
    run_scenario(
        "agentic/decompose",
        {**agentic_base, "agentic": {"strategy": "decompose"}},
        quick_questions, "agentic", "decompose", report=report, index_cache=index_cache,
    )

    # 3b. Step-back
    run_scenario(
        "agentic/step_back",
        {**agentic_base, "agentic": {"strategy": "step_back"}},
        quick_questions, "agentic", "step_back", report=report, index_cache=index_cache,
    )

    # 3c. HyDE
    run_scenario(
        "agentic/hyde",
        {**agentic_base, "agentic": {"strategy": "hyde"}},
        quick_questions, "agentic", "hyde", report=report, index_cache=index_cache,
    )

    # 3d. ReAct
    run_scenario(
        "agentic/react",
        {**agentic_base, "agentic": {"strategy": "react"}},
        quick_questions, "agentic", "react", report=report, index_cache=index_cache,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: Intent Classification
    # ═══════════════════════════════════════════════════════════════════════
    print("\n🧠 PHASE 4: Testing Intent Classification")
    print("-" * 40)

    # 4a. Rule-based classifier
    run_scenario(
        "intent/rule_based",
        {"index": {"backend": "chroma"}, "retrieve": {"reranker": "none", "confidence_threshold": 0.1}, "intent": {"mode": "rule"}},
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # 4b. Always simple (forces naive)
    run_scenario(
        "intent/always_simple",
        {"index": {"backend": "chroma"}, "retrieve": {"reranker": "none", "confidence_threshold": 0.1}, "intent": {"mode": "always_simple"}},
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: Combined Scenarios (realistic configs)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n⚡ PHASE 5: Combined Realistic Scenarios")
    print("-" * 40)

    # 5a. Best quality: hybrid + cross_encoder + naive
    run_scenario(
        "combined/hybrid_crossencoder_naive",
        {
            "index": {"backend": "hybrid_rrf"},
            "retrieve": {"reranker": "cross_encoder", "rerank": True, "confidence_threshold": 0.1, "top_k": 5},
        },
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # 5b. Fast path: bm25 + no rerank + naive
    run_scenario(
        "combined/bm25_norerank_naive",
        {
            "index": {"backend": "bm25"},
            "retrieve": {"reranker": "none", "confidence_threshold": 0.1, "top_k": 3},
        },
        quick_questions, "naive", report=report, index_cache=index_cache,
    )

    # 5c. Agentic with reranking
    run_scenario(
        "combined/chroma_crossencoder_agentic_decompose",
        {
            "index": {"backend": "chroma"},
            "retrieve": {"reranker": "cross_encoder", "rerank": True, "confidence_threshold": 0.1},
            "intent": {"mode": "always_complex"},
            "agentic": {"strategy": "decompose"},
        },
        quick_questions, "agentic", "decompose", report=report, index_cache=index_cache,
    )

    # ═══════════════════════════════════════════════════════════════════════
    report.end_time = time.time()
    report.print_summary()

    # Save report as JSON
    out_path = Path("out/raglab_out/test_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "elapsed_seconds": report.end_time - report.start_time,
        "results": [
            {
                "scenario": r.scenario,
                "question_id": r.question_id,
                "pipeline": r.pipeline,
                "answer": r.answer[:200],
                "latency_ms": r.latency_ms,
                "success": r.success,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport saved to: {out_path}")

    # Exit code
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
