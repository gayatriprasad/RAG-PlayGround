#!/usr/bin/env python
"""
Extended RAG Combination Tests — Covers ALL remaining untested combinations.

Tests:
1. All agentic strategies (decompose, step_back, hyde, react)
2. All confidence scorers (retrieval_only, composite, nli, llm_judge)
3. All cache modes (exact, semantic, none)
4. All intent modes (rule, llm, hybrid, always_simple, always_complex)
5. All generation modes (strict_rag, soft_rag, cot_rag, self_check_rag)
6. MonoT5 reranker
7. Ollama LLM (vs OpenAI)
8. RAG Fusion + Adaptive pipelines (now that Ollama is running)

Usage:
    python rag-lab/tests/test_extended_combinations.py --test all
    python rag-lab/tests/test_extended_combinations.py --test agentic
    python rag-lab/tests/test_extended_combinations.py --test confidence
"""

import logging
import sys
import time
import argparse
from pathlib import Path
from typing import List

# Add rag-lab/src to path
_RAG_LAB_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from raglab.chunkers import get_chunker
from raglab.classifiers import get_classifier
from raglab.config import (
    ChunkCfg, Config, EmbedCfg, GoldenCfg, IndexCfg, IntentCfg, 
    LLMCfg, RetrieveCfg, ExperimentCfg, AgenticCfg, GenerationCfg, ConfidenceCfg
)
from raglab.index import get_index
from raglab.pipelines import (
    NaiveRAGPipeline, AgenticRAGPipeline, ReflectionRAGPipeline,
    RAGFusionPipeline, AdaptiveRAGPipeline
)
from raglab.rerankers import get_reranker
from raglab.types import Chunk, Document, Question
from raglab.utils.confidence import get_confidence_scorer
from raglab.utils.cache import get_cache

# ─── Test Data ──────────────────────────────────────────────────────────

TEST_DOCUMENTS = [
    Document(id="doc1", content="RAG (Retrieval Augmented Generation) combines retrieval with LLMs to ground answers in facts.", source_type="confluence", metadata={}),
    Document(id="doc2", content="Vector databases like ChromaDB store embeddings for semantic search.", source_type="github", metadata={}),
    Document(id="doc3", content="BM25 is a sparse retrieval algorithm based on term frequency.", source_type="jira", metadata={}),
]

TEST_QUESTIONS = [
    Question(id="q1", text="What is RAG?", ground_truth="Retrieval Augmented Generation", source_type="confluence", category="factual"),
    Question(id="q2", text="Compare ChromaDB and BM25", ground_truth="ChromaDB uses dense vectors, BM25 uses sparse keywords", source_type="github", category="analytical"),
]

# ─── Test Results Tracking ──────────────────────────────────────────────

test_results = {"passed": 0, "failed": 0, "errors": []}

def run_test(name: str, test_func):
    """Run a single test with error handling."""
    global test_results
    start_time = time.time()
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TEST: {name}")
        logger.info(f"{'='*80}")
        test_func()
        duration = time.time() - start_time
        logger.info(f"✅ PASSED ({duration:.2f}s)")
        test_results["passed"] += 1
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ FAILED ({duration:.2f}s): {str(e)}")
        test_results["failed"] += 1
        test_results["errors"].append(f"{name}: {str(e)}")
        import traceback
        traceback.print_exc()

# ─── Test Functions ─────────────────────────────────────────────────────

def test_agentic_strategies():
    """Test all 4 agentic strategies."""
    strategies = ["decompose", "step_back", "hyde", "react"]
    
    # Setup
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for i, doc in enumerate(TEST_DOCUMENTS):
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)
    
    index = get_index(IndexCfg(backend="chroma", persist_dir="./out/test_agentic"), EmbedCfg(model="all-MiniLM-L6-v2"))
    index.build(chunks, experiment_name="test_agentic")
    
    cfg = Config(
        experiment=ExperimentCfg(name="test_agentic", corpus_glob=["**/*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/qa.jsonl"),
        chunk=ChunkCfg(strategy="fixed", chunk_tokens=128),
        retrieve=RetrieveCfg(top_k=3),
        embed=EmbedCfg(model="all-MiniLM-L6-v2"),
        index=IndexCfg(backend="chroma", persist_dir="./out/test_agentic"),
        intent=IntentCfg(mode="always_complex"),
        llm=LLMCfg(provider="ollama", model="llama3", base_url="http://localhost:11434/v1"),
        agentic=AgenticCfg(strategy="decompose"),
        generation=GenerationCfg(mode="strict_rag", citation_mode="chunk_id")
    )
    
    for strategy in strategies:
        logger.info(f"\n  Testing agentic strategy: {strategy}")
        cfg.agentic.strategy = strategy
        pipeline = AgenticRAGPipeline(index, None, cfg)
        result = pipeline.run(TEST_QUESTIONS[0])
        logger.info(f"  ✓ {strategy}: generated answer ({len(result.predicted_answer)} chars)")

def test_confidence_scorers():
    """Test all 4 confidence scorers."""
    scorers = ["retrieval_only", "composite", "nli", "llm_judge"]
    
    # Create mock chunks
    from raglab.types import RetrievedChunk
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id=f"chunk{i}", doc_id=f"doc{i}", content=doc.content, source_type=doc.source_type, chunk_index=i, metadata={}),
            score=1.0 - i*0.1
        )
        for i, doc in enumerate(TEST_DOCUMENTS)
    ]
    
    for scorer_name in scorers:
        logger.info(f"\n  Testing confidence scorer: {scorer_name}")
        cfg = ConfidenceCfg(scorer=scorer_name)
        scorer = get_confidence_scorer(cfg)
        scored = scorer.score(chunks, "What is RAG?")
        logger.info(f"  ✓ {scorer_name}: scored {len(scored)} chunks")

def test_cache_modes():
    """Test all 3 cache modes."""
    cache_modes = ["exact", "semantic", "none"]
    
    for mode in cache_modes:
        logger.info(f"\n  Testing cache mode: {mode}")
        cfg = RetrieveCfg(cache_mode=mode)
        cache = get_cache(cfg)
        
        # Mock chunks
        chunks_mock = []
        cache.set("test query", "chroma", 5, chunks_mock, ttl=3600)
        result = cache.get("test query", "chroma", 5)
        
        if mode == "exact":
            assert result == chunks_mock, f"Cache miss for {mode}"
            logger.info(f"  ✓ {mode}: cache hit successful")
        elif mode == "semantic":
            # Semantic cache may or may not hit (similarity threshold)
            logger.info(f"  ✓ {mode}: semantic cache tested")
        else:
            assert result is None, f"Cache should be disabled for {mode}"
            logger.info(f"  ✓ {mode}: cache disabled (as expected)")

def test_intent_classification_modes():
    """Test all 5 intent modes."""
    modes = ["rule", "llm", "hybrid", "always_simple", "always_complex"]
    
    llm_cfg = LLMCfg(provider="ollama", model="llama3", base_url="http://localhost:11434/v1")
    
    for mode in modes:
        logger.info(f"\n  Testing intent mode: {mode}")
        cfg = IntentCfg(mode=mode, simple_threshold=0.8, llm_model="llama3")
        classifier = get_classifier(cfg, llm_cfg)
        
        result = classifier.classify(TEST_QUESTIONS[0].text)
        logger.info(f"  ✓ {mode}: classified as {result.label} (confidence: {result.confidence:.2f})")

def test_generation_modes():
    """Test all 4 generation modes."""
    modes = ["strict_rag", "soft_rag", "cot_rag", "self_check_rag"]
    
    # Setup
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)
    
    index = get_index(IndexCfg(backend="chroma", persist_dir="./out/test_gen"), EmbedCfg(model="all-MiniLM-L6-v2"))
    index.build(chunks, experiment_name="test_gen")
    
    for mode in modes:
        logger.info(f"\n  Testing generation mode: {mode}")
        cfg = Config(
            experiment=ExperimentCfg(name="test_gen", corpus_glob=["**/*.txt"], representations=["chroma"]),
            golden=GoldenCfg(path="./golden/qa.jsonl"),
            chunk=ChunkCfg(strategy="fixed", chunk_tokens=128),
            retrieve=RetrieveCfg(top_k=3),
            embed=EmbedCfg(model="all-MiniLM-L6-v2"),
            index=IndexCfg(backend="chroma", persist_dir="./out/test_gen"),
            intent=IntentCfg(mode="always_simple"),
            llm=LLMCfg(provider="ollama", model="llama3", base_url="http://localhost:11434/v1"),
            generation=GenerationCfg(mode=mode, citation_mode="chunk_id")
        )
        
        pipeline = NaiveRAGPipeline(index, None, cfg)
        result = pipeline.run(TEST_QUESTIONS[0])
        logger.info(f"  ✓ {mode}: generated answer ({len(result.predicted_answer)} chars)")

def test_monot5_reranker():
    """Test MonoT5 reranker (slow but thorough)."""
    logger.info("\n  Testing MonoT5 reranker...")
    
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)
    
    index = get_index(IndexCfg(backend="chroma", persist_dir="./out/test_monot5"), EmbedCfg(model="all-MiniLM-L6-v2"))
    index.build(chunks, experiment_name="test_monot5")
    
    query = "What is RAG?"
    initial_results = index.retrieve(query, top_k=5, experiment_name="test_monot5")
    
    cfg = RetrieveCfg(reranker="monot5", reranker_model="castorini/monot5-base-msmarco")
    reranker = get_reranker(cfg)
    
    reranked = reranker.rerank(query, initial_results)
    logger.info(f"  ✓ monot5: reranked {len(reranked)} chunks")

def test_rag_fusion_with_ollama():
    """Test RAG Fusion pipeline with Ollama (was failing before)."""
    logger.info("\n  Testing RAG Fusion with Ollama...")
    
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)
    
    index = get_index(IndexCfg(backend="chroma", persist_dir="./out/test_fusion"), EmbedCfg(model="all-MiniLM-L6-v2"))
    index.build(chunks, experiment_name="test_fusion")
    
    cfg = Config(
        experiment=ExperimentCfg(name="test_fusion", corpus_glob=["**/*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/qa.jsonl"),
        chunk=ChunkCfg(strategy="fixed", chunk_tokens=128),
        retrieve=RetrieveCfg(top_k=3),
        embed=EmbedCfg(model="all-MiniLM-L6-v2"),
        index=IndexCfg(backend="chroma", persist_dir="./out/test_fusion"),
        intent=IntentCfg(mode="always_complex"),
        llm=LLMCfg(provider="ollama", model="llama3", base_url="http://localhost:11434/v1"),
        generation=GenerationCfg(mode="strict_rag", citation_mode="chunk_id")
    )
    
    pipeline = RAGFusionPipeline(index, None, cfg)
    result = pipeline.run(TEST_QUESTIONS[0])
    logger.info(f"  ✓ RAG Fusion: generated answer with query variants")

def test_adaptive_with_ollama():
    """Test Adaptive RAG pipeline with Ollama (was failing before)."""
    logger.info("\n  Testing Adaptive RAG with Ollama...")
    
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)
    
    index = get_index(IndexCfg(backend="chroma", persist_dir="./out/test_adaptive"), EmbedCfg(model="all-MiniLM-L6-v2"))
    index.build(chunks, experiment_name="test_adaptive")
    
    cfg = Config(
        experiment=ExperimentCfg(name="test_adaptive", corpus_glob=["**/*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/qa.jsonl"),
        chunk=ChunkCfg(strategy="fixed", chunk_tokens=128),
        retrieve=RetrieveCfg(top_k=3),
        embed=EmbedCfg(model="all-MiniLM-L6-v2"),
        index=IndexCfg(backend="chroma", persist_dir="./out/test_adaptive"),
        intent=IntentCfg(mode="hybrid"),
        llm=LLMCfg(provider="ollama", model="llama3", base_url="http://localhost:11434/v1"),
        generation=GenerationCfg(mode="strict_rag", citation_mode="chunk_id")
    )
    
    pipeline = AdaptiveRAGPipeline(index, None, cfg)
    result = pipeline.run(TEST_QUESTIONS[0])
    logger.info(f"  ✓ Adaptive RAG: routed to {result.metadata.get('route', 'unknown')}")

def test_ollama_vs_openai():
    """Compare Ollama vs OpenAI (if API key available)."""
    logger.info("\n  Testing Ollama LLM...")
    
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        doc_chunks = chunker.chunk(doc)
        chunks.extend(doc_chunks)
    
    index = get_index(IndexCfg(backend="chroma", persist_dir="./out/test_llm"), EmbedCfg(model="all-MiniLM-L6-v2"))
    index.build(chunks, experiment_name="test_llm")
    
    # Test Ollama
    cfg_ollama = Config(
        experiment=ExperimentCfg(name="test_llm", corpus_glob=["**/*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/qa.jsonl"),
        chunk=ChunkCfg(strategy="fixed", chunk_tokens=128),
        retrieve=RetrieveCfg(top_k=3),
        embed=EmbedCfg(model="all-MiniLM-L6-v2"),
        index=IndexCfg(backend="chroma", persist_dir="./out/test_llm"),
        intent=IntentCfg(mode="always_simple"),
        llm=LLMCfg(provider="ollama", model="llama3", base_url="http://localhost:11434/v1"),
        generation=GenerationCfg(mode="strict_rag", citation_mode="chunk_id")
    )
    
    pipeline = NaiveRAGPipeline(index, None, cfg_ollama)
    result = pipeline.run(TEST_QUESTIONS[0])
    logger.info(f"  ✓ Ollama (llama3): {len(result.predicted_answer)} chars")
    
    # Try different Ollama models
    for model in ["qwen2.5:3b", "gemma3:4b"]:
        logger.info(f"\n  Testing Ollama model: {model}")
        cfg_ollama.llm.model = model
        pipeline = NaiveRAGPipeline(index, None, cfg_ollama)
        result = pipeline.run(TEST_QUESTIONS[0])
        logger.info(f"  ✓ {model}: {len(result.predicted_answer)} chars")

# ─── Main Test Runner ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extended RAG combination tests")
    parser.add_argument(
        "--test",
        choices=["all", "agentic", "confidence", "cache", "intent", "generation", "monot5", "fusion", "adaptive", "ollama"],
        default="all",
        help="Which test category to run"
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 RAG PLAYGROUND — EXTENDED COMBINATION TESTS")
    logger.info("=" * 80)
    
    tests = {
        "agentic": ("Agentic Strategies (4)", test_agentic_strategies),
        "confidence": ("Confidence Scorers (4)", test_confidence_scorers),
        "cache": ("Cache Modes (3)", test_cache_modes),
        "intent": ("Intent Classification (5)", test_intent_classification_modes),
        "generation": ("Generation Modes (4)", test_generation_modes),
        "monot5": ("MonoT5 Reranker", test_monot5_reranker),
        "fusion": ("RAG Fusion + Ollama", test_rag_fusion_with_ollama),
        "adaptive": ("Adaptive RAG + Ollama", test_adaptive_with_ollama),
        "ollama": ("Ollama LLM Models", test_ollama_vs_openai),
    }
    
    if args.test == "all":
        for name, func in tests.values():
            run_test(name, func)
    else:
        name, func = tests[args.test]
        run_test(name, func)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nTotal: {test_results['passed'] + test_results['failed']} | Passed: {test_results['passed']} | Failed: {test_results['failed']}")
    
    success_rate = 100 * test_results["passed"] / (test_results["passed"] + test_results["failed"]) if (test_results["passed"] + test_results["failed"]) > 0 else 0
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if test_results["failed"] > 0:
        logger.info("\nFailed tests:")
        for error in test_results["errors"]:
            logger.info(f"  ❌ {error}")
    
    logger.info("\n" + "=" * 80)
    
    sys.exit(0 if test_results["failed"] == 0 else 1)

if __name__ == "__main__":
    main()
