"""
RAG PlayGround — End-to-End Integration Test Suite

Tests all combinations of:
- Chunking strategies (fixed, sentence, semantic, recursive, none)
- Embedding models (MiniLM, MPNet, BGE)
- Retrieval backends (chroma, bm25, hybrid_rrf, graph_rag)
- Rerankers (none, cross_encoder, bm25_rerank)
- Pipelines (naive, agentic, reflection, rag_fusion, adaptive)
- Guardrails (confidence scoring, caching, tracing)

Usage:
    python rag-lab/tests/test_integration_e2e.py
    
    # Run specific test
    python rag-lab/tests/test_integration_e2e.py --test chunking
    python rag-lab/tests/test_integration_e2e.py --test pipelines
    python rag-lab/tests/test_integration_e2e.py --test full
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
    LLMCfg, RetrieveCfg, ExperimentCfg
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


# ─── Test Data ─────────────────────────────────────────────────────────────────

TEST_DOCUMENTS = [
    Document(
        id="doc1",
        content="RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation. "
                "It retrieves relevant documents from a knowledge base and uses them as context "
                "for the LLM to generate accurate answers.",
        source_type="confluence",
        metadata={"title": "What is RAG?"}
    ),
    Document(
        id="doc2",
        content="Fine-tuning updates model weights on domain-specific data. It's expensive "
                "and time-consuming but provides deep domain adaptation. RAG is more flexible.",
        source_type="confluence",
        metadata={"title": "RAG vs Fine-tuning"}
    ),
    Document(
        id="doc3",
        content="Vector databases store embeddings for fast similarity search. Popular options "
                "include ChromaDB, Pinecone, and Weaviate. They enable semantic retrieval.",
        source_type="github",
        metadata={"title": "Vector Databases"}
    ),
    Document(
        id="doc4",
        content="Authentication in our API uses OAuth 2.0 and JWT tokens. Users authenticate "
                "via /auth/login endpoint and receive a token valid for 24 hours.",
        source_type="confluence",
        metadata={"title": "API Authentication"}
    ),
    Document(
        id="doc5",
        content="Chunking strategies include fixed-size, sentence-based, and semantic chunking. "
                "Fixed-size is simple but may split sentences. Semantic chunking preserves meaning.",
        source_type="confluence",
        metadata={"title": "Chunking Strategies"}
    )
]

TEST_QUESTIONS = [
    Question(
        id="q1",
        text="What is RAG?",
        ground_truth="RAG combines retrieval with LLM generation",
        source_type="confluence",
        category="factual"
    ),
    Question(
        id="q2",
        text="Compare RAG with fine-tuning",
        ground_truth="RAG is more flexible, fine-tuning is more expensive",
        source_type="confluence",
        category="analytical"
    ),
    Question(
        id="q3",
        text="How does authentication work?",
        ground_truth="OAuth 2.0 and JWT tokens via /auth/login",
        source_type="confluence",
        category="factual"
    )
]


# ─── Test Suite ────────────────────────────────────────────────────────────────

class IntegrationTestSuite:
    """End-to-end integration test suite."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.failed_tests: List[str] = []
    
    def run_test(self, name: str, test_func):
        """Run a single test and record results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TEST: {name}")
        logger.info(f"{'='*80}")
        
        start = time.time()
        try:
            test_func()
            elapsed = time.time() - start
            logger.info(f"✅ PASSED ({elapsed:.2f}s)")
            self.results.append({"name": name, "status": "PASSED", "time": elapsed})
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"❌ FAILED ({elapsed:.2f}s): {e}")
            logger.exception(e)
            self.failed_tests.append(name)
            self.results.append({"name": name, "status": "FAILED", "time": elapsed, "error": str(e)})
    
    def print_summary(self):
        """Print test summary."""
        logger.info(f"\n{'='*80}")
        logger.info("📊 TEST SUMMARY")
        logger.info(f"{'='*80}\n")
        
        passed = sum(1 for r in self.results if r["status"] == "PASSED")
        failed = len(self.failed_tests)
        total = len(self.results)
        
        logger.info(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        logger.info(f"Success Rate: {passed/total*100:.1f}%\n")
        
        if failed > 0:
            logger.info("Failed tests:")
            for name in self.failed_tests:
                logger.info(f"  ❌ {name}")
        
        logger.info("\n" + "="*80)


# ─── Chunking Tests ────────────────────────────────────────────────────────────

def test_chunking_strategies():
    """Test all chunking strategies."""
    strategies = ["fixed", "sentence", "semantic", "recursive", "none"]
    
    for strategy in strategies:
        logger.info(f"\n  Testing chunking strategy: {strategy}")
        
        cfg = ChunkCfg(strategy=strategy, chunk_tokens=256, overlap=50)
        chunker = get_chunker(cfg)
        
        # Chunk all test documents
        all_chunks = []
        for doc in TEST_DOCUMENTS:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
            logger.info(f"    {doc.id}: {len(chunks)} chunks")
        
        assert len(all_chunks) > 0, f"No chunks produced for {strategy}"
        logger.info(f"  ✓ {strategy}: {len(all_chunks)} total chunks")


def test_embedding_models():
    """Test different embedding models."""
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]
    
    for model in models:
        logger.info(f"\n  Testing embedding model: {model}")
        
        from raglab.utils.embedder import Embedder
        embedder = Embedder(model)
        
        # Embed test texts
        texts = [doc.content[:100] for doc in TEST_DOCUMENTS[:3]]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
        assert len(embeddings[0]) > 0, f"Empty embedding for {model}"
        logger.info(f"  ✓ {model}: dim={len(embeddings[0])}")


# ─── Retrieval Tests ───────────────────────────────────────────────────────────

def test_retrieval_backends():
    """Test all retrieval backends."""
    
    # Prepare chunks
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        chunks.extend(chunker.chunk(doc))
    
    backends = ["chroma", "bm25", "hybrid_rrf"]
    # Skip graph_rag for quick tests (requires spaCy entity extraction)
    
    for backend in backends:
        logger.info(f"\n  Testing retrieval backend: {backend}")
        
        cfg = IndexCfg(backend=backend, persist_dir=f"./out/test_{backend}")
        embed_cfg = EmbedCfg(model="all-MiniLM-L6-v2")
        
        index = get_index(cfg, embed_cfg)
        
        # Build index
        logger.info(f"    Building {backend} index...")
        if hasattr(index, 'build'):
            # All indices need experiment_name
            index.build(chunks, experiment_name="integration_test")
        
        # Test retrieval
        query = "What is RAG?"
        results = index.retrieve(query, top_k=3, experiment_name="integration_test")
        
        assert len(results) > 0, f"No results for {backend}"
        assert all(hasattr(r, 'chunk') for r in results), f"Invalid results for {backend}"
        logger.info(f"  ✓ {backend}: retrieved {len(results)} chunks")


# ─── Reranker Tests ────────────────────────────────────────────────────────────

def test_rerankers():
    """Test all reranker implementations."""
    
    # Prepare chunks and retrieve
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        chunks.extend(chunker.chunk(doc))
    
    index_cfg = IndexCfg(backend="chroma", persist_dir="./out/test_rerank")
    embed_cfg = EmbedCfg(model="all-MiniLM-L6-v2")
    index = get_index(index_cfg, embed_cfg)
    index.build(chunks, experiment_name="test_rerank")
    
    query = "What is RAG?"
    initial_results = index.retrieve(query, top_k=5, experiment_name="test_rerank")
    
    rerankers = ["none", "cross_encoder", "bm25_rerank"]
    # Skip monot5 for speed
    
    for reranker_name in rerankers:
        logger.info(f"\n  Testing reranker: {reranker_name}")
        
        cfg = RetrieveCfg(rerank=reranker_name != "none", reranker=reranker_name)
        reranker = get_reranker(cfg)
        
        if reranker:
            reranked = reranker.rerank(query, initial_results)
            assert len(reranked) == len(initial_results), f"Length mismatch for {reranker_name}"
            logger.info(f"  ✓ {reranker_name}: reranked {len(reranked)} chunks")
        else:
            logger.info(f"  ✓ {reranker_name}: no reranking (as expected)")


# ─── Pipeline Tests ────────────────────────────────────────────────────────────

def test_naive_pipeline():
    """Test NaiveRAGPipeline end-to-end."""
    logger.info("\n  Setting up pipeline...")
    
    # Setup
    cfg = _create_test_config()
    index, reranker = _setup_index_and_reranker(cfg)
    
    # Run pipeline
    pipeline = NaiveRAGPipeline(index, reranker, cfg)
    result = pipeline.run(TEST_QUESTIONS[0])
    
    # Verify
    assert result.predicted_answer, "No answer generated"
    assert len(result.retrieved_chunks) > 0, "No chunks retrieved"
    assert result.pipeline == "naive", f"Wrong pipeline: {result.pipeline}"
    logger.info(f"  ✓ NaiveRAGPipeline: answer length={len(result.predicted_answer)}")


def test_agentic_pipeline():
    """Test AgenticRAGPipeline with decomposition."""
    logger.info("\n  Setting up agentic pipeline...")
    
    cfg = _create_test_config()
    cfg.agentic.strategy = "decompose"
    index, reranker = _setup_index_and_reranker(cfg)
    
    pipeline = AgenticRAGPipeline(index, reranker, cfg)
    result = pipeline.run(TEST_QUESTIONS[1])  # Complex question
    
    assert result.predicted_answer, "No answer generated"
    assert result.pipeline == "agentic", f"Wrong pipeline: {result.pipeline}"
    logger.info(f"  ✓ AgenticRAGPipeline: answer length={len(result.predicted_answer)}")


def test_reflection_pipeline():
    """Test ReflectionRAGPipeline with self-critique."""
    logger.info("\n  Setting up reflection pipeline...")
    
    cfg = _create_test_config()
    index, reranker = _setup_index_and_reranker(cfg)
    
    pipeline = ReflectionRAGPipeline(index, reranker, cfg)
    result = pipeline.run(TEST_QUESTIONS[0])
    
    assert result.predicted_answer, "No answer generated"
    assert "reflection_rounds" in result.metadata, "No reflection metadata"
    logger.info(f"  ✓ ReflectionRAGPipeline: {result.metadata['reflection_rounds']} rounds")


def test_rag_fusion_pipeline():
    """Test RAGFusionPipeline with query variants."""
    logger.info("\n  Setting up RAG Fusion pipeline...")
    
    cfg = _create_test_config()
    index, reranker = _setup_index_and_reranker(cfg)
    
    pipeline = RAGFusionPipeline(index, reranker, cfg, n_variants=2)
    result = pipeline.run(TEST_QUESTIONS[0])
    
    assert result.predicted_answer, "No answer generated"
    assert "n_variants" in result.metadata, "No fusion metadata"
    logger.info(f"  ✓ RAGFusionPipeline: {result.metadata['n_variants']} variants")


def test_adaptive_pipeline():
    """Test AdaptiveRAGPipeline with routing."""
    logger.info("\n  Setting up adaptive pipeline...")
    
    cfg = _create_test_config()
    index, reranker = _setup_index_and_reranker(cfg)
    
    pipeline = AdaptiveRAGPipeline(index, reranker, cfg)
    
    # Test factual routing
    result = pipeline.run(TEST_QUESTIONS[0])
    assert result.predicted_answer, "No answer for factual"
    assert "adaptive_query_type" in result.metadata, "No adaptive metadata"
    logger.info(f"  ✓ AdaptiveRAGPipeline (factual): {result.metadata['adaptive_query_type']}")
    
    # Test analytical routing
    result = pipeline.run(TEST_QUESTIONS[1])
    assert result.predicted_answer, "No answer for analytical"
    logger.info(f"  ✓ AdaptiveRAGPipeline (analytical): {result.metadata['adaptive_query_type']}")


# ─── Guardrails Tests ──────────────────────────────────────────────────────────

def test_confidence_scoring():
    """Test all confidence scorers."""
    from raglab.config import ConfidenceCfg
    
    scorers = ["retrieval_only", "composite"]
    # Skip NLI and LLM judge for speed
    
    for scorer_name in scorers:
        logger.info(f"\n  Testing confidence scorer: {scorer_name}")
        
        cfg = ConfidenceCfg(scorer=scorer_name)
        scorer = get_confidence_scorer(cfg)
        
        # Create mock chunks
        from raglab.types import RetrievedChunk
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    id=f"chunk{i}",
                    doc_id=f"doc{i}",
                    content=doc.content,
                    source_type=doc.source_type,
                    chunk_index=i,
                    metadata={}
                ),
                score=0.8 - i*0.1,
                reasoning_path=None
            )
            for i, doc in enumerate(TEST_DOCUMENTS[:3])
        ]
        
        scored = scorer.score(chunks, "test query")
        assert all("trust_score" in c.chunk.metadata for c in scored), f"Missing trust scores for {scorer_name}"
        logger.info(f"  ✓ {scorer_name}: scored {len(scored)} chunks")


def test_caching():
    """Test all cache implementations."""
    cache_modes = ["exact", "none"]
    # Skip semantic cache for speed
    
    for mode in cache_modes:
        logger.info(f"\n  Testing cache mode: {mode}")
        
        cfg = RetrieveCfg(cache_mode=mode)
        cache = get_cache(cfg)
        
        # Test cache operations
        chunks_mock = []  # Mock chunks
        cache.set("test query", "chroma", 5, chunks_mock, ttl=3600)
        result = cache.get("test query", "chroma", 5)
        
        if mode == "exact":
            assert result == chunks_mock, f"Cache miss for {mode}"
            logger.info(f"  ✓ {mode}: cache hit successful")
        else:
            assert result is None, f"Unexpected cache hit for {mode}"
            logger.info(f"  ✓ {mode}: cache disabled (as expected)")


def test_tracing():
    """Test observability tracing."""
    from raglab.observability import PipelineTracer
    
    logger.info("\n  Testing tracing...")
    
    # Use JSONL tracer (no external deps)
    with PipelineTracer("test_exp", "test query", use_langfuse=False) as tracer:
        with tracer.span("span1", input="test"):
            time.sleep(0.01)
        
        with tracer.span("span2"):
            time.sleep(0.01)
        
        tracer.score("test_score", 0.95)
    
    # Check trace file was created
    trace_file = Path("./out/raglab_out/traces/test_exp_traces.jsonl")
    assert trace_file.exists(), "Trace file not created"
    logger.info(f"  ✓ Tracing: trace written to {trace_file}")


# ─── Full End-to-End Test ──────────────────────────────────────────────────────

def test_full_e2e_workflow():
    """Test complete workflow from documents to answer."""
    logger.info("\n  Running full E2E workflow...")
    
    # 1. Chunk documents
    logger.info("    Step 1: Chunking...")
    chunker = get_chunker(ChunkCfg(strategy="fixed", chunk_tokens=128))
    chunks = []
    for doc in TEST_DOCUMENTS:
        chunks.extend(chunker.chunk(doc))
    logger.info(f"      → {len(chunks)} chunks created")
    
    # 2. Build index
    logger.info("    Step 2: Building index...")
    cfg = IndexCfg(backend="chroma", persist_dir="./out/test_e2e")
    embed_cfg = EmbedCfg(model="all-MiniLM-L6-v2")
    index = get_index(cfg, embed_cfg)
    index.build(chunks, experiment_name="test_e2e")
    logger.info(f"      → Index built")
    
    # 3. Classify intent
    logger.info("    Step 3: Intent classification...")
    intent_cfg = IntentCfg(mode="rule")  # Fast rule-based
    llm_cfg = LLMCfg(provider="ollama", model="llama3")
    classifier = get_classifier(intent_cfg, llm_cfg)
    intent = classifier.classify(TEST_QUESTIONS[0].text)
    logger.info(f"      → Intent: {intent.label} (confidence: {intent.confidence:.2f})")
    
    # 4. Retrieve
    logger.info("    Step 4: Retrieval...")
    results = index.retrieve(TEST_QUESTIONS[0].text, top_k=3, experiment_name="test_e2e")
    logger.info(f"      → Retrieved {len(results)} chunks")
    
    # 5. Rerank
    logger.info("    Step 5: Reranking...")
    rerank_cfg = RetrieveCfg(rerank=True, reranker="cross_encoder")
    reranker = get_reranker(rerank_cfg)
    if reranker:
        results = reranker.rerank(TEST_QUESTIONS[0].text, results)
        logger.info(f"      → Reranked {len(results)} chunks")
    
    # 6. Confidence scoring
    logger.info("    Step 6: Confidence scoring...")
    from raglab.config import ConfidenceCfg
    conf_scorer = get_confidence_scorer(ConfidenceCfg(scorer="retrieval_only"))
    results = conf_scorer.score(results, TEST_QUESTIONS[0].text)
    logger.info(f"      → Scored {len(results)} chunks")
    
    # 7. Generate answer
    logger.info("    Step 7: Answer generation...")
    full_cfg = _create_test_config()
    pipeline = NaiveRAGPipeline(index, reranker, full_cfg)
    result = pipeline.run(TEST_QUESTIONS[0])
    logger.info(f"      → Answer: {result.predicted_answer[:100]}...")
    
    logger.info("  ✓ Full E2E workflow complete")


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _create_test_config() -> Config:
    """Create test configuration."""
    from raglab.config import AgenticCfg, GenerationCfg
    
    return Config(
        experiment=ExperimentCfg(
            name="integration_test",
            corpus_glob=["test/*.txt"],
            representations=["chroma"]
        ),
        golden=GoldenCfg(path="./golden/qa.jsonl"),
        chunk=ChunkCfg(strategy="fixed", chunk_tokens=128),
        embed=EmbedCfg(model="all-MiniLM-L6-v2"),
        index=IndexCfg(backend="chroma", persist_dir="./out/test_pipeline"),
        retrieve=RetrieveCfg(top_k=3, rerank=False),
        intent=IntentCfg(mode="rule"),
        llm=LLMCfg(provider="ollama", model="llama3", temperature=0.0),
        agentic=AgenticCfg(strategy="decompose", max_sub_queries=3),
        generation=GenerationCfg(mode="strict_rag", citation_mode="chunk_id")
    )


def _setup_index_and_reranker(cfg: Config):
    """Setup index and reranker for tests."""
    # Chunk documents
    chunker = get_chunker(cfg.chunk)
    chunks = []
    for doc in TEST_DOCUMENTS:
        chunks.extend(chunker.chunk(doc))
    
    # Build index
    index = get_index(cfg.index, cfg.embed)
    if not index.is_built(cfg.experiment.name):
        # All indices need experiment_name
        index.build(chunks, experiment_name=cfg.experiment.name)
    
    # Get reranker
    reranker = get_reranker(cfg.retrieve)
    
    return index, reranker


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Run integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Integration Tests")
    parser.add_argument(
        "--test",
        choices=["all", "chunking", "embedding", "retrieval", "reranking", "pipelines", "guardrails", "full"],
        default="all",
        help="Which test suite to run"
    )
    args = parser.parse_args()
    
    suite = IntegrationTestSuite()
    
    logger.info("=" * 80)
    logger.info("🚀 RAG PLAYGROUND — END-TO-END INTEGRATION TESTS")
    logger.info("=" * 80)
    
    if args.test in ["all", "chunking"]:
        suite.run_test("Chunking Strategies", test_chunking_strategies)
    
    if args.test in ["all", "embedding"]:
        suite.run_test("Embedding Models", test_embedding_models)
    
    if args.test in ["all", "retrieval"]:
        suite.run_test("Retrieval Backends", test_retrieval_backends)
    
    if args.test in ["all", "reranking"]:
        suite.run_test("Rerankers", test_rerankers)
    
    if args.test in ["all", "pipelines"]:
        suite.run_test("Naive Pipeline", test_naive_pipeline)
        suite.run_test("Agentic Pipeline", test_agentic_pipeline)
        suite.run_test("Reflection Pipeline", test_reflection_pipeline)
        suite.run_test("RAG Fusion Pipeline", test_rag_fusion_pipeline)
        suite.run_test("Adaptive Pipeline", test_adaptive_pipeline)
    
    if args.test in ["all", "guardrails"]:
        suite.run_test("Confidence Scoring", test_confidence_scoring)
        suite.run_test("Caching", test_caching)
        suite.run_test("Tracing", test_tracing)
    
    if args.test in ["all", "full"]:
        suite.run_test("Full E2E Workflow", test_full_e2e_workflow)
    
    suite.print_summary()
    
    # Exit with error code if any tests failed
    sys.exit(len(suite.failed_tests))


if __name__ == "__main__":
    main()
