# RAG-PlayGround Implementation Status

**Last Updated:** 2026-05-22  
**Status:** SKILLS 01-05 + 14A-14B Complete ✅

## Completed Skills

### ✅ SKILL 01: Config & Types Foundation
- **Files:** [config.py](rag-lab/src/raglab/config.py), [types.py](rag-lab/src/raglab/types.py)
- **Config Classes:** 9 Pydantic models (ChunkCfg, RetrieveCfg, GoldenCfg, ExperimentCfg, EmbedCfg, IndexCfg, IntentCfg, LLMCfg, BenchmarkCfg)
- **Type Classes:** 6 data models (Document, Chunk, Question, RetrievedChunk, IntentResult, EvalResult)
- **Validation:** All imports working, instantiation tested

### ✅ SKILL 02: Data Loading (EnterpriseRAG-Bench)
- **Files:** [enterprise_bench.py](rag-lab/src/raglab/parsers/enterprise_bench.py)
- **Functions:**
  - `load_questions(cfg)` - JSONL parsing with filtering by source_types/categories/max_questions
  - `load_documents(cfg)` - Loads .txt/.md files from corpus/raw/<source_type>/
  - `download_bench_slice()` - Placeholder for HuggingFace integration
- **Dependencies:** huggingface-hub, datasets
- **Test Data:** 5 questions, 150 sample documents (50 per source type: confluence, github, slack)

### ✅ SKILL 03: Chunking Strategies
- **Files:** [base.py](rag-lab/src/raglab/chunkers/base.py), [fixed.py](rag-lab/src/raglab/chunkers/fixed.py), [semantic.py](rag-lab/src/raglab/chunkers/semantic.py), [sentence.py](rag-lab/src/raglab/chunkers/sentence.py), [__init__.py](rag-lab/src/raglab/chunkers/__init__.py)
- **Implementations:**
  - **FixedChunker:** Token-based chunking with tiktoken (cl100k_base), overlap support
  - **SemanticChunker:** Sentence-level semantic similarity with sentence-transformers, fallback to FixedChunker
  - **SentenceChunker:** Spacy-based sentence boundary detection with token grouping, fallback to sentencizer
- **Factory:** `get_chunker(cfg)` with match/case for strategy selection
- **Dependencies:** tiktoken, spacy (en_core_web_sm model)
- **Tests:** All passed (FixedChunker: 3 chunks with overlap, SentenceChunker: 2 chunks with sentence grouping, SemanticChunker: fallback validated)

### ✅ SKILL 04: Intent Classification
- **Files:** [base.py](rag-lab/src/raglab/classifiers/base.py), [rule_based.py](rag-lab/src/raglab/classifiers/rule_based.py), [llm_classifier.py](rag-lab/src/raglab/classifiers/llm_classifier.py), [__init__.py](rag-lab/src/raglab/classifiers/__init__.py)
- **Implementations:**
  - **RuleClassifier:** Heuristic-based (word count, complex keywords, sentence structure)
  - **LLMClassifier:** GPT-4o-mini/Ollama with JSON-structured prompts
  - **HybridClassifier:** Rule-based fast path → LLM fallback if confidence < threshold
- **Factory:** `get_classifier(cfg, llm_cfg)` with mode selection
- **Dependencies:** openai
- **Tests:** 100% accuracy on 20 test cases (9 basic + 11 edge cases)

### ✅ SKILL 05: Dense Vector Indexing
- **Files:** [embedder.py](rag-lab/src/raglab/utils/embedder.py), [base.py](rag-lab/src/raglab/index/base.py), [chroma_index.py](rag-lab/src/raglab/index/chroma_index.py), [__init__.py](rag-lab/src/raglab/index/__init__.py)
- **Implementations:**
  - **Embedder:** Singleton pattern per model, batch and single embedding, 384-dim vectors (all-MiniLM-L6-v2)
  - **ChromaIndex:** Persistent ChromaDB with cosine similarity, batch indexing (100 chunks), source_type filtering, rebuild detection
- **Factory:** `get_index(cfg, embed_cfg)` for backend selection
- **Dependencies:** chromadb, sentence-transformers
- **Tests:** 12/12 passed (Embedder: 5/5, ChromaIndex: 7/7)

### ✅ SKILL 14A: Hybrid Retrieval (Dense + Sparse)
- **Files:** [hybrid_index.py](rag-lab/src/raglab/index/hybrid_index.py)
- **Implementation:**
  - Combines ChromaDB (dense) + BM25Okapi (sparse)
  - Reciprocal Rank Fusion (RRF) for result merging
  - Persists BM25 index to bm25.pkl
  - Configurable RRF constant via `IndexCfg.rrf_k`
  - Source_type filtering support
- **Config Updates:**
  - `IndexCfg.backend: Literal["chroma", "pageindex", "hybrid"]`
  - `IndexCfg.rrf_k: int = 60`
- **Factory Update:** `get_index()` now handles "hybrid" backend
- **Dependencies:** rank-bm25
- **Tests:** Build, retrieve, RRF fusion, filtering, persistence - all passed

### ✅ SKILL 14B: Ingestion (Normalize + Deduplicate)
- **Files:** [normalizer.py](rag-lab/src/raglab/parsers/normalizer.py)
- **Implementation:**
  - **normalize():** Whitespace collapse, encoding fixes, metadata enrichment (ingested_at, char_count, word_count, version=sha256[:8])
  - **deduplicate():** 
    - Exact dedup by content hash
    - Near-dedup by Jaccard similarity > 0.85 for documents with similar char_count (±5%)
  - Logs removal counts
- **Tests:** Normalization, exact dedup, near-dedup, unique preservation - all passed

## Verification Results

```
✅ SKILL 01: Config & Types
✅ SKILL 02: Data Loading
✅ SKILL 03: Chunking (Fixed, Semantic, Sentence)
✅ SKILL 04: Intent Classification (Rule, LLM, Hybrid)
✅ SKILL 05: Dense Indexing (ChromaDB)
✅ SKILL 14A: Hybrid Retrieval (Dense + BM25 + RRF)
✅ SKILL 14B: Ingestion (Normalize + Deduplicate)
```
 
**Factory Functions Tested:**
- `get_chunker("fixed")` → FixedChunker ✅
- `get_index("chroma")` → ChromaIndex ✅
- `get_index("hybrid")` → HybridIndex ✅

## Dependencies Installed

```toml
dependencies = [
  "pyyaml",
  "pydantic>=2",
  "tqdm",
  "psutil",
  "numpy",
  "pypdf",
  "pdfplumber",
  "python-docx",
  "huggingface-hub",
  "datasets",
  "tiktoken",           # SKILL 03
  "spacy",              # SKILL 03
  "openai",             # SKILL 04
  "chromadb",           # SKILL 05
  "sentence-transformers", # SKILL 05
  "rank-bm25",          # SKILL 14A
]
```

## File Structure

```
rag-lab/src/raglab/
├── __init__.py
├── config.py                 # SKILL 01 ✅
├── types.py                  # SKILL 01 ✅
├── run_experiment.py
├── chunkers/
│   ├── __init__.py          # Factory
│   ├── base.py              # SKILL 03 ✅
│   ├── fixed.py             # SKILL 03 ✅
│   ├── semantic.py          # SKILL 03 ✅
│   └── sentence.py          # SKILL 03 ✅
├── classifiers/
│   ├── __init__.py          # Factory + Hybrid
│   ├── base.py              # SKILL 04 ✅
│   ├── rule_based.py        # SKILL 04 ✅
│   └── llm_classifier.py    # SKILL 04 ✅
├── index/
│   ├── __init__.py          # Factory
│   ├── base.py              # SKILL 05 ✅
│   ├── chroma_index.py      # SKILL 05 ✅
│   └── hybrid_index.py      # SKILL 14A ✅
├── parsers/
│   ├── __init__.py
│   ├── base.py
│   ├── enterprise_bench.py  # SKILL 02 ✅
│   └── normalizer.py        # SKILL 14B ✅
└── utils/
    ├── __init__.py
    ├── embedder.py          # SKILL 05 ✅
    ├── hashing.py
    ├── memory.py
    └── timing.py
```

## Next Steps (Remaining Skills)

From [.github/copilot-skills.md](/.github/copilot-skills.md):

- **SKILL 06:** Reranker Interface & Cross-Encoder
- **SKILL 07:** RAG Pipeline Architecture (Naive + Agentic)
- **SKILL 08:** Evaluation Framework (LLM Judge, Retrieval Recall)
- **SKILL 09:** Experiment Runner & Orchestration
- **SKILL 10:** Results Storage & Reporting
- **SKILL 11:** FastAPI Backend
- **SKILL 12:** Next.js Frontend
- **SKILL 13:** CI/CD (GitHub Actions)
- **SKILL 14C-14F:** Additional advanced features

## Notes

- All tests passing with comprehensive verification
- No paid managed services used (all free tier: ChromaDB local, OpenAI gpt-4o-mini, Ollama fallback)
- Config-driven design: no hardcoded values
- All factory patterns working correctly
- Ready for pipeline integration (SKILL 07+)
