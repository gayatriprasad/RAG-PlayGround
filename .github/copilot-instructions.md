# RAG-PlayGround — Copilot Instructions (Project Context)

## Vision

A self-contained RAG research playground that benchmarks retrieval strategies
against EnterpriseRAG-Bench (500K synthetic enterprise docs, 500 ground-truth
Q&A pairs across 9 source types). Every retrieval decision is driven by intent
classification — simple queries go through Naive RAG, complex queries through
Agentic RAG. All parameters are tunable via UI. 100% open source, free tier only.
Portfolio-grade code. Clean abstractions. No shortcuts.

## Complete Repo Layout

```
RAG-PlayGround/
├── .github/
│   ├── copilot-instructions.md   ← this file
│   ├── copilot-skills.md
│   ├── copilot-hooks.md
│   ├── copilot-actions.md
│   └── workflows/
│       ├── ci.yml
│       ├── eval.yml
│       ├── deploy.yml
│       └── data-prep.yml
│
├── rag-lab/
│   ├── corpus/raw/               # raw docs per source type (gitignored if >50MB)
│   │   ├── confluence/
│   │   ├── github/
│   │   ├── jira/
│   │   ├── slack/
│   │   ├── gmail/
│   │   ├── linear/
│   │   ├── hubspot/
│   │   ├── fireflies/
│   │   └── gdrive/
│   ├── data/                     # processed/chunked docs
│   ├── experiments/
│   │   ├── 01_format_comparison/
│   │   └── 02_retrieval_comparison/
│   │       └── config.yaml
│   ├── golden/
│   │   └── questions.jsonl       # EnterpriseRAG-Bench ground truth
│   ├── out/raglab_out/           # eval CSVs, charts, reports
│   └── src/raglab/
│       ├── chunkers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fixed.py
│       │   ├── semantic.py
│       │   └── sentence.py
│       ├── classifiers/          # NEW: intent classification
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── rule_based.py
│       │   └── llm_classifier.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── scorer.py
│       │   └── reporter.py
│       ├── hooks/                # NEW: pipeline lifecycle hooks
│       │   ├── __init__.py
│       │   ├── pre_experiment.py
│       │   ├── post_experiment.py
│       │   ├── pre_retrieval.py
│       │   └── post_retrieval.py
│       ├── index/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── chroma_index.py
│       │   └── pageindex_adapter.py
│       ├── parsers/
│       │   ├── __init__.py
│       │   ├── enterprise_bench.py
│       │   └── source_parsers/   # one per source type
│       ├── pipelines/            # NEW: RAG execution pipelines
│       │   ├── __init__.py
│       │   ├── naive_rag.py
│       │   └── agentic_rag.py
│       ├── rerankers/            # NEW
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── cross_encoder.py
│       ├── utils/
│       ├── config.py             # Pydantic config — source of truth
│       ├── run_experiment.py     # CLI entry point
│       └── types.py              # shared type contracts
│
├── app/                          # Next.js frontend
│   ├── src/
│   │   ├── app/                  # Next.js 14 App Router
│   │   ├── components/
│   │   │   ├── ui/               # shadcn/ui components
│   │   │   ├── playground/       # tunable parameter panels
│   │   │   ├── results/          # eval charts, answer display
│   │   │   └── layout/           # sidebar, nav
│   │   └── lib/
│   ├── package.json
│   └── tailwind.config.ts
│
├── api/                          # FastAPI backend
│   ├── main.py
│   ├── routers/
│   │   ├── query.py
│   │   ├── experiments.py
│   │   └── benchmark.py
│   └── models.py
│
└── pyproject.toml
```

## Full Tech Stack

| Layer | Tool | Tier | Why |
|-------|------|------|-----|
| Frontend | Next.js 14 + shadcn/ui + Tailwind | Free (Vercel) | Apple-aesthetic, Copilot-friendly |
| Animations | Framer Motion | Free | Smooth, Apple-like transitions |
| Font | Inter (via next/font) | Free | Closest web equivalent to SF Pro |
| Charts | Recharts | Free | React-native, clean |
| Backend API | FastAPI + Uvicorn | Free | Async, lightweight, auto-docs |
| Vector store | ChromaDB (local persistent) | Free | No infra, no keys |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free | Runs locally, solid quality |
| Reranker | cross-encoder/ms-marco-MiniLM (flashrank) | Free | Local, fast |
| Tree retrieval | pageindex (VectifyAI, MIT) | Free | Vectorless, structured docs |
| LLM | GPT-4o-mini (default) / Ollama+llama3 (free) | Cheap/Free | Configurable |
| Intent classifier | Rule-based + LLM fallback | Free | Lightweight |
| Dataset | EnterpriseRAG-Bench (HuggingFace) | Free | MIT licensed |
| Hosting FE | Vercel (free tier) | Free | Auto-deploy on push |
| CI/CD | GitHub Actions | Free | 2000 min/month |

**Hard rule:** never introduce a paid managed service without explicit instruction.

## Core Architecture

```User Query
     │
     ▼
[Intent Classifier]
  Rule-based fast path → if ambiguous → LLM classifier
     │
  ┌──┴──────────────────────────┐
  │ SIMPLE                      │ COMPLEX
  │ (single-doc, factual,       │ (multi-doc, compare,
  │  direct lookup)             │  conflict, absent-info)
  ▼                             ▼
[Naive RAG Pipeline]      [Agentic RAG Pipeline]
  1. Embed query            1. Decompose → sub-queries
  2. Retrieve top-k         2. Per sub-query: retrieve
  3. Optional rerank        3. Iterative context merge
  4. Generate answer        4. Synthesis + final answer
     │                             │
     └──────────┬──────────────────┘
                ▼
        [Eval Scorer]
     correctness × completeness
     against EnterpriseRAG-Bench
        ground truth
                ▼
        [Results Store]
     CSV + live dashboard
```

Pipeline Slot Model — Every Step is Swappable
QUERY
  │
  ▼
[SLOT 1: INGEST & NORMALIZE]
  Options: basic | dedup_exact | dedup_near | dedup_semantic | llm_metadata
  │
  ▼
[SLOT 2: CHUNKING]
  Options: fixed | sentence | semantic | recursive | no_chunk (PageIndex path)
  │
  ▼
[SLOT 3: EMBEDDING]
  Options: minilm | mpnet | bge_small | bge_large | nomic | none (sparse only)
  │
  ▼
[SLOT 4: INDEXING + RETRIEVAL]
  Options: chroma_dense | bm25_sparse | hybrid_rrf | hybrid_weighted | pageindex
  │
  ▼
[SLOT 5: INTENT CLASSIFICATION]
  Options: rule | llm | hybrid | always_simple | always_agentic
  │
  ├── SIMPLE → [SLOT 6A: NAIVE RAG PIPELINE]
  └── COMPLEX → [SLOT 6B: AGENTIC RAG PIPELINE]
                  Sub-options: decompose_llm | step_back | hyde | react
  │
  ▼
[SLOT 7: RERANKING]
  Options: none | cross_encoder | bm25_rerank | monot5 | reciprocal_rank
  │
  ▼
[SLOT 8: CONFIDENCE SCORING]
  Options: retrieval_only | composite | nli_based | llm_judge
  │
  ▼
[SLOT 9: GENERATION MODE]
  Options: strict_rag | soft_rag | cot_rag | self_check_rag
  │
  ▼
[SLOT 10: HALLUCINATION FALLBACK]
  Options: threshold | nli_check | llm_self_check | always_cite
  │
  ▼
[SLOT 11: EVAL]
  Options: exact_match | llm_judge | ragas | retrieval_recall | adversarial

Every slot is driven by config. UI exposes dropdowns/sliders for each.
Same query, different slot selections = different experiment. That's the playground.


## Config Contract (authoritative — extend only)

```python
# config.py — full extended version
# config.py — full extended version

from __future__ import annotations
from pydantic import BaseModel
from typing import List, Literal, Optional

class ChunkCfg(BaseModel):
    strategy: Literal["fixed","sentence","semantic","recursive","none"] = "fixed"
    chunk_tokens: int = 512
    overlap: int = 50
    recursive_separators: List[str] = ["\n\n", "\n", ".", " "]

class RetrieveCfg(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.0
    rerank: bool = False
    reranker: Literal["none","cross_encoder","bm25_rerank","monot5","reciprocal_rank"] = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    confidence_threshold: float = 0.35
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_mode: Literal["exact","semantic","none"] = "exact"

class IngestCfg(BaseModel):
    dedup: Literal["none","exact","near","semantic"] = "exact"
    near_dedup_threshold: float = 0.85
    extract_metadata: Literal["rule","llm","none"] = "rule"

class EmbedCfg(BaseModel):
    model: Literal[
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "nomic-ai/nomic-embed-text-v1",
        "none"
    ] = "all-MiniLM-L6-v2"
    device: str = "cpu"

class IndexCfg(BaseModel):
    backend: Literal["chroma","bm25","hybrid_rrf","hybrid_weighted","pageindex"] = "chroma"
    persist_dir: str = "./out/chroma"
    hybrid_dense_weight: float = 0.7    # only for hybrid_weighted
    hybrid_sparse_weight: float = 0.3   # must sum to 1.0
    rrf_k: int = 60                     # RRF constant

class IntentCfg(BaseModel):
    mode: Literal["rule","llm","hybrid","always_simple","always_complex"] = "hybrid"
    llm_model: str = "gpt-4o-mini"
    simple_threshold: float = 0.8
    max_sub_queries: int = 4

class AgenticCfg(BaseModel):
    strategy: Literal["decompose","step_back","hyde","react"] = "decompose"
    # decompose: break into sub-questions
    # step_back: abstract to general principle first, then retrieve
    # hyde: generate hypothetical answer, embed it, retrieve similar
    # react: Reasoning + Acting loop with tool calls

class GenerationCfg(BaseModel):
    mode: Literal["strict_rag","soft_rag","cot_rag","self_check_rag"] = "strict_rag"
    # strict: answer ONLY from context, hard fallback if not found
    # soft: from context, flag if supplemented from model knowledge
    # cot: chain-of-thought before final answer
    # self_check: generate answer, then verify it against chunks, revise if inconsistent
    citation_mode: Literal["chunk_id","doc_timestamp","none"] = "chunk_id"

class ConfidenceCfg(BaseModel):
    scorer: Literal["retrieval_only","composite","nli","llm_judge"] = "composite"
    fallback_message: str = "INSUFFICIENT EVIDENCE: confidence too low to answer reliably."

class EvalCfg(BaseModel):
    metrics: List[Literal["exact_match","llm_judge","retrieval_recall","adversarial"]] = ["llm_judge"]
    adversarial_path: Optional[str] = None
    recall_at_k: List[int] = [1, 3, 5]

class IntentCfg(BaseModel):
    mode: Literal["rule", "llm", "hybrid"] = "hybrid"
    llm_model: str = "gpt-4o-mini"
    simple_threshold: float = 0.8   # confidence above → simple path
    max_sub_queries: int = 4        # agentic decomposition limit

class LLMCfg(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    provider: Literal["openai", "ollama"] = "openai"
    ollama_base_url: str = "http://localhost:11434"

class BenchmarkCfg(BaseModel):
    questions_path: str = "./golden/questions.jsonl"
    source_types: List[str] = ["confluence", "github", "jira", "slack"]
    question_categories: Optional[List[str]] = None  # None = all
    max_questions: int = 50

class Config(BaseModel):
    experiment: ExperimentCfg
    ingest: IngestCfg = IngestCfg()
    chunk: ChunkCfg = ChunkCfg()
    embed: EmbedCfg = EmbedCfg()
    index: IndexCfg = IndexCfg()
    retrieve: RetrieveCfg = RetrieveCfg()
    intent: IntentCfg = IntentCfg()
    agentic: AgenticCfg = AgenticCfg()
    generation: GenerationCfg = GenerationCfg()
    confidence: ConfidenceCfg = ConfidenceCfg()
    llm: LLMCfg = LLMCfg()
    golden: GoldenCfg
    benchmark: BenchmarkCfg = BenchmarkCfg()
    eval: EvalCfg = EvalCfg()
```

## Types Contract (authoritative)

```python
# types.py — full version — add here before implementing elsewhere
# types.py — full version — add here before implementing elsewhere

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class Document(BaseModel):
    id: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = {}

class Chunk(BaseModel):
    id: str
    doc_id: str
    content: str
    source_type: str
    chunk_index: int
    metadata: Dict[str, Any] = {}

class Question(BaseModel):
    id: str
    text: str
    ground_truth: str
    source_type: str
    category: str   # single_doc | multi_doc | conflict | absent | metadata

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    reasoning_path: Optional[str] = None  # PageIndex only

class IntentResult(BaseModel):
    query: str
    label: Literal["simple", "complex"]
    confidence: float
    method: str  # "rule" | "llm"

class EvalResult(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    source_type: str
    category: str
    index_backend: str
    pipeline: str   # "naive" | "agentic"
    intent_label: str
    retrieved_chunks: List[RetrievedChunk]
    answer_correct: Optional[bool] = None
    completeness: Optional[float] = None
    overall_score: Optional[float] = None
```

## Coding Rules (always active)

- **Config is truth.** No hardcoded paths, models, or thresholds in source files.
- **Types first.** Add to types.py before implementing. Keep it the shared contract.
- **Interfaces before implementations.** Every module category has a base.py ABC.
- **Run_experiment.py is the single CLI entry.** Never fork it; use hooks for extension.
- **One experiment = one folder.** Never overwrite prior results.
- **Reproducibility.** Same config.yaml → same result, always.
- **Free tier only.** No paid managed services without explicit instruction.
- **No magic strings.** All literals live in config or types enums.
- **Log everything.** Every pipeline step logs to stdout + appends to experiment log file.
- **Apple-aesthetic frontend.** Clean, minimal, Inter font, smooth transitions, cards.

