<h1 align="center">RAG PlayGround</h1>

<p align="center">
  <strong>A modular RAG research lab for benchmarking retrieval strategies with intent-driven pipeline routing.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white" alt="Python 3.14" />
  <img src="https://img.shields.io/badge/Next.js-14-black?logo=next.js" alt="Next.js 14" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

---

## What Is This?

A self-contained playground for experimenting with every dimension of a RAG pipeline — from chunking strategies to retrieval backends to reranking algorithms. Every query is routed through an **intent classifier** that decides whether to use a simple single-shot retrieval (Naive RAG) or a multi-step reasoning pipeline (Agentic RAG).

All parameters are tunable via a clean UI. Same query, different config = different experiment. That's the playground.

## Architecture

```
User Query
     │
     ▼
┌─────────────────────┐
│  Intent Classifier   │  rule-based → LLM fallback
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
 SIMPLE    COMPLEX
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Naive RAG│ │ Agentic RAG  │
│         │ │  decompose   │
│ embed → │ │  step_back   │
│ retrieve│ │  hyde        │
│ rerank  │ │  react       │
│ generate│ │              │
└────┬────┘ └──────┬───────┘
     │              │
     └──────┬───────┘
            ▼
     ┌────────────┐
     │  Eval Loop  │
     │ correctness │
     │ completeness│
     └─────────────┘
```

## Slot Model — Every Step Is Swappable

| Slot | Options |
|------|---------|
| **Chunking** | `fixed` · `sentence` · `semantic` · `recursive` · `none` |
| **Embedding** | `all-MiniLM-L6-v2` · `all-mpnet-base-v2` · `bge-small` · `bge-large` · `nomic` |
| **Index & Retrieval** | `chroma` (dense) · `bm25` (sparse) · `hybrid_rrf` · `hybrid_weighted` · `pageindex` |
| **Intent** | `rule` · `llm` · `hybrid` · `always_simple` · `always_complex` |
| **Pipeline** | Naive RAG · Agentic (`decompose` / `step_back` / `hyde` / `react`) |
| **Reranking** | `none` · `cross_encoder` · `bm25_rerank` · `reciprocal_rank` · `monot5` |
| **Generation** | `strict_rag` · `soft_rag` · `cot_rag` · `self_check_rag` |

## Tech Stack

| Layer | Tool | Cost |
|-------|------|------|
| Frontend | Next.js 14 + shadcn/ui + Tailwind + Framer Motion | Free |
| Backend | FastAPI + Uvicorn | Free |
| Vector Store | ChromaDB (local persistent) | Free |
| Sparse Retrieval | rank-bm25 | Free |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384d) | Free |
| Reranker | flashrank (ms-marco-MiniLM-L-12-v2) | Free |
| LLM | Ollama + llama3 (local) / GPT-4o-mini (configurable) | Free/Cheap |
| Charts | Recharts | Free |

**Hard rule:** no paid managed services without explicit instruction.

## Project Structure

```
RAG-PlayGround/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry, CORS, router mounts
│   ├── models.py               # Request/response Pydantic schemas
│   └── routers/
│       ├── query.py            # POST /query — main RAG endpoint
│       ├── experiments.py      # Experiment management
│       └── benchmark.py        # Batch evaluation runs
│
├── app/                        # Next.js 14 frontend
│   └── src/
│       ├── app/                # App Router pages
│       │   ├── playground/     # Interactive query + param tuning
│       │   ├── benchmark/      # Batch eval dashboard
│       │   ├── compare/        # Side-by-side experiment comparison
│       │   └── config/         # Config editor
│       └── components/
│           ├── ui/             # shadcn/ui primitives
│           └── layout/         # Sidebar navigation
│
├── rag-lab/                    # Core RAG engine (Python package)
│   └── src/raglab/
│       ├── chunkers/           # fixed, sentence, semantic
│       ├── classifiers/        # Intent: rule-based + LLM
│       ├── index/              # chroma, bm25, hybrid_rrf, hybrid_weighted, pageindex
│       ├── pipelines/          # naive_rag, agentic_rag
│       ├── rerankers/          # cross_encoder, bm25_rerank, reciprocal_rank, monot5
│       ├── eval/               # Scoring metrics
│       ├── parsers/            # Document parsers (PDF, DOCX, blocks)
│       ├── utils/              # Embedder, hashing, timing, memory
│       ├── config.py           # Pydantic config — single source of truth
│       ├── types.py            # Shared type contracts
│       └── run_experiment.py   # CLI entry point
│
└── .github/
    └── copilot-instructions.md # Full project spec & coding rules
```

## Quick Start

### Prerequisites

- Python 3.12+ 
- Node.js 18+
- [Ollama](https://ollama.ai) with `llama3` pulled (`ollama pull llama3`)

### 1. Backend Setup

```bash
cd rag-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Index your corpus (uses experiment config)
python -m raglab.run_experiment experiments/01_format_comparison/config.yaml
```

### 2. Start API Server

```bash
cd ..  # project root
rag-lab/.venv/bin/uvicorn api.main:app --port 8001 --host 127.0.0.1
```

### 3. Start Frontend

```bash
cd app
npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### 4. Query via API

```bash
curl -s http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What caused the memory leak in the search service?",
    "source_type": "all",
    "index_backend": "hybrid_rrf",
    "reranker": "cross_encoder",
    "rerank": true,
    "pipeline_override": "agentic"
  }' | python -m json.tool
```

## Configuration

All behavior is driven by YAML config. Example (`experiments/01_format_comparison/config.yaml`):

```yaml
experiment:
  name: "01_format_comparison"
  corpus_glob:
    - "corpus/raw/confluence/*.txt"
    - "corpus/raw/github/*.txt"
    - "corpus/raw/slack/*.txt"
  representations: ["text"]

chunk:
  strategy: "fixed"
  chunk_tokens: 512
  overlap: 50

retrieve:
  top_k: 5
  rerank: false
  confidence_threshold: 0.15

intent:
  mode: "hybrid"
  llm_model: "llama3"

llm:
  provider: "ollama"
  model: "llama3"
  ollama_base_url: "http://localhost:11434/v1"
```

## Test Results

Full pipeline validated across **21 scenarios** — all passing:

| Category | Scenarios | Status |
|----------|-----------|--------|
| Index backends | chroma, bm25, hybrid_rrf × 3 queries each | ✅ 9/9 |
| Rerankers | cross_encoder, bm25_rerank, reciprocal_rank | ✅ 3/3 |
| Agentic strategies | decompose, step_back, hyde, react | ✅ 4/4 |
| Intent modes | rule_based, always_simple | ✅ 2/2 |
| Combined configs | various combos | ✅ 3/3 |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Execute RAG query with parameter overrides |
| `GET` | `/experiments` | List all experiments |
| `POST` | `/benchmark/run` | Run batch evaluation |
| `GET` | `/benchmark/results` | Get eval results |

## Design Principles

1. **Config is truth** — no hardcoded paths, models, or thresholds in source
2. **Types first** — shared contracts in `types.py` before implementation
3. **Interfaces before implementations** — every module has a `base.py` ABC
4. **One experiment = one folder** — never overwrite prior results
5. **Reproducibility** — same config → same result, always
6. **Free tier only** — runs entirely on local models

## License

[MIT](LICENSE)