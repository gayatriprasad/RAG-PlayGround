# RAG PlayGround

**A production-ready RAG research lab — benchmark any retrieval strategy, pipeline, or model combination against EnterpriseRAG-Bench.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-22%2F22%20passing-brightgreen)](rag-lab/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What Is This?

A fully-tested RAG research platform where every pipeline step is a swappable slot driven by YAML config. Every query routes through intelligent intent classification — simple queries go to Naive RAG, complex queries to Agentic RAG. All combinations are benchmarked against the [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) ground truth dataset (500 Q&A pairs across 9 enterprise source types).

---

## Architecture

```
User Query
     │
     ▼
┌─────────────────────────────┐
│  Intent Classifier           │  rule · llm · hybrid · always_simple · always_complex
└────────────┬────────────────┘
             │
      ┌──────┴──────────────┐
   SIMPLE               COMPLEX
      │                     │
      ▼                     ▼
┌──────────┐     ┌──────────────────────┐
│ Naive RAG│     │ Agentic RAG           │  decompose · step_back · hyde · react
│          │     ├──────────────────────┤
│ embed    │     │ Reflection RAG        │  generate → critique → refine
│ retrieve │     ├──────────────────────┤
│ rerank   │     │ RAG Fusion            │  N query variants → RRF merge
│ generate │     ├──────────────────────┤
└────┬─────┘     │ Adaptive RAG          │  4-way routing (factual/analytical/generative/conversational)
     │           └──────────┬───────────┘
     │                      │
     └──────────┬───────────┘
                ▼
     ┌─────────────────────┐
     │  Guardrails          │  confidence scoring · caching · tracing
     └──────────┬──────────┘
                ▼
     ┌─────────────────────┐
     │  Eval Loop           │  llm_judge · exact_match · retrieval_recall · adversarial
     └─────────────────────┘
```

---

## Slot Model — Every Step Is Swappable

| Slot | Options |
|---|---|
| **Chunking** | `fixed` · `sentence` · `semantic` · `recursive` · `none` |
| **Embedding** | `all-MiniLM-L6-v2` · `all-mpnet-base-v2` · `bge-small-en-v1.5` · `bge-large` · `nomic` |
| **Retrieval** | `chroma` (dense) · `bm25` (sparse) · `hybrid_rrf` · `hybrid_weighted` · `graph_rag` · `pageindex` |
| **Reranking** | `none` · `cross_encoder` · `bm25_rerank` · `monot5` · `reciprocal_rank` |
| **Intent** | `rule` · `llm` · `hybrid` · `always_simple` · `always_complex` |
| **Pipeline** | Naive · Agentic (`decompose` / `step_back` / `hyde` / `react`) · Reflection · RAG Fusion · Adaptive |
| **Confidence** | `retrieval_only` · `composite` · `nli` · `llm_judge` |
| **Generation** | `strict_rag` · `soft_rag` · `cot_rag` · `self_check_rag` |
| **Cache** | `exact` (SHA-256) · `semantic` (embedding similarity) · `none` |

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Frontend | Next.js 14 + shadcn/ui + Tailwind + Framer Motion | Free |
| Backend | FastAPI + Uvicorn | Free |
| Vector Store | ChromaDB (local persistent) | Free |
| Sparse Retrieval | rank-bm25 | Free |
| Graph Retrieval | NetworkX + spaCy (entity-based) | Free |
| Embeddings | sentence-transformers (MiniLM, MPNet, BGE) | Free |
| Rerankers | flashrank, MonoT5 | Free |
| LLM | Ollama (llama3 / qwen2.5 / gemma3) · GPT-4o-mini | Free / Cheap |
| Multi-Agent | LangGraph | Free |
| Observability | Langfuse (optional) + JSONL tracer | Free |
| MCP Server | fastmcp — Claude Desktop compatible | Free |
| Testing | pytest + custom harness | Free |

> **Hard rule:** no paid managed services. Runs entirely on local models via Ollama.

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai) installed and running

```bash
ollama pull llama3
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

### 1. Backend Setup

```bash
cd rag-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run test suite (22/22 passing)
python tests/test_integration_e2e.py --test all
python tests/test_extended_combinations.py --test all

# Index your corpus
python -m raglab.run_experiment \
  --config experiments/02_retrieval_comparison/config.yaml
```

### 2. Start Ollama

```bash
ollama serve   # run in a separate terminal
# verify: curl http://localhost:11434/v1/models
```

### 3. Start API Server

```bash
# from project root
rag-lab/.venv/bin/uvicorn api.main:app --port 8001 --host 127.0.0.1 --reload
```

### 4. Start Frontend

```bash
cd app
npm install && npm run dev
```

Open [http://localhost:3000/playground](http://localhost:3000/playground)

### 5. Query via API

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

---

## Configuration

All behaviour is driven by YAML config. Same config → same results. Reproducibility guaranteed.

```yaml
# experiments/02_retrieval_comparison/config.yaml
experiment:
  name: "02_retrieval_comparison"
  corpus_glob:
    - "corpus/raw/confluence/*.txt"
    - "corpus/raw/github/*.txt"
  representations: ["chroma", "hybrid_rrf"]

chunk:
  strategy: "fixed"        # fixed · sentence · semantic · recursive · none
  chunk_tokens: 512
  overlap: 50

retrieve:
  top_k: 5
  reranker: "cross_encoder"  # cross_encoder · bm25_rerank · monot5 · none
  rerank: true
  confidence_threshold: 0.35
  cache_mode: "exact"        # exact · semantic · none

intent:
  mode: "hybrid"             # rule · llm · hybrid · always_simple · always_complex
  llm_model: "llama3"
  simple_threshold: 0.8

agentic:
  strategy: "decompose"      # decompose · step_back · hyde · react
  max_sub_queries: 4

generation:
  mode: "strict_rag"         # strict_rag · soft_rag · cot_rag · self_check_rag
  citation_mode: "chunk_id"

llm:
  provider: "ollama"
  model: "llama3"
  ollama_base_url: "http://localhost:11434/v1"   # /v1 suffix required
  temperature: 0.0
  max_tokens: 512

index:
  backend: "hybrid_rrf"    # chroma · bm25 · hybrid_rrf · hybrid_weighted · graph_rag
  persist_dir: "./out/chroma"
  rrf_k: 60

embed:
  model: "all-MiniLM-L6-v2"   # all-MiniLM-L6-v2 · all-mpnet-base-v2 · BAAI/bge-small-en-v1.5
  device: "cpu"

benchmark:
  questions_path: "./golden/questions.jsonl"
  source_types: ["confluence", "github", "jira", "slack"]
  max_questions: 50

eval:
  metrics: ["llm_judge", "retrieval_recall", "exact_match"]
  recall_at_k: [1, 3, 5]
```

---

## Test Results — 22/22 Passing

| Test Suite | Scenarios | Duration | Status |
|---|---|---|---|
| Chunking strategies | 5 strategies × 5 documents | ~11s | ✅ 100% |
| Embedding models | MiniLM · MPNet · BGE | ~21s | ✅ 100% |
| Retrieval backends | chroma · bm25 · hybrid_rrf · graph_rag | ~0.3s | ✅ 100% |
| Rerankers | cross-encoder · bm25 · monot5 · rrf | ~2s | ✅ 100% |
| Pipelines | naive · agentic · reflection · fusion · adaptive | ~15s | ✅ 100% |
| Agentic strategies | decompose · step-back · hyde · react | ~42s | ✅ 100% |
| Confidence scoring | retrieval_only · composite · nli · llm_judge | — | ✅ 100% |
| Intent classification | rule · llm · hybrid · always_simple · always_complex | — | ✅ 100% |
| Hook registry | all 6 lifecycle stages | — | ✅ 100% |
| LangGraph graph | node compilation + stop guard | — | ✅ 100% |

**Performance benchmarks (Ollama / local):**

| Pipeline | Latency |
|---|---|
| Naive RAG (cache hit) | ~1s |
| Naive RAG (cold) | ~30s |
| Agentic RAG — decompose | ~45s |
| Reflection RAG (2 rounds) | ~60s |
| Cache hit reduction | ~90% |

```bash
# Run all tests
python rag-lab/tests/test_integration_e2e.py --test all
python rag-lab/tests/test_extended_combinations.py --test all

# Run specific category
python rag-lab/tests/test_integration_e2e.py --test chunking
python rag-lab/tests/test_integration_e2e.py --test retrieval
python rag-lab/tests/test_extended_combinations.py --test agentic
python rag-lab/tests/test_extended_combinations.py --test confidence
```

See [TEST_COVERAGE_REPORT.md](rag-lab/tests/TEST_COVERAGE_REPORT.md) for full analysis.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Execute RAG query with per-request parameter overrides |
| `GET` | `/experiments` | List all experiments with result summaries |
| `POST` | `/benchmark/run` | Run batch evaluation on a question set |
| `GET` | `/benchmark/results` | Retrieve evaluation results as JSON |
| `GET` | `/health` | Health check |

---

## Key Features

### Multi-Agent Orchestration (LangGraph)
State machine with four dedicated agents and conditional revision loop:
- **Planner** — decomposes complex queries into sub-questions with source_type hints
- **Retriever** — executes per sub-query retrieval with deduplication
- **Synthesizer** — constrained generation with inline `[CHUNK_ID]` citations
- **Critic** — scores answer confidence, flags unsupported claims
- **Stop guard** — forces finalisation after max iterations; prevents infinite loops

### Observability
- Per-query trace: intent → retrieval hops (latency + scores) → rerank delta → generation time + tokens
- Langfuse integration (optional) — faithfulness drift, cost per query, weekly report
- JSONL tracer fallback when Langfuse keys are absent — tracing never skipped
- Answer drift detector — flags when the same question starts returning different answers over time

### Guardrails
- Prompt injection detection (regex patterns, zero cost)
- Toxicity gate (detoxify, local, no API key)
- Context window guard — trims lowest-trust chunks first, never silently truncates
- Hallucination fallback — returns `INSUFFICIENT EVIDENCE` when avg trust score < threshold

### Claude Desktop Integration (MCP)
Expose RAG tools to any MCP-compatible client:
```json
{
  "mcpServers": {
    "rag-playground": {
      "command": "python",
      "args": ["api/mcp_server.py"],
      "env": { "OPENAI_API_KEY": "your-key-here" }
    }
  }
}
```
Tools available: `retrieve` · `ask` · `index_status` · `list_experiments`

See [MCP_SETUP.md](MCP_SETUP.md) for full configuration guide.

---

## Project Structure

```
RAG-PlayGround/
├── CLAUDE.md                        # Agent memory — read by Claude Code natively
├── .github/
│   ├── copilot-instructions.md      # Copilot persistent context
│   ├── copilot-skills.md            # Skills 00–18
│   ├── copilot-hooks.md             # Hooks 01–13
│   ├── copilot-actions.md           # Actions 01–08
│   └── workflows/                   # 8 GitHub Actions workflows
│
├── rag-lab/
│   ├── corpus/raw/                  # Source docs (confluence, github, jira, slack, ...)
│   ├── experiments/                 # One folder per experiment, each with config.yaml
│   ├── golden/questions.jsonl       # EnterpriseRAG-Bench ground truth (immutable)
│   ├── out/raglab_out/              # Eval CSVs, traces, reports
│   └── src/raglab/
│       ├── agents/                  # LangGraph: planner, retriever, synthesizer, critic, graph
│       ├── chunkers/                # 5 strategies + factory
│       ├── classifiers/             # 5 intent modes + factory
│       ├── eval/                    # scorer, reporter
│       ├── hooks/                   # 13 hooks across 6 lifecycle stages + registry
│       ├── index/                   # 6 backends + factory
│       ├── observability/           # Langfuse tracer + JSONL fallback
│       ├── parsers/                 # Document parsers + normalizer
│       ├── pipelines/               # 5 pipelines + base (confidence check, fallback)
│       ├── rerankers/               # 5 methods + factory
│       ├── utils/                   # embedder, cache, confidence, memory
│       ├── config.py                # Pydantic config — single source of truth
│       ├── types.py                 # Shared type contracts
│       └── run_experiment.py        # CLI entry point (typer)
│
├── api/
│   ├── main.py                      # FastAPI app
│   ├── mcp_server.py                # MCP server (Claude Desktop)
│   └── routers/                     # query, experiments, benchmark
│
├── app/                             # Next.js 14 frontend
│   └── src/app/
│       ├── playground/              # Tunable slot controls + answer display
│       ├── benchmark/               # Leaderboard + eval charts
│       ├── compare/                 # A/B side-by-side runs
│       └── config/                  # Config inspector + inline edit
│
└── pyproject.toml
```

---

## Design Principles

1. **Config is truth** — no hardcoded paths, models, or thresholds in source files
2. **Types first** — shared contracts in `types.py` before any implementation
3. **Interfaces before implementations** — every module has a `base.py` ABC
4. **One experiment = one folder** — prior results are never overwritten
5. **Reproducibility** — same `config.yaml` → same result, always
6. **Free tier only** — runs entirely on local models (Ollama)
7. **Test everything** — 100% coverage across all component combinations
8. **Observable by default** — tracing, confidence scores, cache metrics on every query
9. **Agent boundaries are explicit** — retrieval agents never generate; synthesis agents never retrieve
10. **Hooks never modify core logic** — hooks wrap steps, never call LLM directly

---

## Documentation

- [Test Coverage Report](rag-lab/tests/TEST_COVERAGE_REPORT.md)
- [Ollama Integration Notes](OLLAMA_FIXES.md)
- [MCP Setup Guide](MCP_SETUP.md)
- [Copilot Instructions](.github/copilot-instructions.md)

---

## Recent Updates

**v2.0.0 — May 2026**
- 22/22 test suites passing (100% coverage)
- RAG Fusion and Adaptive RAG pipelines
- Multi-agent orchestration with LangGraph (Planner → Retriever → Synthesizer → Critic)
- Reflection RAG with self-critique loop
- Conversation memory for multi-turn sessions
- 4 confidence scoring methods + hallucination fallback
- Semantic query caching
- MCP server for Claude Desktop integration
- Langfuse observability with JSONL fallback
- Validated with llama3, qwen2.5, gemma3, llama3.2, gemma4

---

## License

[MIT](LICENSE)

---

## Acknowledgments

[LangGraph](https://github.com/langchain-ai/langgraph) · [Langfuse](https://langfuse.com) · [ChromaDB](https://www.trychroma.com) · [Ollama](https://ollama.ai) · [sentence-transformers](https://www.sbert.net) · [FastAPI](https://fastapi.tiangolo.com) · [Next.js](https://nextjs.org)

---

<p align="center"><strong>Production-ready · 22/22 tests passing · Zero paid dependencies</strong></p>