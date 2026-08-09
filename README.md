# NeuralBench

**A full-stack RAG research platform — benchmark any retrieval strategy, model, or pipeline with statistical confidence.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-22%2F22%20passing-brightgreen)](rag-lab/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Demo — 90 seconds, one complete flow

> Upload a document → ask a multi-hop question → watch the pipeline trace → see whether the result is statistically significant vs baseline

*[Screen recording — add link here once recorded]*

The flow: corpus loaded from `/upload` → intent classified as COMPLEX → agentic RAG decomposes the query → pipeline story shows each retrieval hop with trust scores → answer with inline citations → significance panel shows whether this config beats the naive baseline with a real p-value and confidence interval.

**Pipeline hierarchy by corpus size:**
- Small corpus (fits in one context window) → CAG (no retrieval step)
- Medium corpus (100–50K chunks) → RAG (all 7 pipeline strategies)
- Large corpus (50K+ chunks, 10,000+ pages) → RLM (LLM writes Python to query data externally)

### 90-second demo script

1. **Upload** — drop a PDF/Markdown doc on `/upload`. Corpus is parsed, chunked, and indexed live.
2. **Simple query** — ask a direct factual question. Intent classifier routes it SIMPLE → naive RAG. Answer returns with inline citations and a confidence score in under a second (cache-warm path).
3. **Multi-hop query** — ask a question that needs two documents to answer. Intent classifier routes it COMPLEX → agentic RAG. The pipeline story panel shows each decomposed sub-question and its own retrieval hop as it happens.
4. **Significance verdict** — open `/compare`, pick the just-run config against the naive baseline. The panel reports a bootstrap 95% CI, a paired Wilcoxon/McNemar p-value, and an effect size — never just a point estimate.
5. **Pipeline trace** — expand the trace panel: every stage (classify → retrieve → rerank → generate → score) is a span with latency and metadata, exported to the configured tracer (JSONL by default, Langfuse/Phoenix/OpenLLMetry optional).

### Talking points for an interview walkthrough

- *"Every comparison in this system produces a confidence interval and a significance test — not just a delta."* Demonstrates statistical rigor as a first-class design constraint, not an afterthought.
- *"The pipeline is fully config-driven — swapping the vector DB, embedding model, or LLM provider is a one-line YAML change, never a code change."* Demonstrates the strategy-pattern architecture and the Module Responsibility Matrix.
- *"The failure modes are enumerated up front — stale index, judge drift, partial runs — and each has a concrete guard in code, not just a comment."* Demonstrates the Failure Mode Register discipline and defensive engineering under real constraints (see `.github/copilot-instructions.md` §5).

### Core path health check

Before any demo or PR, verify the 10-step core path end-to-end:

```bash
make setup                                    # 1. clean environment in < 15 min
make dev                                      # 2. API on :8001 + frontend on :3000
curl http://localhost:8001/health             # 3a. liveness
curl http://localhost:8001/ready              # 3b. readiness — {"db": bool, "vector": bool, "llm": bool, "ready": bool}
open http://localhost:3000/playground         # 4. ask a question, get a streaming answer
make test                                     # 8. 22/22 tests, coverage ≥ 80%
make eval                                     # 10. full benchmark run + significance report
```

If `/ready` reports `false` for any indicator, `make dev` did not finish starting cleanly — check the corresponding service (Postgres/SQLite, the vector index, and Ollama) before demoing.



## What this is — and what it isn't

NeuralBench is a RAG research platform built to answer one question empirically: **does this configuration change actually improve things, or is it noise?**

Every comparison produces a confidence interval, a paired significance test, and an effect size. A point estimate alone is never reported. This matters because most RAG benchmarks don't test whether their differences are real.

**It is not** a production RAG system, a no-code tool, or a managed service. It is a research platform for practitioners who want to understand RAG from the inside.

---

## What's built vs what's planned

Honest status. A reviewer who finds this themselves loses trust; being told upfront earns it.

### Core — working, tested, demoed

| Component | Detail | Status |
|---|---|---|
| 8 RAG pipelines | Naive, Agentic ×4 (decompose/step-back/HyDE/ReAct), Reflection, RAG Fusion, Adaptive, CAG, RLM | ✅ |
| 5 chunking strategies | Fixed, sentence, semantic, recursive, none (PageIndex path) | ✅ |
| 4 index backends | ChromaDB, BM25, Hybrid RRF, GraphRAG | ✅ |
| Multi-agent graph | LangGraph: planner → retriever → synthesizer → critic | ✅ |
| Evaluation | RAGAS, LLM-judge, recall@k, adversarial probes | ✅ |
| Statistical significance | Bootstrap CIs, paired Wilcoxon/McNemar, Benjamini-Hochberg | ✅ |
| Judge calibration | Cohen's kappa vs human labels, position-bias check | ✅ |
| LLM providers | Ollama (local, free), GPT-4o-mini, Groq | ✅ |
| Observability | JSONL tracer + Langfuse integration | ✅ |
| Frontend | Next.js 14, streaming UI, pipeline story, embedding viz | ✅ |
| Backend | FastAPI, async, SSE streaming, rate limiting, circuit breaker | ✅ |
| Database | Postgres + pgvector, analytical SQL (window fns, CTEs, percentiles) | ✅ |
| 22/22 tests | Unit + integration + contract tests | ✅ |

### Extended — planned and specified, implementation in progress

| Component | Detail | Status |
|---|---|---|
| RLM pipeline (arXiv:2512.24601) | Corpus-as-variable in Python REPL, code-generated retrieval, sub-model delegation, RestrictedPython sandbox | ✅ |
| Cloud vector DBs | Pinecone, Weaviate, Qdrant, Milvus, pgvector | 🔄 |
| Improvement loop | Eval → synthesize gaps → fine-tune embeddings → re-benchmark — backend, API, and frontend (`/improve`: recall heatmap, loop progress, history timeline) done and tested | ✅ |
| Additional LLM providers | Gemini, OpenRouter, Grok, Anthropic | ✅ |
| BYOC upload | PDF, DOCX, CSV ingest + user's own Q&A golden set | 🔄 |
| Challenge mode | Goal-driven guided learning for students | 🔄 |
| Arize Phoenix + OpenLLMetry | Alternative observability backends | ✅ |
| /learn page | Inline concept glossary with "Try it" links | 🔄 |
| Marker/Surya OCR parsing + OCR quality metric | Structured PDF parsing with graceful fallback, CER/WER scoring | ✅ |
| CAG, ColBERT index, agentic state validation, semantic memory | Skill 52 sub-parts | ✅ |
| SIE embedder + quantization | int8/binary embedding quantization | ✅ |
| Agentic eval metrics | Step-level, trajectory, and consistency scoring | ✅ |
| HITL grading UI | Judge calibration + uncertainty sampling annotation queues | ✅ |
| Confidence calibration | Reliability diagram, ECE, Platt/isotonic/temperature recalibration | ✅ |

Architecture and specifications for all extended components are in [`.github/copilot-skills.md`](.github/copilot-skills.md).

---

## Core architecture

```
Query
  │
  ▼
Intent Classifier ──→ SIMPLE ──→ Naive RAG
  │                                  │
  └──────────→ COMPLEX ──→ Agentic RAG (decompose/HyDE/ReAct/step-back)
                                     │
                              Reflection / Fusion / Adaptive
                                     │
                              Confidence scoring → Hallucination fallback
                                     │
                              Statistical significance (every comparison)
                                     │
                              Postgres + pgvector (analytical SQL dashboard)
```

Every result: confidence interval · significance verdict · pipeline trace · citations.

---

## The four pillars

| Pillar | What it demonstrates | Key files |
|---|---|---|
| **Frontend** | Streaming Next.js UI, UMAP viz, Apple-grade design | `app/` |
| **Backend** | Async FastAPI, strategy-pattern model registry, LangGraph orchestration | `api/`, `src/raglab/pipelines/`, `src/raglab/models/` |
| **Networking** | Retry/backoff, circuit breaker, SSE, connection pooling | `src/raglab/net/` |
| **Database** | Postgres + pgvector, analytical SQL (window fns, CTEs, percentiles) | `src/raglab/db/` |

---

## Quick start

**Prerequisites:** Python 3.12+, Node.js 20, Ollama

```bash
ollama pull llama3

make setup   # venv + deps + node_modules + .env from example
make dev     # API on :8001 + frontend on :3000
make test    # 22/22 tests (≥80% coverage gate)
```

Open [http://localhost:3000/playground](http://localhost:3000/playground).

Full setup details in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Configuration

Everything is driven by YAML config. Same config → same results. Reproducibility is enforced.

```yaml
# experiments/02_retrieval_comparison/config.yaml
experiment:
  name: "02_retrieval_comparison"
chunk:
  strategy: "fixed"          # fixed · sentence · semantic · recursive · none
retrieve:
  top_k: 5
  reranker: "cross_encoder"  # none · cross_encoder · bm25_rerank · monot5
  cache_mode: "exact"
intent:
  mode: "hybrid"             # rule · llm · hybrid · always_simple · always_complex
agentic:
  strategy: "decompose"      # decompose · step_back · hyde · react
index:
  backend: "hybrid_rrf"      # chroma · bm25 · hybrid_rrf · graph_rag
llm:
  provider: "ollama"         # ollama · openai · groq · gemini · openrouter
  model: "llama3"
stats:
  bootstrap_samples: 10000
  significance_alpha: 0.05
  multiple_comparison: "benjamini_hochberg"
```

---

## Test results

```
Core integration tests:   13/13 ✅
Extended combination tests: 9/9 ✅
Total:                   22/22 ✅   Coverage ≥ 80%
```

```bash
make test
# or targeted:
python rag-lab/tests/test_integration_e2e.py --test retrieval
python rag-lab/tests/test_extended_combinations.py --test agentic
```

---

## Statistical rigor

The benchmark won't report a delta without testing whether it's real.

Every A-vs-B comparison produces:
- **Bootstrap 95% CI** on the delta (10,000 resamples, fixed seed — reproducible)
- **Paired significance test** — McNemar for binary correctness, Wilcoxon signed-rank for continuous scores
- **Multiple-comparison correction** — Benjamini-Hochberg when comparing >2 configs
- **Effect size** — Cohen's d (continuous) or risk difference (binary)
- **Slice check** — Simpson's-paradox guard: aggregate winner must hold per source_type and per category

Judge validity is measured separately: Cohen's kappa and position-bias flip rate against a hand-labeled calibration sample.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | RAG query, streaming supported |
| `GET` | `/experiments` | List experiments with result summaries |
| `POST` | `/benchmark/run` | Batch evaluation |
| `GET` | `/benchmark/results` | Results as JSON with significance verdicts |
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Readiness (checks DB, vector index, LLM) |

---

## Project structure

```
NeuralBench/
├── CLAUDE.md               # Agent memory — read by Claude Code natively
├── ARCHITECTURE.md         # Layered design, data flow, error taxonomy
├── CONTRIBUTING.md         # Definition of Done, responsibility matrix
├── Makefile                # setup / dev / test / lint / eval
├── docs/adr/               # Architecture Decision Records (001–008)
├── rag-lab/src/raglab/
│   ├── pipelines/          # 6 RAG strategies
│   ├── agents/             # LangGraph multi-agent graph
│   ├── index/              # 13 vector DB backends
│   ├── eval/               # scorer, reporter, significance, judge calibration
│   ├── models/             # 9 LLM provider clients
│   ├── db/                 # Postgres + pgvector + analytical SQL
│   ├── net/                # Networking resilience layer
│   └── improvement/        # Self-improving RAG flywheel
├── api/                    # FastAPI + MCP server
└── app/                    # Next.js 14 frontend
```

---

## Design principles

1. Config is truth — no hardcoded paths, models, or thresholds
2. Every comparison has a confidence interval and significance verdict
3. All SQL is parameterized — injection test in CI proves it
4. Two prompt injection surfaces protected — query scan (Hook 10) and document content scan (DocumentInjectionScanHook) — indirect injection via BYOC documents is flagged and mitigated in generation context
5. Dependencies scanned weekly — Dependabot (pip + npm + Actions) and pip-audit in CI
6. OSS tier is self-contained — Ollama + ChromaDB + SQLite, zero API keys
7. Reproducibility — same `config.yaml` → same result, always
8. Agent boundaries are explicit — retrieval agents never generate

---

## Architecture decisions

Key choices are documented as Architecture Decision Records in [`docs/adr/`](docs/adr/):

- **ADR-001** — Strategy pattern for every pipeline slot
- **ADR-002** — LangGraph over AutoGen/CrewAI
- **ADR-003** — SQLite default, Postgres optional
- **ADR-005** — RRF over weighted fusion (parameter-free)
- **ADR-008** — Custom networking layer over a client library

---

## Acknowledgments

[LangGraph](https://github.com/langchain-ai/langgraph) · [Langfuse](https://langfuse.com) · [ChromaDB](https://www.trychroma.com) · [Ollama](https://ollama.ai) · [sentence-transformers](https://www.sbert.net) · [FastAPI](https://fastapi.tiangolo.com) · [Next.js](https://nextjs.org) · [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)

---

## License

[MIT](LICENSE)

---

<p align="center">
<strong>Every comparison has a confidence interval. Every claim is falsifiable.</strong>
</p>