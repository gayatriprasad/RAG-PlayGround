# RAG-PlayGround

## What this repo is
A RAG research playground benchmarking retrieval strategies against
EnterpriseRAG-Bench. Every pipeline step is a swappable slot driven by config.

## Repo map
- rag-lab/src/raglab/    → core Python library (chunkers, index, pipelines, eval)
- api/                   → FastAPI backend
- app/                   → Next.js frontend
- .github/               → Copilot instructions, skills, hooks, actions

## Rules that never break
1. Config is truth — no hardcoded paths or model names in source files
2. types.py is the contract — add here before implementing elsewhere
3. BaseIndex, BaseChunker, BaseReranker must be implemented, never instantiated
4. One experiment = one folder — never overwrite prior results
5. All pipeline steps log to stdout + experiment log file

## What good output looks like
- A new index backend: implements BaseIndex, registered in factory, appears in UI dropdown
- A new chunker: implements BaseChunker, registered in factory, config enum updated
- A new eval metric: implements BaseMetric, registered in BenchmarkScorer

## What to never do
- Import from a specific pipeline inside config.py or types.py
- Hardcode "gpt-4o-mini" anywhere outside config.py
- Write to out/ without using the experiment name as a prefix
