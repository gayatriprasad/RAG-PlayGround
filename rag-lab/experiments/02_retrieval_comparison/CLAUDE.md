# Experiment 02 — Retrieval Comparison

## Purpose
Compare ChromaDB (dense), BM25 (sparse), HybridRRF, and PageIndex
on EnterpriseRAG-Bench questions across 4 source types.

## Config files
- config.yaml → chroma backend
- config_pageindex.yaml → pageindex backend
- config_hybrid.yaml → hybrid_rrf backend

## Expected outputs in out/raglab_out/
- 02_retrieval_comparison_*_scores.csv
- 02_retrieval_comparison_*_report.md
- 02_retrieval_comparison_*_traces.jsonl

## Do not modify
- golden/questions.jsonl (ground truth — immutable)
- corpus/raw/ (raw docs — immutable)
