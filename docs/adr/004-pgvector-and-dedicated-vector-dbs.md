# ADR-004: pgvector AND dedicated vector DBs, both supported

## Status
Accepted

## Context
Vector search backends span a spectrum: fully local (ChromaDB, FAISS, BM25),
self-hosted-but-relational (pgvector), self-hosted-dedicated (Milvus), and managed
cloud (Pinecone, Weaviate, Qdrant, Zilliz). Users have different constraints —
some want zero infra, some already run Postgres, some want managed scale.

## Decision
Implement `BaseIndex` for all of the above via `index/` factory dispatch. Do not
pick one as "the" production backend; treat backend choice as a pipeline slot
like any other, benchmarkable against the same golden set.

## Consequences
Users can literally A/B test "is pgvector's HNSW as good as Pinecone's managed
index for my corpus" with one config line and get a statistically-grounded
answer (ADR-005, Skill 43). This is directly aligned with the platform's premise
— vector DB choice becomes an empirical question, not a vendor pitch.

## Alternatives considered
- **pgvector only**: simplest to maintain, but forecloses the "which vector DB is
  actually faster/better for you" comparison that is core to the product's value.
- **Cloud-only (Pinecone/Weaviate/Qdrant)**: breaks the $0 OSS-tier requirement.
