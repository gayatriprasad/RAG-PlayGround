# ADR-005: Reciprocal Rank Fusion over weighted score fusion (hybrid default)

## Status
Accepted

## Context
Hybrid retrieval (dense + sparse) needs a way to combine two ranked lists with
incomparable score scales — cosine similarity (dense) and BM25 scores (sparse)
are not on the same numeric scale, so naive weighted summation requires
per-corpus tuning to avoid one signal dominating.

## Decision
`hybrid_rrf.py` (Reciprocal Rank Fusion, `score = sum(1 / (k + rank))`) is the
default hybrid backend. `hybrid_weighted.py` (tunable `hybrid_dense_weight` /
`hybrid_sparse_weight`) is offered as an explicit alternative for users who want
to tune the blend for their corpus.

## Consequences
RRF requires zero tuning and is embedding-model-independent — it only needs rank
positions, not raw scores, so it degrades gracefully across corpora and embedding
models without per-corpus calibration. The `rrf_k` constant (default 60, standard
in IR literature) is the only knob. Users who need finer control still have the
weighted-fusion escape hatch.

## Alternatives considered
- **Weighted-sum-only**: requires per-corpus tuning of weights to avoid one
  signal dominating due to scale mismatch — rejected as the default because it
  isn't "swap and go."
- **Learned fusion (a small reranking model over both signals)**: adds a training
  dependency and defeats the zero-infra OSS-tier goal — kept out of scope.
