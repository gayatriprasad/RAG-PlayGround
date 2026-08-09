# ADR-001: Strategy pattern for every pipeline slot

## Status
Accepted

## Context
NeuralBench's core promise is that every pipeline step (chunker, index, reranker,
classifier, LLM provider, prompt strategy) is swappable via one config line. That
requires a uniform way to add a new implementation without touching call sites.

## Decision
Every module category defines a `base.py` Abstract Base Class. Concrete
implementations live in one file each and are registered in a `get_<thing>(cfg)`
factory function that dispatches on a config `Literal`. Pipelines and the API only
ever import the factory, never a concrete class.

## Consequences
Adding a backend is: implement the ABC, register it in the factory, add the
`Literal` value to `config.py`. No pipeline code changes. This is the single
mechanism that makes 13 vector DBs, 8 pipelines, and 5 chunkers config-driven
rather than requiring branching logic scattered across the codebase.

## Alternatives considered
- **Plugin registry with entry_points (setuptools)**: more indirection than needed
  for a project with a fixed, curated set of backends; rejected for simplicity.
- **Inheritance-free duck typing**: faster to write initially, but loses the
  compile-time/import-time guarantee that every backend implements the full
  contract — rejected because contract violations should fail loudly, not at
  runtime deep in a pipeline.
