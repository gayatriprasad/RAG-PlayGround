# ADR-007: Config-as-truth with Pydantic

## Status
Accepted

## Context
Reproducibility (NFR: same config + seed → identical result) and "no hardcoded
paths/models/thresholds in source" (Coding Rule 1) both require a single,
type-validated source of truth for every tunable parameter across chunking,
retrieval, generation, and evaluation.

## Decision
`config.py` defines one Pydantic `Config` model composed of per-concern sub-models
(`ChunkCfg`, `RetrieveCfg`, `ModelRegistryCfg`, etc.). Every module reads its
settings from a `Cfg` object passed in at construction time — never from a global,
an environment variable read inline, or a magic literal. `types.py` mirrors this
discipline for data contracts (`Document`, `Chunk`, `EvalResult`, ...).

## Consequences
A YAML file fully determines a run's behavior; two people running the same
`config.yaml` get the same pipeline shape. Config validation happens at
`Config(**yaml_dict)` parse time — invalid configs fail immediately at startup
(`ConfigError`), not mid-run. The cost is upfront ceremony: every new tunable
parameter must be added to `config.py` before it can be used anywhere else.

## Alternatives considered
- **argparse / plain dicts**: no schema validation, no fail-fast on typos in a
  config file, no IDE autocomplete — rejected for reliability.
- **Environment-variable-driven config**: works for secrets (and is used for
  exactly that — DSNs, API keys) but doesn't scale to dozens of structured,
  nested tunables — rejected as the general mechanism.
