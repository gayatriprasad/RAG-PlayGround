# Skill 59 — Dedup query.py overrides against PRESET_FIELD_MAP

Date: 2026-08-11

## Scope
Deduplicate query-time override logic in api/routers/query.py so it reuses
raglab.config PRESET_FIELD_MAP + apply_preset(), eliminating duplicate
field-mapping logic that could drift from Skill 58 preset behavior.

## Implemented changes
- Updated imports in query router to include PRESET_FIELD_MAP and apply_preset.
- Added internal helper: _apply_query_overrides(cfg, req).
- Replaced hand-rolled override block in query() with one call to the helper.
- Preserved explicit standalone rerank boolean override.
- Intentionally excluded chunk_strategy from live query overrides (build-time field).

## Behavior parity check (critical)
Reranker side-effect parity is preserved:
- Old behavior: req.reranker not None/"none" forced cfg.retrieve.rerank = True.
- New behavior: apply_preset() branch for retrieve.reranker still forces
  cfg.retrieve.rerank = True.

No observable behavior change for supported live overrides; only the mapping
source of truth changed to PRESET_FIELD_MAP.

## Regression tests added
File: api/tests/test_query_overrides.py

- test_query_override_matches_preset_field_map
  Verifies each live-applicable key (index_backend, top_k, reranker,
  intent_mode, llm_provider, llm_model) yields exactly the same resulting
  Config as apply_preset(base_cfg, {key: value}).

- test_query_chunk_strategy_override_is_intentionally_ignored
  Documents and enforces the intentional no-op for chunk_strategy at query time.

- test_query_reranker_override_still_sets_rerank_flag
  Guards the reranker => rerank=True side effect.

## Verification
Command:
- python -m pytest api/tests/test_query_overrides.py -q

Result:
- 3 passed
