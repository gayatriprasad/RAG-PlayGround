# Config Contract Fix — copilot-instructions.md

Date: 2026-08-11

## Scope
Update the embedded config contract sample in .github/copilot-instructions.md
so it matches the live raglab config contract and no longer references stale
VectorDBCfg.

## Applied edits
- Replaced the class Config(BaseModel) block with the corrected field list:
  includes benchmark, rlm, improvement, observability and uses index: IndexCfg,
  llm: LLMCfg.
- Removed the standalone stale class VectorDBCfg(BaseModel) block because it
  is no longer the real type name in code and was a second source of confusion.

## Notes
- This is a documentation contract correction only; no runtime code path in
  rag-lab is modified by this file update.
- The goal is to stop future generated code from importing nonexistent
  VectorDBCfg from raglab.config.
