# ADR-002: LangGraph over AutoGen/CrewAI for multi-agent orchestration

## Status
Accepted

## Context
The agentic RAG pipeline needs a planner → retriever → synthesizer → critic loop
with conditional branching (e.g., re-retrieve if the critic rejects an answer).
Several multi-agent frameworks exist (AutoGen, CrewAI, LangGraph).

## Decision
Use LangGraph with a flat state graph (single level of conditional edges, no
nested sub-graphs spawning further agents). Agents are graph nodes; the shared
state object carries the question, retrieved chunks, draft answer, and critique.

## Consequences
LangGraph's explicit state machine model makes the agentic loop's control flow
inspectable and traceable — each node emits a trace span, and the graph itself
documents the possible transitions. The flat-graph constraint (Module
Responsibility Matrix: "must not spawn nested agents") keeps depth-1 debuggability:
a run's trace is always a single linear-ish path through named nodes, never a
recursive tree of unknown depth.

## Alternatives considered
- **AutoGen**: strong for open-ended multi-agent conversation, but its
  conversation-driven model (agents freely messaging each other) is harder to
  constrain to a fixed, auditable pipeline shape — rejected for auditability.
- **CrewAI**: role-based abstraction is a good fit conceptually, but the
  framework was less mature for fine-grained conditional-edge control at the
  time of this decision — rejected for control-flow precision.
