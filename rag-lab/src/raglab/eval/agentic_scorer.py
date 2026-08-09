"""
Agentic evaluation metrics — Skill 55.

Standard EvalResult scoring (exact_match/llm_judge/retrieval_recall) treats
an agentic run as a black box: question in, answer out. That misses *how*
the agent got there — a lucky answer from a wasteful 4-iteration trajectory
scores the same as an efficient 1-shot answer. AgenticEvalScorer adds three
layers on top of the base EvalResult, using the RAGState/trace produced by
agents/graph.py's run_agentic_graph():

1. StepQualityScorer — scores the plan/retrieval/critique steps individually.
2. TrajectoryScorer  — how many steps/iterations it took to reach the answer.
3. ConsistencyScorer — agreement across repeated runs of the same question
   (only computed when >1 run is supplied — this is expensive since it
   requires re-running the full graph N times per question).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from raglab.types import AgenticEvalResult, ConsistencyScore, EvalResult, StepScore, TrajectoryScore

logger = logging.getLogger(__name__)


class StepQualityScorer:
    """Scores the individual plan / retrieval / critique steps of one
    agentic graph run, using the trace + intermediate state LangGraph
    already produces — no extra LLM calls needed."""

    def score_plan(self, state: Dict[str, Any]) -> StepScore:
        plan = state.get("retrieval_plan") or []
        trace = state.get("trace", {})
        strategy = trace.get("planning_strategy", "unknown")
        n_subqueries = len(plan)
        score = 1.0 if n_subqueries >= 1 else 0.0
        return StepScore(
            step_type="plan",
            score=score,
            metric_scores={"n_subqueries": float(n_subqueries)},
            notes=f"strategy={strategy}",
        )

    def score_retrieval(self, state: Dict[str, Any]) -> StepScore:
        chunks = state.get("retrieved_chunks") or []
        n_chunks = len(chunks)
        avg_relevance = sum(c.score for c in chunks) / n_chunks if n_chunks else 0.0
        score = 1.0 if n_chunks > 0 else 0.0
        return StepScore(
            step_type="retrieval",
            score=score,
            metric_scores={"n_chunks": float(n_chunks), "avg_relevance": avg_relevance},
            notes=f"retrieved {n_chunks} chunks",
        )

    def score_critique(self, state: Dict[str, Any]) -> StepScore:
        trace = state.get("trace", {})
        confidence = float(trace.get("critique_confidence", 0.0))
        errors = float(trace.get("critique_errors", 0))
        unsupported = float(trace.get("critique_unsupported", 0))
        return StepScore(
            step_type="critique",
            score=confidence,
            metric_scores={"errors": errors, "unsupported_claims": unsupported},
            notes=f"confidence={confidence:.2f}",
        )

    def score_all(self, state: Dict[str, Any]) -> List[StepScore]:
        return [self.score_plan(state), self.score_retrieval(state), self.score_critique(state)]


class TrajectoryScorer:
    """
    Scores how efficiently an agentic run reached its final answer.

    steps_to_answer: 1 (initial pass) + revision iterations.
    wasted_retrievals: each revision loop re-retrieves; if the final answer
        still scored low, those extra retrievals bought nothing.
    revision_rounds: the 'iteration' counter LangGraph's should_revise loop
        maintains.
    """

    def score(self, state: Dict[str, Any], overall_score: float) -> TrajectoryScore:
        iteration = int(state.get("iteration", 0))
        steps_to_answer = iteration + 1
        wasted_retrievals = iteration if overall_score < 0.5 else 0
        efficiency = overall_score / steps_to_answer if steps_to_answer > 0 else 0.0
        return TrajectoryScore(
            steps_to_answer=steps_to_answer,
            wasted_retrievals=wasted_retrievals,
            revision_rounds=iteration,
            trajectory_efficiency=efficiency,
        )


class ConsistencyScorer:
    """
    Measures agreement across N repeated runs of the same question — do
    independent runs converge on the same answer and the same retrieval
    plan? Requires an embedder; degrades to `None` (not computed) if one
    cannot be constructed (e.g. sandboxed environment without model access).
    """

    def __init__(self, embed_cfg=None):
        self.embed_cfg = embed_cfg
        self._embedder = None

    def _get_embedder(self):
        if self.embed_cfg is None:
            return None
        if self._embedder is None:
            try:
                from raglab.utils.embedder import get_embedder

                self._embedder = get_embedder(self.embed_cfg)
            except Exception as e:
                logger.warning(f"ConsistencyScorer: embedder unavailable ({e}); skipping consistency scoring")
                return None
        return self._embedder

    def _avg_pairwise_cosine(self, vectors: List[List[float]]) -> float:
        if len(vectors) < 2:
            return 1.0
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(self._cosine(vectors[i], vectors[j]))
        return sum(sims) / len(sims) if sims else 1.0

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(
        self, final_answers: List[str], plans: List[List[str]], overall_scores: List[float]
    ) -> Optional[ConsistencyScore]:
        if len(final_answers) < 2:
            return None
        embedder = self._get_embedder()
        if embedder is None:
            return None

        answer_vectors = [embedder.embed_one(a) for a in final_answers if a]
        answer_consistency = self._avg_pairwise_cosine(answer_vectors)

        plan_texts = [" ".join(p) for p in plans if p]
        plan_vectors = [embedder.embed_one(t) for t in plan_texts]
        plan_consistency = self._avg_pairwise_cosine(plan_vectors)

        mean_score = sum(overall_scores) / len(overall_scores)
        score_variance = sum((s - mean_score) ** 2 for s in overall_scores) / len(overall_scores)

        return ConsistencyScore(
            n_runs=len(final_answers),
            answer_consistency=answer_consistency,
            plan_consistency=plan_consistency,
            score_variance=score_variance,
            reliable=score_variance < 0.05,
        )


class AgenticEvalScorer:
    """Orchestrates StepQualityScorer + TrajectoryScorer + (optional)
    ConsistencyScorer into a single AgenticEvalResult."""

    def __init__(self, embed_cfg=None):
        self.step_scorer = StepQualityScorer()
        self.trajectory_scorer = TrajectoryScorer()
        self.consistency_scorer = ConsistencyScorer(embed_cfg=embed_cfg)

    def score(
        self, base_result: EvalResult, states: List[Dict[str, Any]], overall_scores: Optional[List[float]] = None
    ) -> AgenticEvalResult:
        """
        Args:
            base_result: the already-scored EvalResult for the primary run.
            states: one or more final RAGState dicts (from run_agentic_graph)
                for the same question — pass more than one to enable
                consistency scoring.
            overall_scores: overall_score per state, when states has >1 entry
                (defaults to repeating base_result.overall_score).
        """
        if not states:
            raise ValueError("AgenticEvalScorer.score() requires at least one RAGState")

        primary_state = states[0]
        step_scores = self.step_scorer.score_all(primary_state)
        overall = base_result.overall_score if base_result.overall_score is not None else 0.0
        trajectory = self.trajectory_scorer.score(primary_state, overall)

        consistency = None
        if len(states) > 1:
            final_answers = [s.get("final_answer") or "" for s in states]
            plans = [s.get("retrieval_plan") or [] for s in states]
            scores = overall_scores if overall_scores is not None else [overall] * len(states)
            consistency = self.consistency_scorer.score(final_answers, plans, scores)

        return AgenticEvalResult(
            base_result=base_result,
            step_scores=step_scores,
            trajectory=trajectory,
            consistency=consistency,
        )
