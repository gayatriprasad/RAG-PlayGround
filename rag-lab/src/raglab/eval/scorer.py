"""
Evaluation scorer with multiple metrics.
Scores EvalResult objects for correctness and completeness.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from raglab.config import EvalCfg, LLMCfg
from raglab.types import EvalResult

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def score(self, result: EvalResult) -> EvalResult:
        """Score a single EvalResult and return it with updated fields."""
        pass


class ExactMatchMetric(BaseMetric):
    """
    Simple exact match metric. No LLM call needed.
    answer_correct = ground_truth.lower().strip() in predicted.lower()
    completeness = 1.0 if correct else 0.0
    """
    
    def score(self, result: EvalResult) -> EvalResult:
        gt = result.ground_truth.lower().strip()
        pred = result.predicted_answer.lower()
        
        correct = gt in pred
        result.answer_correct = correct
        result.completeness = 1.0 if correct else 0.0
        result.overall_score = float(correct)
        
        logger.debug(
            f"ExactMatch q={result.question_id}: "
            f"correct={correct}"
        )
        return result


class LLMJudgeMetric(BaseMetric):
    """
    LLM-based evaluation metric. Two LLM calls per result:
    1. Correctness: YES/NO
    2. Completeness: 0.0-1.0
    """
    
    def __init__(self, llm_cfg: LLMCfg):
        """
        Initialize with LLM configuration.
        
        Args:
            llm_cfg: LLMCfg for making LLM calls
        """
        self.llm_cfg = llm_cfg
        self._client = None
    
    @property
    def client(self):
        """Lazy-initialize LLM client."""
        if self._client is None:
            from raglab.pipelines.naive_rag import build_llm_client
            self._client = build_llm_client(self.llm_cfg)
        return self._client
    
    def score(self, result: EvalResult) -> EvalResult:
        # Call 1: Correctness
        correct = self._judge_correctness(result)
        result.answer_correct = correct
        
        # Call 2: Completeness
        completeness = self._judge_completeness(result)
        result.completeness = completeness
        
        # Overall score
        result.overall_score = float(correct) * completeness
        
        logger.debug(
            f"LLMJudge q={result.question_id}: "
            f"correct={correct}, completeness={completeness:.2f}, "
            f"overall={result.overall_score:.2f}"
        )
        return result
    
    def _judge_correctness(self, result: EvalResult) -> bool:
        """Ask LLM if the predicted answer is correct."""
        messages = [
            {
                "role": "system",
                "content": "You are an evaluation judge. Reply YES or NO only."
            },
            {
                "role": "user",
                "content": (
                    f"Does the predicted answer correctly answer the question "
                    f"given the ground truth?\n\n"
                    f"Question: {result.question}\n"
                    f"Ground truth: {result.ground_truth}\n"
                    f"Predicted answer: {result.predicted_answer}\n\n"
                    f"Reply YES or NO only."
                )
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_cfg.model,
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            logger.warning(f"LLM correctness call failed: {e}")
            return False
    
    def _judge_completeness(self, result: EvalResult) -> float:
        """Ask LLM what fraction of ground truth is captured."""
        messages = [
            {
                "role": "system",
                "content": "You are an evaluation judge. Reply with a decimal number only."
            },
            {
                "role": "user",
                "content": (
                    f"What fraction (0.0-1.0) of the ground truth information "
                    f"is captured in the predicted answer?\n\n"
                    f"Ground truth: {result.ground_truth}\n"
                    f"Predicted answer: {result.predicted_answer}\n\n"
                    f"Reply with a decimal number only (e.g., 0.75)."
                )
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_cfg.model,
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip()
            score = float(answer)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM completeness call failed: {e}")
            return 0.0


class RetrievalRecallMetric(BaseMetric):
    """
    Checks if ground-truth answer text appears in any retrieved chunk content.
    recall@k for each k in EvalCfg.recall_at_k.
    No LLM call needed.
    """
    
    def __init__(self, eval_cfg: EvalCfg):
        """
        Initialize with eval configuration.
        
        Args:
            eval_cfg: EvalCfg with recall_at_k list
        """
        self.recall_at_k = eval_cfg.recall_at_k
    
    def score(self, result: EvalResult) -> EvalResult:
        gt_lower = result.ground_truth.lower().strip()
        chunks = result.retrieved_chunks
        
        # Check recall at each k
        recall_results = {}
        for k in self.recall_at_k:
            top_k_chunks = chunks[:k]
            found = any(
                gt_lower in chunk.chunk.content.lower()
                for chunk in top_k_chunks
            )
            recall_results[str(k)] = found
        
        # Store in metadata dict
        if not result.metadata:
            result.metadata = {}
        result.metadata["recall_at_k"] = recall_results
        
        # Set completeness/answer_correct if not already set by another metric
        if result.completeness is None:
            # Use recall@5 as proxy for completeness
            max_k = max(self.recall_at_k)
            result.completeness = 1.0 if recall_results.get(str(max_k), False) else 0.0
        
        if result.answer_correct is None:
            result.answer_correct = recall_results.get("1", False)
        
        if result.overall_score is None:
            result.overall_score = float(any(recall_results.values()))
        
        logger.debug(
            f"RetrievalRecall q={result.question_id}: "
            f"recall@k={recall_results}"
        )
        return result


class AdversarialMetric(BaseMetric):
    """
    Tests pipeline behavior on adversarial probes.
    Loads probes from JSONL file and checks if responses match expected behavior.
    """
    
    def __init__(self, eval_cfg: EvalCfg):
        """
        Initialize with eval configuration.
        
        Args:
            eval_cfg: EvalCfg with adversarial_path
        """
        self.adversarial_path = eval_cfg.adversarial_path
        self.probes = self._load_probes()
    
    def _load_probes(self) -> list:
        """Load adversarial probes from JSONL file."""
        if not self.adversarial_path:
            return []
        
        from pathlib import Path
        path = Path(self.adversarial_path)
        if not path.exists():
            logger.warning(f"Adversarial probes file not found: {path}")
            return []
        
        probes = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        probes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Loaded {len(probes)} adversarial probes")
        return probes
    
    def score(self, result: EvalResult) -> EvalResult:
        """
        Score based on adversarial behavior.
        For normal questions, this is a pass-through.
        """
        if not self.probes:
            return result
        
        predicted = result.predicted_answer.lower()
        
        # Check if this question matches an adversarial probe
        for probe in self.probes:
            if probe.get("query", "").lower() in result.question.lower():
                expected = probe.get("expected_behavior", "answer")
                
                if expected == "refuse":
                    # Should refuse to answer
                    correct = any(
                        phrase in predicted
                        for phrase in ["not found", "cannot answer", "insufficient", "i don't know"]
                    )
                elif expected == "flag_uncertainty":
                    # Should flag low confidence
                    correct = any(
                        phrase in predicted
                        for phrase in ["not sure", "uncertain", "may not be", "insufficient"]
                    )
                else:  # "answer"
                    correct = "not found" not in predicted
                
                result.answer_correct = correct
                if result.overall_score is None:
                    result.overall_score = float(correct)
                break
        
        return result


class BenchmarkScorer:
    """
    Main scorer that orchestrates multiple metrics.
    """
    
    def __init__(self, eval_cfg: EvalCfg, llm_cfg: LLMCfg):
        """
        Initialize BenchmarkScorer.
        
        Args:
            eval_cfg: EvalCfg with metrics list
            llm_cfg: LLMCfg for LLM-based metrics
        """
        self.eval_cfg = eval_cfg
        self.llm_cfg = llm_cfg
        self.metrics = self._build_metrics()
    
    def _build_metrics(self) -> List[BaseMetric]:
        """Build metric instances from config."""
        metrics = []
        for m in self.eval_cfg.metrics:
            match m:
                case "exact_match":
                    metrics.append(ExactMatchMetric())
                case "llm_judge":
                    metrics.append(LLMJudgeMetric(self.llm_cfg))
                case "retrieval_recall":
                    metrics.append(RetrievalRecallMetric(self.eval_cfg))
                case "adversarial":
                    metrics.append(AdversarialMetric(self.eval_cfg))
                case _:
                    logger.warning(f"Unknown metric: {m}")
        
        logger.info(f"BenchmarkScorer initialized with metrics: {[type(m).__name__ for m in metrics]}")
        return metrics
    
    def score(self, results: List[EvalResult]) -> List[EvalResult]:
        """
        Score all results with configured metrics.
        
        Args:
            results: List of EvalResult objects from pipeline runs
            
        Returns:
            List of EvalResult objects with scores populated
        """
        logger.info(f"Scoring {len(results)} results with {len(self.metrics)} metrics")
        
        scored_results = []
        for i, result in enumerate(results):
            for metric in self.metrics:
                result = metric.score(result)
            scored_results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Scored {i + 1}/{len(results)} results")
        
        logger.info(f"Scoring complete. Mean overall_score: {self._mean_score(scored_results):.3f}")
        return scored_results
    
    def _mean_score(self, results: List[EvalResult]) -> float:
        """Calculate mean overall score."""
        scores = [r.overall_score for r in results if r.overall_score is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    def to_dataframe(self, results: List[EvalResult]):
        """
        Convert results to a pandas DataFrame.
        
        Columns include ALL slot selections so you can pivot any way you want:
        - question_id, question, source_type, category
        - pipeline, index_backend, agentic_strategy
        - reranker, cache_mode, intent_label
        - ground_truth, predicted_answer
        - answer_correct, completeness, overall_score
        - recall_at_1, recall_at_3, recall_at_5
        - latency_ms
        
        Args:
            results: List of scored EvalResult objects
            
        Returns:
            pandas DataFrame
        """
        import pandas as pd
        
        rows = []
        for r in results:
            row = {
                "question_id": r.question_id,
                "question": r.question,
                "source_type": r.source_type,
                "category": r.category,
                "pipeline": r.pipeline,
                "index_backend": r.index_backend,
                "intent_label": r.intent_label,
                "ground_truth": r.ground_truth,
                "predicted_answer": r.predicted_answer,
                "answer_correct": r.answer_correct,
                "completeness": r.completeness,
                "overall_score": r.overall_score,
            }
            
            # Extract metadata fields
            metadata = r.metadata if hasattr(r, 'metadata') and r.metadata else {}
            
            # Agentic strategy (from agentic pipeline metadata)
            row["agentic_strategy"] = metadata.get("strategy", "none")
            
            # Reranker type (from metadata if set, otherwise "none")
            row["reranker"] = metadata.get("reranker", "none")
            
            # Cache mode (from metadata if set, otherwise "none")
            row["cache_mode"] = metadata.get("cache_mode", "none")
            
            # Latency (from metadata if set, otherwise 0)
            row["latency_ms"] = metadata.get("latency_ms", 0)
            
            # Recall@k scores (from RetrievalRecallMetric metadata)
            recall_at_k = metadata.get("recall_at_k", {})
            row["recall_at_1"] = recall_at_k.get("1", None)
            row["recall_at_3"] = recall_at_k.get("3", None)
            row["recall_at_5"] = recall_at_k.get("5", None)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
