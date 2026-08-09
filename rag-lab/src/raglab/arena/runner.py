"""
Arena Runner — run experiments across multiple models and compare results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List

from raglab.config import Config, ModelRegistryCfg
from raglab.types import EvalResult, Question

logger = logging.getLogger(__name__)


class ArenaRunner:
    """
    Model comparison arena: run same questions through multiple models.

    For each question × model:
      - Swap cfg.llm = model_config
      - Run pipeline
      - Collect EvalResult

    API providers run in parallel (asyncio.gather), Ollama sequential (local GPU).
    """

    def __init__(self, pipeline_cfg: Config):
        """
        Args:
            pipeline_cfg: Base configuration with all pipeline settings
        """
        self.base_cfg = pipeline_cfg

    def run(
        self,
        questions: List[Question],
        models: List[ModelRegistryCfg],
    ) -> "ArenaResult":
        """
        Run arena evaluation across multiple models.

        Args:
            questions: List of questions to evaluate
            models: List of model configurations to compare

        Returns:
            ArenaResult with per-model results and leaderboard
        """
        from raglab.types import ArenaResult

        logger.info(
            f"Starting arena run: {len(questions)} questions × {len(models)} models"
        )

        # Separate Ollama (sequential) from API models (parallel)
        ollama_models = [m for m in models if m.provider == "ollama"]
        api_models = [m for m in models if m.provider != "ollama"]

        all_results: Dict[str, List[EvalResult]] = {}

        # Run Ollama models sequentially (local GPU)
        for model_cfg in ollama_models:
            model_id = f"{model_cfg.provider}/{model_cfg.model}"
            logger.info(f"Running Ollama model: {model_id}")
            results = self._run_model(questions, model_cfg)
            all_results[model_id] = results

        # Run API models in parallel
        if api_models:
            logger.info(f"Running {len(api_models)} API models in parallel")
            api_results = asyncio.run(self._run_api_models_parallel(questions, api_models))
            all_results.update(api_results)

        # Build leaderboard
        leaderboard = self._build_leaderboard(all_results)
        winner_by_category = self._compute_category_winners(all_results)

        arena_result = ArenaResult(
            models=[f"{m.provider}/{m.model}" for m in models],
            results=all_results,
            leaderboard=leaderboard,
            winner_by_category=winner_by_category,
        )

        logger.info(f"Arena run complete: {len(all_results)} models evaluated")
        return arena_result

    def _run_model(
        self, questions: List[Question], model_cfg: ModelRegistryCfg
    ) -> List[EvalResult]:
        """Run pipeline for all questions with a specific model."""
        from raglab.pipelines.naive_rag import NaiveRAGPipeline
        from raglab.index import get_index

        # Clone config and swap LLM
        cfg = self.base_cfg.model_copy(deep=True)
        cfg.llm = model_cfg

        # Build index if needed (shared across models)
        index = get_index(cfg.index, cfg.embed)
        if not index.is_built(cfg.experiment.name):
            logger.warning("Index not built, skipping model")
            return []

        # Initialize pipeline
        pipeline = NaiveRAGPipeline(
            index=index,
            reranker=None,  # TODO: support reranker
            cfg=cfg,
        )

        # Run each question
        results = []
        for i, question in enumerate(questions):
            try:
                start = time.perf_counter()
                result = pipeline.run(question)
                elapsed = time.perf_counter() - start

                # Add timing metadata
                result.metadata["latency_ms"] = int(elapsed * 1000)
                result.metadata["model"] = f"{model_cfg.provider}/{model_cfg.model}"

                results.append(result)
                logger.info(
                    f"  [{i+1}/{len(questions)}] {question.id}: "
                    f"{elapsed*1000:.0f}ms"
                )
            except Exception as e:
                logger.error(f"Failed on question {question.id}: {e}")
                continue

        return results

    async def _run_api_models_parallel(
        self, questions: List[Question], models: List[ModelRegistryCfg]
    ) -> Dict[str, List[EvalResult]]:
        """Run multiple API-based models in parallel."""
        tasks = [
            asyncio.to_thread(self._run_model, questions, model_cfg)
            for model_cfg in models
        ]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for model_cfg, results in zip(models, results_list):
            model_id = f"{model_cfg.provider}/{model_cfg.model}"
            if isinstance(results, Exception):
                logger.error(f"Model {model_id} failed: {results}")
                result_dict[model_id] = []
            else:
                result_dict[model_id] = results

        return result_dict

    def _build_leaderboard(
        self, results: Dict[str, List[EvalResult]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build leaderboard: {model_id: {metric: score}}.

        Metrics:
          - accuracy: fraction correct (answer_correct == True)
          - avg_score: mean overall_score
          - avg_latency_ms: mean latency
        """
        leaderboard = {}

        for model_id, eval_results in results.items():
            if not eval_results:
                leaderboard[model_id] = {
                    "accuracy": 0.0,
                    "avg_score": 0.0,
                    "avg_latency_ms": 0.0,
                }
                continue

            correct = sum(1 for r in eval_results if r.answer_correct)
            scores = [r.overall_score for r in eval_results if r.overall_score is not None]
            latencies = [
                r.metadata.get("latency_ms", 0)
                for r in eval_results
                if "latency_ms" in r.metadata
            ]

            leaderboard[model_id] = {
                "accuracy": correct / len(eval_results) if eval_results else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            }

        return leaderboard

    def _compute_category_winners(
        self, results: Dict[str, List[EvalResult]]
    ) -> Dict[str, str]:
        """Find best model per question category."""
        # Group by category
        category_scores: Dict[str, Dict[str, List[float]]] = {}

        for model_id, eval_results in results.items():
            for result in eval_results:
                cat = result.category
                if cat not in category_scores:
                    category_scores[cat] = {}
                if model_id not in category_scores[cat]:
                    category_scores[cat][model_id] = []

                if result.overall_score is not None:
                    category_scores[cat][model_id].append(result.overall_score)

        # Find winner per category
        winners = {}
        for cat, model_scores in category_scores.items():
            best_model = None
            best_avg = -1.0
            for model_id, scores in model_scores.items():
                avg = sum(scores) / len(scores) if scores else 0.0
                if avg > best_avg:
                    best_avg = avg
                    best_model = model_id

            if best_model:
                winners[cat] = best_model

        return winners
