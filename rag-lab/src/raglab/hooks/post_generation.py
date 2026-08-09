"""
Post-generation hooks: run after an LLM generation call completes.
"""

import logging
from typing import Optional

from raglab.config import Config
from raglab.hooks.base import PostGenerationHook
from raglab.types import EvalResult, Question
from raglab.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def pricing_model_id(cfg: Config) -> str:
    """Map an `llm.provider`/`llm.model` config pair to a CostTracker.PRICING key."""
    provider = cfg.llm.provider
    model = cfg.llm.model
    if provider == "groq":
        return f"groq/{model}"
    if provider == "ollama":
        return "ollama"
    return model


class CostRecordingHook(PostGenerationHook):
    """
    HOOK 10: Records token usage and cost for every generation call via a
    shared CostTracker (Skill 27). Uses each pipeline's own LLM client to
    approximate token counts, since pipelines don't yet return exact usage.
    """

    def __init__(self, tracker: Optional[CostTracker] = None):
        self.tracker = tracker or CostTracker()

    def run(self, question: Question, result: EvalResult, latency_ms: float, cfg: Config) -> None:
        if not getattr(cfg.cost, "track", True):
            return

        from raglab.models import get_llm

        try:
            client = get_llm(cfg.llm)
            input_tokens = client.count_tokens(question.text)
            output_tokens = client.count_tokens(result.predicted_answer)
        except Exception as e:
            logger.warning(f"CostRecordingHook: token counting failed ({e}), skipping record")
            return

        cost_usd = self.tracker.record(
            model_id=pricing_model_id(cfg),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int(latency_ms),
            stage="generation",
        )

        # Denormalize onto the result so DBWriter/eval_result_to_row can persist
        # per-question cost/latency/model without a second lookup (Skill 29/30).
        result.metadata["cost_usd"] = cost_usd
        result.metadata["latency_ms"] = latency_ms
        result.metadata.setdefault("model", cfg.llm.model)
