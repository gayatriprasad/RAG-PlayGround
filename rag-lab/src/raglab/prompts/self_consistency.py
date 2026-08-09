"""
Self-Consistency Prompt Strategy — majority vote over multiple samples.

Runs zero-shot n_samples times at higher temperature (0.5-0.7).
Final answer = majority vote across samples.
All samples stored in metadata for analysis.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List

from raglab.prompts.base import BasePromptStrategy
from raglab.prompts.zero_shot import ZeroShotPrompt
from raglab.types import RetrievedChunk

logger = logging.getLogger(__name__)


class SelfConsistencyPrompt(BasePromptStrategy):
    """
    Self-consistency: generate multiple answers and majority-vote.

    Uses ZeroShotPrompt as the base, samples n times at higher temperature,
    then picks the most common answer.

    Note: This strategy requires special handling in the pipeline —
    the caller must invoke the LLM multiple times. Use `build_messages()`
    for the prompt and `aggregate_responses()` for voting.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._base = ZeroShotPrompt(cfg)
        self.n_samples = getattr(cfg, "n_samples", 5)
        self.temperature = 0.6  # Higher temp for diverse samples

    def build_messages(
        self, query: str, chunks: List[RetrievedChunk], cfg=None
    ) -> List[dict]:
        """Build messages (same as zero-shot). Caller invokes n_samples times."""
        return self._base.build_messages(query, chunks, cfg)

    def get_sampling_params(self) -> dict:
        """Return temperature and n_samples for the caller."""
        return {
            "temperature": self.temperature,
            "n_samples": self.n_samples,
        }

    def aggregate_responses(self, responses: List[str]) -> str:
        """
        Majority vote across multiple responses.

        Args:
            responses: List of n_samples raw LLM responses

        Returns:
            Most common answer (normalized)
        """
        if not responses:
            return ""

        # Normalize answers for comparison
        normalized = [self._normalize(r) for r in responses]

        # Count occurrences
        counter = Counter(normalized)
        most_common = counter.most_common(1)[0]

        logger.info(
            f"Self-consistency: {len(responses)} samples, "
            f"winner='{most_common[0][:50]}...' ({most_common[1]}/{len(responses)} votes)"
        )

        # Return the original (un-normalized) version of the winning answer
        winning_normalized = most_common[0]
        for orig, norm in zip(responses, normalized):
            if norm == winning_normalized:
                return orig.strip()

        return responses[0].strip()

    def parse_response(self, response: str) -> str:
        """For single-response use, just strip."""
        return response.strip()

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip whitespace, remove citations)."""
        import re
        text = text.strip().lower()
        # Remove citation markers like [chunk_id]
        text = re.sub(r"\[[\w\-_]+\]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text
