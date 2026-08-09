"""
Pre-generation hooks: run immediately before an LLM generation call.
"""

import logging

from raglab.config import Config
from raglab.hooks.base import PreGenerationHook
from raglab.types import Question

logger = logging.getLogger(__name__)


class GenerationLoggerHook(PreGenerationHook):
    """HOOK 09: Logs which model/pipeline is about to generate an answer."""

    def run(self, question: Question, cfg: Config) -> None:
        logger.debug(f"Q={question.id}: generating with model='{cfg.llm.model}'")
