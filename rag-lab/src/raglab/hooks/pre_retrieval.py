"""
Pre-retrieval hooks: query cleaning and PII detection.
"""

import logging
import re
from typing import List

from raglab.config import Config
from raglab.hooks.base import PreRetrievalHook

logger = logging.getLogger(__name__)


class QueryCleanerHook(PreRetrievalHook):
    """
    HOOK 03: Cleans and normalizes query before retrieval.
    """

    def run(self, query: str, cfg: Config) -> str:
        # 1. Strip whitespace
        cleaned = query.strip()

        # 2. Collapse repeated whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 3. Append ? if question pattern without punctuation
        if cleaned and not cleaned.endswith(("?", ".", "!")):
            question_words = ["what", "how", "why", "when", "where", "who", "which", "is", "are", "do", "does", "can"]
            if any(cleaned.lower().startswith(w) for w in question_words):
                cleaned += "?"

        # 4. Truncate to 512 chars
        if len(cleaned) > 512:
            logger.warning(f"Query truncated from {len(cleaned)} to 512 chars")
            cleaned = cleaned[:512]

        return cleaned


class PIIDetectorHook(PreRetrievalHook):
    """
    HOOK 04: Detects PII patterns in queries (log only, no modification).
    """

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }

    def run(self, query: str, cfg: Config) -> str:
        detected = []
        for name, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, query):
                detected.append(name)

        if detected:
            logger.warning(f"PII pattern detected in query: {detected}")

        # Return query unchanged — this is a playground, user controls their data
        return query
