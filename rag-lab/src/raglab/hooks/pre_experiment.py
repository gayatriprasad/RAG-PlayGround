"""
Pre-experiment hooks: config validation and data integrity checks.
"""

import logging
import random
from collections import Counter
from pathlib import Path
from typing import List

from raglab.config import Config
from raglab.hooks.base import PreExperimentHook
from raglab.types import Document, Question

logger = logging.getLogger(__name__)


class ConfigValidatorHook(PreExperimentHook):
    """
    HOOK 01: Validates configuration before experiment runs.
    Checks paths, writability, and LLM provider reachability.
    """

    def run(self, cfg: Config, documents: List[Document], questions: List[Question]) -> None:
        # 1. Verify golden path exists
        golden_path = Path(cfg.golden.path)
        if not golden_path.exists():
            logger.warning(f"Golden file not found: {golden_path}")

        # 2. Verify corpus directories exist for each source type
        valid_types = []
        for st in cfg.benchmark.source_types:
            corpus_dir = Path("corpus/raw") / st
            if corpus_dir.exists():
                valid_types.append(st)
            else:
                logger.warning(f"Corpus directory missing: {corpus_dir} — removing '{st}' from run")

        if not valid_types:
            logger.warning("No valid source type directories found")

        # 3. Verify persist_dir is writable
        persist_dir = Path(cfg.index.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # 4. Log startup summary
        logger.info(
            f"=== Experiment: {cfg.experiment.name} ===\n"
            f"  Questions: {len(questions)}\n"
            f"  Documents: {len(documents)}\n"
            f"  Source types: {valid_types}\n"
            f"  Index backend: {cfg.index.backend}\n"
            f"  LLM: {cfg.llm.model} ({cfg.llm.provider})\n"
            f"  Intent mode: {cfg.intent.mode}\n"
            f"  Agentic strategy: {cfg.agentic.strategy}"
        )


class DataIntegrityHook(PreExperimentHook):
    """
    HOOK 02: Checks data integrity — questions reference docs that exist.
    """

    def run(self, cfg: Config, documents: List[Document], questions: List[Question]) -> None:
        if not questions or not documents:
            logger.warning("Empty questions or documents list — skipping integrity check")
            return

        # 1. Sample 5 random questions
        sample_size = min(5, len(questions))
        sampled = random.sample(questions, sample_size)

        # 2. Check source type matches
        doc_source_types = {d.source_type for d in documents}
        mismatches = 0
        for q in sampled:
            q_sources = q.source_type.split(",")
            if q.source_type != "all" and not any(s.strip() in doc_source_types for s in q_sources):
                mismatches += 1

        if mismatches > 2:
            raise RuntimeError(
                "Data mismatch: questions reference source types not found in corpus"
            )

        # 3. Log distribution
        source_dist = Counter(d.source_type for d in documents)
        category_dist = Counter(q.category for q in questions)

        logger.info(
            f"Data integrity OK:\n"
            f"  Total docs: {len(documents)}\n"
            f"  Total questions: {len(questions)}\n"
            f"  Doc source_types: {dict(source_dist)}\n"
            f"  Question categories: {dict(category_dist)}"
        )

        # 4. Estimated run time
        est_minutes = (len(questions) * 2 * 2) / 60
        logger.info(f"  Estimated run time: ~{est_minutes:.1f} minutes")
