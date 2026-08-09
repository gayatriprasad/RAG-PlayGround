"""
Dataset loader — unified entry point for all question sources.

Supports three layers configured via DatasetCfg:
  - bench: EnterpriseRAG-Bench golden set (500 immutable)
  - synthetic: LLM-generated from corpus (1000 target)
  - beir: BEIR benchmark subsets (500 target)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from raglab.types import Document, Question

logger = logging.getLogger(__name__)


def load_bench(bench_path: str, cfg=None) -> List[Question]:
    """
    Load questions from the EnterpriseRAG-Bench golden set.
    
    Args:
        bench_path: Path to questions.jsonl
        cfg: Optional config for filtering
        
    Returns:
        List of Question objects
    """
    path = Path(bench_path)
    if not path.exists():
        logger.warning(f"Bench file not found: {path}")
        return []

    questions: List[Question] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions.append(
                    Question(
                        id=data.get("id", ""),
                        text=data.get("question", data.get("text", "")),
                        ground_truth=data.get("answer", data.get("ground_truth", "")),
                        source_type=data.get("source_type", "unknown"),
                        category=data.get("category", "unknown"),
                    )
                )
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Skipping malformed line: {e}")

    logger.info(f"Loaded {len(questions)} questions from bench ({path})")
    return questions


def deduplicate(questions: List[Question], threshold: float = 0.9) -> List[Question]:
    """
    Deduplicate questions using text similarity.
    
    Uses a simple approach: normalize text and check for high overlap.
    For exact duplicates, uses set-based dedup on question text.
    For near-duplicates (threshold < 1.0), uses Jaccard similarity on word sets.
    
    Args:
        questions: List of questions to deduplicate
        threshold: Similarity threshold for near-dedup (0.9 = very similar)
        
    Returns:
        Deduplicated list of questions
    """
    if not questions:
        return []

    seen_texts: set = set()
    unique: List[Question] = []

    for q in questions:
        normalized = q.text.lower().strip()

        # Exact dedup
        if normalized in seen_texts:
            continue

        # Near-dedup via Jaccard similarity on word sets
        q_words = set(normalized.split())
        is_dup = False

        if threshold < 1.0:
            for seen in seen_texts:
                seen_words = set(seen.split())
                if not q_words or not seen_words:
                    continue
                intersection = q_words & seen_words
                union = q_words | seen_words
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard >= threshold:
                    is_dup = True
                    break

        if not is_dup:
            seen_texts.add(normalized)
            unique.append(q)

    removed = len(questions) - len(unique)
    if removed > 0:
        logger.info(f"Deduplicated: removed {removed} duplicate questions")

    return unique


def load_all(
    cfg,
    docs: Optional[List[Document]] = None,
    llm_client=None,
) -> List[Question]:
    """
    Load questions from all configured layers.
    
    Layers (from DatasetCfg.layers):
      - "bench": Load from bench_path (EnterpriseRAG-Bench, 500 questions)
      - "synthetic": Generate from corpus docs (1000 target)
      - "beir": Load from BEIR subsets (500 target)
    
    Args:
        cfg: Config or DatasetCfg with layer settings
        docs: Documents for synthetic generation (required if "synthetic" in layers)
        llm_client: Optional LLM client for synthetic generation
        
    Returns:
        Combined, deduplicated list of questions capped at max_questions
    """
    # Handle both full Config and DatasetCfg
    dataset_cfg = getattr(cfg, "dataset", cfg)
    layers = getattr(dataset_cfg, "layers", ["bench"])
    max_questions = getattr(dataset_cfg, "max_questions", 500)

    questions: List[Question] = []

    # Layer 1: Bench (EnterpriseRAG-Bench golden set)
    if "bench" in layers:
        bench_path = getattr(dataset_cfg, "bench_path", "./golden/questions.jsonl")
        bench_questions = load_bench(bench_path)
        questions.extend(bench_questions)
        logger.info(f"Layer 'bench': {len(bench_questions)} questions")

    # Layer 2: Synthetic (LLM-generated from corpus)
    if "synthetic" in layers:
        synthetic_path = getattr(
            dataset_cfg, "synthetic_path", "./golden/questions_synthetic.jsonl"
        )
        synth_file = Path(synthetic_path)

        if synth_file.exists():
            # Load previously generated synthetic questions
            synth_questions = load_bench(synthetic_path)
            questions.extend(synth_questions)
            logger.info(
                f"Layer 'synthetic': loaded {len(synth_questions)} from cache ({synth_file})"
            )
        elif docs:
            # Generate new synthetic questions
            from raglab.datasets.synthesizer import DatasetSynthesizer

            synthesizer = DatasetSynthesizer()
            synth_questions = synthesizer.generate(
                docs=docs, cfg=cfg, llm_client=llm_client
            )
            # Save for next time
            synthesizer.save(synth_questions, synthetic_path)
            questions.extend(synth_questions)
            logger.info(f"Layer 'synthetic': generated {len(synth_questions)} questions")
        else:
            logger.warning(
                "Layer 'synthetic' requested but no docs provided and no cache file found"
            )

    # Layer 3: BEIR subsets
    if "beir" in layers:
        beir_path = getattr(dataset_cfg, "beir_path", "./golden/questions_beir.jsonl")
        beir_file = Path(beir_path)

        if beir_file.exists():
            beir_questions = load_bench(beir_path)
            questions.extend(beir_questions)
            logger.info(
                f"Layer 'beir': loaded {len(beir_questions)} from cache ({beir_file})"
            )
        else:
            from raglab.datasets.beir_loader import BEIRLoader

            beir_subsets = getattr(dataset_cfg, "beir_subsets", ["msmarco", "hotpotqa"])
            beir_questions = BEIRLoader().load(beir_subsets)

            # Cache for next time
            if beir_questions:
                Path(beir_path).parent.mkdir(parents=True, exist_ok=True)
                with open(beir_path, "w", encoding="utf-8") as f:
                    for q in beir_questions:
                        f.write(q.model_dump_json() + "\n")

            questions.extend(beir_questions)
            logger.info(f"Layer 'beir': loaded {len(beir_questions)} questions")

    # Deduplicate across all layers
    questions = deduplicate(questions, threshold=0.9)

    # Cap at max_questions
    if len(questions) > max_questions:
        logger.info(f"Capping from {len(questions)} to {max_questions} questions")
        questions = questions[:max_questions]

    logger.info(
        f"Dataset loaded: {len(questions)} total questions "
        f"(layers={layers}, max={max_questions})"
    )
    return questions
