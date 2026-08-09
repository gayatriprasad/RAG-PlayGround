"""
BEIR Benchmark Loader — loads questions from BEIR subsets via HuggingFace.

Supports: msmarco, hotpotqa, nq, fiqa
Maps to Question schema for unified evaluation.
"""

from __future__ import annotations

import logging
from typing import List

from raglab.types import Question

logger = logging.getLogger(__name__)

# Category mapping per dataset
_CATEGORY_MAP = {
    "hotpotqa": "multi_hop",
    "msmarco": "factual",
    "nq": "factual",
    "fiqa": "analytical",
}


class BEIRLoader:
    """
    Load questions from BEIR benchmark subsets via HuggingFace datasets.
    
    Each subset is loaded, mapped to the Question schema, and capped
    at max_per_subset entries.
    """

    SUPPORTED_SUBSETS = ["msmarco", "hotpotqa", "nq", "fiqa"]

    def load(
        self,
        subsets: List[str],
        max_per_subset: int = 250,
    ) -> List[Question]:
        """
        Load questions from BEIR subsets.
        
        Args:
            subsets: List of BEIR subset names to load
            max_per_subset: Maximum questions per subset
            
        Returns:
            List of Question objects from all requested subsets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )
            return []

        all_questions: List[Question] = []

        for subset in subsets:
            if subset not in self.SUPPORTED_SUBSETS:
                logger.warning(f"Unsupported BEIR subset: {subset}, skipping")
                continue

            try:
                questions = self._load_subset(subset, max_per_subset)
                all_questions.extend(questions)
                logger.info(f"Loaded {len(questions)} questions from BEIR/{subset}")
            except Exception as e:
                logger.warning(f"Failed to load BEIR/{subset}: {e}")

        logger.info(f"Total BEIR questions loaded: {len(all_questions)}")
        return all_questions

    def _load_subset(self, subset: str, max_per_subset: int) -> List[Question]:
        """Load a single BEIR subset."""
        from datasets import load_dataset

        # BEIR datasets on HuggingFace follow this pattern
        dataset_name = f"BeIR/{subset}"

        try:
            ds = load_dataset(dataset_name, "queries", split="queries", trust_remote_code=True)
        except Exception:
            # Fallback: try loading as standard split
            try:
                ds = load_dataset(dataset_name, split="test", trust_remote_code=True)
            except Exception:
                # Try alternative naming
                ds = load_dataset(
                    "BeIR/beir", subset, split="queries", trust_remote_code=True
                )

        category = _CATEGORY_MAP.get(subset, "factual")
        questions: List[Question] = []

        for i, row in enumerate(ds):
            if i >= max_per_subset:
                break

            # Extract query text — field names vary by dataset
            query_text = row.get("text", row.get("query", row.get("question", "")))
            if not query_text:
                continue

            # Extract answer if available
            answer = row.get("answer", row.get("answers", ""))
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            q_id = row.get("_id", row.get("id", str(i)))

            questions.append(
                Question(
                    id=f"beir_{subset}_{q_id}",
                    text=query_text,
                    ground_truth=answer if answer else "[BEIR - passage retrieval]",
                    source_type=subset,
                    category=category,
                )
            )

        return questions
