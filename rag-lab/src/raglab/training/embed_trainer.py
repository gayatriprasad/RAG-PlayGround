"""
Embedding Fine-Tuner — Skill 20

Fine-tunes a sentence-transformer embedding model on domain-specific
(question, relevant_chunk) pairs using Multiple Negatives Ranking Loss.
Evaluates recall@k before/after to quantify improvement.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from raglab.types import Chunk, Question

logger = logging.getLogger(__name__)


class EmbeddingFineTuner:
    """
    Fine-tunes sentence-transformer models for domain-specific retrieval.

    Workflow:
      1. prepare_training_data() — build (anchor, positive, negative) triplets
      2. train() — fine-tune using MultipleNegativesRankingLoss
      3. evaluate() — measure recall@k delta vs base model
    """

    def prepare_training_data(
        self,
        questions: List[Question],
        chunks: List[Chunk],
        negatives_per_positive: int = 1,
        seed: int = 42,
    ) -> List:
        """
        Build training examples from questions and their ground-truth chunks.

        For each question, finds the chunk containing the ground truth answer.
        Creates InputExample(texts=[question.text, chunk.content], label=1.0).
        Negative sampling: pair with random non-relevant chunk.

        Args:
            questions: List of Question objects with ground_truth field
            chunks: All available chunks in the corpus
            negatives_per_positive: Number of negative examples per positive
            seed: Random seed for reproducibility

        Returns:
            List of InputExample triplets for training
        """
        from sentence_transformers import InputExample

        rng = random.Random(seed)
        examples: List[InputExample] = []

        # Build lookup: find best-matching chunk for each question
        chunk_contents = {c.id: c.content.lower() for c in chunks}

        for question in questions:
            gt_lower = question.ground_truth.lower()

            # Find chunk containing the ground truth (best overlap)
            best_chunk = None
            best_overlap = 0
            for chunk in chunks:
                # Simple overlap: count shared words
                gt_words = set(gt_lower.split())
                chunk_words = set(chunk.content.lower().split())
                overlap = len(gt_words & chunk_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk = chunk

            if best_chunk is None or best_overlap == 0:
                continue

            # Positive example
            examples.append(
                InputExample(
                    texts=[question.text, best_chunk.content],
                    label=1.0,
                )
            )

            # Negative examples — random non-relevant chunks
            negative_pool = [c for c in chunks if c.id != best_chunk.id]
            if negative_pool:
                negatives = rng.sample(
                    negative_pool,
                    min(negatives_per_positive, len(negative_pool)),
                )
                for neg in negatives:
                    examples.append(
                        InputExample(
                            texts=[question.text, neg.content],
                            label=0.0,
                        )
                    )

        logger.info(
            f"Prepared {len(examples)} training examples from "
            f"{len(questions)} questions"
        )
        return examples

    def train(
        self,
        base_model: str,
        examples: List,
        output_path: str,
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        evaluation_steps: int = 500,
    ) -> str:
        """
        Fine-tune a sentence-transformer model with MultipleNegativesRankingLoss.

        Args:
            base_model: HuggingFace model name or path (e.g. "all-MiniLM-L6-v2")
            examples: List of InputExample from prepare_training_data()
            output_path: Directory to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            warmup_steps: Linear warmup steps
            evaluation_steps: Steps between evaluations

        Returns:
            Path to the saved fine-tuned model
        """
        from sentence_transformers import SentenceTransformer, losses
        from torch.utils.data import DataLoader

        logger.info(f"Loading base model: {base_model}")
        model = SentenceTransformer(base_model)

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Training for {epochs} epochs, batch_size={batch_size}, "
            f"{len(examples)} examples"
        )

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(output),
            show_progress_bar=True,
        )

        # Save training metadata
        meta = {
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_examples": len(examples),
            "output_path": str(output),
        }
        meta_path = output / "training_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"Fine-tuned model saved to: {output}")
        return str(output)

    def evaluate(
        self,
        model_path: str,
        base_model: str,
        questions: List[Question],
        chunks: List[Chunk],
        recall_at_k: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare retrieval recall@k between base and fine-tuned model.

        Rebuilds index with each model, runs retrieval for all questions,
        and computes recall@k against ground truth.

        Args:
            model_path: Path to fine-tuned model
            base_model: Original model name for comparison
            questions: Evaluation questions with ground_truth
            chunks: All corpus chunks
            recall_at_k: List of k values (default [1, 3, 5])

        Returns:
            {
                "base_model": {"recall@1": ..., "recall@3": ..., "recall@5": ...},
                "fine_tuned": {"recall@1": ..., "recall@3": ..., "recall@5": ...},
                "delta": {"recall@1": ..., "recall@3": ..., "recall@5": ...}
            }
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if recall_at_k is None:
            recall_at_k = [1, 3, 5]

        results = {}

        for label, model_name in [("base_model", base_model), ("fine_tuned", model_path)]:
            logger.info(f"Evaluating {label}: {model_name}")
            model = SentenceTransformer(model_name)

            # Embed all chunks
            chunk_texts = [c.content for c in chunks]
            chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)

            # Evaluate recall for each question
            recall_scores = {f"recall@{k}": [] for k in recall_at_k}

            for question in questions:
                query_emb = model.encode([question.text])[0]

                # Compute similarities
                similarities = np.dot(chunk_embeddings, query_emb) / (
                    np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
                    + 1e-10
                )

                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1]

                # Check if ground truth is in retrieved chunks
                gt_lower = question.ground_truth.lower()

                for k in recall_at_k:
                    top_k_chunks = [chunks[i] for i in top_indices[:k]]
                    hit = any(
                        gt_lower in chunk.content.lower()
                        for chunk in top_k_chunks
                    )
                    recall_scores[f"recall@{k}"].append(1.0 if hit else 0.0)

            # Average recall scores
            results[label] = {
                metric: sum(scores) / len(scores) if scores else 0.0
                for metric, scores in recall_scores.items()
            }

        # Compute delta
        results["delta"] = {
            metric: results["fine_tuned"][metric] - results["base_model"][metric]
            for metric in results["base_model"]
        }

        logger.info(f"Evaluation results: {json.dumps(results, indent=2)}")
        return results
