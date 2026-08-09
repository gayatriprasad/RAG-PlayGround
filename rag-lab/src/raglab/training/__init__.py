"""
Embedding fine-tuning module.

Provides EmbeddingFineTuner for domain-adaptive training using
contrastive learning on (question, relevant_chunk) pairs.
"""

from raglab.training.embed_trainer import EmbeddingFineTuner

__all__ = ["EmbeddingFineTuner"]
