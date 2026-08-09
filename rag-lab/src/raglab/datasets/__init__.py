"""
Dataset loading and expansion module.

Supports three layers:
  - bench: EnterpriseRAG-Bench golden set (500 immutable questions)
  - synthetic: LLM-generated questions from corpus (Skill 19)
  - beir: BEIR benchmark subsets from HuggingFace
"""

from raglab.datasets.synthesizer import DatasetSynthesizer
from raglab.datasets.beir_loader import BEIRLoader
from raglab.datasets.loader import load_all, deduplicate

__all__ = [
    "DatasetSynthesizer",
    "BEIRLoader",
    "load_all",
    "deduplicate",
]
