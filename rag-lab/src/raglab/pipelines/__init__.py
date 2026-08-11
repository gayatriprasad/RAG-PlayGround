"""
Pipeline implementations for RAG execution.
"""

from raglab.pipelines.naive_rag import NaiveRAGPipeline, build_llm_client
from raglab.pipelines.agentic_rag import AgenticRAGPipeline
from raglab.pipelines.reflection_rag import ReflectionRAGPipeline  # SKILL 16
from raglab.pipelines.rag_fusion import RAGFusionPipeline  # SKILL 17
from raglab.pipelines.adaptive_rag import AdaptiveRAGPipeline  # SKILL 17
from raglab.pipelines.cag import CacheAugmentedPipeline  # SKILL 52
from raglab.pipelines.rlm import RLMPipeline  # SKILL 54

__all__ = [
    "NaiveRAGPipeline",
    "AgenticRAGPipeline",
    "ReflectionRAGPipeline",
    "RAGFusionPipeline",
    "AdaptiveRAGPipeline",
    "CacheAugmentedPipeline",
    "RLMPipeline",
    "build_llm_client",
]
