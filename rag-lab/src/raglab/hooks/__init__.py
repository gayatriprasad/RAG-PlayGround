"""
Pipeline lifecycle hooks.
All hooks extend the pipeline without touching core orchestration logic.
"""

from raglab.hooks.base import (
    PreExperimentHook,
    PostExperimentHook,
    PreRetrievalHook,
    PostRetrievalHook,
    PreGenerationHook,
    PostGenerationHook,
)
from raglab.hooks.pre_experiment import ConfigValidatorHook, DataIntegrityHook
from raglab.hooks.pre_retrieval import QueryCleanerHook, PIIDetectorHook
from raglab.hooks.post_retrieval import ScoreLoggerHook, DiversityFilterHook
from raglab.hooks.pre_generation import GenerationLoggerHook
from raglab.hooks.post_generation import CostRecordingHook
from raglab.hooks.post_experiment import ResultArchiverHook, MarkdownReporterHook


class HookRegistry:
    """Registry to collect and run hooks in order."""

    def __init__(self):
        self.pre_experiment: list[PreExperimentHook] = []
        self.post_experiment: list[PostExperimentHook] = []
        self.pre_retrieval: list[PreRetrievalHook] = []
        self.post_retrieval: list[PostRetrievalHook] = []
        self.pre_generation: list[PreGenerationHook] = []
        self.post_generation: list[PostGenerationHook] = []

    def run_pre_experiment(self, cfg, documents, questions):
        for hook in self.pre_experiment:
            hook.run(cfg, documents, questions)

    def run_post_experiment(self, cfg, results):
        for hook in self.post_experiment:
            hook.run(cfg, results)

    def run_pre_retrieval(self, query, cfg):
        for hook in self.pre_retrieval:
            query = hook.run(query, cfg)
        return query

    def run_post_retrieval(self, query, chunks, cfg):
        for hook in self.post_retrieval:
            chunks = hook.run(query, chunks, cfg)
        return chunks

    def run_pre_generation(self, question, cfg):
        for hook in self.pre_generation:
            hook.run(question, cfg)

    def run_post_generation(self, question, result, latency_ms, cfg):
        for hook in self.post_generation:
            hook.run(question, result, latency_ms, cfg)


def get_default_registry() -> HookRegistry:
    """Create a HookRegistry with the default hooks enabled."""
    registry = HookRegistry()
    registry.pre_experiment.append(ConfigValidatorHook())
    registry.pre_experiment.append(DataIntegrityHook())
    registry.pre_retrieval.append(QueryCleanerHook())
    registry.pre_retrieval.append(PIIDetectorHook())
    registry.post_retrieval.append(ScoreLoggerHook())
    registry.post_retrieval.append(DiversityFilterHook())
    registry.pre_generation.append(GenerationLoggerHook())
    registry.post_generation.append(CostRecordingHook())
    registry.post_experiment.append(ResultArchiverHook())
    registry.post_experiment.append(MarkdownReporterHook())
    return registry


__all__ = [
    "PreExperimentHook",
    "PostExperimentHook",
    "PreRetrievalHook",
    "PostRetrievalHook",
    "PreGenerationHook",
    "PostGenerationHook",
    "ConfigValidatorHook",
    "DataIntegrityHook",
    "QueryCleanerHook",
    "PIIDetectorHook",
    "ScoreLoggerHook",
    "DiversityFilterHook",
    "GenerationLoggerHook",
    "CostRecordingHook",
    "ResultArchiverHook",
    "MarkdownReporterHook",
    "HookRegistry",
    "get_default_registry",
]
