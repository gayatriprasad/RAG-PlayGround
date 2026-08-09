"""
Unit tests for raglab.improvement.loop.ImprovementLoop (Skill 46).

The orchestration logic (diagnose -> generate -> validate -> version) is
exercised for real. The heavy steps (embedding fine-tuning, index rebuild,
re-benchmarking) are injected as stubs/mocks so this test suite stays fast
and network-free, per this repo's testing conventions.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

import pytest

from raglab.config import Config, ExperimentCfg, GoldenCfg, ImprovementCfg
from raglab.improvement.loop import ImprovementLoop
from raglab.types import Chunk, Document, EvalResult


class _StubLLMClient:
    """Returns one canned factual Q&A regardless of prompt — good enough to
    exercise DatasetSynthesizer.generate()'s real JSON-parsing path."""

    def complete(self, messages, **kwargs):
        return json.dumps(
            [
                {
                    "question": "What does the runbook say about the deployment process?",
                    "answer": "the deployment process is fully automated",
                    "category": "factual",
                    "difficulty": "easy",
                }
            ]
        )


class _StubFineTuner:
    def prepare_training_data(self, questions, chunks, **kwargs):
        return [("anchor", "positive")] * len(questions)

    def train(self, base_model, examples, output_path, **kwargs):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path


class _FakeEmbedder:
    def __init__(self, model_name):
        pass

    def embed(self, texts):
        return [[1.0, 0.0] if "automated" in t.lower() else [0.0, 1.0] for t in texts]


def _make_cfg(tmp_path, min_recall_threshold=0.7) -> Config:
    cfg = Config(
        experiment=ExperimentCfg(name="test", corpus_glob=[], representations=[]),
        golden=GoldenCfg(path="./golden/questions.jsonl"),
    )
    cfg.improvement = ImprovementCfg(
        min_recall_threshold=min_recall_threshold,
        min_slice_size=3,
        reports_dir=str(tmp_path / "improvement"),
        models_dir=str(tmp_path / "models"),
    )
    return cfg


def _make_result(question_id, source_type, category, overall_score, recall_3):
    return EvalResult(
        question_id=question_id,
        question="q",
        ground_truth="gt",
        predicted_answer="pred",
        source_type=source_type,
        category=category,
        index_backend="chroma",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        overall_score=overall_score,
        metadata={"recall_at_k": {"3": recall_3}},
    )


def test_no_gap_returns_report_without_running_pipeline(tmp_path):
    cfg = _make_cfg(tmp_path, min_recall_threshold=0.0)  # nothing counts as a gap
    baseline = [_make_result(f"q{i}", "confluence", "single_doc", 0.9, 0.95) for i in range(5)]

    loop = ImprovementLoop(cfg, run_id="run1")
    report = loop.run(baseline, docs=[], chunks=[])

    assert report.gap_slices == []
    assert report.iteration == 1
    assert "nothing to improve" in report.recommendation.lower()
    report_path = Path(cfg.improvement.reports_dir) / "iter_1" / "report.json"
    assert report_path.exists()


def test_full_loop_with_gap_generates_and_versions_report(tmp_path, monkeypatch):
    import raglab.utils.embedder as embedder_module

    monkeypatch.setattr(embedder_module, "Embedder", _FakeEmbedder)

    cfg = _make_cfg(tmp_path, min_recall_threshold=0.7)
    baseline = [
        _make_result(f"q{i}", "confluence", "multi_doc", 0.4, 0.2) for i in range(5)
    ]
    docs = [
        Document(
            id="d0",
            content="this paragraph describes an automated deployment process in detail here today. " * 3,
            source_type="confluence",
        )
    ]
    chunks = [Chunk(id="c0", doc_id="d0", content=docs[0].content, source_type="confluence", chunk_index=0)]

    fake_results = [
        _make_result(f"q{i}", "confluence", "multi_doc", 0.9, 0.9) for i in range(5)
    ]

    loop = ImprovementLoop(
        cfg,
        run_id="run1",
        fine_tuner=_StubFineTuner(),
        llm_client=_StubLLMClient(),
        rebuild_index_fn=lambda model_path, out_dir: None,
        rerun_pipeline_fn=lambda questions, index_dir: fake_results,
    )
    report = loop.run(baseline, docs=docs, chunks=chunks)

    assert report.iteration == 1
    assert len(report.gap_slices) == 1
    assert report.n_synthetic_pairs_generated >= 1
    assert report.fine_tuned_model_path is not None
    assert report.significance is not None
    assert report.significance.delta == pytest.approx(-0.5)
    assert "iteration 1" in report.recommendation.lower() or "iter_1" in report.recommendation.lower()

    report_path = Path(cfg.improvement.reports_dir) / "iter_1" / "report.json"
    assert report_path.exists()
    questions_path = Path(cfg.improvement.reports_dir) / "iter_1" / "questions.jsonl"
    assert questions_path.exists()


def test_iteration_number_increments_and_never_overwrites(tmp_path):
    cfg = _make_cfg(tmp_path, min_recall_threshold=0.0)
    baseline = [_make_result(f"q{i}", "confluence", "single_doc", 0.9, 0.95) for i in range(5)]

    loop1 = ImprovementLoop(cfg, run_id="run1")
    report1 = loop1.run(baseline, docs=[], chunks=[])
    assert report1.iteration == 1

    loop2 = ImprovementLoop(cfg, run_id="run1")
    report2 = loop2.run(baseline, docs=[], chunks=[])
    assert report2.iteration == 2  # increments, does not overwrite iter_1

    assert (Path(cfg.improvement.reports_dir) / "iter_1" / "report.json").exists()
    assert (Path(cfg.improvement.reports_dir) / "iter_2" / "report.json").exists()


def test_no_pairs_pass_validation_reports_honestly(tmp_path, monkeypatch):
    import raglab.utils.embedder as embedder_module

    monkeypatch.setattr(embedder_module, "Embedder", _FakeEmbedder)

    cfg = _make_cfg(tmp_path, min_recall_threshold=0.7)
    baseline = [_make_result(f"q{i}", "confluence", "multi_doc", 0.4, 0.2) for i in range(5)]
    docs = [Document(id="d0", content="short", source_type="confluence")]  # too short to generate from

    loop = ImprovementLoop(cfg, run_id="run1", llm_client=_StubLLMClient())
    report = loop.run(baseline, docs=docs, chunks=[])

    assert report.n_pairs_passed_validation == 0
    assert report.fine_tuned_model_path is None
    assert "no synthetic pairs passed validation" in report.recommendation.lower()
