"""
Unit tests for DatasetSynthesizer.validate_generated (Skill 44 — synthetic QA validation).

The embedding-based answerability check is monkeypatched to a deterministic
fake embedder so these tests don't require downloading a real
sentence-transformers model or network access.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

from raglab.datasets.synthesizer import DatasetSynthesizer
from raglab.types import Document, Question


class _FakeEmbedder:
    """Deterministic fake: embeds each text as a one-hot vector keyed by
    whether it contains the word 'supported'. This lets us control which
    paragraphs "match" a given ground_truth in a test without any real model."""

    def __init__(self, model_name):
        pass

    def embed(self, texts):
        return [[1.0, 0.0] if "supported" in t.lower() else [0.0, 1.0] for t in texts]


def _patch_embedder(monkeypatch):
    import raglab.utils.embedder as embedder_module

    monkeypatch.setattr(embedder_module, "Embedder", _FakeEmbedder)


def test_rejects_degenerate_short_question():
    synth = DatasetSynthesizer()
    questions = [Question(id="q0", text="What?", ground_truth="answer", source_type="confluence", category="factual")]
    kept, rejected, report = synth.validate_generated(questions, docs=[])

    assert kept == []
    assert len(rejected) == 1
    assert "degenerate" in rejected[0]["reason"]


def test_rejects_answer_leaking_question():
    synth = DatasetSynthesizer()
    questions = [
        Question(
            id="q0",
            text="Is the answer really the secret code alpha七?",
            ground_truth="the secret code alpha七",
            source_type="confluence",
            category="factual",
        )
    ]
    kept, rejected, report = synth.validate_generated(questions, docs=[])
    assert kept == []
    assert "leaks" in rejected[0]["reason"]


def test_adversarial_must_have_not_found_ground_truth():
    synth = DatasetSynthesizer()
    questions = [
        Question(
            id="q0",
            text="What is the secret launch code mentioned in the doc?",
            ground_truth="the actual code",  # should be NOT FOUND for adversarial
            source_type="confluence",
            category="adversarial",
        )
    ]
    kept, rejected, report = synth.validate_generated(questions, docs=[])
    assert kept == []
    assert rejected[0]["reason"].startswith("category_mismatch")


def test_adversarial_with_not_found_is_kept_without_answerability_check():
    synth = DatasetSynthesizer()
    questions = [
        Question(
            id="q0",
            text="What is the CEO's home address per this document?",
            ground_truth="NOT FOUND",
            source_type="confluence",
            category="adversarial",
        )
    ]
    # No docs at all -> would fail answerability if checked, but adversarial skips it.
    kept, rejected, report = synth.validate_generated(questions, docs=[])
    assert len(kept) == 1
    assert rejected == []


def test_answerability_rejects_when_no_corpus_support(monkeypatch):
    _patch_embedder(monkeypatch)
    synth = DatasetSynthesizer()
    docs = [Document(id="d0", content="unrelated paragraph one\n\nunrelated paragraph two", source_type="confluence")]
    questions = [
        Question(
            id="q0",
            text="What does the runbook say about the process?",
            ground_truth="a fact that is not supported anywhere",
            source_type="confluence",
            category="factual",
        )
    ]
    kept, rejected, report = synth.validate_generated(questions, docs=docs)
    assert kept == []
    assert rejected[0]["reason"].startswith("unanswerable")


def test_answerability_keeps_when_corpus_supports(monkeypatch):
    _patch_embedder(monkeypatch)
    synth = DatasetSynthesizer()
    docs = [Document(id="d0", content="this paragraph is supported by evidence", source_type="confluence")]
    questions = [
        Question(
            id="q0",
            text="What does the runbook document say about the deployment steps?",
            ground_truth="this is a supported fact",
            source_type="confluence",
            category="factual",
            difficulty="easy",
        )
    ]
    kept, rejected, report = synth.validate_generated(questions, docs=docs)
    assert len(kept) == 1
    assert report["difficulty_spread"] == {"easy": 1}


def test_report_summarizes_counts_and_reasons():
    synth = DatasetSynthesizer()
    questions = [
        Question(id="q0", text="Bad?", ground_truth="x", source_type="confluence", category="factual"),
        Question(
            id="q1",
            text="What is the launch code in this doc?",
            ground_truth="secret",
            source_type="confluence",
            category="adversarial",
        ),
    ]
    kept, rejected, report = synth.validate_generated(questions, docs=[])
    assert report["n_total"] == 2
    assert report["n_kept"] == 0
    assert report["n_rejected"] == 2
    assert "degenerate" in report["rejected_by_reason"]
    assert "category_mismatch" in report["rejected_by_reason"]
