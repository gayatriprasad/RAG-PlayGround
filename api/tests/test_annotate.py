"""Tests for Skill 56 — HITL grading API router (/annotate)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag-lab" / "src"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import annotate


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(annotate, "_CALIBRATION_SAMPLE_PATH", tmp_path / "judge_calibration_sample.jsonl")
    monkeypatch.setattr(annotate, "_UNCERTAINTY_ANNOTATIONS_PATH", tmp_path / "uncertainty_annotations.jsonl")
    monkeypatch.setattr(annotate, "_OUT_DIR", tmp_path / "out")

    app = FastAPI()
    app.include_router(annotate.router)
    return TestClient(app)


def _write_calibration_sample(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_calibration_queue_returns_404_when_no_sample_exists(client):
    response = client.get("/annotate/queue", params={"mode": "calibration"})
    assert response.status_code == 404


def test_calibration_queue_returns_first_unlabeled_row(client, monkeypatch):
    _write_calibration_sample(
        annotate._CALIBRATION_SAMPLE_PATH,
        [
            {"question_id": "q1", "question": "Q1?", "human_correct": True, "human_completeness": 1.0},
            {"question_id": "q2", "question": "Q2?", "human_correct": None, "human_completeness": None},
        ],
    )
    response = client.get("/annotate/queue", params={"mode": "calibration"})
    assert response.status_code == 200
    body = response.json()
    assert body["item"]["question_id"] == "q2"
    assert body["progress"] == {"labeled": 1, "total": 2}


def test_calibration_submit_persists_label(client):
    _write_calibration_sample(
        annotate._CALIBRATION_SAMPLE_PATH,
        [{"question_id": "q1", "question": "Q1?", "human_correct": None, "human_completeness": None}],
    )
    response = client.post(
        "/annotate/submit",
        json={"mode": "calibration", "question_id": "q1", "human_correct": True, "human_completeness": 0.8},
    )
    assert response.status_code == 200
    assert response.json()["progress"] == {"labeled": 1, "total": 1}

    rows = [json.loads(line) for line in open(annotate._CALIBRATION_SAMPLE_PATH)]
    assert rows[0]["human_correct"] is True
    assert rows[0]["human_completeness"] == 0.8


def test_calibration_submit_unknown_question_id_returns_404(client):
    _write_calibration_sample(annotate._CALIBRATION_SAMPLE_PATH, [{"question_id": "q1", "human_correct": None}])
    response = client.post(
        "/annotate/submit",
        json={"mode": "calibration", "question_id": "unknown", "human_correct": True, "human_completeness": 0.5},
    )
    assert response.status_code == 404


def test_submit_rejects_out_of_range_completeness(client):
    response = client.post(
        "/annotate/submit",
        json={"mode": "calibration", "question_id": "q1", "human_correct": True, "human_completeness": 1.5},
    )
    assert response.status_code == 400


def test_uncertainty_queue_requires_experiment_param(client):
    response = client.get("/annotate/queue", params={"mode": "uncertainty"})
    assert response.status_code == 400


def test_uncertainty_queue_returns_404_without_results_csv(client):
    response = client.get("/annotate/queue", params={"mode": "uncertainty", "experiment": "nope"})
    assert response.status_code == 404


def test_uncertainty_queue_and_submit_end_to_end(client, monkeypatch):
    import pandas as pd

    exp_dir = annotate._OUT_DIR / "myexp"
    exp_dir.mkdir(parents=True)
    df = pd.DataFrame(
        [
            {"question_id": "q1", "question": "Q1?", "ground_truth": "gt1", "predicted_answer": "p1", "overall_score": 0.5},
            {"question_id": "q2", "question": "Q2?", "ground_truth": "gt2", "predicted_answer": "p2", "overall_score": 0.95},
        ]
    )
    df.to_csv(exp_dir / "myexp_results.csv", index=False)

    response = client.get("/annotate/queue", params={"mode": "uncertainty", "experiment": "myexp"})
    assert response.status_code == 200
    body = response.json()
    assert body["item"]["question_id"] == "q1"  # 0.5 is in the ambiguous band; 0.95 is not
    assert body["progress"] == {"labeled": 0, "total": 1}

    submit = client.post(
        "/annotate/submit",
        json={"mode": "uncertainty", "question_id": "q1", "human_correct": False, "human_completeness": 0.4, "experiment": "myexp"},
    )
    assert submit.status_code == 200

    response2 = client.get("/annotate/queue", params={"mode": "uncertainty", "experiment": "myexp"})
    assert response2.json()["item"] is None
    assert response2.json()["progress"] == {"labeled": 1, "total": 1}


def test_unknown_mode_rejected(client):
    response = client.get("/annotate/queue", params={"mode": "bogus"})
    assert response.status_code == 400
