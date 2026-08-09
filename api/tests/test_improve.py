"""Tests for the Skill 46 improvement-loop API router (/improve/*)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag-lab" / "src"))

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import improve
from raglab.config import Config, ExperimentCfg, GoldenCfg, ImprovementCfg


def _make_cfg(**improvement_overrides):
    return Config(
        experiment=ExperimentCfg(name="exp1", corpus_glob=["*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/questions.jsonl"),
        improvement=ImprovementCfg(min_recall_threshold=0.7, min_slice_size=3, **improvement_overrides),
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(improve, "_OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(improve, "find_experiment_config", lambda experiment=None: tmp_path / "config.yaml")

    app = FastAPI()
    app.include_router(improve.router)
    return TestClient(app)


def _write_results(out_dir: Path, experiment: str, rows):
    exp_dir = out_dir / experiment
    exp_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(exp_dir / f"{experiment}_results.csv", index=False)


def _rows_with_recall(source_type, category, overall_score, recall_3, n=3):
    return [
        {
            "question_id": f"{source_type}_{category}_{i}",
            "question": "q",
            "ground_truth": "gt",
            "predicted_answer": "pred",
            "source_type": source_type,
            "category": category,
            "index_backend": "chroma",
            "pipeline": "naive",
            "intent_label": "simple",
            "overall_score": overall_score,
            "recall_at_3": recall_3,
        }
        for i in range(n)
    ]


def test_improve_status_404_without_results(client, monkeypatch):
    monkeypatch.setattr(improve, "load_config", lambda path: _make_cfg())
    response = client.get("/improve/status", params={"experiment": "exp1"})
    assert response.status_code == 404


def test_improve_status_reports_gap(client, monkeypatch):
    monkeypatch.setattr(improve, "load_config", lambda path: _make_cfg())
    rows = _rows_with_recall("confluence", "multi_doc", 0.5, 0.3)
    _write_results(improve._OUT_DIR, "exp1", rows)

    response = client.get("/improve/status", params={"experiment": "exp1"})
    assert response.status_code == 200
    body = response.json()
    assert body["should_run"] is True
    assert len(body["gap_slices"]) == 1
    assert body["gap_slices"][0]["source_type"] == "confluence"


def test_improve_heatmap_returns_all_slices(client, monkeypatch):
    monkeypatch.setattr(improve, "load_config", lambda path: _make_cfg())
    rows = _rows_with_recall("confluence", "multi_doc", 0.5, 0.3) + _rows_with_recall(
        "github", "single_doc", 0.9, 0.95
    )
    _write_results(improve._OUT_DIR, "exp1", rows)

    response = client.get("/improve/heatmap", params={"experiment": "exp1"})
    assert response.status_code == 200
    body = response.json()
    assert body["min_recall_threshold"] == 0.7
    slices = {(s["source_type"], s["category"]): s for s in body["slices"]}
    assert slices[("confluence", "multi_doc")]["gap"] is True
    assert slices[("github", "single_doc")]["gap"] is False


def test_improve_reports_empty_when_no_reports_dir(client, monkeypatch):
    monkeypatch.setattr(improve, "load_config", lambda path: _make_cfg())
    response = client.get("/improve/reports", params={"experiment": "exp1"})
    assert response.status_code == 200
    assert response.json()["reports"] == []


def test_improve_report_by_iteration_404_when_missing(client, monkeypatch):
    monkeypatch.setattr(improve, "load_config", lambda path: _make_cfg())
    response = client.get("/improve/reports/1", params={"experiment": "exp1"})
    assert response.status_code == 404
