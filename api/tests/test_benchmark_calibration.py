"""Tests for Skill 57 — /benchmark/calibration API endpoint."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag-lab" / "src"))

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import benchmark


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_OUT_DIR", tmp_path)
    app = FastAPI()
    app.include_router(benchmark.router)
    return TestClient(app)


def _write_results(out_dir: Path, experiment: str, rows):
    exp_dir = out_dir / experiment
    exp_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(exp_dir / f"{experiment}_results.csv", index=False)


def test_calibration_404_when_no_results(client):
    response = client.get("/benchmark/calibration", params={"experiment": "nope"})
    assert response.status_code == 404


def test_calibration_400_when_missing_columns(client):
    _write_results(benchmark._OUT_DIR, "exp1", [{"question_id": "q1", "question": "Q?"}])
    response = client.get("/benchmark/calibration", params={"experiment": "exp1"})
    assert response.status_code == 400


def test_calibration_returns_curve_and_diagram(client):
    rows = [
        {"question_id": f"q{i}", "question": f"Q{i}?", "overall_score": 0.9, "answer_correct": (i % 2 == 0)}
        for i in range(10)
    ]
    _write_results(benchmark._OUT_DIR, "exp2", rows)

    response = client.get("/benchmark/calibration", params={"experiment": "exp2", "n_bins": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["n_questions"] == 10
    assert len(body["curve"]["bins"]) == 11
    assert body["curve"]["ece"] > 0.1  # 0.9 confidence but only 50% correct -> overconfident
    assert "points" in body["diagram"]
