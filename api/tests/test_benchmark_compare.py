"""Tests for Skill 43 — /benchmark/compare API endpoint.

Wires raglab.eval.significance.compare_from_records() into the API so the
UI never has to report a raw percentage delta without a CI/p-value/verdict.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag-lab" / "src"))

import numpy as np
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


def _rows(seed: int, low: float, high: float, n: int = 20):
    rng = np.random.default_rng(seed)
    return [
        {
            "question_id": f"q{i}",
            "overall_score": float(v),
            "answer_correct": bool(rng.integers(0, 2)),
        }
        for i, v in enumerate(rng.uniform(low, high, n))
    ]


def test_compare_404_when_baseline_missing(client):
    response = client.get(
        "/benchmark/compare", params={"baseline": "nope", "candidate": "also_nope"}
    )
    assert response.status_code == 404


def test_compare_400_on_unsupported_metric(client):
    _write_results(benchmark._OUT_DIR, "a", _rows(1, 0.3, 0.6))
    _write_results(benchmark._OUT_DIR, "b", _rows(2, 0.6, 0.9))
    response = client.get(
        "/benchmark/compare",
        params={"baseline": "a", "candidate": "b", "metric": "not_a_real_metric"},
    )
    assert response.status_code == 400


def test_compare_returns_significance_result(client):
    _write_results(benchmark._OUT_DIR, "baseline_exp", _rows(1, 0.3, 0.6))
    _write_results(benchmark._OUT_DIR, "candidate_exp", _rows(2, 0.6, 0.9))

    response = client.get(
        "/benchmark/compare",
        params={"baseline": "baseline_exp", "candidate": "candidate_exp", "metric": "overall_score"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["config_a"] == "baseline_exp"
    assert body["config_b"] == "candidate_exp"
    assert body["n_questions"] == 20
    assert "ci_lower" in body and "ci_upper" in body
    assert "p_value" in body
    assert "verdict" in body and body["verdict"] != ""
    # candidate has strictly higher scores by construction -> B should win
    assert "B" in body["verdict"] or body["delta"] < 0


def test_compare_uses_mcnemar_for_binary_metric(client):
    _write_results(benchmark._OUT_DIR, "a2", _rows(3, 0.3, 0.6))
    _write_results(benchmark._OUT_DIR, "b2", _rows(4, 0.6, 0.9))

    response = client.get(
        "/benchmark/compare",
        params={"baseline": "a2", "candidate": "b2", "metric": "answer_correct"},
    )
    assert response.status_code == 200
    assert response.json()["test_used"] == "mcnemar"
