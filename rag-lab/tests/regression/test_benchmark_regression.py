"""
Benchmark regression test — Skill 49(C).

Compares the most recent real benchmark run (if one exists on disk) against
`baseline_scores.json` and asserts the score hasn't regressed by more than
`tolerance`. If the baseline hasn't been initialized yet (n_questions == 0,
a real "not yet run" state — never fabricated), this test warns and skips
rather than failing or inventing numbers, matching Skill 50(I)'s CI pattern.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

_BASELINE_PATH = Path(__file__).resolve().parent / "baseline_scores.json"
_RAG_LAB_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"


def _load_baseline() -> dict:
    return json.loads(_BASELINE_PATH.read_text())


def test_baseline_file_exists_and_is_well_formed():
    baseline = _load_baseline()
    for key in ("experiment", "n_questions", "overall_score_mean", "tolerance"):
        assert key in baseline


def test_benchmark_score_has_not_regressed_vs_baseline():
    baseline = _load_baseline()

    if baseline["n_questions"] == 0:
        pytest.skip(
            "Baseline not yet initialized (n_questions=0 — 'not yet run', not a fabricated "
            "zero score). Run `make eval` then `python tests/regression/update_baseline.py` "
            "to initialize it honestly."
        )

    result_csv = _OUT_DIR / baseline["experiment"] / f"{baseline['experiment']}_results.csv"
    if not result_csv.exists():
        pytest.skip(f"No current results at {result_csv} — run `make eval` first.")

    import pandas as pd

    df = pd.read_csv(result_csv)
    assert len(df) > 0, "Results CSV exists but is empty — treat as a failed run, not a valid comparison."

    current_mean = float(df["overall_score"].mean())
    baseline_mean = baseline["overall_score_mean"]
    tolerance = baseline["tolerance"]

    assert current_mean >= baseline_mean - tolerance, (
        f"Benchmark regression detected: current overall_score mean {current_mean:.4f} "
        f"is more than {tolerance} below baseline {baseline_mean:.4f}."
    )
