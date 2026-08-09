"""
Tests for Skill 50G — partial-run guard (check_run_completeness / PartialRunError).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.eval.scorer import check_run_completeness
from raglab.types import PartialRunError


def test_full_completion_returns_true():
    assert check_run_completeness(n_scored=10, n_total=10) is True


def test_above_threshold_returns_true():
    assert check_run_completeness(n_scored=9, n_total=10, min_fraction=0.9) is True


def test_below_threshold_returns_false():
    assert check_run_completeness(n_scored=5, n_total=10, min_fraction=0.9) is False


def test_zero_total_raises_partial_run_error():
    with pytest.raises(PartialRunError):
        check_run_completeness(n_scored=0, n_total=0)
