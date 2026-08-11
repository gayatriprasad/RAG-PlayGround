"""
Contract tests — Tier 1 (offline, every-commit).

Checks that config values that reference external registries are actually
valid keys in those registries. These tests are fast, need no network, and
run in the main CI gate alongside everything else.

Supersedes the single inline check in test_skill62_matrix_fixes.py — that
test still exists as a regression guard; this suite is the forward-looking
canonical location for this class of test.
"""
import glob
import os

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", ".."))  # rag-lab/


def _load_all_configs():
    """Yield (rel_path, parsed_yaml) for every preset and experiment config."""
    patterns = [
        os.path.join(_REPO_ROOT, "presets", "*.yaml"),
        os.path.join(_REPO_ROOT, "experiments", "**", "config*.yaml"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            with open(path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                yield os.path.relpath(path, _REPO_ROOT), data


# ---------------------------------------------------------------------------
# FlashRank reranker_model contract
# ---------------------------------------------------------------------------

def test_reranker_model_values_are_valid_flashrank_keys():
    """Every reranker_model in every preset/experiment config must be a real
    FlashRank key — catches exactly the class of bug from Skill 62 (a value
    that was neither a valid FlashRank key nor a correctly-spelled HF path).

    Error message names the file and the bad value explicitly.
    """
    from flashrank.Config import model_file_map  # verified import path

    valid_keys = set(model_file_map.keys())
    invalid = []

    for rel_path, data in _load_all_configs():
        model = (data.get("retrieve") or {}).get("reranker_model")
        if model and model not in valid_keys:
            invalid.append((rel_path, model))

    assert not invalid, (
        "Invalid reranker_model values (not in FlashRank model_file_map):\n"
        + "\n".join(f"  {path}: {model!r}  (valid: {sorted(valid_keys)})" for path, model in invalid)
    )
