"""
Contract tests — Tier 2 (network-dependent, periodic only).

These tests check config values that reference live external services with
no offline-checkable registry. They skip cleanly when the required service
is unreachable — they are NOT part of the fast CI path.

Excluded from the main pytest run via:
  -k "not test_live_registries"
which mirrors the existing pattern for test_integration_e2e and
test_extended_combinations in ci.yml.
"""
import glob
import os
from typing import Generator

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers (shared with Tier 1 — kept local to avoid cross-test deps)
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, "..", ".."))


def _all_config_field(field_path: list[str]) -> list[tuple[str, str]]:
    """Collect (rel_path, value) for a nested field from all presets/configs."""
    patterns = [
        os.path.join(_REPO_ROOT, "presets", "*.yaml"),
        os.path.join(_REPO_ROOT, "experiments", "**", "config*.yaml"),
    ]
    results = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            node = data
            for key in field_path:
                node = (node or {}).get(key)
            if node and isinstance(node, str):
                results.append((os.path.relpath(path, _REPO_ROOT), node))
    return results


def _ollama_is_reachable() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/version", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Ollama model reachability
# ---------------------------------------------------------------------------

@pytest.mark.test_live_registries
def test_ollama_models_are_pullable():
    """Every llm.model for provider=ollama should be known to the daemon.

    Skips cleanly when no Ollama daemon is reachable — this is a periodic
    check, not a hard gate. Run locally or in a periodic workflow.
    """
    if not _ollama_is_reachable():
        pytest.skip("no Ollama daemon reachable at localhost:11434")

    import subprocess

    patterns = [
        os.path.join(_REPO_ROOT, "presets", "*.yaml"),
        os.path.join(_REPO_ROOT, "experiments", "**", "config*.yaml"),
    ]
    unknown = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            llm = data.get("llm") or {}
            provider = llm.get("llm_provider") or llm.get("provider", "ollama")
            model = llm.get("llm_model") or llm.get("model")
            if provider != "ollama" or not model:
                continue
            result = subprocess.run(
                ["ollama", "show", model],
                capture_output=True, timeout=10
            )
            if result.returncode != 0:
                unknown.append((os.path.relpath(path, _REPO_ROOT), model))

    assert not unknown, (
        "Ollama models not known to local daemon (run 'ollama pull <model>'):\n"
        + "\n".join(f"  {p}: {m!r}" for p, m in unknown)
    )


# ---------------------------------------------------------------------------
# HuggingFace embed model existence
# ---------------------------------------------------------------------------

@pytest.mark.test_live_registries
def test_embed_models_exist_on_huggingface():
    """Every embed.model (non-ollama prefix, non-openai prefix) should resolve
    on HuggingFace Hub.

    Skips cleanly when no network is available.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Probe connectivity with a lightweight call
        api.list_models(limit=1)
    except Exception:
        pytest.skip("HuggingFace Hub not reachable")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    missing = []
    seen: set[str] = set()

    for rel_path, model in _all_config_field(["embed", "model"]):
        # Skip special prefixes that aren't HF repo IDs
        if model in ("none",) or model.startswith(("ollama/", "openai/", "sie/")):
            continue
        # Skip sentence-transformers short names (no org/repo slash) — they're
        # library-internal aliases that sentence-transformers resolves, not direct
        # HF repo IDs checkable via hub API.
        if "/" not in model:
            continue
        if model in seen:
            continue
        seen.add(model)
        try:
            api.model_info(model, timeout=10)
        except RepositoryNotFoundError:
            missing.append((rel_path, model))
        except Exception:
            pass  # transient network error — don't fail the test

    assert not missing, (
        "Embedding models not found on HuggingFace Hub:\n"
        + "\n".join(f"  {p}: {m!r}" for p, m in missing)
    )
