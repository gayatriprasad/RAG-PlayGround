"""
Skill 62 regression tests:
  1. reranker_model values across all configs/presets are valid FlashRank keys.
  2. Rebuild-in-place on ChromaIndex returns a fresh collection handle (not stale).
"""
import glob
import os
import tempfile

import pytest
import yaml


# Valid keys copied from flashrank's model_file_map (avoids hard import dependency in CI)
_VALID_FLASHRANK_KEYS = {
    "ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2",
    "ms-marco-MultiBERT-L-12",
    "rank-T5-flan",
    "ce-esci-MiniLM-L12-v2",
    "rank_zephyr_7b_v1_full",
    "miniReranker_arabic_v1",
}

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_reranker_models():
    """Collect every reranker_model value from presets and experiment configs."""
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
            model = (data.get("retrieve") or {}).get("reranker_model")
            if model:
                results.append((os.path.relpath(path, _REPO_ROOT), model))
    return results


def test_reranker_model_defaults_are_valid_flashrank_keys():
    """All reranker_model overrides must be keys in FlashRank's model_file_map."""
    entries = _load_reranker_models()
    assert entries, "No reranker_model entries found — check glob paths"
    invalid = [
        (path, model)
        for path, model in entries
        if model not in _VALID_FLASHRANK_KEYS
    ]
    assert not invalid, (
        "Invalid reranker_model values (not in FlashRank model_file_map):\n"
        + "\n".join(f"  {path}: {model!r}" for path, model in invalid)
    )


def test_rebuild_in_place_returns_fresh_collection_handle():
    """build() after a stale-collection delete must return the new chunk count, not error."""
    from raglab.config import Config, EmbedCfg, IndexCfg, ExperimentCfg, GoldenCfg
    from raglab.index.chroma_index import ChromaIndex
    from raglab.types import Chunk

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(
            experiment=ExperimentCfg(
                name="skill62_test",
                corpus_glob=["corpus/raw/**/*.txt"],
                representations=["chroma"],
            ),
            golden=GoldenCfg(path="./golden/questions.jsonl"),
        )
        cfg.index.persist_dir = tmpdir
        embed_cfg = EmbedCfg(model="all-MiniLM-L6-v2")

        index = ChromaIndex(cfg.index, embed_cfg)
        exp_name = "skill62_rebuild_test"

        def make_chunks(n: int, offset: int = 0):
            return [
                Chunk(
                    id=f"c{offset + i}",
                    doc_id="d1",
                    content=f"content {offset + i}",
                    source_type="test",
                    chunk_index=i,
                )
                for i in range(n)
            ]

        # First build: 5 chunks
        index.build(make_chunks(5), exp_name)
        assert index.collection.count() == 5

        # Second build: 8 chunks — triggers the delete-and-rebuild path
        index.build(make_chunks(8, offset=10), exp_name)
        assert index.collection.count() == 8, (
            "Stale collection handle — second build() returned old count instead of 8"
        )
