"""
Slot-registry regression tests — Skill 49(B).

Guards Coding Rule: "Every backend/strategy name in a config Literal matches
its file name and factory case." Rather than fragile filename string
matching, this inspects the actual factory `match` statements (source of
truth for "is this backend wired up") and cross-checks against the config
Literal (source of truth for "is this backend advertised").
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import inspect

from raglab.config import ChunkCfg, IndexCfg, ModelRegistryCfg


def _match_case_values(source: str) -> set:
    """Extract the string literals used in `case "x":` / `case "x" | "y":` lines."""
    values = set()
    for line in source.splitlines():
        m = re.match(r'\s*case\s+(.+?):\s*$', line)
        if not m:
            continue
        clause = m.group(1)
        if clause.strip() == "_":
            continue
        for literal_match in re.finditer(r'"([^"]+)"', clause):
            values.add(literal_match.group(1))
    return values


def _literal_values(model_cls, field_name) -> set:
    annotation = model_cls.model_fields[field_name].annotation
    return set(annotation.__args__)


def test_chunk_strategy_literal_matches_factory_cases():
    from raglab.chunkers import get_chunker as _  # noqa: F401
    import raglab.chunkers as chunkers_module

    source = inspect.getsource(chunkers_module)
    factory_cases = _match_case_values(source)
    literal_values = _literal_values(ChunkCfg, "strategy")

    assert literal_values == factory_cases, (
        f"ChunkCfg.strategy Literal {literal_values} does not match "
        f"chunkers/__init__.py factory cases {factory_cases}"
    )


def test_index_backend_literal_matches_factory_cases():
    import raglab.index as index_module

    source = inspect.getsource(index_module)
    factory_cases = _match_case_values(source)
    literal_values = _literal_values(IndexCfg, "backend")

    assert literal_values == factory_cases, (
        f"IndexCfg.backend Literal {literal_values} does not match "
        f"index/__init__.py factory cases {factory_cases}"
    )


def test_llm_provider_literal_matches_factory_cases():
    import raglab.models.factory as factory_module

    source = inspect.getsource(factory_module)
    factory_cases = _match_case_values(source)
    literal_values = _literal_values(ModelRegistryCfg, "provider")

    assert literal_values == factory_cases, (
        f"ModelRegistryCfg.provider Literal {literal_values} does not match "
        f"models/factory.py factory cases {factory_cases}"
    )


def test_every_index_backend_module_file_exists():
    """Every non-hybrid, non-composite backend name should have a same-named
    (or clearly-named) module file under index/ — sanity check against typos."""
    index_dir = Path(__file__).resolve().parents[2] / "src" / "raglab" / "index"
    existing_files = {p.stem for p in index_dir.glob("*.py")}

    expected_file_by_backend = {
        "chroma": "chroma_index",
        "bm25": "bm25_index",
        "hybrid_rrf": "hybrid_rrf",
        "hybrid_weighted": "hybrid_weighted",
        "hybrid": "hybrid_index",
        "faiss": "faiss_index",
        "pageindex": "pageindex_adapter",
        "graph_rag": "graph_rag",
        "pgvector": "pgvector_index",
        "milvus": "milvus_index",
        "pinecone": "pinecone_index",
        "weaviate": "weaviate_index",
        "qdrant": "qdrant_index",
    }
    for backend, filename in expected_file_by_backend.items():
        assert filename in existing_files, f"Expected index/{filename}.py for backend '{backend}'"
