# config.py — full extended version

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from raglab.types import ConfigError

class ChunkCfg(BaseModel):
    strategy: Literal["fixed", "sentence", "semantic", "recursive", "none"] = "fixed"
    chunk_tokens: int = 512
    overlap: int = 50
    recursive_separators: List[str] = ["\n\n", "\n", ". ", " ", ""]

class RetrieveCfg(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.0
    rerank: bool = False
    reranker: Literal["none", "cross_encoder", "bm25_rerank", "monot5", "reciprocal_rank"] = "none"
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    confidence_threshold: float = 0.35
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_mode: Literal["exact", "semantic", "none"] = "exact"

class IngestCfg(BaseModel):
    parser: Literal["auto", "pdfplumber", "tesseract", "marker", "surya"] = "auto"
    dedup: Literal["none", "exact", "near", "semantic"] = "exact"
    near_dedup_threshold: float = 0.85
    extract_metadata: Literal["rule", "llm", "none"] = "rule"

class GoldenCfg(BaseModel):
    path: str

class ExperimentCfg(BaseModel):
    name: str
    corpus_glob: List[str]
    representations: List[str]

class EmbedCfg(BaseModel):
    model: str = "all-MiniLM-L6-v2"  # HF model name or local path (fine-tuned)
    device: str = "cpu"
    quantization: Literal["none", "int8", "binary"] = "none"  # Skill 53B
    sie_base_url: str = "http://localhost:8080"  # only used with sie/* models (Skill 53A)

class IndexCfg(BaseModel):
    backend: Literal[
        "chroma", "bm25", "hybrid_rrf", "hybrid_weighted", "hybrid",
        "faiss", "pageindex", "graph_rag", "colbert",
        "pgvector", "milvus", "pinecone", "weaviate", "qdrant", "zilliz",
    ] = "chroma"
    persist_dir: str = "./out/chroma"
    rrf_k: int = 60
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    # FAISS params
    faiss_index_type: Literal["flat", "ivf_flat", "ivf_pq", "hnsw"] = "flat"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10
    faiss_m: int = 32
    # Milvus / Zilliz params
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_token: Optional[str] = None
    milvus_collection: str = "neuralbench"
    # Pinecone params
    pinecone_index_name: str = "neuralbench"
    pinecone_region: str = "us-east-1"
    # Weaviate params
    weaviate_class: str = "NeuralBench"
    # Qdrant params
    qdrant_collection: str = "neuralbench"
    # pgvector params
    pgvector_table: str = "chunks"

class IntentCfg(BaseModel):
    mode: Literal["rule", "llm", "hybrid", "always_simple", "always_complex"] = "hybrid"
    llm_model: str = "gpt-4o-mini"
    simple_threshold: float = 0.8
    max_sub_queries: int = 4

class AgenticCfg(BaseModel):
    strategy: Literal["decompose", "step_back", "hyde", "react"] = "decompose"

class GenerationCfg(BaseModel):
    mode: Literal["strict_rag", "soft_rag", "cot_rag", "self_check_rag"] = "strict_rag"
    citation_mode: Literal["chunk_id", "doc_timestamp", "none"] = "chunk_id"

class ConfidenceCfg(BaseModel):
    scorer: Literal["retrieval_only", "composite", "nli", "llm_judge"] = "composite"
    fallback_message: str = "INSUFFICIENT EVIDENCE: confidence too low to answer reliably."

class EvalCfg(BaseModel):
    metrics: List[Literal["exact_match", "llm_judge", "retrieval_recall", "adversarial", "ocr_quality", "agentic_quality", "calibration"]] = ["llm_judge"]
    adversarial_path: Optional[str] = None
    recall_at_k: List[int] = [1, 3, 5]
    # Agentic eval settings (Skill 55)
    agentic_consistency_runs: int = 1  # set to 3+ to enable consistency scoring (expensive: N full graph runs per question)
    # Calibration settings (Skill 57)
    calibration_n_bins: int = 10

class ModelRegistryCfg(BaseModel):
    """Universal LLM config — supports all providers via Skill 21 registry."""
    provider: Literal["ollama", "openai", "anthropic", "groq", "hf", "lmstudio", "grok", "openrouter", "gemini"] = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None  # read from env if None
    context_window: int = 8192
    max_tokens: int = 512
    temperature: float = 0.0


# Backward-compat alias
LLMCfg = ModelRegistryCfg

class PromptCfg(BaseModel):
    """Prompt Engineering Lab config — Skill 23."""
    strategy: Literal["zero_shot", "few_shot", "cot", "self_consistency", "medprompt"] = "zero_shot"
    n_examples: int = 3            # for few_shot / medprompt
    n_samples: int = 5             # for self_consistency / medprompt
    temperature_sweep: List[float] = [0.0]
    prompt_version: str = "v1"     # tracked in prompts/ folder
    system_prompt_file: Optional[str] = None
    citation_mode: Literal["chunk_id", "doc_timestamp", "none"] = "chunk_id"

class DatasetCfg(BaseModel):
    """Dataset Expander config — Skill 26: 2000 questions (500 bench + 1000 synthetic + 500 BEIR)."""
    layers: List[Literal["bench", "synthetic", "beir"]] = ["bench"]
    bench_path: str = "./golden/questions.jsonl"         # 500, immutable
    synthetic_path: str = "./golden/questions_synthetic.jsonl"  # Skill 19
    beir_path: str = "./golden/questions_beir.jsonl"     # BEIR subset
    beir_subsets: List[str] = ["msmarco", "hotpotqa"]
    max_questions: int = 500           # total cap across all layers
    source_types: List[str] = ["confluence", "github", "jira", "slack"]

class BenchmarkCfg(BaseModel):
    questions_path: str = "./golden/questions.jsonl"
    source_types: List[str] = ["confluence", "github", "jira", "slack"]
    question_categories: Optional[List[str]] = None
    max_questions: int = 50

class CostCfg(BaseModel):
    """Cost & Latency tracking config — Skill 27."""
    track: bool = True
    alert_threshold_usd: float = 0.05  # warn if single query exceeds
    # pricing per 1M tokens (input/output) — update as providers change
    pricing: dict = {
        "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
        "gpt-4o":           {"input": 2.50,  "output": 10.0},
        "claude-3-haiku":   {"input": 0.25,  "output": 1.25},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.0},
        "groq/llama3-70b":  {"input": 0.59,  "output": 0.79},
        "ollama":           {"input": 0.0,   "output": 0.0},
    }

class DatabaseCfg(BaseModel):
    """Database Layer config — Skill 29."""
    backend: Literal["sqlite", "postgres"] = "sqlite"
    # SQLite settings
    sqlite_path: str = "./out/neuralbench.db"
    # Postgres settings (DSN from env DATABASE_URL only, not config)
    dsn: Optional[str] = None  # read from DATABASE_URL env if None
    pool_min_size: int = 2
    pool_max_size: int = 10
    enable_pgvector: bool = False  # enable pgvector tables + indexes

class NetworkCfg(BaseModel):
    """Networking resilience layer config — Skill 31 (PILLAR 3)."""
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0
    pool_max_connections: int = 20
    pool_max_keepalive: int = 10
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    circuit_breaker_threshold: int = 5     # consecutive failures before opening
    circuit_breaker_cooldown_s: float = 30.0
    rate_limit_per_minute: int = 60        # default inbound API limit
    rate_limit_arena_per_minute: int = 10  # stricter limit for expensive /arena calls

class CorpusCfg(BaseModel):
    """Bring-your-own-corpus config — Skill 33."""
    source: Literal["bench", "upload", "mixed"] = "bench"
    upload_dir: str = "./corpus/uploads"
    auto_detect_source_type: bool = True
    user_questions_path: Optional[str] = None

class ChallengeCfg(BaseModel):
    """Guided challenge mode config — Skill 34."""
    challenges_path: str = "./challenges/challenges.json"

class ExportCfg(BaseModel):
    """Export & share config — Skill 35. Nothing to configure yet."""
    pass

class StatsCfg(BaseModel):
    """Statistical significance layer config — Skill 43."""
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    alpha: float = 0.05
    min_effect_size: float = 0.05          # minimum |delta| to call a result practically significant
    continuous_test: Literal["wilcoxon", "paired_t"] = "wilcoxon"
    correction_method: Literal["benjamini_hochberg", "bonferroni", "none"] = "benjamini_hochberg"
    min_judge_kappa: float = 0.6           # Skill 44 — judge calibration reliability threshold
    enforce_slice_check: bool = True       # Skill 44 — refuse aggregate-only "winner" claims

class ImprovementCfg(BaseModel):
    """Closed-loop improvement config — Skill 46."""
    auto_trigger: bool = True              # set False on shared instances (SECURITY.md)
    min_recall_threshold: float = 0.7      # a source_type x category slice below this is a "gap"
    min_slice_size: int = 3                # slices with fewer questions are skipped (too noisy)
    max_iterations: int = 5
    fine_tune_epochs: int = 3
    fine_tune_base_model: str = "all-MiniLM-L6-v2"
    reports_dir: str = "./out/improvement"
    models_dir: str = "./models"

class RLMCfg(BaseModel):
    """
    Recursive Language Model pipeline config — Skill 54. Only relevant when
    the corpus is too large for retrieval-based RAG or CAG to handle well
    (frontend surfaces this once dataset.max_documents exceeds ~5K).
    """
    max_iterations: int = 5
    max_tokens_per_slice: int = 4096
    sub_model: str = "llama3"
    sub_provider: Literal["ollama", "openai", "groq"] = "ollama"
    max_code_rewrites: int = 2
    corpus_preview_chars: int = 500

class ObservabilityCfg(BaseModel):
    """Tracer backend selection — Skill 47C."""
    backend: Literal["jsonl", "langfuse", "phoenix", "openllmetry"] = "jsonl"
    project_name: str = "neuralbench"
    langfuse_host: str = "https://cloud.langfuse.com"
    phoenix_port: int = 6006

class Config(BaseModel):
    experiment: ExperimentCfg
    ingest: IngestCfg = IngestCfg()
    chunk: ChunkCfg = ChunkCfg()
    retrieve: RetrieveCfg = RetrieveCfg()
    golden: GoldenCfg
    embed: EmbedCfg = EmbedCfg()
    index: IndexCfg = IndexCfg()
    intent: IntentCfg = IntentCfg()
    agentic: AgenticCfg = AgenticCfg()
    generation: GenerationCfg = GenerationCfg()
    confidence: ConfidenceCfg = ConfidenceCfg()
    prompt: PromptCfg = PromptCfg()
    llm: LLMCfg = LLMCfg()
    dataset: DatasetCfg = DatasetCfg()  # Skill 26
    benchmark: BenchmarkCfg = BenchmarkCfg()
    eval: EvalCfg = EvalCfg()
    cost: CostCfg = CostCfg()  # Skill 27
    db: DatabaseCfg = DatabaseCfg()  # Skill 29
    net: NetworkCfg = NetworkCfg()  # Skill 31
    corpus: CorpusCfg = CorpusCfg()  # Skill 33
    challenge: ChallengeCfg = ChallengeCfg()  # Skill 34
    export: ExportCfg = ExportCfg()  # Skill 35
    stats: StatsCfg = StatsCfg()  # Skill 43
    improvement: ImprovementCfg = ImprovementCfg()  # Skill 46
    observability: ObservabilityCfg = ObservabilityCfg()  # Skill 47C
    rlm: RLMCfg = RLMCfg()  # Skill 54


# ─── Skill 58: Preset-aware config loading ─────────────────────────────────
#
# `rag-lab/presets/*.yaml` files (beginner.yaml, max_recall.yaml, ...) are
# flat, one-click *fragments* — a handful of top-level keys plus a name and
# description. They were never meant to satisfy the full `Config` contract
# on their own (they have no `experiment` or `golden` sections, both of
# which are required). Loading one directly via `Config(**raw)` therefore
# raises a wall of confusing pydantic `ValidationError`s.
#
# `PRESET_FIELD_MAP` declares exactly which flat preset key maps to which
# nested Config field. `apply_preset()` overlays a preset fragment onto an
# already-valid base `Config`. `is_preset_shaped()` lets callers detect a
# preset file being mistakenly passed where a full config is expected, so
# they can raise one clear `ConfigError` instead of a raw pydantic crash.

PRESET_FIELD_MAP: Dict[str, Tuple[str, str]] = {
    "index_backend": ("index", "backend"),
    "chunk_strategy": ("chunk", "strategy"),
    "top_k": ("retrieve", "top_k"),
    "reranker": ("retrieve", "reranker"),
    "intent_mode": ("intent", "mode"),
    "llm_provider": ("llm", "provider"),
    "llm_model": ("llm", "model"),
}

# Sections that a full experiment Config must have but a preset fragment
# never provides. Used by is_preset_shaped() to recognize preset files.
_CONFIG_REQUIRED_SECTIONS = ("experiment", "golden")


def is_preset_shaped(raw: Dict[str, Any]) -> bool:
    """Return True if `raw` looks like a one-click preset fragment rather
    than a full experiment Config: it is missing the required top-level
    sections (`experiment`, `golden`) but contains at least one of the
    flat keys a preset is known to carry.
    """
    if not isinstance(raw, dict):
        return False
    missing_required = any(section not in raw for section in _CONFIG_REQUIRED_SECTIONS)
    has_preset_keys = any(key in raw for key in PRESET_FIELD_MAP)
    return missing_required and has_preset_keys


def apply_preset(cfg: "Config", preset: Dict[str, Any]) -> "Config":
    """Apply a flat preset fragment (as loaded from `rag-lab/presets/*.yaml`)
    on top of an already-valid base `Config`, returning a new `Config` with
    the requested fields overridden. Unknown/metadata keys (`name`,
    `description`) are ignored. Never mutates `cfg` in place.
    """
    new_cfg = cfg.model_copy(deep=True)
    for key, value in preset.items():
        if key not in PRESET_FIELD_MAP:
            continue
        section_name, field_name = PRESET_FIELD_MAP[key]
        section = getattr(new_cfg, section_name)
        setattr(section, field_name, value)
        # Selecting a real reranker implies reranking should be turned on.
        if section_name == "retrieve" and field_name == "reranker" and value not in (None, "none"):
            new_cfg.retrieve.rerank = True
    return new_cfg


def load_config_with_preset(
    config_path: str, preset: Optional[str] = None, presets_dir: Optional[str] = None
) -> "Config":
    """Load a full base Config from `config_path`, optionally layering a
    preset fragment on top. Raises `ConfigError` with an actionable message
    if `config_path` itself looks like a preset fragment (the historical
    "the bug" this skill fixes) or if a requested preset file can't be found.
    """
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if is_preset_shaped(raw):
        preset_keys = ", ".join(k for k in PRESET_FIELD_MAP if k in raw)
        raise ConfigError(
            f"'{path}' looks like a one-click preset fragment (found keys: "
            f"{preset_keys}), not a full experiment Config — it is missing "
            f"required sections {_CONFIG_REQUIRED_SECTIONS}. Presets only "
            f"override a handful of fields and cannot be used as --config "
            f"directly.\nFix: pass a full base config via --config and layer "
            f"the preset on top with --preset, e.g.\n"
            f"  raglab-run --config experiments/02_retrieval_comparison/config.yaml --preset {path.stem}"
        )

    cfg = Config(**raw)

    if preset:
        preset_path = Path(preset)
        if not preset_path.exists():
            candidate_dir = Path(presets_dir) if presets_dir else Path(__file__).resolve().parents[2] / "presets"
            candidate = candidate_dir / f"{preset}.yaml"
            if candidate.exists():
                preset_path = candidate
            else:
                raise ConfigError(f"Preset file not found: {preset} (looked in {candidate})")

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_raw = yaml.safe_load(f) or {}
        cfg = apply_preset(cfg, preset_raw)

    return cfg
