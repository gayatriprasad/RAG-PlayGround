# config.py — full extended version

from __future__ import annotations
from pydantic import BaseModel
from typing import List, Literal, Optional

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
    model: Literal[
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "nomic-ai/nomic-embed-text-v1",
        "none"
    ] = "all-MiniLM-L6-v2"
    device: str = "cpu"

class IndexCfg(BaseModel):
    backend: Literal["chroma", "pageindex", "bm25", "hybrid_rrf", "hybrid_weighted", "hybrid"] = "chroma"
    persist_dir: str = "./out/chroma"
    rrf_k: int = 60
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3

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
    metrics: List[Literal["exact_match", "llm_judge", "retrieval_recall", "adversarial"]] = ["llm_judge"]
    adversarial_path: Optional[str] = None
    recall_at_k: List[int] = [1, 3, 5]

class LLMCfg(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    provider: Literal["openai", "ollama"] = "openai"
    ollama_base_url: str = "http://localhost:11434"

class BenchmarkCfg(BaseModel):
    questions_path: str = "./golden/questions.jsonl"
    source_types: List[str] = ["confluence", "github", "jira", "slack"]
    question_categories: Optional[List[str]] = None
    max_questions: int = 50

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
    llm: LLMCfg = LLMCfg()
    benchmark: BenchmarkCfg = BenchmarkCfg()
    eval: EvalCfg = EvalCfg()
