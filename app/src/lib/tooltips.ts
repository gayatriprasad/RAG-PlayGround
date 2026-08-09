/**
 * Parameter tooltip copy (Skill 38B) — shown via the ⓘ icon next to every
 * playground control. Keyed by the config field name used in the UI.
 */

export interface ParamTooltip {
  what: string
  when: string
  example?: string
}

export const PARAM_TOOLTIPS: Record<string, ParamTooltip> = {
  index_backend: {
    what: "Which vector store retrieves candidate chunks for a query.",
    when: "Use hybrid_rrf when queries mix exact keywords and concepts; chroma for a simple dense baseline; bm25 for pure keyword search.",
    example: "chroma, bm25, hybrid_rrf, faiss, pageindex",
  },
  chunk_strategy: {
    what: "How source documents are split into retrievable chunks (set at index-build time).",
    when: "fixed is simplest; sentence/semantic preserve meaning boundaries better for prose; recursive adapts to document structure.",
    example: "fixed, sentence, semantic, recursive",
  },
  chunk_tokens: {
    what: "Target size of each chunk, in tokens.",
    when: "Smaller chunks (128-256) improve precision; larger chunks (512-1024) preserve more context per retrieval.",
  },
  overlap: {
    what: "Number of tokens shared between consecutive chunks.",
    when: "Higher overlap reduces the chance a fact gets split across a chunk boundary, at the cost of more redundant storage.",
  },
  top_k: {
    what: "Number of chunks retrieved and passed to the LLM as context.",
    when: "Raise it for complex/multi-doc questions; lower it to reduce cost and latency for simple lookups.",
  },
  rerank: {
    what: "Whether retrieved chunks are re-scored by a second, more precise model before generation.",
    when: "Enable for higher-stakes queries where retrieval precision matters more than latency.",
  },
  reranker: {
    what: "Which reranking model reorders the initial candidates.",
    when: "cross_encoder is the highest quality but slowest; reciprocal_rank is a cheap fusion-only option.",
    example: "none, cross_encoder, bm25_rerank, monot5, reciprocal_rank",
  },
  intent_mode: {
    what: "How the query gets classified as 'simple' vs 'complex', which chooses the naive vs agentic pipeline.",
    when: "hybrid (rule + LLM fallback) is the best default; always_simple/always_complex are useful for isolating pipeline behavior in comparisons.",
    example: "rule, llm, hybrid, always_simple, always_complex",
  },
  llm_provider: {
    what: "Which LLM backend generates the final answer.",
    when: "ollama is free and local; openai/anthropic/groq need an API key but are typically higher quality/faster.",
    example: "ollama, openai, anthropic, groq, hf, lmstudio",
  },
  llm_model: {
    what: "The specific model name requested from the selected provider.",
    when: "Match this to a model your provider actually supports — e.g. llama3 for ollama, gpt-4o-mini for openai.",
  },
  confidence_threshold: {
    what: "Minimum confidence score required before the pipeline will answer instead of returning the fallback message.",
    when: "Raise it to reduce hallucination risk on ambiguous queries; lower it if the pipeline is too conservative.",
  },
  cache_mode: {
    what: "How query results are cached to avoid recomputation.",
    when: "exact only reuses results for identical queries; semantic reuses results for paraphrased queries; none disables caching entirely.",
    example: "exact, semantic, none",
  },
  prompt_strategy: {
    what: "The prompting technique used to build the LLM's instructions.",
    when: "cot (chain-of-thought) helps with multi-step reasoning; few_shot helps steer output format; zero_shot is the simplest/fastest.",
    example: "zero_shot, few_shot, cot, self_consistency, medprompt",
  },
  generation_mode: {
    what: "How strictly the LLM must ground its answer in retrieved context.",
    when: "strict_rag refuses to use outside knowledge; soft_rag allows it but flags it; self_check_rag verifies its own answer against the chunks before returning it.",
    example: "strict_rag, soft_rag, cot_rag, self_check_rag",
  },
  agentic_strategy: {
    what: "The multi-step reasoning strategy used for complex queries.",
    when: "decompose breaks the question into sub-questions; hyde generates a hypothetical answer first to improve retrieval; react interleaves reasoning with tool calls.",
    example: "decompose, step_back, hyde, react",
  },
  confidence_scorer: {
    what: "How the pipeline computes its confidence in an answer.",
    when: "composite blends multiple signals; nli checks entailment between answer and context; llm_judge asks a model to self-rate.",
    example: "retrieval_only, composite, nli, llm_judge",
  },
  embed_model: {
    what: "The embedding model used to convert text into vectors for retrieval.",
    when: "Larger models (bge-large, e5-large) retrieve more accurately but are slower to embed; MiniLM is the fastest baseline.",
    example: "all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5, intfloat/e5-large-v2, ollama/nomic-embed-text, openai/text-embedding-3-small",
  },
  faiss_index_type: {
    what: "The FAISS ANN index structure used for similarity search.",
    when: "flat is exact but slow at scale; ivf_flat/ivf_pq trade some accuracy for speed on large corpora; hnsw is a fast approximate default.",
    example: "flat, ivf_flat, ivf_pq, hnsw",
  },
  hybrid_dense_weight: {
    what: "Weight given to the dense (embedding) retriever in a weighted hybrid search.",
    when: "Raise it when queries are conceptual; lower it (raising sparse weight) when queries rely on exact keywords/codes.",
  },
  rrf_k: {
    what: "The 'k' constant in Reciprocal Rank Fusion, controlling how strongly top ranks dominate the fused score.",
    when: "Lower values (e.g. 10-20) sharpen the influence of top-ranked results; the default 60 is a well-established RRF baseline.",
  },
  recall_at_k: {
    what: "The set of k values used to compute retrieval recall during evaluation.",
    when: "Include multiple k's (1, 3, 5) to see how quickly the correct chunk surfaces in the ranking.",
  },
  citation_mode: {
    what: "How the LLM is instructed to cite its sources in the generated answer.",
    when: "chunk_id is the most precise and machine-parseable; doc_timestamp is more human-readable; none disables citations.",
    example: "chunk_id, doc_timestamp, none",
  },
  source_type: {
    what: "Which corpus source (confluence, github, jira, slack, etc.) the question/document comes from.",
    when: "Filter sample questions or corpus by source_type to test retrieval on a specific document style.",
  },
  temperature: {
    what: "Sampling randomness for the LLM's output.",
    when: "0.0 is fully deterministic — best for reproducible benchmarking. Raise it for more varied/creative generations.",
  },
  similarity_threshold: {
    what: "Minimum similarity score a chunk must meet to be included in retrieval results.",
    when: "Raise it to filter out weak matches; keep it at 0 to always return the top_k best-available chunks.",
  },
}
