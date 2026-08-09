/**
 * Concept cards for the /learn page (Skill 40A). Each card explains one RAG
 * concept in plain language, with an analogy and a "Try it" deep link that
 * pre-configures the playground to demonstrate the concept live.
 */

export interface Concept {
  id: string
  title: string
  definition: string
  analogy: string
  tryIt: {
    label: string
    href: string
  }
}

export const CONCEPTS: Concept[] = [
  {
    id: "chunking",
    title: "Chunking",
    definition:
      "Splitting a long document into smaller pieces so each one can be independently embedded and retrieved. Chunk boundaries matter — splitting mid-sentence or mid-list-item loses meaning.",
    analogy:
      "Like cutting a book into index cards for a library catalog — cut them badly and a card ends mid-sentence, useless on its own.",
    tryIt: { label: "Compare chunking strategies", href: "/viz?tab=chunking" },
  },
  {
    id: "embedding",
    title: "Embeddings",
    definition:
      "A numeric vector representation of text such that semantically similar text ends up with similar vectors, enabling similarity search instead of exact keyword matching.",
    analogy:
      "Like plotting every sentence on a map by meaning — 'dog' and 'puppy' land near each other, 'dog' and 'stock market' land far apart.",
    tryIt: { label: "Explore the embedding space", href: "/viz?tab=embeddings" },
  },
  {
    id: "bm25_vs_dense",
    title: "BM25 (sparse) vs Dense retrieval",
    definition:
      "BM25 ranks documents by exact keyword overlap and term frequency. Dense retrieval ranks by embedding similarity, capturing meaning even without shared words.",
    analogy:
      "BM25 is Ctrl+F with smart ranking; dense retrieval is a librarian who understands what you mean, not just what you typed.",
    tryIt: { label: "Try both in the playground", href: "/playground?preset=research_compare" },
  },
  {
    id: "hybrid_rrf",
    title: "Hybrid retrieval (Reciprocal Rank Fusion)",
    definition:
      "Combines BM25 and dense retrieval rankings into one list by giving each result a score based on 1/(k + rank) in each list, then summing — rewarding results that rank well in both.",
    analogy: "Like combining two judges' scorecards — a contestant ranked well by both judges wins overall, even if neither ranked them #1.",
    tryIt: { label: "Run with hybrid_rrf", href: "/playground?preset=max_recall" },
  },
  {
    id: "faiss_index_types",
    title: "FAISS index types",
    definition:
      "flat does exact brute-force search (slow but perfectly accurate). ivf_flat/ivf_pq cluster vectors first and only search nearby clusters (fast, approximate). hnsw builds a navigable graph for fast approximate search at any scale.",
    analogy:
      "flat is checking every house in a city; ivf/hnsw is asking a local who already knows which neighborhood to check.",
    tryIt: { label: "Compare index backends", href: "/compare" },
  },
  {
    id: "reranking",
    title: "Reranking",
    definition:
      "A second, more expensive model re-scores the initial retrieved candidates (e.g. with a cross-encoder that reads the query and chunk together) to produce a more precise final ranking.",
    analogy: "Retrieval is a quick skim of resumes to shortlist candidates; reranking is the in-depth interview that decides the final order.",
    tryIt: { label: "Toggle reranking on", href: "/playground?preset=production_balanced" },
  },
  {
    id: "intent_classification",
    title: "Intent classification",
    definition:
      "Deciding whether a query is 'simple' (single fact lookup) or 'complex' (multi-document, comparative, or requiring reasoning) to route it to the naive or agentic pipeline.",
    analogy:
      "Like a helpdesk triage — quick questions go straight to the FAQ page, complicated ones get escalated to a specialist.",
    tryIt: { label: "See intent routing live", href: "/playground" },
  },
  {
    id: "agentic_strategies",
    title: "Agentic RAG strategies",
    definition:
      "For complex queries: decompose breaks a question into sub-questions retrieved separately; step_back asks a more general question first; hyde generates a hypothetical answer to retrieve against; react interleaves reasoning and tool calls.",
    analogy: "Like a research assistant who breaks a big question into smaller ones, checks a few angles, then synthesizes a final report.",
    tryIt: { label: "Ask a complex question", href: "/playground?sample=complex" },
  },
  {
    id: "confidence_scoring",
    title: "Confidence scoring",
    definition:
      "A composite signal (retrieval score, NLI entailment, or LLM self-judgment) estimating how trustworthy a generated answer is, used to trigger a fallback when evidence is weak.",
    analogy: "Like a weather forecast's confidence interval — the system tells you not just the answer, but how sure it is.",
    tryIt: { label: "See trust scores in action", href: "/playground" },
  },
  {
    id: "generation_modes",
    title: "Generation modes",
    definition:
      "strict_rag answers only from retrieved context; soft_rag allows supplementing with model knowledge (flagged); cot_rag reasons step-by-step before answering; self_check_rag verifies its own draft answer against the context and revises if inconsistent.",
    analogy: "strict_rag is an open-book exam with the book required; soft_rag lets you use outside knowledge but you must say so.",
    tryIt: { label: "Try self_check_rag", href: "/playground" },
  },
  {
    id: "benchmark_scores",
    title: "What benchmark scores mean",
    definition:
      "overall_score blends correctness (does the answer match ground truth?) and completeness (does it cover all required points?). Scores are compared across configs with bootstrap confidence intervals — a difference isn't 'real' until it's statistically significant.",
    analogy:
      "Like a school grade with a rubric — getting the right final number matters, but so does showing all your work.",
    tryIt: { label: "View the benchmark dashboard", href: "/benchmark" },
  },
]
