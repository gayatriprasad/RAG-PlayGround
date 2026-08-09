/**
 * Hardcoded, deterministic insight rules (Skill 39B/39C) — NOT LLM-based.
 * Operates on the grouped stats and per-row data already returned by
 * GET /benchmark/results.
 */

export interface GroupStat {
  count: number
  avg_score: number
}

export interface BenchmarkRow {
  question_id: string
  source_type: string
  category: string
  pipeline: string
  index_backend?: string
  intent_label?: string
  answer_correct: boolean | string
  completeness: number
  overall_score: number
}

/** Auto-insight captions above the category/pipeline charts (Skill 39B). */
export function categoryInsight(byCategory: Record<string, GroupStat>): string | null {
  const entries = Object.entries(byCategory)
  if (entries.length < 2) return null

  const sorted = [...entries].sort((a, b) => a[1].avg_score - b[1].avg_score)
  const [worstName, worst] = sorted[0]
  const [bestName, best] = sorted[sorted.length - 1]
  const gap = best.avg_score - worst.avg_score

  if (gap < 0.1) {
    return "Scores are fairly consistent across question categories — no single category is dragging down the average."
  }
  return `"${worstName}" questions score ${(worst.avg_score * 100).toFixed(0)}%, ` +
    `${(gap * 100).toFixed(0)} points below "${bestName}" (${(best.avg_score * 100).toFixed(0)}%) — ` +
    `worth investigating retrieval/generation quality specifically for "${worstName}" cases.`
}

export function pipelineInsight(byPipeline: Record<string, GroupStat>): string | null {
  const naive = byPipeline["naive"]
  const agentic = byPipeline["agentic"]
  if (!naive || !agentic) return null

  const diff = agentic.avg_score - naive.avg_score
  if (Math.abs(diff) < 0.05) {
    return "Naive and agentic pipelines score about the same here — the extra decomposition cost of agentic RAG isn't paying off on this question set."
  }
  if (diff > 0) {
    return `Agentic RAG outscores naive by ${(diff * 100).toFixed(0)} points (${(agentic.avg_score * 100).toFixed(0)}% vs ${(naive.avg_score * 100).toFixed(0)}%) — the extra decomposition step is helping on complex questions.`
  }
  return `Naive RAG actually outscores agentic by ${(-diff * 100).toFixed(0)} points here — the agentic pipeline's added complexity isn't translating into better answers on this set.`
}

/** Per-row failure hypothesis (Skill 39C) — click-to-expand on a table row. */
export function failureHypothesis(row: BenchmarkRow): string {
  const correct = row.answer_correct === true || row.answer_correct === "True"
  if (correct) {
    return "Answered correctly — no failure to diagnose."
  }

  if (row.completeness >= 0.5) {
    return (
      "Likely a generation failure: completeness is reasonably high " +
      `(${(row.completeness * 100).toFixed(0)}%), suggesting relevant content was retrieved, ` +
      "but the final answer didn't match the expected ground truth closely enough — " +
      "check phrasing, format, or whether the LLM misread the retrieved context."
    )
  }
  if (row.completeness > 0) {
    return (
      "Mixed signal: partial completeness " +
      `(${(row.completeness * 100).toFixed(0)}%) suggests only some relevant content was retrieved — ` +
      "consider raising top_k or trying a different index backend for this question's source type."
    )
  }
  return (
    "Likely a retrieval failure: completeness is near zero, meaning the retrieved chunks " +
    "didn't contain the information needed — try a different chunking strategy, a larger top_k, " +
    "or check whether the source document is even in the corpus."
  )
}

/** Auto-suggested name for a saved run, based on its config (Skill 39D). */
export function suggestRunName(config: {
  index_backend?: string
  pipeline?: string
  reranker?: string
  llm_model?: string
}): string {
  const parts = [
    config.index_backend,
    config.reranker && config.reranker !== "none" ? `${config.reranker}-rerank` : null,
    config.llm_model,
  ].filter(Boolean)
  return parts.length > 0 ? parts.join("_") : "unnamed_run";
}
