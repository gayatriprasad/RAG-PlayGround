"use client";

import { CheckCircle2, Search, SlidersHorizontal, Sparkles, Zap } from "lucide-react";

export interface PipelineStoryResult {
  pipeline: string;
  intent_label: string;
  intent_confidence: number;
  reranker: string;
  retrieved_chunks: { score: number }[];
  latency_ms: number;
}

/** Ordered walkthrough of what the pipeline actually did (Skill 38C). */
export function PipelineStory({ result }: { result: PipelineStoryResult }) {
  const topScore = result.retrieved_chunks.reduce(
    (max, c) => Math.max(max, c.score),
    0
  );

  const steps = [
    {
      icon: Sparkles,
      label: "Intent",
      detail: `${result.intent_label} (${(result.intent_confidence * 100).toFixed(0)}% confidence)`,
    },
    {
      icon: Search,
      label: "Retrieval",
      detail:
        result.retrieved_chunks.length > 0
          ? `${result.retrieved_chunks.length} chunks retrieved (top score ${topScore.toFixed(3)})`
          : "No chunks retrieved",
    },
    ...(result.reranker && result.reranker !== "none"
      ? [
          {
            icon: SlidersHorizontal,
            label: "Rerank",
            detail: `Reordered candidates using ${result.reranker}`,
          },
        ]
      : []),
    {
      icon: result.pipeline === "naive" ? Zap : Sparkles,
      label: "Generation",
      detail: `${result.pipeline} pipeline, ${result.latency_ms}ms total`,
    },
    {
      icon: CheckCircle2,
      label: "Done",
      detail: "Answer returned with citations below",
    },
  ];

  return (
    <div className="flex flex-col gap-0">
      {steps.map((step, i) => (
        <div key={step.label} className="flex gap-3">
          <div className="flex flex-col items-center">
            <div className="h-7 w-7 shrink-0 rounded-full bg-primary/10 text-primary flex items-center justify-center">
              <step.icon className="h-3.5 w-3.5" />
            </div>
            {i < steps.length - 1 && <div className="w-px flex-1 bg-border my-1" />}
          </div>
          <div className="pb-4">
            <p className="text-sm font-medium leading-tight">{step.label}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{step.detail}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
