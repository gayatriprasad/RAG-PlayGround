"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Send, Loader2, ArrowLeftRight, GitFork, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { apiPost } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";

interface QueryResponse {
  answer: string;
  pipeline: string;
  intent_label: string;
  intent_confidence: number;
  retrieved_chunks: { chunk: { content: string; source_type: string }; score: number }[];
  latency_ms: number;
}

interface PipelineConfig {
  label: string;
  intent_mode: string;
  reranker: string;
  top_k: number;
}

const PRESETS: Record<string, PipelineConfig> = {
  naive_basic: { label: "Naive (no rerank)", intent_mode: "always_simple", reranker: "none", top_k: 5 },
  naive_rerank: { label: "Naive + Rerank", intent_mode: "always_simple", reranker: "cross_encoder", top_k: 5 },
  agentic_basic: { label: "Agentic (no rerank)", intent_mode: "always_complex", reranker: "none", top_k: 5 },
  agentic_rerank: { label: "Agentic + Rerank", intent_mode: "always_complex", reranker: "cross_encoder", top_k: 5 },
  hybrid: { label: "Hybrid (auto-route)", intent_mode: "hybrid", reranker: "none", top_k: 5 },
};

export default function ComparePage() {
  const [query, setQuery] = useState("");
  const [leftPreset, setLeftPreset] = useState("naive_basic");
  const [rightPreset, setRightPreset] = useState("agentic_basic");
  // Set once the user forks the left config onto the right side (Skill 37C) —
  // when non-null, the right panel becomes independently editable instead of
  // following a named preset.
  const [rightCustom, setRightCustom] = useState<PipelineConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [leftResult, setLeftResult] = useState<QueryResponse | null>(null);
  const [rightResult, setRightResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function forkLeftToRight() {
    setRightCustom({ ...PRESETS[leftPreset], label: `Custom (forked from ${PRESETS[leftPreset].label})` });
  }

  const rightConfig = rightCustom ?? PRESETS[rightPreset];

  async function handleCompare() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setLeftResult(null);
    setRightResult(null);

    const left = PRESETS[leftPreset];
    const right = rightConfig;

    try {
      const [l, r] = await Promise.all([
        apiPost<QueryResponse>("/query", {
          question: query,
          top_k: left.top_k,
          intent_mode: left.intent_mode,
          reranker: left.reranker,
        }),
        apiPost<QueryResponse>("/query", {
          question: query,
          top_k: right.top_k,
          intent_mode: right.intent_mode,
          reranker: right.reranker,
        }),
      ]);
      setLeftResult(l);
      setRightResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function ResultPanel({ result }: { result: QueryResponse | null }) {
    if (loading) {
      return (
        <div className="space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      );
    }
    if (!result) {
      return (
        <p className="text-sm text-muted-foreground text-center py-8">
          Run a comparison to see results
        </p>
      );
    }
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="space-y-4"
      >
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">{result.pipeline}</Badge>
          <Badge variant="outline">{result.latency_ms}ms</Badge>
          <Badge variant="outline">
            {result.intent_label} ({(result.intent_confidence * 100).toFixed(0)}%)
          </Badge>
        </div>
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{result.answer}</p>
        {result.retrieved_chunks.length > 0 && (
          <>
            <Separator />
            <p className="text-xs text-muted-foreground font-medium">
              {result.retrieved_chunks.length} chunks retrieved
            </p>
            {result.retrieved_chunks.slice(0, 3).map((rc, i) => (
              <div key={i} className="p-2 rounded bg-muted/50 border border-border">
                <p className="text-xs text-muted-foreground line-clamp-2">{rc.chunk.content}</p>
              </div>
            ))}
          </>
        )}
      </motion.div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <header className="h-14 flex items-center px-6 border-b border-border shrink-0">
        <h1 className="text-lg font-semibold tracking-tight">Compare Pipelines</h1>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Query + Controls */}
        <Card className="shadow-sm border-border">
          <CardContent className="pt-5 space-y-4">
            <Textarea
              placeholder="Enter a question to compare across pipelines..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="min-h-[60px] resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && e.metaKey) handleCompare();
              }}
            />
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <Label className="text-xs text-muted-foreground mb-1 block">Left Pipeline</Label>
                <Select value={leftPreset} onValueChange={(v) => v && setLeftPreset(v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.entries(PRESETS).map(([k, v]) => (
                      <SelectItem key={k} value={k}>{v.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-col items-center gap-1 mt-5">
                <ArrowLeftRight className="h-4 w-4 text-muted-foreground" />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 px-1.5 text-[10px] text-muted-foreground"
                  onClick={forkLeftToRight}
                  title="Copy the left config to the right side, then tweak one param"
                >
                  <GitFork className="h-3 w-3 mr-1" /> Fork
                </Button>
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <Label className="text-xs text-muted-foreground block">Right Pipeline</Label>
                  {rightCustom && (
                    <button
                      onClick={() => setRightCustom(null)}
                      className="text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-0.5"
                    >
                      <RotateCcw className="h-2.5 w-2.5" /> reset
                    </button>
                  )}
                </div>
                {rightCustom ? (
                  <div className="space-y-2 p-2 rounded-md border border-dashed border-primary/40 bg-primary/5">
                    <p className="text-[10px] font-medium text-primary">{rightCustom.label}</p>
                    <Select
                      value={rightCustom.intent_mode}
                      onValueChange={(v) => v && setRightCustom({ ...rightCustom, intent_mode: v })}
                    >
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="always_simple">Always Simple</SelectItem>
                        <SelectItem value="always_complex">Always Complex</SelectItem>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select
                      value={rightCustom.reranker}
                      onValueChange={(v) => v && setRightCustom({ ...rightCustom, reranker: v })}
                    >
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">No Rerank</SelectItem>
                        <SelectItem value="cross_encoder">Cross Encoder</SelectItem>
                        <SelectItem value="bm25_rerank">BM25 Rerank</SelectItem>
                      </SelectContent>
                    </Select>
                    <div>
                      <Label className="text-[10px] text-muted-foreground">Top K: {rightCustom.top_k}</Label>
                      <Slider
                        value={[rightCustom.top_k]}
                        onValueChange={(v) => setRightCustom({ ...rightCustom, top_k: Array.isArray(v) ? v[0] : v })}
                        min={1}
                        max={20}
                        step={1}
                      />
                    </div>
                  </div>
                ) : (
                  <Select value={rightPreset} onValueChange={(v) => v && setRightPreset(v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.entries(PRESETS).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              <Button onClick={handleCompare} disabled={loading || !query.trim()} className="mt-5">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Compare"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="pt-5">
              <p className="text-sm font-medium text-destructive">{toFriendlyError(new Error(error)).title}</p>
              <p className="text-xs text-destructive/80 mt-1">{toFriendlyError(new Error(error)).description}</p>
            </CardContent>
          </Card>
        )}

        {/* Side-by-side Results */}
        <div className="grid grid-cols-2 gap-6">
          <Card className="shadow-sm border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">{PRESETS[leftPreset].label}</CardTitle>
            </CardHeader>
            <CardContent>
              <ResultPanel result={leftResult} />
            </CardContent>
          </Card>

          <Card className="shadow-sm border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">{rightConfig.label}</CardTitle>
            </CardHeader>
            <CardContent>
              <ResultPanel result={rightResult} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
