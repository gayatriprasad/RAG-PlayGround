"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Send, Loader2, ArrowLeftRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { apiPost } from "@/lib/api";

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
  const [loading, setLoading] = useState(false);
  const [leftResult, setLeftResult] = useState<QueryResponse | null>(null);
  const [rightResult, setRightResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleCompare() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setLeftResult(null);
    setRightResult(null);

    const left = PRESETS[leftPreset];
    const right = PRESETS[rightPreset];

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

  function ResultPanel({ result, preset }: { result: QueryResponse | null; preset: string }) {
    const config = PRESETS[preset];
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
              <ArrowLeftRight className="h-4 w-4 text-muted-foreground mt-5" />
              <div className="flex-1">
                <Label className="text-xs text-muted-foreground mb-1 block">Right Pipeline</Label>
                <Select value={rightPreset} onValueChange={(v) => v && setRightPreset(v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.entries(PRESETS).map(([k, v]) => (
                      <SelectItem key={k} value={k}>{v.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
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
              <p className="text-sm text-destructive">{error}</p>
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
              <ResultPanel result={leftResult} preset={leftPreset} />
            </CardContent>
          </Card>

          <Card className="shadow-sm border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">{PRESETS[rightPreset].label}</CardTitle>
            </CardHeader>
            <CardContent>
              <ResultPanel result={rightResult} preset={rightPreset} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
