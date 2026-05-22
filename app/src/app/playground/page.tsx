"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Loader2, Sparkles, Zap, Lightbulb } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { apiPost } from "@/lib/api";

interface QueryResponse {
  answer: string;
  pipeline: string;
  intent_label: string;
  intent_confidence: number;
  retrieved_chunks: { chunk: { content: string; source_type: string }; score: number }[];
  latency_ms: number;
}

export default function PlaygroundPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Config controls
  const [topK, setTopK] = useState(5);
  const [chunkStrategy, setChunkStrategy] = useState("fixed");
  const [intentMode, setIntentMode] = useState("hybrid");
  const [reranker, setReranker] = useState("none");

  const sampleQuestions = [
    { text: "What hashing algorithm does the auth service use for passwords?", category: "confluence" },
    { text: "What was the root cause of the 502 errors on the users endpoint?", category: "github" },
    { text: "What caused the memory leak in the search service?", category: "github" },
    { text: "What was the resolution for the Elasticsearch disk space issue?", category: "slack" },
    { text: "What subscription tiers does the payment service offer?", category: "github" },
    { text: "Compare the incident response times for SEV-1 vs SEV-2.", category: "confluence" },
    { text: "What monitoring tools are used across all services at Acme?", category: "multi-doc" },
    { text: "How does the quarterly key rotation process work?", category: "slack" },
  ];

  async function handleSubmit() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await apiPost<QueryResponse>("/query", {
        question: query,
        top_k: topK,
        chunk_strategy: chunkStrategy,
        intent_mode: intentMode,
        reranker,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="h-14 flex items-center px-6 border-b border-border shrink-0">
        <h1 className="text-lg font-semibold tracking-tight">Playground</h1>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel — Config */}
        <aside className="w-[280px] border-r border-border p-5 overflow-y-auto space-y-6 shrink-0">
          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Chunking Strategy
            </Label>
            <Select value={chunkStrategy} onValueChange={(v) => v && setChunkStrategy(v)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="fixed">Fixed</SelectItem>
                <SelectItem value="sentence">Sentence</SelectItem>
                <SelectItem value="semantic">Semantic</SelectItem>
                <SelectItem value="recursive">Recursive</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Top K: {topK}
            </Label>
            <Slider
              value={[topK]}
              onValueChange={(v) => setTopK(Array.isArray(v) ? v[0] : v)}
              min={1}
              max={20}
              step={1}
            />
          </div>

          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Intent Classification
            </Label>
            <Select value={intentMode} onValueChange={(v) => v && setIntentMode(v)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="rule">Rule-based</SelectItem>
                <SelectItem value="llm">LLM</SelectItem>
                <SelectItem value="hybrid">Hybrid</SelectItem>
                <SelectItem value="always_simple">Always Simple</SelectItem>
                <SelectItem value="always_complex">Always Complex</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Reranker
            </Label>
            <Select value={reranker} onValueChange={(v) => v && setReranker(v)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="cross_encoder">Cross Encoder</SelectItem>
                <SelectItem value="bm25_rerank">BM25 Rerank</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 flex flex-col p-6 gap-6 overflow-y-auto">
          {/* Query Input */}
          <Card className="shadow-sm border-border">
            <CardContent className="pt-5">
              <div className="flex gap-3">
                <Textarea
                  placeholder="Ask a question about your documents..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="min-h-[80px] resize-none"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && e.metaKey) handleSubmit();
                  }}
                />
                <Button
                  onClick={handleSubmit}
                  disabled={loading || !query.trim()}
                  className="h-auto px-4"
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Press ⌘+Enter to submit
              </p>
            </CardContent>
          </Card>

          {/* Sample Questions */}
          {!result && !loading && (
            <Card className="shadow-sm border-border">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-amber-500" />
                  <CardTitle className="text-sm font-medium">Try a sample question</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-2">
                  {sampleQuestions.map((sq, i) => (
                    <button
                      key={i}
                      onClick={() => setQuery(sq.text)}
                      className="text-left px-3 py-2 rounded-lg border border-border hover:bg-muted/50 hover:border-primary/30 transition-colors group"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-foreground/80 group-hover:text-foreground">
                          {sq.text}
                        </span>
                        <Badge variant="outline" className="text-[10px] ml-2 shrink-0">
                          {sq.category}
                        </Badge>
                      </div>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Loading State */}
          {loading && (
            <Card className="shadow-sm border-border">
              <CardContent className="pt-5 space-y-3">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-4 w-2/3" />
              </CardContent>
            </Card>
          )}

          {/* Error */}
          {error && (
            <Card className="shadow-sm border-destructive/50 bg-destructive/5">
              <CardContent className="pt-5">
                <p className="text-sm text-destructive">{error}</p>
              </CardContent>
            </Card>
          )}

          {/* Result */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-4"
              >
                {/* Answer Card */}
                <Card className="shadow-sm border-border">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">Answer</CardTitle>
                      <div className="flex gap-2">
                        <Badge variant="secondary" className="text-xs">
                          {result.pipeline === "naive" ? (
                            <Zap className="h-3 w-3 mr-1" />
                          ) : (
                            <Sparkles className="h-3 w-3 mr-1" />
                          )}
                          {result.pipeline}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {result.intent_label} ({(result.intent_confidence * 100).toFixed(0)}%)
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {result.latency_ms}ms
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                      {result.answer}
                    </p>
                  </CardContent>
                </Card>

                {/* Retrieved Chunks */}
                {result.retrieved_chunks.length > 0 && (
                  <Card className="shadow-sm border-border">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">
                        Retrieved Chunks ({result.retrieved_chunks.length})
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {result.retrieved_chunks.map((rc, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-lg bg-muted/50 border border-border"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <Badge variant="secondary" className="text-xs">
                              {rc.chunk.source_type}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              score: {rc.score.toFixed(3)}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground line-clamp-3">
                            {rc.chunk.content}
                          </p>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
