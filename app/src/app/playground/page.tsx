"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Loader2, Sparkles, Zap, Lightbulb, AlertCircle, RefreshCw, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ParamTooltip } from "@/components/param-tooltip";
import { PipelineStory } from "@/components/pipeline-story";
import { CitationText } from "@/components/citation-text";
import { apiGet, apiPost, API_BASE } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";

interface RetrievedChunkItem {
  chunk_id: string;
  content: string;
  source_type: string;
  score: number;
  chunk: { content: string; source_type: string };
}

interface QueryResponse {
  answer: string;
  pipeline: string;
  intent_label: string;
  intent_confidence: number;
  retrieved_chunks: RetrievedChunkItem[];
  latency_ms: number;
  reranker: string;
}

interface PresetSummary {
  id: string;
  name: string;
  description: string;
}

interface PresetDetail {
  index_backend?: string;
  chunk_strategy?: string;
  top_k?: number;
  reranker?: string;
  intent_mode?: string;
  llm_provider?: string;
  llm_model?: string;
}

interface ChunkEstimate {
  n_documents: number;
  total_tokens: number;
  estimated_chunks: number;
  approximate: boolean;
}

interface ConfigSnapshot {
  indexBackend: string;
  pipelineOverride: string;
  topK: number;
  chunkStrategy: string;
  llmModel: string;
  intentMode: string;
  reranker: string;
  llmProvider: string;
  chunkTokens: number;
  overlap: number;
}

export default function PlaygroundPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(true);
  const [streamingAnswer, setStreamingAnswer] = useState<string | null>(null);

  // Config controls — the 5 "Simple" params (Skill 37D)
  const [indexBackend, setIndexBackend] = useState("chroma");
  const [pipelineOverride, setPipelineOverride] = useState("auto");
  const [topK, setTopK] = useState(5);
  const [chunkStrategy, setChunkStrategy] = useState("fixed");
  const [llmModel, setLlmModel] = useState("llama3");

  // Advanced-only params
  const [advanced, setAdvanced] = useState(false);
  const [intentMode, setIntentMode] = useState("hybrid");
  const [reranker, setReranker] = useState("none");
  const [llmProvider, setLlmProvider] = useState("ollama");
  const [chunkTokens, setChunkTokens] = useState(512);
  const [overlap, setOverlap] = useState(50);

  // Presets (Skill 37E)
  const [presets, setPresets] = useState<PresetSummary[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("");

  // Live chunk-count preview (Skill 39E)
  const [chunkEstimate, setChunkEstimate] = useState<ChunkEstimate | null>(null);
  const [estimateError, setEstimateError] = useState<string | null>(null);

  // Config history — last 10 distinct configs used, with undo (Skill 36)
  const [configHistory, setConfigHistory] = useState<ConfigSnapshot[]>([]);

  function currentConfigSnapshot(): ConfigSnapshot {
    return {
      indexBackend,
      pipelineOverride,
      topK,
      chunkStrategy,
      llmModel,
      intentMode,
      reranker,
      llmProvider,
      chunkTokens,
      overlap,
    };
  }

  function applyConfigSnapshot(snap: ConfigSnapshot) {
    setIndexBackend(snap.indexBackend);
    setPipelineOverride(snap.pipelineOverride);
    setTopK(snap.topK);
    setChunkStrategy(snap.chunkStrategy);
    setLlmModel(snap.llmModel);
    setIntentMode(snap.intentMode);
    setReranker(snap.reranker);
    setLlmProvider(snap.llmProvider);
    setChunkTokens(snap.chunkTokens);
    setOverlap(snap.overlap);
  }

  function recordConfigHistory() {
    const snap = currentConfigSnapshot();
    setConfigHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last && JSON.stringify(last) === JSON.stringify(snap)) return prev;
      return [...prev, snap].slice(-10);
    });
  }

  function undoLastConfig() {
    setConfigHistory((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice(0, -1);
      applyConfigSnapshot(prev[prev.length - 1]);
      return next;
    });
  }

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

  useEffect(() => {
    apiGet<{ presets: PresetSummary[] }>("/presets")
      .then((data) => setPresets(data.presets))
      .catch(() => setPresets([]));
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      const params = new URLSearchParams({
        chunk_tokens: String(chunkTokens),
        overlap: String(overlap),
        strategy: chunkStrategy,
      });
      apiGet<ChunkEstimate>(`/corpus/chunk-estimate?${params}`)
        .then((data) => {
          setChunkEstimate(data);
          setEstimateError(null);
        })
        .catch((e) => setEstimateError(e instanceof Error ? e.message : "Estimate failed"));
    }, 300);
    return () => clearTimeout(timer);
  }, [chunkStrategy, chunkTokens, overlap]);

  async function applyPreset(id: string) {
    setSelectedPreset(id);
    if (!id) return;
    try {
      const preset = await apiGet<PresetDetail>(`/presets/${id}`);
      if (preset.index_backend) setIndexBackend(preset.index_backend);
      if (preset.chunk_strategy) setChunkStrategy(preset.chunk_strategy);
      if (preset.top_k) setTopK(preset.top_k);
      if (preset.reranker) setReranker(preset.reranker);
      if (preset.intent_mode) setIntentMode(preset.intent_mode);
      if (preset.llm_provider) setLlmProvider(preset.llm_provider);
      if (preset.llm_model) setLlmModel(preset.llm_model);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load preset");
    }
  }

  async function handleSubmit() {
    if (!query.trim()) return;
    recordConfigHistory();
    setLoading(true);
    setError(null);
    setResult(null);
    setStreamingAnswer(null);

    const basePayload = {
      question: query,
      top_k: topK,
      chunk_strategy: chunkStrategy,
      intent_mode: intentMode,
      reranker,
      index_backend: indexBackend,
      pipeline_override: pipelineOverride === "auto" ? undefined : pipelineOverride,
      llm_provider: llmProvider,
      llm_model: llmModel || undefined,
    };

    if (!streaming) {
      try {
        const res = await apiPost<QueryResponse>("/query", basePayload);
        setResult({ ...res, reranker });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
      return;
    }

    // Streaming path (Skill 32) — fetch + ReadableStream, since EventSource
    // doesn't support POST bodies. Falls back to non-streaming on failure.
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...basePayload, stream: true }),
      });
      if (!res.ok || !res.body) throw new Error(`API error: ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let meta: Partial<QueryResponse> = {};
      const startedAt = performance.now();

      setStreamingAnswer("");

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.replace(/^data: /, "").trim();
          if (!line) continue;
          if (line === "[DONE]") continue;

          try {
            const evt = JSON.parse(line);
            if (evt.meta) {
              meta = {
                pipeline: evt.meta.pipeline,
                intent_label: evt.meta.pipeline,
                intent_confidence: evt.meta.intent_confidence,
                retrieved_chunks: (evt.meta.retrieved_chunks || []).map((c: any) => ({
                  chunk_id: c.chunk_id,
                  content: "",
                  source_type: c.source_type,
                  score: c.score,
                  chunk: { content: "", source_type: c.source_type },
                })),
              };
            } else if (evt.token) {
              answer += evt.token;
              setStreamingAnswer(answer);
            } else if (evt.error) {
              throw new Error(evt.error);
            }
          } catch {
            // ignore malformed partial JSON chunk
          }
        }
      }

      setResult({
        answer,
        pipeline: meta.pipeline || "naive",
        intent_label: meta.intent_label || meta.pipeline || "simple",
        intent_confidence: meta.intent_confidence ?? 1,
        retrieved_chunks: meta.retrieved_chunks || [],
        latency_ms: Math.round(performance.now() - startedAt),
        reranker,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Streaming failed");
    } finally {
      setLoading(false);
      setStreamingAnswer(null);
    }
  }

  const friendlyError = error ? toFriendlyError(new Error(error)) : null;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="h-14 flex items-center justify-between px-6 border-b border-border shrink-0">
        <h1 className="text-lg font-semibold tracking-tight">Playground</h1>
        {configHistory.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={undoLastConfig}
            aria-label="Undo last config change"
            title="Revert to the previous config used"
          >
            <RotateCcw className="h-3.5 w-3.5 mr-1.5" /> Undo config ({configHistory.length})
          </Button>
        )}
      </header>

      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left Panel — Config */}
        <aside className="w-full md:w-[300px] max-h-[45vh] md:max-h-none border-b md:border-b-0 md:border-r border-border p-5 overflow-y-auto space-y-6 shrink-0">
          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Preset
            </Label>
            <Select value={selectedPreset} onValueChange={(v) => v && applyPreset(v)}>
              <SelectTrigger><SelectValue placeholder="Choose a starting point…" /></SelectTrigger>
              <SelectContent>
                {presets.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex rounded-lg border border-border p-1 gap-1">
            <button
              onClick={() => setAdvanced(false)}
              className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-colors ${!advanced ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Simple
            </button>
            <button
              onClick={() => setAdvanced(true)}
              className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-colors ${advanced ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Advanced
            </button>
          </div>

          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
              Index Backend <ParamTooltip param="index_backend" />
            </Label>
            <Select value={indexBackend} onValueChange={(v) => v && setIndexBackend(v)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="chroma">Chroma (dense)</SelectItem>
                <SelectItem value="bm25">BM25 (sparse)</SelectItem>
                <SelectItem value="hybrid_rrf">Hybrid RRF</SelectItem>
                <SelectItem value="hybrid_weighted">Hybrid Weighted</SelectItem>
                <SelectItem value="pageindex">PageIndex</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Pipeline
            </Label>
            <Select value={pipelineOverride} onValueChange={(v) => v && setPipelineOverride(v)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto (classify intent)</SelectItem>
                <SelectItem value="naive">Force Naive</SelectItem>
                <SelectItem value="agentic">Force Agentic</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
              Top K: {topK} <ParamTooltip param="top_k" />
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
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
              Chunking Strategy <ParamTooltip param="chunk_strategy" />
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
            <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
              LLM Model <ParamTooltip param="llm_model" />
            </Label>
            <Input value={llmModel} onChange={(e) => setLlmModel(e.target.value)} placeholder="llama3" />
          </div>

          {advanced && (
            <>
              <div className="space-y-3">
                <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
                  LLM Provider <ParamTooltip param="llm_provider" />
                </Label>
                <Select value={llmProvider} onValueChange={(v) => v && setLlmProvider(v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ollama">Ollama</SelectItem>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="groq">Groq</SelectItem>
                    <SelectItem value="hf">HuggingFace</SelectItem>
                    <SelectItem value="lmstudio">LM Studio</SelectItem>
                  </SelectContent>
                </Select>
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
                <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
                  Reranker <ParamTooltip param="reranker" />
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

              <div className="space-y-3">
                <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
                  Chunk Size: {chunkTokens} <ParamTooltip param="chunk_tokens" />
                </Label>
                <Slider
                  value={[chunkTokens]}
                  onValueChange={(v) => setChunkTokens(Array.isArray(v) ? v[0] : v)}
                  min={64}
                  max={1024}
                  step={64}
                />
                <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
                  Overlap: {overlap} <ParamTooltip param="overlap" />
                </Label>
                <Slider
                  value={[overlap]}
                  onValueChange={(v) => setOverlap(Array.isArray(v) ? v[0] : v)}
                  min={0}
                  max={200}
                  step={10}
                />
                {chunkEstimate && (
                  <p className="text-xs text-muted-foreground">
                    ≈ {chunkEstimate.estimated_chunks.toLocaleString()} chunks across{" "}
                    {chunkEstimate.n_documents} docs
                    {chunkEstimate.approximate ? " (approximate for this strategy)" : ""}
                  </p>
                )}
                {estimateError && (
                  <p className="text-xs text-destructive">{estimateError}</p>
                )}
              </div>

              <div className="space-y-3">
                <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Streaming
                </Label>
                <Button
                  type="button"
                  variant={streaming ? "default" : "outline"}
                  size="sm"
                  className="w-full"
                  onClick={() => setStreaming((s) => !s)}
                >
                  {streaming ? "On — live token stream" : "Off — full response"}
                </Button>
              </div>
            </>
          )}
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
                  aria-label="Submit query"
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

          {/* Streaming answer (live tokens) */}
          {loading && streamingAnswer !== null && (
            <Card className="shadow-sm border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Answer</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {streamingAnswer}
                  <span className="inline-block w-1.5 h-4 bg-primary/70 ml-0.5 animate-pulse align-middle" />
                </p>
              </CardContent>
            </Card>
          )}

          {/* Loading State */}
          {loading && streamingAnswer === null && (
            <Card className="shadow-sm border-border">
              <CardContent className="pt-5 space-y-3">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-4 w-2/3" />
              </CardContent>
            </Card>
          )}

          {/* Error */}
          {friendlyError && (
            <Card className="shadow-sm border-destructive/50 bg-destructive/5">
              <CardContent className="pt-5 flex items-start gap-3">
                <AlertCircle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-destructive">{friendlyError.title}</p>
                  <p className="text-xs text-destructive/80 mt-1">{friendlyError.description}</p>
                </div>
                {friendlyError.retryable && (
                  <Button variant="outline" size="sm" onClick={handleSubmit}>
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Retry
                  </Button>
                )}
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
                className="grid grid-cols-3 gap-4"
              >
                <div className="col-span-2 space-y-4">
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
                      <CitationText
                        text={result.answer}
                        chunks={result.retrieved_chunks.map((rc) => ({
                          chunk_id: rc.chunk_id,
                          content: rc.content || rc.chunk?.content || "",
                          source_type: rc.source_type || rc.chunk?.source_type || "",
                          score: rc.score,
                        }))}
                      />
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
                            id={`chunk-${rc.chunk_id}`}
                            className="p-3 rounded-lg bg-muted/50 border border-border"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <Badge variant="secondary" className="text-xs">
                                {rc.chunk?.source_type || rc.source_type}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                score: {rc.score.toFixed(3)}
                              </span>
                            </div>
                            <p className="text-xs text-muted-foreground line-clamp-3">
                              {rc.chunk?.content || rc.content}
                            </p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  )}
                </div>

                {/* Pipeline Story (Skill 38C) — default open, not collapsed */}
                <Card className="shadow-sm border-border h-fit">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Pipeline Story</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <PipelineStory result={result} />
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
