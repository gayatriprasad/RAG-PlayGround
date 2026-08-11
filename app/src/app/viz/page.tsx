"use client";

import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";
import { Info, X } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";

interface EmbeddingPoint {
  x: number;
  y: number;
  id: string;
  source_type: string;
  is_query: boolean;
  chunk_preview: string;
  trust_score: number;
}

interface EmbeddingData {
  points: EmbeddingPoint[];
  method: string;
  n_chunks: number;
  n_queries: number;
}

interface ChunkPreview {
  index: number;
  content: string;
  n_chars: number;
}

interface ChunkingStrategyResult {
  n_chunks?: number;
  chunks?: ChunkPreview[];
  mid_clause_splits?: number;
  mid_list_splits?: number;
  error?: string;
}

const PALETTE = ["#0066CC", "#34C759", "#FF9500", "#AF52DE", "#FF2D55", "#5AC8FA", "#8E8E93"];

function colorForCategory(value: string, categories: string[]): string {
  const idx = categories.indexOf(value);
  return PALETTE[idx % PALETTE.length];
}

function colorForTrust(score: number): string {
  // 0 = red, 1 = green
  const hue = Math.max(0, Math.min(1, score)) * 120;
  return `hsl(${hue}, 70%, 45%)`;
}

/** Euclidean distance in the 2D projection — used for the query-nearest-chunk insight. */
function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export default function VizPage() {
  const [projectionMethod, setProjectionMethod] = useState("umap");
  const [colorBy, setColorBy] = useState<"source_type" | "trust_score">("source_type");
  const [showInfoCard, setShowInfoCard] = useState(true);
  const [query, setQuery] = useState("");
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [loadingEmbedding, setLoadingEmbedding] = useState(false);
  const [embeddingError, setEmbeddingError] = useState<string | null>(null);

  const [chunkingDoc, setChunkingDoc] = useState("");
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(["fixed", "sentence"]);
  const [chunkingData, setChunkingData] = useState<Record<string, ChunkingStrategyResult> | null>(null);
  const [loadingChunking, setLoadingChunking] = useState(false);
  const [chunkingError, setChunkingError] = useState<string | null>(null);

  const strategies = ["fixed", "sentence", "semantic", "recursive"];

  const loadEmbeddingSpace = async () => {
    setLoadingEmbedding(true);
    setEmbeddingError(null);
    try {
      const response = await fetch(`${API_BASE}/viz/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method: projectionMethod,
          queries: query.trim() ? [query.trim()] : [],
        }),
      });

      if (response.ok) {
        setEmbeddingData(await response.json());
      } else {
        setEmbeddingError(`API error: ${response.status} ${await response.text()}`);
      }
    } catch (error) {
      setEmbeddingError(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setLoadingEmbedding(false);
    }
  };

  const visualizeChunking = async () => {
    if (!chunkingDoc) return;

    setLoadingChunking(true);
    setChunkingError(null);
    try {
      const response = await fetch(`${API_BASE}/viz/chunking`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document: chunkingDoc,
          strategies: selectedStrategies,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setChunkingData(data.strategies);
      } else {
        setChunkingError(`API error: ${response.status} ${await response.text()}`);
      }
    } catch (error) {
      setChunkingError(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setLoadingChunking(false);
    }
  };

  const toggleStrategy = (strategy: string) => {
    setSelectedStrategies((prev) => {
      if (prev.includes(strategy)) {
        return prev.filter((s) => s !== strategy);
      } else if (prev.length < 3) {
        return [...prev, strategy];
      }
      return prev;
    });
  };

  const sourceTypes = useMemo(
    () =>
      embeddingData
        ? Array.from(new Set(embeddingData.points.filter((p) => !p.is_query).map((p) => p.source_type))).sort()
        : [],
    [embeddingData]
  );

  /** Auto-insight caption for the embedding plot (Skill 40B) — deterministic, not LLM-based. */
  const embeddingInsight = useMemo(() => {
    if (!embeddingData) return null;
    const chunkPoints = embeddingData.points.filter((p) => !p.is_query);
    const queryPoints = embeddingData.points.filter((p) => p.is_query);

    if (queryPoints.length > 0 && chunkPoints.length > 0) {
      const q = queryPoints[0];
      const nearest = [...chunkPoints].sort((a, b) => distance(a, q) - distance(b, q))[0];
      return `Your query's nearest chunk in this 2D projection is from "${nearest.source_type}" — ` +
        `distance is only approximate since ${embeddingData.method.toUpperCase()} distorts real embedding-space distances.`;
    }
    if (sourceTypes.length > 1) {
      return `${chunkPoints.length} chunks projected across ${sourceTypes.length} source types (${sourceTypes.join(", ")}) using ${embeddingData.method.toUpperCase()}.`;
    }
    return `${chunkPoints.length} chunks projected using ${embeddingData.method.toUpperCase()}.`;
  }, [embeddingData, sourceTypes]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Visualization Lab</h1>
        <p className="text-muted-foreground">Explore embedding space and chunking strategies</p>
      </div>

      <Tabs defaultValue="embedding">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="embedding">Embedding Space</TabsTrigger>
          <TabsTrigger value="chunking">Chunking Preview</TabsTrigger>
        </TabsList>

        {/* Tab 1: Embedding Space */}
        <TabsContent value="embedding" className="space-y-4">
          {showInfoCard && (
            <Card className="border-primary/30 bg-primary/5">
              <CardContent className="pt-5 flex items-start gap-3">
                <Info className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                <div className="flex-1 text-sm text-muted-foreground space-y-1">
                  <p className="font-medium text-foreground">What am I looking at?</p>
                  <p>
                    Each point is a document chunk, projected from high-dimensional embedding
                    space down to 2D using {projectionMethod.toUpperCase()}. Points that sit close
                    together were embedded as semantically similar by the model — clusters often
                    correspond to a source type or topic. If you run a query, it appears as a
                    highlighted point, and its nearest neighbours are the chunks most likely to be
                    retrieved for that question.
                  </p>
                </div>
                <button
                  onClick={() => setShowInfoCard(false)}
                  aria-label="Dismiss explanation"
                  className="text-muted-foreground hover:text-foreground shrink-0"
                >
                  <X className="h-4 w-4" />
                </button>
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader>
              <CardTitle>2D Embedding Projection</CardTitle>
              <CardDescription>
                Visualize semantic relationships between chunks
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="method">Projection Method</Label>
                  <Select value={projectionMethod} onValueChange={(v) => v && setProjectionMethod(v)}>
                    <SelectTrigger id="method">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="umap">UMAP (best overall)</SelectItem>
                      <SelectItem value="tsne">t-SNE (local clusters)</SelectItem>
                      <SelectItem value="pca">PCA (fast, linear)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="color">Color By</Label>
                  <Select value={colorBy} onValueChange={(v) => v && setColorBy(v as "source_type" | "trust_score")}>
                    <SelectTrigger id="color">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="source_type">Source Type</SelectItem>
                      <SelectItem value="trust_score">Trust Score</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="query_embed">Query (optional)</Label>
                  <Input
                    id="query_embed"
                    placeholder="Overlay query on plot"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
              </div>

              <Button
                onClick={loadEmbeddingSpace}
                disabled={loadingEmbedding}
                className="w-full"
              >
                {loadingEmbedding ? "Projecting..." : "Load Embedding Space"}
              </Button>

              {embeddingError && (
                <p className="text-sm text-destructive">{toFriendlyError(embeddingError).description}</p>
              )}

              {embeddingData && (
                <div className="border rounded-lg p-4 bg-muted/10">
                  <div className="flex justify-between mb-2">
                    <div>
                      <p className="text-sm font-medium">
                        {embeddingData.n_chunks} chunks, {embeddingData.n_queries} queries
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Method: {embeddingData.method}
                      </p>
                    </div>
                  </div>

                  {embeddingInsight && (
                    <p className="text-xs text-muted-foreground leading-relaxed mb-3">{embeddingInsight}</p>
                  )}

                  <ResponsiveContainer width="100%" height={384}>
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                      <XAxis type="number" dataKey="x" tick={{ fontSize: 11 }} name="x" />
                      <YAxis type="number" dataKey="y" tick={{ fontSize: 11 }} name="y" />
                      <ZAxis range={[40, 40]} />
                      <RechartsTooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={({ active, payload }) => {
                          if (!active || !payload || !payload.length) return null;
                          const p = payload[0].payload as EmbeddingPoint;
                          return (
                            <div className="bg-foreground text-background text-xs rounded-md px-2.5 py-1.5 shadow-md max-w-64">
                              <p className="font-medium mb-0.5">
                                {p.is_query ? "Query" : p.source_type}
                                {!p.is_query && ` · trust ${p.trust_score.toFixed(2)}`}
                              </p>
                              <p className="text-background/80 line-clamp-3">{p.chunk_preview}</p>
                            </div>
                          );
                        }}
                      />
                      <Scatter data={embeddingData.points.filter((p) => !p.is_query)} shape="circle">
                        {embeddingData.points
                          .filter((p) => !p.is_query)
                          .map((p, i) => (
                            <Cell
                              key={i}
                              fill={colorBy === "trust_score" ? colorForTrust(p.trust_score) : colorForCategory(p.source_type, sourceTypes)}
                            />
                          ))}
                      </Scatter>
                      <Scatter
                        data={embeddingData.points.filter((p) => p.is_query)}
                        shape="star"
                        fill="#FF3B30"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>

                  {/* Legend */}
                  <div className="mt-4 flex gap-2 flex-wrap">
                    {colorBy === "source_type" ? (
                      sourceTypes.map((source) => (
                        <Badge key={source} variant="outline">
                          <span
                            className="w-3 h-3 rounded-full inline-block mr-1"
                            style={{ backgroundColor: colorForCategory(source, sourceTypes) }}
                          />
                          {source}
                        </Badge>
                      ))
                    ) : (
                      <>
                        <Badge variant="outline">
                          <span className="w-3 h-3 rounded-full inline-block mr-1" style={{ backgroundColor: colorForTrust(0) }} />
                          low trust
                        </Badge>
                        <Badge variant="outline">
                          <span className="w-3 h-3 rounded-full inline-block mr-1" style={{ backgroundColor: colorForTrust(1) }} />
                          high trust
                        </Badge>
                      </>
                    )}
                    {embeddingData.n_queries > 0 && (
                      <Badge variant="outline">
                        <span className="w-3 h-3 rounded-full inline-block mr-1 bg-[#FF3B30]" />
                        query
                      </Badge>
                    )}
                  </div>
                </div>
              )}

              {!embeddingData && !loadingEmbedding && !embeddingError && (
                <EmptyState
                  title="No projection yet"
                  description="Click Load Embedding Space to project this experiment's chunk embeddings to 2D."
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 2: Chunking Preview */}
        <TabsContent value="chunking" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Chunking Strategy Comparison</CardTitle>
              <CardDescription>
                See how different strategies split your document
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="doc">Document Text</Label>
                <textarea
                  id="doc"
                  className="w-full h-32 p-3 mt-1 border rounded-md font-mono text-sm"
                  placeholder="Paste document text to visualize chunking strategies..."
                  value={chunkingDoc}
                  onChange={(e) => setChunkingDoc(e.target.value)}
                />
              </div>

              <div>
                <Label>Strategies (select up to 3)</Label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                  {strategies.map((strategy) => (
                    <div
                      key={strategy}
                      className={`p-3 rounded-lg border cursor-pointer text-center transition-colors ${
                        selectedStrategies.includes(strategy)
                          ? "border-primary bg-primary/10"
                          : "border-border hover:border-primary/50"
                      } ${selectedStrategies.length >= 3 && !selectedStrategies.includes(strategy) ? "opacity-50 cursor-not-allowed" : ""}`}
                      onClick={() => toggleStrategy(strategy)}
                    >
                      <span className="font-medium capitalize">{strategy}</span>
                    </div>
                  ))}
                </div>
              </div>

              <Button
                onClick={visualizeChunking}
                disabled={loadingChunking || !chunkingDoc || selectedStrategies.length === 0}
                className="w-full"
              >
                {loadingChunking ? "Processing..." : `Visualize (${selectedStrategies.length} strategies)`}
              </Button>

              {chunkingError && (
                <p className="text-sm text-destructive">{toFriendlyError(chunkingError).description}</p>
              )}

              {chunkingData && (
                <div className="space-y-4">
                  {selectedStrategies.map((strategy) => {
                    const result = chunkingData[strategy];
                    if (!result || result.error) {
                      return (
                        <Card key={strategy}>
                          <CardHeader>
                            <CardTitle className="text-lg capitalize">{strategy}</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm text-destructive">{result?.error || "No result"}</p>
                          </CardContent>
                        </Card>
                      );
                    }
                    const chunks = result.chunks || [];
                    const avgChars = chunks.length
                      ? Math.round(chunks.reduce((s, c) => s + c.n_chars, 0) / chunks.length)
                      : 0;
                    const hasBoundaryIssues = (result.mid_clause_splits ?? 0) > 0 || (result.mid_list_splits ?? 0) > 0;
                    return (
                      <Card key={strategy}>
                        <CardHeader>
                          <CardTitle className="text-lg capitalize">{strategy}</CardTitle>
                          <CardDescription>
                            {chunks.length} chunks, avg {avgChars} chars/chunk
                          </CardDescription>
                          {/* Boundary quality diagnosis caption (Skill 40C) */}
                          <p className={`text-xs mt-1 ${hasBoundaryIssues ? "text-amber-600" : "text-muted-foreground"}`}>
                            {hasBoundaryIssues
                              ? `⚠ ${result.mid_clause_splits ?? 0} boundary(ies) cut mid-sentence` +
                                (result.mid_list_splits ? `, ${result.mid_list_splits} cut mid-list-item` : "") +
                                " — meaning may be lost at those chunk edges."
                              : "No mid-sentence or mid-list-item boundary splits detected."}
                          </p>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2 max-h-60 overflow-y-auto">
                            {chunks.map((chunk, idx) => (
                              <div
                                key={idx}
                                className="p-3 rounded border"
                                style={{
                                  backgroundColor: `hsl(${(idx * 360) / chunks.length}, 70%, 95%)`,
                                  borderColor: `hsl(${(idx * 360) / chunks.length}, 70%, 70%)`,
                                }}
                              >
                                <div className="flex justify-between mb-1">
                                  <Badge variant="outline">Chunk {idx + 1}</Badge>
                                  <Badge variant="secondary">{chunk.n_chars} chars</Badge>
                                </div>
                                <p className="text-sm font-mono">{chunk.content}</p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

