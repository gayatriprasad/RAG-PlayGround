"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { toast } from "sonner";
import {
  Trophy,
  Clock,
  Target,
  AlertCircle,
  Download,
  Save,
  Link as LinkIcon,
  ChevronDown,
  ChevronRight,
  Trash2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { CalibrationCard } from "@/components/benchmark/CalibrationCard";
import { apiGet, API_BASE } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";
import {
  categoryInsight,
  pipelineInsight,
  failureHypothesis,
  suggestRunName,
  type BenchmarkRow,
} from "@/lib/insights";

interface ExperimentMeta {
  name: string;
  config_path: string;
  has_results: boolean;
}

interface BenchmarkResults {
  experiment: string;
  total_questions: number;
  average_score: number;
  by_category: Record<string, { count: number; avg_score: number }>;
  by_pipeline: Record<string, { count: number; avg_score: number }>;
  by_source_type: Record<string, { count: number; avg_score: number }>;
  rows: BenchmarkRow[];
}

interface SavedRun {
  id: string;
  name: string;
  experiment: string;
  average_score: number;
  total_questions: number;
  saved_at: number;
}

const CHART_COLORS = ["#0066CC", "#34C759", "#FF9500", "#AF52DE", "#FF2D55"];
const SAVED_RUNS_KEY = "nb_saved_runs";
const EXPORT_FORMATS: { format: "markdown" | "csv" | "html" | "json"; label: string }[] = [
  { format: "markdown", label: "Markdown" },
  { format: "csv", label: "CSV" },
  { format: "html", label: "HTML" },
  { format: "json", label: "JSON" },
];

function loadSavedRuns(): SavedRun[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(window.localStorage.getItem(SAVED_RUNS_KEY) || "[]");
  } catch {
    return [];
  }
}

export default function BenchmarkPage() {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [results, setResults] = useState<BenchmarkResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Baseline comparison (Skill 39A)
  const [baselineExp, setBaselineExp] = useState<string>("");
  const [baselineResults, setBaselineResults] = useState<BenchmarkResults | null>(null);

  // Failure analysis table (Skill 39C)
  const [showOnlyFailures, setShowOnlyFailures] = useState(false);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  // Saved runs (Skill 39D)
  const [savedRuns, setSavedRuns] = useState<SavedRun[]>([]);

  useEffect(() => {
    setSavedRuns(loadSavedRuns());
    apiGet<{ experiments: ExperimentMeta[] }>("/experiments")
      .then((data) => {
        setExperiments(data.experiments);
        if (data.experiments.length > 0) {
          setSelectedExp(data.experiments[0].name);
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedExp) return;
    setLoading(true);
    setError(null);
    apiGet<BenchmarkResults>(`/benchmark/results?experiment=${encodeURIComponent(selectedExp)}`)
      .then(setResults)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [selectedExp]);

  useEffect(() => {
    if (!baselineExp) {
      setBaselineResults(null);
      return;
    }
    apiGet<BenchmarkResults>(`/benchmark/results?experiment=${encodeURIComponent(baselineExp)}`)
      .then(setBaselineResults)
      .catch(() => setBaselineResults(null));
  }, [baselineExp]);

  function saveCurrentRun() {
    if (!results) return;
    const dominantPipeline = Object.entries(results.by_pipeline).sort(
      (a, b) => b[1].count - a[1].count
    )[0]?.[0];
    const dominantIndex = results.rows[0]?.index_backend;
    const name = suggestRunName({
      index_backend: dominantIndex,
      pipeline: dominantPipeline,
    });
    const run: SavedRun = {
      id: crypto.randomUUID(),
      name,
      experiment: results.experiment,
      average_score: results.average_score,
      total_questions: results.total_questions,
      saved_at: Date.now(),
    };
    const next = [run, ...savedRuns].slice(0, 20);
    setSavedRuns(next);
    window.localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next));
    toast.success(`Saved run "${name}"`);
  }

  function deleteSavedRun(id: string) {
    const next = savedRuns.filter((r) => r.id !== id);
    setSavedRuns(next);
    window.localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next));
  }

  async function copyShareLink() {
    if (!selectedExp) return;
    try {
      const data = await apiGet<{ token: string; url: string }>(
        `/share/config?experiment=${encodeURIComponent(selectedExp)}`
      );
      const fullUrl = `${window.location.origin}${data.url}`;
      await navigator.clipboard.writeText(fullUrl);
      toast.success("Share link copied to clipboard");
    } catch (e) {
      toast.error(toFriendlyError(e).title);
    }
  }

  const categoryData = results
    ? Object.entries(results.by_category).map(([name, v]) => ({
        name,
        score: +(v.avg_score * 100).toFixed(1),
        count: v.count,
      }))
    : [];

  const pipelineData = results
    ? Object.entries(results.by_pipeline).map(([name, v]) => ({
        name,
        score: +(v.avg_score * 100).toFixed(1),
        count: v.count,
      }))
    : [];

  const catInsight = results ? categoryInsight(results.by_category) : null;
  const pipeInsight = results ? pipelineInsight(results.by_pipeline) : null;
  const baselineScorePct = baselineResults ? +(baselineResults.average_score * 100).toFixed(1) : null;
  const deltaVsBaseline =
    results && baselineResults ? (results.average_score - baselineResults.average_score) * 100 : null;

  const visibleRows = results
    ? showOnlyFailures
      ? results.rows.filter((r) => !(r.answer_correct === true || r.answer_correct === "True"))
      : results.rows
    : [];

  return (
    <div className="h-full flex flex-col">
      <header className="h-14 flex items-center justify-between px-6 border-b border-border shrink-0 gap-3">
        <h1 className="text-lg font-semibold tracking-tight shrink-0">Benchmark Results</h1>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {selectedExp && (
            <>
              <Button variant="outline" size="sm" onClick={saveCurrentRun} disabled={!results}>
                <Save className="h-4 w-4 mr-1.5" /> Save run
              </Button>
              <Button variant="outline" size="sm" onClick={copyShareLink}>
                <LinkIcon className="h-4 w-4 mr-1.5" /> Copy link
              </Button>
              {EXPORT_FORMATS.map((f) => (
                <a
                  key={f.format}
                  href={`${API_BASE}/export/run/${encodeURIComponent(selectedExp)}?format=${f.format}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <Button variant="outline" size="sm">
                    <Download className="h-3.5 w-3.5 mr-1" /> {f.label}
                  </Button>
                </a>
              ))}
            </>
          )}
          <Select value={baselineExp} onValueChange={(v) => setBaselineExp(v === "__none__" ? "" : v || "")}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="vs baseline…" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__none__">No baseline</SelectItem>
              {experiments.map((exp) => (
                <SelectItem key={exp.name} value={exp.name}>
                  {exp.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={selectedExp} onValueChange={(v) => v && setSelectedExp(v)}>
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="Select experiment" />
            </SelectTrigger>
            <SelectContent>
              {experiments.map((exp) => (
                <SelectItem key={exp.name} value={exp.name}>
                  {exp.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {loading && (
          <div className="grid grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-28 rounded-xl" />
            ))}
          </div>
        )}

        {error && (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="pt-5 flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
              <div>
                <p className="text-sm font-medium text-destructive">{toFriendlyError(error).title}</p>
                <p className="text-xs text-destructive/80 mt-1">{toFriendlyError(error).description}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {!loading && !error && experiments.length === 0 && (
          <EmptyState
            title="No experiments found"
            description="Run an experiment via the CLI (make eval) to see benchmark results here."
          />
        )}

        {results && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
            className="space-y-6"
          >
            {/* Score Cards */}
            <div className="grid grid-cols-3 gap-4">
              <Card className="shadow-sm border-border">
                <CardContent className="pt-5">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Trophy className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-2xl font-semibold">
                          {(results.average_score * 100).toFixed(1)}%
                        </p>
                        {deltaVsBaseline !== null && (
                          <Badge
                            variant="outline"
                            className={deltaVsBaseline >= 0 ? "text-emerald-600 border-emerald-600/30" : "text-destructive border-destructive/30"}
                          >
                            {deltaVsBaseline >= 0 ? "+" : ""}
                            {deltaVsBaseline.toFixed(1)} pts vs baseline
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Avg Score{deltaVsBaseline !== null ? " (delta not statistically tested)" : ""}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-sm border-border">
                <CardContent className="pt-5">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-chart-2/10 flex items-center justify-center">
                      <Target className="h-5 w-5 text-chart-2" />
                    </div>
                    <div>
                      <p className="text-2xl font-semibold">
                        {results.total_questions}
                      </p>
                      <p className="text-xs text-muted-foreground">Questions</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-sm border-border">
                <CardContent className="pt-5">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center">
                      <Clock className="h-5 w-5 text-chart-3" />
                    </div>
                    <div>
                      <p className="text-2xl font-semibold">
                        {Object.keys(results.by_category).length}
                      </p>
                      <p className="text-xs text-muted-foreground">Categories</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-6">
              {/* By Category */}
              <Card className="shadow-sm border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Score by Category</CardTitle>
                  {catInsight && (
                    <p className="text-xs text-muted-foreground leading-relaxed mt-1">{catInsight}</p>
                  )}
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={categoryData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          borderRadius: 8,
                          border: "1px solid #E5E5E5",
                          fontSize: 12,
                        }}
                      />
                      {baselineScorePct !== null && (
                        <ReferenceLine
                          y={baselineScorePct}
                          stroke="#FF3B30"
                          strokeDasharray="4 4"
                          label={{ value: `baseline ${baselineScorePct}%`, fontSize: 10, position: "insideTopRight" }}
                        />
                      )}
                      <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                        {categoryData.map((_, i) => (
                          <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* By Pipeline */}
              <Card className="shadow-sm border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Score by Pipeline</CardTitle>
                  {pipeInsight && (
                    <p className="text-xs text-muted-foreground leading-relaxed mt-1">{pipeInsight}</p>
                  )}
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={pipelineData} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          borderRadius: 8,
                          border: "1px solid #E5E5E5",
                          fontSize: 12,
                        }}
                      />
                      {baselineScorePct !== null && (
                        <ReferenceLine
                          y={baselineScorePct}
                          stroke="#FF3B30"
                          strokeDasharray="4 4"
                          label={{ value: `baseline ${baselineScorePct}%`, fontSize: 10, position: "insideTopRight" }}
                        />
                      )}
                      <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                        {pipelineData.map((_, i) => (
                          <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Source Type Breakdown */}
            {results.by_source_type && Object.keys(results.by_source_type).length > 0 && (
              <Card className="shadow-sm border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">By Source Type</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-3">
                    {Object.entries(results.by_source_type).map(([type, v]) => (
                      <div
                        key={type}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/50 border border-border"
                      >
                        <Badge variant="secondary" className="text-xs">{type}</Badge>
                        <span className="text-sm font-medium">
                          {(v.avg_score * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-muted-foreground">({v.count})</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Confidence Calibration (Skill 57) */}
            <CalibrationCard experiment={selectedExp} />

            {/* Failure Analysis Table (Skill 39C) */}
            {results.rows && results.rows.length > 0 && (
              <Card className="shadow-sm border-border">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">
                      Question Results ({visibleRows.length})
                    </CardTitle>
                    <Button
                      variant={showOnlyFailures ? "default" : "outline"}
                      size="sm"
                      onClick={() => setShowOnlyFailures((s) => !s)}
                    >
                      {showOnlyFailures ? "Showing failures only" : "Show only failures"}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-1">
                  {visibleRows.length === 0 ? (
                    <p className="text-sm text-muted-foreground py-4 text-center">
                      No failing questions — everything answered correctly.
                    </p>
                  ) : (
                    visibleRows.map((row) => {
                      const correct = row.answer_correct === true || row.answer_correct === "True";
                      const isOpen = expandedRow === row.question_id;
                      return (
                        <div key={row.question_id} className="border border-border rounded-lg overflow-hidden">
                          <button
                            className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-muted/50 transition-colors"
                            onClick={() => setExpandedRow(isOpen ? null : row.question_id)}
                          >
                            {isOpen ? (
                              <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                            )}
                            <Badge variant={correct ? "secondary" : "outline"} className={correct ? "text-xs" : "text-xs text-destructive border-destructive/30"}>
                              {correct ? "correct" : "incorrect"}
                            </Badge>
                            <span className="text-xs text-muted-foreground shrink-0">{row.category}</span>
                            <span className="text-xs text-muted-foreground shrink-0">{row.pipeline}</span>
                            <span className="text-xs font-mono truncate flex-1">{row.question_id}</span>
                            <span className="text-xs text-muted-foreground shrink-0">
                              {(row.overall_score * 100).toFixed(0)}%
                            </span>
                          </button>
                          {isOpen && (
                            <div className="px-3 pb-3 pt-1 border-t border-border bg-muted/30">
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                {failureHypothesis(row)}
                              </p>
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}
                </CardContent>
              </Card>
            )}

            {/* Saved Runs (Skill 39D) */}
            {savedRuns.length > 0 && (
              <Card className="shadow-sm border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Saved Runs</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {savedRuns.map((run) => (
                    <div
                      key={run.id}
                      className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/50 border border-border"
                    >
                      <button
                        className="flex items-center gap-3 text-left flex-1"
                        onClick={() => setSelectedExp(run.experiment)}
                      >
                        <span className="text-sm font-medium">{run.name}</span>
                        <span className="text-xs text-muted-foreground">{run.experiment}</span>
                        <span className="text-xs text-muted-foreground">
                          {(run.average_score * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {new Date(run.saved_at).toLocaleString()}
                        </span>
                      </button>
                      <Button variant="ghost" size="sm" onClick={() => deleteSavedRun(run.id)}>
                        <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                      </Button>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
}

