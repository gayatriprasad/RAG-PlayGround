"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { AlertTriangle, CheckCircle2, Circle, Loader2, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { apiGet, apiPost } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";
import { cn } from "@/lib/utils";

interface HeatmapSlice {
  source_type: string;
  category: string;
  recall_at_3: number;
  n_questions: number;
  gap: boolean;
}

interface HeatmapResponse {
  experiment: string;
  min_recall_threshold: number;
  slices: HeatmapSlice[];
}

interface StatusResponse {
  experiment: string;
  should_run: boolean;
  reason: string;
  gap_slices: { source_type: string; category: string; recall_at_3: string }[];
  auto_trigger: boolean;
}

interface ReportSummary {
  iteration: number;
  n_pairs_passed_validation: number;
  recommendation: string;
  delta: number | null;
  significant: boolean | null;
}

const LOOP_STEPS = [
  "Diagnose recall gaps",
  "Generate synthetic pairs",
  "Fine-tune embeddings",
  "Re-index",
  "Re-benchmark",
  "Compare significance",
  "Check prompt regression",
  "Version report",
];

export default function ImprovePage() {
  const [experiment, setExperiment] = useState("");
  const [heatmap, setHeatmap] = useState<HeatmapResponse | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selectedCell, setSelectedCell] = useState<HeatmapSlice | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(() => {
    const exp = experiment.trim();
    if (!exp) {
      setHeatmap(null);
      setStatus(null);
      setReports([]);
      return Promise.resolve();
    }
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ experiment: exp });
    return Promise.all([
      apiGet<HeatmapResponse>(`/improve/heatmap?${params.toString()}`),
      apiGet<StatusResponse>(`/improve/status?${params.toString()}`),
      apiGet<{ reports: ReportSummary[] }>(`/improve/reports?${params.toString()}`),
    ])
      .then(([heatmapData, statusData, reportsData]) => {
        setHeatmap(heatmapData);
        setStatus(statusData);
        setReports(reportsData.reports);
        setSelectedCell(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load improvement data");
        setHeatmap(null);
        setStatus(null);
        setReports([]);
      })
      .finally(() => setLoading(false));
  }, [experiment]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const runCycle = async (full: boolean) => {
    const exp = experiment.trim();
    if (!exp) return;
    setRunning(true);
    setError(null);
    try {
      const params = new URLSearchParams({ experiment: exp, full: String(full) });
      await apiPost(`/improve/run?${params.toString()}`, {});
      toast.success(full ? "Improvement cycle complete" : "Diagnosis complete");
      await fetchAll();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Improvement run failed";
      setError(message);
      toast.error(toFriendlyError(err instanceof Error ? err : new Error(message)).title);
    } finally {
      setRunning(false);
    }
  };

  const sourceTypes = Array.from(new Set(heatmap?.slices.map((s) => s.source_type) ?? [])).sort();
  const categories = Array.from(new Set(heatmap?.slices.map((s) => s.category) ?? [])).sort();
  const cellFor = (sourceType: string, category: string) =>
    heatmap?.slices.find((s) => s.source_type === sourceType && s.category === category);

  return (
    <div className="max-w-5xl mx-auto py-10 px-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Improve</h1>
        <p className="text-sm text-muted-foreground mt-1">
          The self-improving flywheel — diagnose recall gaps, generate targeted synthetic pairs,
          fine-tune embeddings, re-index, re-benchmark, and report the delta with statistical
          significance. Nothing is overwritten; every iteration is versioned.
        </p>
      </div>

      <div className="flex items-end gap-3 max-w-md">
        <div className="flex-1 space-y-1.5">
          <Label htmlFor="experiment">Experiment</Label>
          <Input
            id="experiment"
            placeholder="e.g. 02_retrieval_comparison"
            value={experiment}
            onChange={(e) => setExperiment(e.target.value)}
          />
        </div>
        <Button variant="outline" onClick={() => fetchAll()} disabled={loading || !experiment.trim()}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
        </Button>
      </div>

      {error && (
        <p className="text-sm text-destructive">{toFriendlyError(new Error(error)).description}</p>
      )}

      {/* Panel 1 — Current state: recall heatmap */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Recall Heatmap</CardTitle>
          <CardDescription>
            source_type × category → recall@3. Red cells fall below the{" "}
            {heatmap ? heatmap.min_recall_threshold.toFixed(2) : "—"} threshold.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!heatmap || heatmap.slices.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {experiment.trim() ? "No scored slices found for this experiment yet." : "Enter an experiment to load its recall heatmap."}
            </p>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="text-sm border-collapse">
                  <thead>
                    <tr>
                      <th className="text-left pr-4 pb-2 font-medium text-muted-foreground">source_type</th>
                      {categories.map((c) => (
                        <th key={c} className="px-2 pb-2 font-medium text-muted-foreground text-center">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sourceTypes.map((st) => (
                      <tr key={st}>
                        <td className="pr-4 py-1 font-medium">{st}</td>
                        {categories.map((cat) => {
                          const cell = cellFor(st, cat);
                          return (
                            <td key={cat} className="p-1">
                              {cell ? (
                                <button
                                  onClick={() => setSelectedCell(cell)}
                                  className={cn(
                                    "w-20 h-10 rounded-md text-xs font-semibold flex items-center justify-center transition-colors",
                                    cell.gap
                                      ? "bg-destructive/15 text-destructive hover:bg-destructive/25"
                                      : "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-500/25"
                                  )}
                                >
                                  {cell.recall_at_3.toFixed(2)}
                                </button>
                              ) : (
                                <div className="w-20 h-10 rounded-md bg-muted/40" />
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {selectedCell && (
                <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm flex gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
                  <p>
                    Gap detected in <strong>{selectedCell.source_type} / {selectedCell.category}</strong>.{" "}
                    {selectedCell.n_questions} questions scored here with mean recall@3 ={" "}
                    {selectedCell.recall_at_3.toFixed(3)}. Likely cause: retrieval isn&apos;t surfacing
                    relevant chunks for this slice — fine-tuning embeddings on targeted synthetic pairs
                    for this combination is the next step.
                  </p>
                </div>
              )}

              <div className="flex items-center gap-3">
                <Button onClick={() => runCycle(true)} disabled={running || !experiment.trim()}>
                  {running ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Run improvement cycle
                </Button>
                <Button variant="outline" onClick={() => runCycle(false)} disabled={running || !experiment.trim()}>
                  Diagnose only
                </Button>
                {status && (
                  <span className="text-xs text-muted-foreground">{status.reason}</span>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                &quot;Run improvement cycle&quot; executes the full loop (fine-tune + re-index +
                re-benchmark) — slow and resource-intensive. &quot;Diagnose only&quot; just checks for
                gaps and writes a report if there&apos;s nothing to improve.
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {/* Panel 2 — Loop progress */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Loop Progress</CardTitle>
          <CardDescription>
            {running
              ? "Running the improvement cycle now — this can take a while on real embedding fine-tuning."
              : "The improvement loop runs synchronously; steps below show what it does end-to-end."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3">
            {LOOP_STEPS.map((label, i) => (
              <div key={label} className="flex items-center gap-3">
                {running ? (
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
                ) : (
                  <CheckCircle2
                    className={cn(
                      "h-4 w-4 shrink-0",
                      reports.length > 0 ? "text-emerald-600" : "text-muted-foreground/40"
                    )}
                  />
                )}
                <span className="text-sm">
                  Step {i + 1}: {label}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Panel 3 — Improvement history */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Improvement History</CardTitle>
          <CardDescription>Each iteration is versioned — nothing is overwritten.</CardDescription>
        </CardHeader>
        <CardContent>
          {reports.length === 0 ? (
            <p className="text-sm text-muted-foreground">No improvement iterations recorded yet.</p>
          ) : (
            <div className="flex flex-col gap-0">
              {reports.map((r, i) => (
                <div key={r.iteration} className="flex gap-3">
                  <div className="flex flex-col items-center">
                    <div className="h-7 w-7 shrink-0 rounded-full bg-primary/10 text-primary flex items-center justify-center">
                      {r.significant === true ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                      ) : r.significant === false ? (
                        <Circle className="h-3.5 w-3.5" />
                      ) : (
                        <Circle className="h-3.5 w-3.5" />
                      )}
                    </div>
                    {i < reports.length - 1 && <div className="w-px flex-1 bg-border my-1" />}
                  </div>
                  <div className="pb-4 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="text-sm font-medium">Iteration {r.iteration}</p>
                      <Badge variant={r.significant ? "default" : "secondary"}>
                        {r.significant === null ? "n/a" : r.significant ? "significant ✓" : "not significant ✗"}
                      </Badge>
                      {r.delta !== null && (
                        <span className="text-xs text-muted-foreground">Δ {r.delta.toFixed(3)}</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">{r.recommendation}</p>
                    <Link href="/compare" className="text-xs text-primary hover:underline mt-1 inline-block">
                      Compare to baseline →
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
