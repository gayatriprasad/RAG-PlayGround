"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiGet, apiPost } from "@/lib/api";
import { QuestionCard, AnnotationItem } from "@/components/annotate/QuestionCard";
import { CompletenessSlider } from "@/components/annotate/CompletenessSlider";
import { AnnotationProgress } from "@/components/annotate/AnnotationProgress";

type Mode = "calibration" | "uncertainty";

interface QueueResponse {
  mode: Mode;
  item: AnnotationItem | null;
  progress: { labeled: number; total: number };
}

export default function AnnotatePage() {
  const [mode, setMode] = useState<Mode>("calibration");
  const [experiment, setExperiment] = useState("");
  const [queue, setQueue] = useState<QueueResponse | null>(null);
  const [completeness, setCompleteness] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchQueue = useCallback(() => {
    if (mode === "uncertainty" && !experiment.trim()) {
      setQueue(null);
      return Promise.resolve();
    }
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ mode });
    if (mode === "uncertainty") params.set("experiment", experiment.trim());
    return apiGet<QueueResponse>(`/annotate/queue?${params.toString()}`)
      .then((data) => {
        setQueue(data);
        setCompleteness(0.5);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load annotation queue");
        setQueue(null);
      })
      .finally(() => setLoading(false));
  }, [mode, experiment]);

  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);

  const submit = async (humanCorrect: boolean) => {
    if (!queue?.item) return;
    setSubmitting(true);
    setError(null);
    try {
      await apiPost("/annotate/submit", {
        mode,
        question_id: queue.item.question_id,
        human_correct: humanCorrect,
        human_completeness: completeness,
        experiment: mode === "uncertainty" ? experiment.trim() : undefined,
      });
      await fetchQueue();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit annotation");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-10 px-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Annotate</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Human-in-the-loop grading — label judge calibration samples or ambiguous, low-confidence
          predictions to keep the LLM judge honest.
        </p>
      </div>

      <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
        <TabsList>
          <TabsTrigger value="calibration">Judge Calibration</TabsTrigger>
          <TabsTrigger value="uncertainty">Uncertainty Sampling</TabsTrigger>
        </TabsList>
      </Tabs>

      {mode === "uncertainty" && (
        <div className="space-y-1.5 max-w-sm">
          <Label htmlFor="experiment">Experiment</Label>
          <Input
            id="experiment"
            placeholder="e.g. 02_retrieval_comparison"
            value={experiment}
            onChange={(e) => setExperiment(e.target.value)}
            onBlur={fetchQueue}
          />
        </div>
      )}

      {queue && <AnnotationProgress labeled={queue.progress.labeled} total={queue.progress.total} />}

      {error && (
        <Card className="border-destructive/50">
          <CardContent className="pt-6 text-sm text-destructive">{error}</CardContent>
        </Card>
      )}

      {loading && <p className="text-sm text-muted-foreground">Loading…</p>}

      {!loading && !error && queue && !queue.item && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Queue complete</CardTitle>
            <CardDescription>Every item in this queue has been labeled.</CardDescription>
          </CardHeader>
        </Card>
      )}

      {!loading && queue?.item && (
        <div className="space-y-4">
          <QuestionCard item={queue.item} />
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Your judgment</CardTitle>
              <CardDescription>Is the predicted answer correct, and how complete is it?</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <CompletenessSlider value={completeness} onChange={setCompleteness} />
              <div className="flex gap-3">
                <Button disabled={submitting} onClick={() => submit(true)} className="flex-1">
                  Correct
                </Button>
                <Button
                  disabled={submitting}
                  onClick={() => submit(false)}
                  variant="destructive"
                  className="flex-1"
                >
                  Incorrect
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
