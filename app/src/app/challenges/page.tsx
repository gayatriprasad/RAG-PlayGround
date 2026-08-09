"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Circle, Lightbulb, Loader2, Trophy } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";
import { apiGet, apiPost } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";

interface Challenge {
  id: string;
  title: string;
  difficulty: string;
  goal: string;
  metric: string;
  operator: string;
  target: number;
  hint: string;
  concept: string;
}

interface ChallengeResult {
  challenge_id: string;
  passed: boolean;
  actual: number;
  target: number;
  operator: string;
  message: string;
}

const STORAGE_KEY = "neuralbench_challenges_completed";

function loadCompleted(): Record<string, boolean> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveCompleted(state: Record<string, boolean>) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

const difficultyColor: Record<string, string> = {
  beginner: "bg-emerald-500/10 text-emerald-600 border-emerald-500/30",
  intermediate: "bg-amber-500/10 text-amber-600 border-amber-500/30",
  advanced: "bg-rose-500/10 text-rose-600 border-rose-500/30",
};

export default function ChallengesPage() {
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState<Record<string, boolean>>({});
  const [showHint, setShowHint] = useState<Record<string, boolean>>({});
  const [checking, setChecking] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, ChallengeResult>>({});

  useEffect(() => {
    setCompleted(loadCompleted());
    apiGet<{ challenges: Challenge[] }>("/challenges")
      .then((data) => setChallenges(data.challenges))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load challenges"))
      .finally(() => setLoading(false));
  }, []);

  async function checkChallenge(id: string) {
    setChecking(id);
    try {
      const result = await apiPost<ChallengeResult>(`/challenges/${id}/check`, {});
      setResults((prev) => ({ ...prev, [id]: result }));
      if (result.passed) {
        const next = { ...completed, [id]: true };
        setCompleted(next);
        saveCompleted(next);
      }
    } catch (e) {
      setResults((prev) => ({
        ...prev,
        [id]: {
          challenge_id: id,
          passed: false,
          actual: 0,
          target: 0,
          operator: "",
          message: e instanceof Error ? e.message : "Check failed — run the experiment first.",
        },
      }));
    } finally {
      setChecking(null);
    }
  }

  const nCompleted = Object.values(completed).filter(Boolean).length;

  return (
    <div className="container mx-auto p-6 space-y-6 max-w-3xl">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Trophy className="h-7 w-7 text-amber-500" /> Guided Challenges
        </h1>
        <p className="text-muted-foreground">
          Learn RAG tradeoffs hands-on. Tune your pipeline config, run an experiment, then check your progress here.
        </p>
        {challenges.length > 0 && (
          <div className="mt-3 flex items-center gap-2">
            <div className="h-2 flex-1 max-w-xs bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${(nCompleted / challenges.length) * 100}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground">
              {nCompleted}/{challenges.length} complete
            </span>
          </div>
        )}
      </div>

      {loading && <p className="text-sm text-muted-foreground">Loading challenges...</p>}
      {error && <p className="text-sm text-destructive">{toFriendlyError(error).description}</p>}

      {!loading && !error && challenges.length === 0 && (
        <EmptyState
          title="No challenges available"
          description="Challenges are loaded from rag-lab/challenges/challenges.json — none were found there."
        />
      )}

      <div className="grid grid-cols-1 gap-4">
        {challenges.map((c) => {
          const result = results[c.id];
          const isDone = completed[c.id];
          return (
            <Card key={c.id} className={isDone ? "border-emerald-500/40" : ""}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {isDone ? (
                      <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                    ) : (
                      <Circle className="h-5 w-5 text-muted-foreground" />
                    )}
                    <CardTitle className="text-base">{c.title}</CardTitle>
                  </div>
                  <Badge variant="outline" className={`text-xs ${difficultyColor[c.difficulty] || ""}`}>
                    {c.difficulty}
                  </Badge>
                </div>
                <CardDescription>{c.goal}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2 flex-wrap">
                  <Button
                    size="sm"
                    onClick={() => checkChallenge(c.id)}
                    disabled={checking === c.id}
                  >
                    {checking === c.id ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                    Check
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowHint((prev) => ({ ...prev, [c.id]: !prev[c.id] }))}
                  >
                    <Lightbulb className="h-4 w-4 mr-1" /> Stuck?
                  </Button>
                </div>

                {showHint[c.id] && (
                  <p className="text-sm text-muted-foreground bg-muted/50 rounded-lg p-3">{c.hint}</p>
                )}

                {result && (
                  <p className={`text-sm ${result.passed ? "text-emerald-600" : "text-muted-foreground"}`}>
                    {result.message}
                  </p>
                )}

                {(result?.passed || isDone) && (
                  <p className="text-sm bg-emerald-500/10 text-emerald-700 rounded-lg p-3">
                    <b>Concept:</b> {c.concept}
                  </p>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
