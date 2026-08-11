"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiPost } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";

export default function PromptLabPage() {
  const [strategy, setStrategy] = useState("zero_shot");
  const [nExamples, setNExamples] = useState(3);
  const [nSamples, setNSamples] = useState(5);
  const [temperature, setTemperature] = useState([0.0]);
  const [version, setVersion] = useState("v1");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const strategies = [
    { value: "zero_shot", label: "Zero Shot (baseline)" },
    { value: "few_shot", label: "Few Shot" },
    { value: "cot", label: "Chain of Thought" },
    { value: "self_consistency", label: "Self-Consistency" },
    { value: "medprompt", label: "Medprompt" },
  ];

  const runSingleQuery = async () => {
    if (!query) return;

    setRunning(true);
    setResults([]);
    setError(null);

    try {
      const data = await apiPost<any>("/prompt-lab/run", {
        query,
        strategy,
        n_examples: nExamples,
        n_samples: nSamples,
        temperature: temperature[0],
        version,
        system_prompt: systemPrompt || null,
      });
      setResults([data]);
    } catch (e) {
      setError(toFriendlyError(e).description);
    } finally {
      setRunning(false);
    }
  };

  const runBenchmark = async () => {
    setRunning(true);
    setResults([]);
    setError(null);

    try {
      const data = await apiPost<any>("/prompt-lab/benchmark", {
        strategy,
        n_examples: nExamples,
        n_samples: nSamples,
        temperature: temperature[0],
        version,
        system_prompt: systemPrompt || null,
        max_questions: 20,
      });
      setResults(data.results || []);
    } catch (e) {
      setError(toFriendlyError(e).description);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Prompt Engineering Lab</h1>
        <p className="text-muted-foreground">Test and compare prompt strategies</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left: Strategy Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Strategy Config</CardTitle>
            <CardDescription>Tune prompt parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="strategy">Strategy</Label>
              <Select value={strategy} onValueChange={(v) => v && setStrategy(v)}>
                <SelectTrigger id="strategy">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {strategies.map((s) => (
                    <SelectItem key={s.value} value={s.value}>
                      {s.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {(strategy === "few_shot" || strategy === "medprompt") && (
              <div>
                <Label htmlFor="n_examples">Examples: {nExamples}</Label>
                <Slider
                  id="n_examples"
                  min={1}
                  max={10}
                  step={1}
                  value={[nExamples]}
                  onValueChange={(val) => setNExamples(Array.isArray(val) ? val[0] : val)}
                  className="mt-2"
                />
              </div>
            )}

            {(strategy === "self_consistency" || strategy === "medprompt") && (
              <div>
                <Label htmlFor="n_samples">Samples: {nSamples}</Label>
                <Slider
                  id="n_samples"
                  min={3}
                  max={10}
                  step={1}
                  value={[nSamples]}
                  onValueChange={(val) => setNSamples(Array.isArray(val) ? val[0] : val)}
                  className="mt-2"
                />
              </div>
            )}

            <div>
              <Label htmlFor="temperature">Temperature: {temperature[0].toFixed(1)}</Label>
              <Slider
                id="temperature"
                min={0}
                max={1}
                step={0.1}
                value={temperature}
                onValueChange={(val) => setTemperature(Array.isArray(val) ? [...val] : [val])}
                className="mt-2"
              />
            </div>

            <div>
              <Label htmlFor="version">Version</Label>
              <Select value={version} onValueChange={(v) => v && setVersion(v)}>
                <SelectTrigger id="version">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="v1">v1 (default)</SelectItem>
                  <SelectItem value="v2">v2</SelectItem>
                  <SelectItem value="v3">v3 (experimental)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="system_prompt">Custom System Prompt (optional)</Label>
              <Textarea
                id="system_prompt"
                placeholder="Override default system prompt..."
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="mt-1 font-mono text-sm"
                rows={6}
              />
            </div>
          </CardContent>
        </Card>

        {/* Right: Query + Results */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Test Query</CardTitle>
            <CardDescription>Run single query or benchmark</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="single">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="single">Single Query</TabsTrigger>
                <TabsTrigger value="benchmark">Benchmark (20 questions)</TabsTrigger>
              </TabsList>

              <TabsContent value="single" className="space-y-4">
                <div>
                  <Label htmlFor="query">Question</Label>
                  <Textarea
                    id="query"
                    placeholder="What is the difference between RAG and fine-tuning?"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="mt-1"
                    rows={4}
                  />
                </div>
                <Button onClick={runSingleQuery} disabled={running || !query} className="w-full">
                  {running ? "Running..." : "Run Query"}
                </Button>
                {error && <p className="text-sm text-destructive">{error}</p>}

                {results.length > 0 && results[0].answer && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Answer</CardTitle>
                      <div className="flex gap-2 text-sm">
                        <Badge variant="outline">{Math.round(results[0].latency_ms)}ms</Badge>
                        <Badge variant="outline">{results[0].n_chunks} chunks</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="whitespace-pre-wrap text-sm">{results[0].answer}</p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="benchmark" className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Run this strategy against 20 questions from the benchmark dataset.
                </p>
                <Button onClick={runBenchmark} disabled={running} className="w-full">
                  {running ? "Running..." : "Run Benchmark"}
                </Button>
                {error && <p className="text-sm text-destructive">{error}</p>}

                {results.length > 0 && !results[0].answer && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Benchmark Results</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="font-medium">Accuracy:</span>
                          <span>{((results.filter((r) => r.correct).length / results.length) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="font-medium">Avg Score:</span>
                          <span>{(results.reduce((s, r) => s + (r.score || 0), 0) / results.length).toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="font-medium">Avg Latency:</span>
                          <span>{(results.reduce((s, r) => s + r.latency_ms, 0) / results.length).toFixed(0)}ms</span>
                        </div>
                      </div>

                      <div className="mt-4">
                        <h4 className="font-medium mb-2">Per-Question Results</h4>
                        <div className="max-h-60 overflow-y-auto space-y-1">
                          {results.map((r, idx) => (
                            <div key={idx} className="flex justify-between text-sm border-b py-1">
                              <span className="truncate flex-1">{r.question}</span>
                              <Badge variant={r.correct ? "default" : "destructive"} className="ml-2">
                                {r.correct ? "✓" : "✗"}
                              </Badge>
                            </div>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
