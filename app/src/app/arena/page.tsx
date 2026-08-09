"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { apiPost } from "@/lib/api";
import { toFriendlyError } from "@/lib/errors";

interface ArenaModelResult {
  model_id: string;
  answer: string;
  latency_ms: number;
  error?: string | null;
}

export default function ArenaPage() {
  const [question, setQuestion] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<ArenaModelResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const availableModels = [
    { id: "ollama/llama3", name: "Llama 3 (Ollama)", provider: "ollama", available: true },
    { id: "openai/gpt-4o-mini", name: "GPT-4o Mini", provider: "openai", available: false },
    { id: "openai/gpt-4o", name: "GPT-4o", provider: "openai", available: false },
    { id: "anthropic/claude-3-haiku", name: "Claude 3 Haiku", provider: "anthropic", available: false },
    { id: "groq/llama3-70b", name: "Llama 3 70B (Groq)", provider: "groq", available: false },
  ];

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((id) => id !== modelId) : [...prev, modelId]
    );
  };

  const runComparison = async () => {
    if (!question || selectedModels.length === 0) return;

    setRunning(true);
    setResults([]);
    setError(null);

    try {
      const data = await apiPost<{ question: string; results: ArenaModelResult[] }>(
        "/arena/run",
        { question, models: selectedModels }
      );
      setResults(data.results || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setRunning(false);
    }
  };

  const copyAsMarkdown = () => {
    let md = `# Model Comparison Arena\n\n**Question:** ${question}\n\n`;
    results.forEach((result) => {
      md += `## ${result.model_id}\n\n`;
      md += `**Answer:** ${result.error ? `_Error: ${result.error}_` : result.answer}\n\n`;
      md += `**Latency:** ${result.latency_ms.toFixed(0)}ms\n\n`;
      md += `---\n\n`;
    });
    navigator.clipboard.writeText(md);
    alert("Copied to clipboard!");
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Model Comparison Arena</h1>
          <p className="text-muted-foreground">Compare LLM responses side-by-side</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left: Model Selection */}
        <Card>
          <CardHeader>
            <CardTitle>Select Models</CardTitle>
            <CardDescription>Choose models to compare</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {availableModels.map((model) => (
              <div
                key={model.id}
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedModels.includes(model.id)
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/50"
                }`}
                onClick={() => toggleModel(model.id)}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{model.name}</span>
                  {model.available ? (
                    <Badge variant="default" className="bg-green-500">
                      ✓
                    </Badge>
                  ) : (
                    <Badge variant="secondary">Key Required</Badge>
                  )}
                </div>
                <div className="text-xs text-muted-foreground mt-1">{model.provider}</div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Center: Question Input + Run */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Question</CardTitle>
            <CardDescription>Enter your question for all models</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="question">Your Question</Label>
              <Input
                id="question"
                placeholder="What is the difference between precision and recall?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="mt-1"
              />
            </div>
            <Button
              onClick={runComparison}
              disabled={running || !question || selectedModels.length === 0}
              className="w-full"
            >
              {running ? "Running..." : `Run Comparison (${selectedModels.length} models)`}
            </Button>
          </CardContent>
        </Card>
      </div>

      {error && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="pt-5">
            <p className="text-sm font-medium text-destructive">{toFriendlyError(error).title}</p>
            <p className="text-xs text-destructive/80 mt-1">{toFriendlyError(error).description}</p>
          </CardContent>
        </Card>
      )}

      {/* Results Grid */}
      {results.length > 0 && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold">Results</h2>
            <Button onClick={copyAsMarkdown} variant="outline">
              Copy as Markdown
            </Button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.map((result, idx) => (
              <Card key={idx}>
                <CardHeader>
                  <CardTitle className="text-lg">{result.model_id}</CardTitle>
                  <div className="flex gap-2 text-sm text-muted-foreground">
                    <Badge variant="outline">{result.latency_ms.toFixed(0)}ms</Badge>
                    {result.error && <Badge variant="destructive">Error</Badge>}
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm whitespace-pre-wrap">
                    {result.error ? `Error: ${result.error}` : result.answer}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Latency Leaderboard */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Leaderboard</CardTitle>
              <CardDescription>Ranked by latency (no ground truth for ad-hoc questions)</CardDescription>
            </CardHeader>
            <CardContent>
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Rank</th>
                    <th className="text-left p-2">Model</th>
                    <th className="text-right p-2">Latency (ms)</th>
                    <th className="text-right p-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[...results]
                    .sort((a, b) => a.latency_ms - b.latency_ms)
                    .map((result, idx) => (
                      <tr key={idx} className="border-b">
                        <td className="p-2">{idx + 1}</td>
                        <td className="p-2 font-medium">{result.model_id}</td>
                        <td className="p-2 text-right">{result.latency_ms.toFixed(0)}</td>
                        <td className="p-2 text-right">{result.error ? "Error" : "OK"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
