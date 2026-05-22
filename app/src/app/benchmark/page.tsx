"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Trophy, Clock, Target, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { apiGet } from "@/lib/api";

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
}

const CHART_COLORS = ["#0066CC", "#34C759", "#FF9500", "#AF52DE", "#FF2D55"];

export default function BenchmarkPage() {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [results, setResults] = useState<BenchmarkResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
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

  return (
    <div className="h-full flex flex-col">
      <header className="h-14 flex items-center justify-between px-6 border-b border-border shrink-0">
        <h1 className="text-lg font-semibold tracking-tight">Benchmark Results</h1>
        <Select value={selectedExp} onValueChange={(v) => v && setSelectedExp(v)}>
          <SelectTrigger className="w-[240px]">
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
            <CardContent className="pt-5 flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-destructive" />
              <p className="text-sm text-destructive">{error}</p>
            </CardContent>
          </Card>
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
                      <p className="text-2xl font-semibold">
                        {(results.average_score * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-muted-foreground">Avg Score</p>
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
          </motion.div>
        )}
      </div>
    </div>
  );
}
