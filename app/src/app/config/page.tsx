"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { FileCode, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { apiGet } from "@/lib/api";

interface ExperimentMeta {
  name: string;
  config_path: string;
  has_results: boolean;
}

interface ConfigResponse {
  experiment: string;
  config: Record<string, unknown>;
  raw_yaml: string;
}

export default function ConfigPage() {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [config, setConfig] = useState<ConfigResponse | null>(null);
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
    apiGet<ConfigResponse>(`/experiments/${encodeURIComponent(selectedExp)}/config`)
      .then(setConfig)
      .catch((e) => {
        // If endpoint doesn't exist, show raw path info
        setError(e.message);
        setConfig(null);
      })
      .finally(() => setLoading(false));
  }, [selectedExp]);

  return (
    <div className="h-full flex flex-col">
      <header className="h-14 flex items-center justify-between px-6 border-b border-border shrink-0">
        <h1 className="text-lg font-semibold tracking-tight">Configuration</h1>
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

      <div className="flex-1 overflow-y-auto p-6">
        {loading && (
          <div className="space-y-3">
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-64 w-full rounded-xl" />
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

        {config && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="shadow-sm border-border">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <FileCode className="h-4 w-4 text-primary" />
                  <CardTitle className="text-sm font-medium">
                    {selectedExp}/config.yaml
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <pre className="text-xs leading-relaxed font-mono bg-muted/50 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap border border-border">
                  {config.raw_yaml || JSON.stringify(config.config, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {!loading && !error && !config && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <FileCode className="h-10 w-10 text-muted-foreground/50 mb-3" />
            <p className="text-sm text-muted-foreground">
              Select an experiment to view its configuration
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
