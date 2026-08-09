"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { apiGet } from "@/lib/api";

interface CalibrationResponse {
  experiment: string;
  n_questions: number;
  curve: {
    ece: number;
    overconfident_bins: number[];
    underconfident_bins: number[];
  };
  diagram: {
    points: { predicted: number; actual: number; count: number }[];
    diagonal: { x: number; y: number }[];
  };
}

function eceTrafficLight(ece: number): { label: string; className: string } {
  if (ece < 0.05) return { label: "Well calibrated", className: "bg-green-500/15 text-green-600 dark:text-green-400" };
  if (ece <= 0.1) return { label: "Slightly miscalibrated", className: "bg-yellow-500/15 text-yellow-600 dark:text-yellow-400" };
  return { label: "Poorly calibrated", className: "bg-red-500/15 text-red-600 dark:text-red-400" };
}

export function CalibrationCard({ experiment }: { experiment: string }) {
  const [data, setData] = useState<CalibrationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!experiment) return;
    setLoading(true);
    setError(null);
    apiGet<CalibrationResponse>(`/benchmark/calibration?experiment=${encodeURIComponent(experiment)}`)
      .then(setData)
      .catch((e) => {
        setData(null);
        setError(e instanceof Error ? e.message : "Failed to load calibration data");
      })
      .finally(() => setLoading(false));
  }, [experiment]);

  if (loading || error || !data || data.diagram.points.length === 0) return null;

  const chartData = data.diagram.points.map((p) => ({
    predicted: p.predicted,
    actual: p.actual,
    perfect: p.predicted,
    count: p.count,
  }));
  const traffic = eceTrafficLight(data.curve.ece);
  const pctCorrectAtHighConfidence = Math.round(
    (data.diagram.points[data.diagram.points.length - 1]?.actual ?? 0) * 100
  );

  return (
    <Card className="shadow-sm border-border">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Confidence Calibration</CardTitle>
          <Badge className={traffic.className}>
            ECE {data.curve.ece.toFixed(3)} — {traffic.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="predicted" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
            <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(2) : v)} />
            <Legend />
            <Line
              type="monotone"
              dataKey="perfect"
              name="Perfect calibration"
              stroke="var(--muted-foreground)"
              strokeDasharray="4 4"
              dot={false}
            />
            <Line type="monotone" dataKey="actual" name="Actual accuracy" stroke="var(--primary)" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-muted-foreground mt-2">
          Your eval scores are {traffic.label.toLowerCase()}. At the highest confidence bin, the
          system is actually correct {pctCorrectAtHighConfidence}% of the time.
        </p>
      </CardContent>
    </Card>
  );
}
