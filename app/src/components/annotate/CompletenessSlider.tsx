import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

export function CompletenessSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm">Completeness</Label>
        <span className="text-sm font-medium tabular-nums">{value.toFixed(2)}</span>
      </div>
      <Slider
        value={[value]}
        min={0}
        max={1}
        step={0.05}
        onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
      />
    </div>
  );
}
