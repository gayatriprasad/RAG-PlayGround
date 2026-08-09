export function AnnotationProgress({ labeled, total }: { labeled: number; total: number }) {
  const pct = total > 0 ? Math.round((labeled / total) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Progress</span>
        <span>
          {labeled} / {total} labeled ({pct}%)
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-primary transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
