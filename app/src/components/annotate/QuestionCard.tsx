import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface AnnotationItem {
  question_id: string;
  question: string;
  ground_truth?: string;
  predicted_answer?: string;
  overall_score?: number;
}

export function QuestionCard({ item }: { item: AnnotationItem }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base font-medium">Question {item.question_id}</CardTitle>
        {typeof item.overall_score === "number" && (
          <Badge variant="secondary">judge score: {item.overall_score.toFixed(2)}</Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Question</p>
          <p className="text-sm">{item.question}</p>
        </div>
        {item.ground_truth && (
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Ground Truth</p>
            <p className="text-sm">{item.ground_truth}</p>
          </div>
        )}
        {item.predicted_answer && (
          <div>
            <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Predicted Answer</p>
            <p className="text-sm whitespace-pre-wrap">{item.predicted_answer}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
