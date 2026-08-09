"""Export & share — Skill 35.

RunExporter turns an experiment's results DataFrame into shareable report
formats (markdown / csv / html / json). encode_config/decode_config produce
shareable config links — never encoding secrets (Coding Rule 16).
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fields that must NEVER be encoded into a shareable link (Coding Rule 16).
_SECRET_FIELDS = {
    ("llm", "api_key"),
    ("db", "dsn"),
    ("index", "milvus_token"),
}


class RunExporter:
    """Exports an experiment run (results DataFrame + optional config) to
    portable report formats."""

    def to_markdown(self, experiment_name: str, df, cfg: Optional[Any] = None) -> str:
        lines = [f"# NeuralBench Run Report — {experiment_name}", ""]
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")

        if cfg is not None:
            lines.append("## Configuration")
            lines.append(f"- Index backend: `{cfg.index.backend}`")
            lines.append(f"- LLM: `{cfg.llm.provider}/{cfg.llm.model}`")
            lines.append(f"- Chunk strategy: `{cfg.chunk.strategy}`")
            lines.append(f"- Top-k: `{cfg.retrieve.top_k}`")
            lines.append("")

        lines.append("## Summary")
        n = len(df)
        lines.append(f"- Questions: {n}")
        if "overall_score" in df.columns:
            lines.append(f"- Mean overall score: {df['overall_score'].mean():.3f}")
        if "answer_correct" in df.columns:
            lines.append(f"- Correct: {int(df['answer_correct'].fillna(False).sum())}/{n}")
        if "latency_ms" in df.columns:
            lines.append(f"- Mean latency: {df['latency_ms'].mean():.0f}ms")
        lines.append("")

        if "category" in df.columns and "overall_score" in df.columns:
            lines.append("## Score by category")
            lines.append("")
            lines.append("| Category | Mean score | N |")
            lines.append("|---|---|---|")
            for cat, group in df.groupby("category"):
                lines.append(f"| {cat} | {group['overall_score'].mean():.3f} | {len(group)} |")
            lines.append("")

        lines.append("## Per-question results")
        lines.append("")
        for _, row in df.iterrows():
            lines.append(f"### {row.get('question_id', '')}")
            lines.append(f"**Question:** {row.get('question', '')}")
            lines.append("")
            lines.append(f"**Ground truth:** {row.get('ground_truth', '')}")
            lines.append("")
            lines.append(f"**Predicted answer:** {row.get('predicted_answer', '')}")
            lines.append("")
            score = row.get("overall_score")
            lines.append(f"Score: {score if score is not None else 'n/a'}  ")
            lines.append(f"Pipeline: {row.get('pipeline', 'n/a')}  ")
            lines.append("")

        return "\n".join(lines)

    def to_csv(self, df) -> str:
        return df.to_csv(index=False)

    def to_html(self, experiment_name: str, df, cfg: Optional[Any] = None) -> str:
        n = len(df)
        mean_score = df["overall_score"].mean() if "overall_score" in df.columns else None
        rows_html = []
        for _, row in df.iterrows():
            rows_html.append(
                "<tr>"
                f"<td>{_esc(row.get('question_id', ''))}</td>"
                f"<td>{_esc(row.get('question', ''))}</td>"
                f"<td>{_esc(row.get('predicted_answer', ''))}</td>"
                f"<td>{_esc(row.get('overall_score', ''))}</td>"
                f"<td>{_esc(row.get('pipeline', ''))}</td>"
                "</tr>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeuralBench Run Report — {_esc(experiment_name)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border: 1px solid #e2e2e2; padding: 8px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f7f7f7; }}
  .summary {{ display: flex; gap: 2rem; margin: 1rem 0; }}
  .stat {{ background: #f7f7f7; border-radius: 8px; padding: 12px 20px; }}
  .stat b {{ display: block; font-size: 1.4rem; }}
</style>
</head>
<body>
  <h1>NeuralBench Run Report</h1>
  <p>Experiment: <b>{_esc(experiment_name)}</b> — generated {datetime.now(timezone.utc).isoformat()}</p>
  <div class="summary">
    <div class="stat"><b>{n}</b>Questions</div>
    <div class="stat"><b>{f'{mean_score:.3f}' if mean_score is not None else 'n/a'}</b>Mean score</div>
  </div>
  <table>
    <thead><tr><th>ID</th><th>Question</th><th>Answer</th><th>Score</th><th>Pipeline</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""

    def to_json(self, experiment_name: str, df, cfg: Optional[Any] = None) -> str:
        payload: Dict[str, Any] = {
            "experiment": experiment_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_questions": len(df),
            "results": json.loads(df.to_json(orient="records")),
        }
        if cfg is not None:
            payload["config"] = _strip_secrets(cfg.model_dump())
        return json.dumps(payload, indent=2)


def _esc(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _strip_secrets(data: Dict[str, Any]) -> Dict[str, Any]:
    stripped = json.loads(json.dumps(data))  # deep copy
    for section, field in _SECRET_FIELDS:
        if section in stripped and isinstance(stripped[section], dict):
            stripped[section].pop(field, None)
    return stripped


def encode_config(cfg: Any) -> str:
    """Base64url-encode a Config's non-secret fields for a shareable link.

    Never encodes API keys or DSNs (Coding Rule 16) — only pipeline / model /
    retrieval parameters.
    """
    data = _strip_secrets(cfg.model_dump())
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_config(token: str):
    """Decode a shareable config token back into a Config object."""
    from raglab.config import Config

    raw = base64.urlsafe_b64decode(token.encode("ascii"))
    data = json.loads(raw)
    return Config(**data)
