"""
Post-experiment hooks: result archiving and markdown report generation.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from raglab.config import Config
from raglab.hooks.base import PostExperimentHook
from raglab.types import EvalResult

logger = logging.getLogger(__name__)


class ResultArchiverHook(PostExperimentHook):
    """
    HOOK 07: Archives results with timestamp for reproducibility.
    """

    def run(self, cfg: Config, results: List[EvalResult]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("out/raglab_out")
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{cfg.experiment.name}_{timestamp}"

        # Save full results as JSONL
        jsonl_path = out_dir / f"{prefix}_full.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")

        logger.info(f"Results archived to {out_dir}/")


class MarkdownReporterHook(PostExperimentHook):
    """
    HOOK 08: Generates a markdown report of experiment results.
    """

    def run(self, cfg: Config, results: List[EvalResult]) -> None:
        if not results:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        out_dir = Path("out/raglab_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{cfg.experiment.name}_report.md"

        scored = [r for r in results if r.overall_score is not None]
        if not scored:
            return

        avg_score = sum(r.overall_score for r in scored) / len(scored)
        correct_count = sum(1 for r in scored if r.answer_correct)

        # Top/bottom 5
        sorted_results = sorted(scored, key=lambda r: r.overall_score or 0, reverse=True)
        top5 = sorted_results[:5]
        bottom5 = sorted_results[-5:]

        lines = [
            f"# Experiment: {cfg.experiment.name}",
            f"**Run date:** {timestamp}",
            f"**Config:** index={cfg.index.backend}, intent={cfg.intent.mode}, "
            f"agentic={cfg.agentic.strategy}",
            "",
            "## Overall Scores",
            f"- Total questions: {len(scored)}",
            f"- Correct: {correct_count}/{len(scored)} ({100*correct_count/len(scored):.1f}%)",
            f"- Mean overall_score: {avg_score:.3f}",
            "",
            "## Top 5 Questions",
            "| Question | Pipeline | Score |",
            "|----------|----------|-------|",
        ]
        for r in top5:
            q_preview = r.question[:60] + "..." if len(r.question) > 60 else r.question
            lines.append(f"| {q_preview} | {r.pipeline} | {r.overall_score:.3f} |")

        lines.extend([
            "",
            "## Bottom 5 Questions",
            "| Question | Pipeline | Score |",
            "|----------|----------|-------|",
        ])
        for r in bottom5:
            q_preview = r.question[:60] + "..." if len(r.question) > 60 else r.question
            lines.append(f"| {q_preview} | {r.pipeline} | {r.overall_score:.3f} |")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report saved: {report_path}")
