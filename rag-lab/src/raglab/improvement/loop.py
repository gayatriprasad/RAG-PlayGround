"""
Improvement Loop — Skill 46 (the flywheel).

Wires together modules that already exist as their own skills into one
closed-loop cycle: eval -> diagnose gaps -> generate targeted synthetic pairs
-> fine-tune embeddings -> re-index -> re-benchmark -> report delta with
statistical significance. One loop run = one iteration. Nothing is
overwritten — each iteration's report is versioned under
`cfg.improvement.reports_dir/iter_{n}/`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

from raglab.config import Config
from raglab.datasets.synthesizer import DatasetSynthesizer
from raglab.eval.significance import compare
from raglab.eval.validity import SliceChecker
from raglab.improvement.scheduler import find_gap_slices
from raglab.training.embed_trainer import EmbeddingFineTuner
from raglab.types import Chunk, Document, EvalResult, ImprovementReport, Question

logger = logging.getLogger(__name__)


class ImprovementLoop:
    """
    Closed-loop RAG improvement cycle. Each collaborator is injectable so the
    orchestration logic (diagnose/generate/compare/version) can be tested
    without requiring a real embedding-model training run or a live LLM.
    """

    def __init__(
        self,
        cfg: Config,
        run_id: str,
        synthesizer: Optional[DatasetSynthesizer] = None,
        fine_tuner: Optional[EmbeddingFineTuner] = None,
        llm_client=None,
        rebuild_index_fn: Optional[Callable[[str, str], None]] = None,
        rerun_pipeline_fn: Optional[Callable[[List[Question], str], List[EvalResult]]] = None,
    ):
        """
        Args:
            cfg: full Config (uses cfg.improvement, cfg.stats, cfg.embed).
            run_id: the baseline run this iteration is diagnosing/improving.
            synthesizer: DatasetSynthesizer instance (default: a fresh one).
            fine_tuner: EmbeddingFineTuner instance (default: a fresh one).
            llm_client: LLM client used for synthetic question generation.
            rebuild_index_fn: (model_path, out_dir) -> None. Rebuilds the
                vector index with the fine-tuned model. Injected so tests
                don't need to actually embed/persist a real index.
            rerun_pipeline_fn: (questions, index_dir) -> List[EvalResult].
                Re-runs the benchmark on the given questions against the
                rebuilt index. Injected for the same reason.
        """
        self.cfg = cfg
        self.run_id = run_id
        self.synthesizer = synthesizer or DatasetSynthesizer()
        self.fine_tuner = fine_tuner or EmbeddingFineTuner()
        self.llm_client = llm_client
        self._rebuild_index_fn = rebuild_index_fn
        self._rerun_pipeline_fn = rerun_pipeline_fn
        self.iteration = self._next_iteration()

    def _next_iteration(self) -> int:
        reports_dir = Path(self.cfg.improvement.reports_dir)
        if not reports_dir.exists():
            return 1
        existing = [
            int(p.name.split("_")[-1])
            for p in reports_dir.glob("iter_*")
            if p.name.split("_")[-1].isdigit()
        ]
        return (max(existing) + 1) if existing else 1

    def run(
        self,
        baseline_results: List[EvalResult],
        docs: List[Document],
        chunks: List[Chunk],
    ) -> ImprovementReport:
        # Step 1 — DIAGNOSE: find recall gaps in baseline results.
        gap_slices = find_gap_slices(baseline_results, self.cfg.improvement)
        if not gap_slices:
            return self._no_gap_report(baseline_results)

        gap_source_types = {g["source_type"] for g in gap_slices}
        gap_docs = [d for d in docs if d.source_type in gap_source_types]

        # Step 2 — GENERATE: synthetic Q&A targeted at the gap slices, then
        # quality-gate with the Skill 44 validity checks before use.
        generated = self.synthesizer.generate(gap_docs, self.cfg.dataset, llm_client=self.llm_client)
        kept, rejected, validation_report = self.synthesizer.validate_generated(generated, gap_docs)

        report_dir = Path(self.cfg.improvement.reports_dir) / f"iter_{self.iteration}"
        report_dir.mkdir(parents=True, exist_ok=True)
        self._save_questions(kept, report_dir / "questions.jsonl")

        if not kept:
            return self._finalize(
                gap_slices=gap_slices,
                n_generated=len(generated),
                n_passed=0,
                fine_tuned_model_path=None,
                significance=None,
                slice_check=None,
                recommendation=(
                    "No synthetic pairs passed validation — cannot fine-tune this "
                    "iteration. Check corpus coverage for the gap source types: "
                    f"{sorted(gap_source_types)}."
                ),
                report_dir=report_dir,
            )

        # Step 3 — FINE-TUNE embeddings on the targeted pairs.
        examples = self.fine_tuner.prepare_training_data(kept, chunks)
        model_dir = Path(self.cfg.improvement.models_dir) / f"embed_iter_{self.iteration}"
        fine_tuned_path = self.fine_tuner.train(
            base_model=self.cfg.improvement.fine_tune_base_model,
            examples=examples,
            output_path=str(model_dir),
            epochs=self.cfg.improvement.fine_tune_epochs,
        )

        # Step 4 — RE-INDEX with the fine-tuned model. Never overwrites the
        # baseline index — a fresh directory per iteration.
        index_out_dir = f"./out/chroma_iter_{self.iteration}"
        if self._rebuild_index_fn is not None:
            self._rebuild_index_fn(fine_tuned_path, index_out_dir)

        # Step 5 — RE-BENCHMARK on the SAME question set as the baseline run
        # (not the newly synthesized pairs) so Step 6's paired comparison has
        # shared question_ids to align on.
        baseline_questions = self._baseline_questions(baseline_results)
        iter_results: List[EvalResult] = []
        if self._rerun_pipeline_fn is not None:
            iter_results = self._rerun_pipeline_fn(baseline_questions, index_out_dir)

        significance = None
        slice_check = None
        recommendation = "Re-benchmark step was not run (no rerun_pipeline_fn supplied) — cannot compare."

        if iter_results:
            # Step 6 — COMPARE: is the improvement statistically real?
            significance = compare(
                baseline_results, iter_results, "overall_score", self.cfg.stats, "baseline", f"iter_{self.iteration}"
            )
            slice_check = SliceChecker().check_slices(
                {"baseline": baseline_results, f"iter_{self.iteration}": iter_results},
                "overall_score",
                self.cfg.improvement.min_slice_size,
            )
            recommendation = self._recommendation(significance, slice_check)

        # Step 7 — PROMPT REGRESSION: left to the caller (run_experiment.py)
        # to populate via a separate significance.compare() call when the
        # prompt version on disk is newer than the one recorded on the
        # baseline run — not computed here, since it needs prompt-version
        # bookkeeping this loop does not own.

        return self._finalize(
            gap_slices=gap_slices,
            n_generated=len(generated),
            n_passed=len(kept),
            fine_tuned_model_path=fine_tuned_path,
            significance=significance,
            slice_check=slice_check,
            recommendation=recommendation,
            report_dir=report_dir,
        )

    def _recommendation(self, significance, slice_check) -> str:
        if not significance.significant:
            return (
                f"Iteration {self.iteration}: no significant improvement "
                f"(p={significance.p_value:.3f}). Consider more training data "
                "or a different base embedding model."
            )
        if slice_check is not None and not slice_check.consistent:
            return (
                f"Iteration {self.iteration}: significant on aggregate "
                f"(delta={significance.delta:+.3f}, p={significance.p_value:.3f}) "
                f"but {slice_check.warning} Do not deploy without reviewing per-slice detail."
            )
        return (
            f"Deploy iter_{self.iteration} model — significant "
            f"{significance.delta:+.3f} on overall_score "
            f"(p={significance.p_value:.3f}, effect_size={significance.effect_size:.2f})."
        )

    def _no_gap_report(self, baseline_results: List[EvalResult]) -> ImprovementReport:
        report_dir = Path(self.cfg.improvement.reports_dir) / f"iter_{self.iteration}"
        return self._finalize(
            gap_slices=[],
            n_generated=0,
            n_passed=0,
            fine_tuned_model_path=None,
            significance=None,
            slice_check=None,
            recommendation=(
                f"No slice fell below recall@3 threshold "
                f"{self.cfg.improvement.min_recall_threshold} — nothing to improve this iteration."
            ),
            report_dir=report_dir,
        )

    def _finalize(
        self,
        gap_slices,
        n_generated,
        n_passed,
        fine_tuned_model_path,
        significance,
        slice_check,
        recommendation,
        report_dir: Path,
    ) -> ImprovementReport:
        report = ImprovementReport(
            iteration=self.iteration,
            baseline_run_id=self.run_id,
            gap_slices=gap_slices,
            n_synthetic_pairs_generated=n_generated,
            n_pairs_passed_validation=n_passed,
            fine_tuned_model_path=fine_tuned_model_path,
            significance=significance,
            slice_check=slice_check,
            prompt_regression=None,
            recommendation=recommendation,
        )
        # Step 8 — VERSION: write once, never overwrite a prior iteration.
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.json"
        if report_path.exists():
            raise FileExistsError(
                f"{report_path} already exists — iterations are versioned and never overwritten."
            )
        report_path.write_text(report.model_dump_json(indent=2))
        logger.info(f"Improvement iteration {self.iteration} report written to {report_path}")
        return report

    def _save_questions(self, questions: List[Question], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q.model_dump_json() + "\n")

    @staticmethod
    def _baseline_questions(baseline_results: List[EvalResult]) -> List[Question]:
        """Reconstruct the Question objects the baseline run was scored on,
        so the re-benchmark step evaluates the identical question set and
        Step 6's paired significance test has shared question_ids to align on."""
        return [
            Question(
                id=r.question_id,
                text=r.question,
                ground_truth=r.ground_truth,
                source_type=r.source_type,
                category=r.category,
            )
            for r in baseline_results
        ]

