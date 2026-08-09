"""
Judge calibration — Skill 44(A).

The eval pipeline leans on LLM-as-judge (eval/scorer.py's LLMJudgeMetric), which
has documented position, verbosity, and self-preference biases. If the judge is
wrong, every score the platform reports is unfalsifiable. This module validates
the judge against a human-labeled sample before its scores are trusted.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Optional

from raglab.config import StatsCfg
from raglab.types import CalibrationResult, EvalResult

logger = logging.getLogger(__name__)


class JudgeCalibrator:
    """Builds a human-labeling sample from real eval results, then computes
    judge-vs-human agreement once the sample has been hand-labeled."""

    def build_sample(
        self, results: List[EvalResult], n: int = 40, output_path: str = "golden/judge_calibration_sample.jsonl"
    ) -> Path:
        """Write a stratified sample (across source_type + category +
        correct/incorrect) to a JSONL file for a human to fill in `human_correct`
        and `human_completeness` by hand.
        """
        scored = [r for r in results if r.answer_correct is not None and r.completeness is not None]
        if not scored:
            raise ValueError("build_sample requires already-scored EvalResults (answer_correct/completeness set).")

        strata: dict[tuple, List[EvalResult]] = {}
        for r in scored:
            key = (r.source_type, r.category, bool(r.answer_correct))
            strata.setdefault(key, []).append(r)

        rng = random.Random(42)  # fixed seed -> reproducible sample (Coding Rule 6)
        per_stratum = max(1, n // max(1, len(strata)))

        sample: List[EvalResult] = []
        for group in strata.values():
            take = group if len(group) <= per_stratum else rng.sample(group, per_stratum)
            sample.extend(take)
        if len(sample) > n:
            sample = rng.sample(sample, n)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in sample:
                row = {
                    "question_id": r.question_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "predicted_answer": r.predicted_answer,
                    "judge_correct": r.answer_correct,
                    "judge_completeness": r.completeness,
                    "human_correct": None,
                    "human_completeness": None,
                }
                f.write(json.dumps(row) + "\n")

        logger.info(f"Wrote {len(sample)} rows for human labeling to {path}")
        return path

    def calibrate(self, sample_path: str, cfg: StatsCfg, llm_client=None) -> CalibrationResult:
        """Load a human-labeled sample and compute judge validity metrics.

        Rows with `human_correct`/`human_completeness` still null are skipped
        (not yet labeled) — calibrate() only scores the rows a human has filled in.

        If `llm_client` is given, position-bias is actually measured by
        re-running the judge with the ground_truth/predicted_answer
        presentation order swapped in the prompt; otherwise flip_rate is
        reported as unmeasured (None-equivalent 0.0, with a caveat noting
        this rather than implying "verified unbiased").
        """
        rows = self._load_labeled_rows(sample_path)
        if not rows:
            raise ValueError(
                f"No human-labeled rows found in {sample_path} — fill in "
                "human_correct/human_completeness before calling calibrate()."
            )

        kappa = self._cohens_kappa(rows)
        completeness_corr = self._completeness_correlation(rows)
        flip_rate, position_bias_measured = self._position_bias_flip_rate(rows, llm_client)
        reliable = kappa >= cfg.min_judge_kappa

        caveats = []
        if not reliable:
            caveats.append(
                f"Judge agreement with humans is low (kappa={kappa:.2f}) — "
                "treat absolute scores with caution; relative comparisons "
                "(A vs B on the same questions) are more robust than absolute scores."
            )
        else:
            caveats.append("Judge is well-calibrated against the human sample.")
        if not position_bias_measured:
            caveats.append(
                "Position-bias flip rate was not measured (no llm_client supplied to calibrate())."
            )

        return CalibrationResult(
            n_samples=len(rows),
            cohens_kappa=kappa,
            completeness_correlation=completeness_corr,
            position_bias_flip_rate=flip_rate,
            reliable=reliable,
            caveat=" ".join(caveats),
        )

    def _load_labeled_rows(self, sample_path: str) -> List[dict]:
        path = Path(sample_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration sample not found: {path}")

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("human_correct") is not None:
                    rows.append(row)
        return rows

    def _cohens_kappa(self, rows: List[dict]) -> float:
        from sklearn.metrics import cohen_kappa_score

        judge = [bool(r["judge_correct"]) for r in rows]
        human = [bool(r["human_correct"]) for r in rows]
        if len(set(judge)) == 1 and len(set(human)) == 1:
            # Both raters agree on every row and never disagree — perfect agreement,
            # but sklearn's kappa is undefined (0/0) in this degenerate case.
            return 1.0 if judge == human else 0.0
        return float(cohen_kappa_score(judge, human))

    def _completeness_correlation(self, rows: List[dict]) -> float:
        from scipy.stats import spearmanr

        judge_completeness = [r["judge_completeness"] for r in rows if r.get("human_completeness") is not None]
        human_completeness = [r["human_completeness"] for r in rows if r.get("human_completeness") is not None]
        if len(judge_completeness) < 2:
            return 0.0
        corr, _ = spearmanr(judge_completeness, human_completeness)
        return float(corr) if corr == corr else 0.0  # NaN guard (constant input)

    def _position_bias_flip_rate(self, rows: List[dict], llm_client=None) -> tuple[float, bool]:
        """Fraction of verdicts that flip when the judge prompt's presentation
        order (ground_truth vs predicted_answer) is swapped for the same content.

        Returns (flip_rate, measured). If `llm_client` is None, returns
        (0.0, False) — NOT measured, distinct from "measured and found to be 0".
        """
        if llm_client is None:
            return 0.0, False

        flips = 0
        for row in rows:
            original_verdict = bool(row["judge_correct"])
            swapped_verdict = self._judge_correctness_swapped(
                question=row["question"],
                ground_truth=row["ground_truth"],
                predicted_answer=row["predicted_answer"],
                llm_client=llm_client,
            )
            if original_verdict != swapped_verdict:
                flips += 1
        return flips / len(rows), True

    def _judge_correctness_swapped(
        self, question: str, ground_truth: str, predicted_answer: str, llm_client
    ) -> bool:
        """Same correctness prompt as eval/scorer.py's LLMJudgeMetric, but with
        the ground_truth/predicted_answer presentation order swapped."""
        messages = [
            {"role": "system", "content": "You are an evaluation judge. Reply YES or NO only."},
            {
                "role": "user",
                "content": (
                    f"Does the predicted answer correctly answer the question "
                    f"given the ground truth?\n\n"
                    f"Question: {question}\n"
                    f"Predicted answer: {predicted_answer}\n"
                    f"Ground truth: {ground_truth}\n\n"
                    f"Reply YES or NO only."
                ),
            },
        ]
        try:
            answer = llm_client.complete(messages, temperature=0.0, max_tokens=10).strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            logger.warning(f"Position-bias swapped judge call failed: {e}")
            return False
