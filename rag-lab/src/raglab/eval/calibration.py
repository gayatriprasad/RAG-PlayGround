"""
Uncertainty calibration — Skill 57.

Bootstrap CIs (significance.py) answer "is this difference real?" Calibration
answers a different question: "when we say 0.8 confidence, is the model
right 80% of the time?" Uncalibrated confidence scores lead to bad decisions
downstream — trusting high-confidence wrong answers, distrusting low-
confidence correct ones.
"""

from __future__ import annotations

import logging
from typing import List, Literal

import numpy as np

from raglab.types import CalibrationCurve, EvalResult

logger = logging.getLogger(__name__)

_EPSILON = 1e-9
_MISCALIBRATION_THRESHOLD = 0.05  # |predicted - actual| gap to flag a bin as over/underconfident


def _scored_pairs(results: List[EvalResult]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (predicted_confidence, correct_label) arrays from results that
    have both overall_score and answer_correct scored."""
    scored = [r for r in results if r.overall_score is not None and r.answer_correct is not None]
    if not scored:
        raise ValueError(
            "UncertaintyCalibrator requires already-scored EvalResults "
            "(overall_score/answer_correct set) — score results with BenchmarkScorer first."
        )
    predicted = np.array([float(r.overall_score) for r in scored])
    actual = np.array([float(bool(r.answer_correct)) for r in scored])
    return predicted, actual


class UncertaintyCalibrator:
    """Evaluates whether confidence scores (EvalResult.overall_score) are
    calibrated against actual correctness (EvalResult.answer_correct)."""

    def calibration_curve(self, results: List[EvalResult], n_bins: int = 10) -> CalibrationCurve:
        """Bin results by predicted confidence. For each bin, compute mean
        predicted confidence and actual accuracy. A perfectly calibrated
        system's per-bin points sit on the diagonal y=x."""
        predicted, actual = _scored_pairs(results)
        edges = np.linspace(0.0, 1.0, n_bins + 1)

        mean_predicted = [0.0] * n_bins
        actual_accuracy = [0.0] * n_bins
        bin_counts = [0] * n_bins
        overconfident_bins: List[int] = []
        underconfident_bins: List[int] = []

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (predicted >= lo) & (predicted <= hi) if i == n_bins - 1 else (predicted >= lo) & (predicted < hi)
            count = int(mask.sum())
            bin_counts[i] = count
            if count == 0:
                continue

            bin_predicted = float(predicted[mask].mean())
            bin_actual = float(actual[mask].mean())
            mean_predicted[i] = bin_predicted
            actual_accuracy[i] = bin_actual

            gap = bin_predicted - bin_actual
            if gap > _MISCALIBRATION_THRESHOLD:
                overconfident_bins.append(i)
            elif -gap > _MISCALIBRATION_THRESHOLD:
                underconfident_bins.append(i)

        total = sum(bin_counts)
        ece = (
            sum(count * abs(p - a) for count, p, a in zip(bin_counts, mean_predicted, actual_accuracy)) / total
            if total
            else 0.0
        )

        return CalibrationCurve(
            bins=[float(e) for e in edges],
            mean_predicted=mean_predicted,
            actual_accuracy=actual_accuracy,
            bin_counts=bin_counts,
            ece=ece,
            overconfident_bins=overconfident_bins,
            underconfident_bins=underconfident_bins,
        )

    def expected_calibration_error(self, results: List[EvalResult], n_bins: int = 10) -> float:
        """Weighted average of |predicted_confidence - actual_accuracy| across
        bins. Lower = better calibrated. ECE > 0.1 is a warning, > 0.2 is a problem."""
        return self.calibration_curve(results, n_bins=n_bins).ece

    def reliability_diagram(self, curve: CalibrationCurve) -> dict:
        """Chart-ready data for a reliability diagram: predicted vs actual
        per (non-empty) bin, plus the perfect-calibration diagonal reference."""
        points = [
            {"predicted": p, "actual": a, "count": c}
            for p, a, c in zip(curve.mean_predicted, curve.actual_accuracy, curve.bin_counts)
            if c > 0
        ]
        return {
            "points": points,
            "diagonal": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
            "ece": curve.ece,
            "overconfident_bins": curve.overconfident_bins,
            "underconfident_bins": curve.underconfident_bins,
        }

    def recalibrate(
        self,
        results: List[EvalResult],
        method: Literal["platt", "isotonic", "temperature"],
    ) -> List[EvalResult]:
        """Adjust predicted scores to improve calibration. Returns new
        EvalResult objects with the recalibrated score stashed in
        `metadata["recalibrated_score"]` — this is an analysis tool, the
        original `overall_score` is never overwritten."""
        predicted, actual = _scored_pairs(results)
        if len(set(actual.tolist())) < 2:
            raise ValueError(
                "recalibrate requires both correct and incorrect examples in `results` "
                "to fit a calibration mapping."
            )

        if method == "platt":
            recalibrated = self._platt_scale(predicted, actual)
        elif method == "isotonic":
            recalibrated = self._isotonic_scale(predicted, actual)
        elif method == "temperature":
            recalibrated = self._temperature_scale(predicted, actual)
        else:
            raise ValueError(f"Unknown recalibration method: {method!r}")

        scored = [r for r in results if r.overall_score is not None and r.answer_correct is not None]
        return [
            r.model_copy(
                update={
                    "metadata": {
                        **r.metadata,
                        "recalibrated_score": float(new_score),
                        "recalibration_method": method,
                    }
                }
            )
            for r, new_score in zip(scored, recalibrated)
        ]

    @staticmethod
    def _platt_scale(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Logistic regression on predicted scores vs actual binary labels."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(predicted.reshape(-1, 1), actual)
        return model.predict_proba(predicted.reshape(-1, 1))[:, 1]

    @staticmethod
    def _isotonic_scale(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Non-parametric monotonic mapping from predicted score to accuracy."""
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        return model.fit_transform(predicted, actual)

    @staticmethod
    def _temperature_scale(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Divide the score's logit by a learned temperature T (T>1 softens,
        T<1 sharpens), fit by minimizing negative log-likelihood."""
        from scipy.optimize import minimize_scalar

        clipped = np.clip(predicted, _EPSILON, 1 - _EPSILON)
        logits = np.log(clipped / (1 - clipped))

        def _nll(temperature: float) -> float:
            probs = 1 / (1 + np.exp(-logits / temperature))
            probs = np.clip(probs, _EPSILON, 1 - _EPSILON)
            return -float(np.mean(actual * np.log(probs) + (1 - actual) * np.log(1 - probs)))

        fit = minimize_scalar(_nll, bounds=(0.05, 20.0), method="bounded")
        temperature = float(fit.x)
        return 1 / (1 + np.exp(-logits / temperature))
