"""
Slice-analysis guard — Skill 44(B).

Guards against Simpson's paradox: an aggregate "config A wins" claim can hide
the fact that config A actually loses within every individual slice (source
type, category) once the data is broken down. The reporter must refuse to
print an aggregate-only "best config" claim when this guard fires.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from raglab.types import EvalResult, SliceCheckResult

logger = logging.getLogger(__name__)


class SliceChecker:
    """Computes the aggregate winner and the winner within each source_type
    and category slice, and flags disagreement between them."""

    def check_slices(
        self, configs: Dict[str, List[EvalResult]], metric: str, min_slice_size: int = 3
    ) -> SliceCheckResult:
        """
        Args:
            configs: {config_name: [EvalResult, ...]} — same question set per config.
            metric: "overall_score" | "completeness" | "answer_correct"
            min_slice_size: slices with fewer than this many questions are skipped
                (too few samples to trust a "winner" within that slice).
        """
        if len(configs) < 2:
            raise ValueError("check_slices requires at least 2 configs to compare.")

        aggregate_winner = self._winner(configs, metric)

        per_slice_winners: Dict[str, str] = {}
        for slice_key, slice_configs in self._group_by_slice(configs, metric).items():
            if not self._enough_samples(slice_configs, min_slice_size):
                continue
            per_slice_winners[slice_key] = self._winner(slice_configs, metric)

        consistent = all(winner == aggregate_winner for winner in per_slice_winners.values())

        warning = None
        if not consistent:
            losing_slices = [k for k, w in per_slice_winners.items() if w != aggregate_winner]
            warning = (
                f"Config '{aggregate_winner}' wins on aggregate but loses on {losing_slices} — "
                "Simpson's paradox risk. Do not report the aggregate alone."
            )

        return SliceCheckResult(
            metric=metric,
            aggregate_winner=aggregate_winner,
            per_slice_winners=per_slice_winners,
            consistent=consistent,
            warning=warning,
        )

    def _winner(self, configs: Dict[str, List[EvalResult]], metric: str) -> str:
        means = {name: self._mean(results, metric) for name, results in configs.items() if results}
        if not means:
            raise ValueError("No results available to determine a winner.")
        return max(means, key=means.get)

    def _mean(self, results: List[EvalResult], metric: str) -> float:
        values = [float(getattr(r, metric)) for r in results if getattr(r, metric) is not None]
        return sum(values) / len(values) if values else 0.0

    def _group_by_slice(
        self, configs: Dict[str, List[EvalResult]], metric: str
    ) -> Dict[str, Dict[str, List[EvalResult]]]:
        """Build {slice_key: {config_name: [results in that slice]}} for both
        source_type slices (e.g. "source_type=confluence") and category slices
        (e.g. "category=multi_doc")."""
        slices: Dict[str, Dict[str, List[EvalResult]]] = {}
        for config_name, results in configs.items():
            for r in results:
                for slice_key in (f"source_type={r.source_type}", f"category={r.category}"):
                    slices.setdefault(slice_key, {}).setdefault(config_name, []).append(r)
        return slices

    def _enough_samples(self, slice_configs: Dict[str, List[EvalResult]], min_size: int) -> bool:
        return all(len(results) >= min_size for results in slice_configs.values())
