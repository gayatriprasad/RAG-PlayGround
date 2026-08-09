"""
Statistical significance layer — Skill 43 (the DS backbone).

Decides whether a benchmark difference between two configs is REAL or noise.
These are paired comparisons — config A and config B answer the SAME
questions — so paired tests apply and are more powerful than unpaired ones.

Never report a bare delta. Every comparison returns a SignificanceResult with
a bootstrap confidence interval, a paired significance test, an effect size,
and a human-readable verdict.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Sequence

import numpy as np

from raglab.config import StatsCfg
from raglab.types import EvalResult, SignificanceResult

logger = logging.getLogger(__name__)

MetricName = Literal["overall_score", "answer_correct", "completeness"]


def _metric_value(result: EvalResult, metric: str) -> float:
    """Extract a numeric metric value from an EvalResult, coercing bools to 0/1."""
    value = getattr(result, metric)
    if value is None:
        raise ValueError(
            f"EvalResult {result.question_id!r} has no scored value for metric {metric!r} "
            "— results must be scored (BenchmarkScorer) before comparison."
        )
    return float(value)


def _align_paired(
    results_a: Sequence[EvalResult], results_b: Sequence[EvalResult], metric: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align two result sets by question_id (paired design) and extract metric arrays."""
    by_id_a = {r.question_id: r for r in results_a}
    by_id_b = {r.question_id: r for r in results_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))
    if not shared_ids:
        raise ValueError("No shared question_id values between results_a and results_b — cannot pair.")

    a_vals = np.array([_metric_value(by_id_a[qid], metric) for qid in shared_ids])
    b_vals = np.array([_metric_value(by_id_b[qid], metric) for qid in shared_ids])
    return a_vals, b_vals, len(shared_ids)


def bootstrap_ci(scores: List[float], cfg: StatsCfg) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (mean, ci_lower, ci_upper)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires at least one score.")

    rng = np.random.default_rng(42)  # fixed seed -> reproducible (Coding Rule 6)
    boots = np.array(
        [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(cfg.bootstrap_samples)]
    )
    alpha_tail = (1 - cfg.confidence_level) / 2 * 100
    lo = float(np.percentile(boots, alpha_tail))
    hi = float(np.percentile(boots, 100 - alpha_tail))
    return float(arr.mean()), lo, hi


def _bootstrap_paired_delta_ci(
    a_vals: np.ndarray, b_vals: np.ndarray, cfg: StatsCfg
) -> tuple[float, float]:
    """Bootstrap CI on the paired delta (mean_a - mean_b), resampling question indices."""
    rng = np.random.default_rng(42)
    n = a_vals.size
    diffs = a_vals - b_vals
    boots = np.array(
        [rng.choice(diffs, size=n, replace=True).mean() for _ in range(cfg.bootstrap_samples)]
    )
    alpha_tail = (1 - cfg.confidence_level) / 2 * 100
    lo = float(np.percentile(boots, alpha_tail))
    hi = float(np.percentile(boots, 100 - alpha_tail))
    return lo, hi


def _cohens_d(diffs: np.ndarray) -> float:
    """Cohen's d for paired continuous data: mean diff / sd of differences."""
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diffs.mean() / sd)


def _verdict(
    metric: str, delta: float, p_value: float, effect_size: float, alpha: float, significant: bool
) -> str:
    direction = "B" if delta < 0 else "A"
    magnitude = abs(delta)
    if not significant:
        return f"No significant difference (p={p_value:.3f}) — the {magnitude:.1%} gap on {metric} is within noise."
    return (
        f"{direction} significantly better on {metric} "
        f"(p={p_value:.3f}, effect_size={effect_size:.2f}, delta={magnitude:.1%})"
    )


def compare(
    results_a: List[EvalResult],
    results_b: List[EvalResult],
    metric: MetricName,
    cfg: StatsCfg,
    config_a_name: str = "A",
    config_b_name: str = "B",
) -> SignificanceResult:
    """Paired comparison of config A vs config B on `metric`.

    Chooses the test by metric type:
    - "answer_correct" (binary) -> McNemar's test on the discordant pairs
    - "overall_score" / "completeness" (continuous, bounded 0-1) -> Wilcoxon
      signed-rank (default) or paired t-test, per cfg.continuous_test
    """
    a_vals, b_vals, _n = _align_paired(results_a, results_b, metric)
    return _compare_arrays(a_vals, b_vals, metric, cfg, config_a_name, config_b_name)


def _paired_continuous_test(
    a_vals: np.ndarray, b_vals: np.ndarray, cfg: StatsCfg
) -> tuple[float, float, str]:
    from scipy import stats

    diffs = a_vals - b_vals
    effect_size = _cohens_d(diffs)

    if np.allclose(diffs, 0):
        # Degenerate distribution guard — scipy's wilcoxon/ttest_rel raise/NaN on all-zero diffs.
        return 1.0, 0.0, cfg.continuous_test

    if cfg.continuous_test == "paired_t":
        _, p_value = stats.ttest_rel(a_vals, b_vals)
        return float(p_value), effect_size, "paired_t"

    _, p_value = stats.wilcoxon(a_vals, b_vals)
    return float(p_value), effect_size, "wilcoxon"


def _mcnemar(a_vals: np.ndarray, b_vals: np.ndarray) -> tuple[float, float, str]:
    from statsmodels.stats.contingency_tables import mcnemar

    a_bool = a_vals.astype(bool)
    b_bool = b_vals.astype(bool)
    # discordant pairs: A right/B wrong (b10), A wrong/B right (b01)
    b10 = int(np.sum(a_bool & ~b_bool))
    b01 = int(np.sum(~a_bool & b_bool))
    table = [[0, b01], [b10, 0]]

    if b10 + b01 == 0:
        # No discordant pairs at all — configs agree on every question, no signal to test.
        return 1.0, 0.0, "mcnemar"

    result = mcnemar(table, exact=(b10 + b01 < 25), correction=True)
    n = a_vals.size
    risk_difference = (b10 - b01) / n  # positive => A wins more discordant pairs
    return float(result.pvalue), float(risk_difference), "mcnemar"


def correct_pvalues(results: List[SignificanceResult], cfg: StatsCfg) -> List[SignificanceResult]:
    """Multiple-comparison correction for >2-config comparisons (arena / nightly matrix).

    Mutates and returns new SignificanceResult objects with p_value_corrected set
    and `significant`/`practically_significant`/`verdict` re-derived from the
    corrected p-value.
    """
    if not results:
        return []
    if cfg.correction_method == "none":
        return results

    from statsmodels.stats.multitest import multipletests

    raw_pvalues = [r.p_value for r in results]
    method = "fdr_bh" if cfg.correction_method == "benjamini_hochberg" else "bonferroni"
    reject, corrected, _, _ = multipletests(raw_pvalues, alpha=cfg.alpha, method=method)

    corrected_results = []
    for r, p_corr, sig in zip(results, corrected, reject):
        practically_significant = bool(sig) and abs(r.delta) > cfg.min_effect_size
        verdict = _verdict(r.metric, r.delta, float(p_corr), r.effect_size, cfg.alpha, bool(sig))
        corrected_results.append(
            r.model_copy(
                update={
                    "p_value_corrected": float(p_corr),
                    "significant": bool(sig),
                    "practically_significant": practically_significant,
                    "verdict": verdict,
                }
            )
        )
    return corrected_results


def significance_matrix(
    configs: Dict[str, List[EvalResult]], metric: MetricName, cfg: StatsCfg
) -> List[SignificanceResult]:
    """All pairwise comparisons among N configs, multiple-comparison corrected.

    Powers the arena leaderboard: each cell shows the corrected significance,
    not just which point estimate is higher.
    """
    names = list(configs.keys())
    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            pairwise.append(
                compare(
                    configs[name_a],
                    configs[name_b],
                    metric,
                    cfg,
                    config_a_name=name_a,
                    config_b_name=name_b,
                )
            )
    return correct_pvalues(pairwise, cfg)


def compare_from_records(
    records_a: List[dict],
    records_b: List[dict],
    metric: MetricName,
    cfg: StatsCfg,
    config_a_name: str = "A",
    config_b_name: str = "B",
) -> SignificanceResult:
    """Same as `compare()`, but accepts plain dict rows (e.g. from a benchmark
    results CSV loaded via pandas) instead of full EvalResult objects — avoids
    needing to reconstruct retrieved_chunks/ground_truth/etc. just to compare
    two already-scored runs. Rows must have `question_id` and the metric field.
    """
    by_id_a = {r["question_id"]: r for r in records_a}
    by_id_b = {r["question_id"]: r for r in records_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))
    if not shared_ids:
        raise ValueError("No shared question_id values between records_a and records_b — cannot pair.")

    a_vals = np.array([float(by_id_a[qid][metric]) for qid in shared_ids])
    b_vals = np.array([float(by_id_b[qid][metric]) for qid in shared_ids])
    return _compare_arrays(a_vals, b_vals, metric, cfg, config_a_name, config_b_name)


def _compare_arrays(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    metric: str,
    cfg: StatsCfg,
    config_a_name: str,
    config_b_name: str,
) -> SignificanceResult:
    """Shared core: given two aligned numeric arrays, run the paired test and
    build a SignificanceResult. Both `compare()` and `compare_from_records()`
    funnel into this after extracting/aligning their respective input shapes."""
    n = a_vals.size
    mean_a, mean_b = float(a_vals.mean()), float(b_vals.mean())
    delta = mean_a - mean_b
    ci_lower, ci_upper = _bootstrap_paired_delta_ci(a_vals, b_vals, cfg)

    if metric == "answer_correct":
        p_value, effect_size, test_used = _mcnemar(a_vals, b_vals)
    else:
        p_value, effect_size, test_used = _paired_continuous_test(a_vals, b_vals, cfg)

    significant = p_value < cfg.alpha
    practically_significant = significant and abs(delta) > cfg.min_effect_size
    verdict = _verdict(metric, delta, p_value, effect_size, cfg.alpha, significant)

    return SignificanceResult(
        config_a=config_a_name,
        config_b=config_b_name,
        metric=metric,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        p_value_corrected=None,
        effect_size=float(effect_size),
        test_used=test_used,
        n_questions=n,
        significant=significant,
        practically_significant=practically_significant,
        verdict=verdict,
    )
