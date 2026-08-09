"""
update_baseline.py — Skill 49(C).

Honestly re-initializes tests/regression/baseline_scores.json from a real
benchmark results CSV. Never fabricates a score — refuses to run if the
results CSV doesn't exist or is empty.

Usage:
    python tests/regression/update_baseline.py --experiment 02_retrieval_comparison
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

_RAG_LAB_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"
_BASELINE_PATH = Path(__file__).resolve().parent / "baseline_scores.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    result_csv = _OUT_DIR / args.experiment / f"{args.experiment}_results.csv"
    if not result_csv.exists():
        raise SystemExit(
            f"No results found at {result_csv}. Run `make eval` (or "
            f"`python -m raglab.run_experiment --config experiments/{args.experiment}/config.yaml`) first."
        )

    import pandas as pd

    df = pd.read_csv(result_csv)
    if len(df) == 0:
        raise SystemExit(f"{result_csv} is empty — refusing to write a fabricated baseline.")

    baseline = {
        "_comment": (
            "Regression baseline for `make eval`'s default experiment. Populated by "
            "scripts/update_baseline.py after a real benchmark run — never hand-edited "
            "with fabricated numbers."
        ),
        "experiment": args.experiment,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(df),
        "overall_score_mean": round(float(df["overall_score"].mean()), 4) if "overall_score" in df.columns else 0.0,
        "recall_at_3_mean": round(float(df["recall_at_3"].mean()), 4) if "recall_at_3" in df.columns else 0.0,
        "tolerance": args.tolerance,
    }

    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote baseline: {baseline}")


if __name__ == "__main__":
    main()
