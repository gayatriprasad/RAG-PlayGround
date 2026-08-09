"""
create_stratified_sample.py — Skill 49(D)/Skill 50(H).

Builds a stratified regression sample (N per category) from the golden
question set, so nightly/PR regression runs use a representative slice
instead of an unstratified random sample (see Failure Mode Register:
"20-question sample is representative").

Usage:
    python tests/regression/create_stratified_sample.py \
        --golden ./golden/questions.jsonl \
        --out ./tests/regression/stratified_sample.jsonl \
        --per-category 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def create_stratified_sample(golden_path: Path, per_category: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = defaultdict(list)

    with open(golden_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            by_category[item.get("category", "unknown")].append(item)

    sample: list[dict] = []
    for category, items in sorted(by_category.items()):
        rng.shuffle(items)
        sample.extend(items[:per_category])

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.golden.exists():
        raise SystemExit(f"Golden file not found: {args.golden}")

    sample = create_stratified_sample(args.golden, args.per_category, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(sample)} stratified questions to {args.out}")


if __name__ == "__main__":
    main()
