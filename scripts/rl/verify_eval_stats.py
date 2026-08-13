"""Recompute published per-episode evaluation statistics.

The Stage-A fixed diagnostic publishes per-episode rows plus summary headers.
This script verifies every published count/mean/maximum and displays sample
standard deviations for comparison with the technical report, so the numbers
can be re-derived without NVIDIA Isaac Sim.

Usage:
    python scripts/rl/verify_eval_stats.py [path/to/eval.json]

The default path is runs/eval_stageA_clean.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "runs" / "eval_stageA_clean.json"

# Fields recomputed from per-episode rows and compared to the summary header.
SUMMARY_FIELDS = {
    "mean_return": ("return", np.mean),
    "mean_collected": ("collected", np.mean),
    "max_collected": ("collected", np.max),
    "mean_scored": ("scored", np.mean),
    "max_scored": ("scored", np.max),
}
REQUIRED_ROW_FIELDS = {"return", "collected", "scored"}


def recompute_block(block: dict) -> dict[str, float]:
    """Recompute summary statistics from the per_episode rows of one block."""
    rows = block.get("per_episode")
    if not isinstance(rows, list) or not rows:
        raise ValueError("per_episode is empty; cannot recompute summaries")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"per_episode row {index} is not an object")
        missing = REQUIRED_ROW_FIELDS.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"per_episode row {index} is missing: {names}")
    values = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in REQUIRED_ROW_FIELDS
    }
    recomputed: dict[str, float] = {}
    recomputed["episodes"] = float(len(rows))
    for summary_key, (row_key, reducer) in SUMMARY_FIELDS.items():
        recomputed[summary_key] = float(reducer(values[row_key]))
    # Sample standard deviation for the headline "mean +/- SD" columns.
    recomputed["sd_return"] = float(values["return"].std(ddof=1))
    recomputed["sd_collected"] = float(values["collected"].std(ddof=1))
    recomputed["sd_scored"] = float(values["scored"].std(ddof=1))
    return recomputed


def check_block(name: str, block: dict) -> list[str]:
    """Return a list of mismatch descriptions for one policy block."""
    problems: list[str] = []
    recomputed = recompute_block(block)
    required_headers = {"episodes", *SUMMARY_FIELDS}
    for summary_key in sorted(required_headers):
        if summary_key not in block:
            problems.append(f"{name}: missing summary header {summary_key}")
            continue
        reported_value = block[summary_key]
        # Headers are published rounded to 2 decimals (same as the report).
        if round(recomputed[summary_key], 2) != float(reported_value):
            problems.append(
                f"{name}: {summary_key} reported={reported_value} "
                f"recomputed={recomputed[summary_key]:.4f} (rounded {round(recomputed[summary_key], 2)})"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_EVAL)
    args = ap.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("top-level evaluation JSON must be an object")
    problems: list[str] = []
    for name, block in data.items():
        if not isinstance(block, dict) or "per_episode" not in block:
            print(f"skip: {name} (no per-episode rows)")
            continue
        recomputed = recompute_block(block)
        problems.extend(check_block(name, block))
        print(
            f"{name:12s} collected={recomputed['mean_collected']:.2f} "
            f"+/- {recomputed['sd_collected']:.2f} | "
            f"return={recomputed['mean_return']:.2f} +/- {recomputed['sd_return']:.2f} | "
            f"scored={recomputed['mean_scored']:.2f} | "
            f"max_collected={recomputed['max_collected']:.0f}"
        )

    if problems:
        print("\nFAILED checks:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAll published summary statistics match the per-episode rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
