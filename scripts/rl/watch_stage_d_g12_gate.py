#!/usr/bin/env python3
"""Stop Gen-12 after its first reliable deterministic bucket if it regresses."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


WINDOWS = ((0.0, 30.0), (55.0, 80.0), (105.0, 130.0), (130.0, 160.1))


def window_scores(row: dict[str, Any]) -> list[int]:
    totals = [0 for _ in WINDOWS]
    for event in row.get("timeline") or []:
        if event.get("ev") != "score":
            continue
        timestamp = float(event.get("t", -1.0))
        count = int(event.get("q", 0) or 0) + int(event.get("u", 0) or 0)
        for index, (start, end) in enumerate(WINDOWS):
            if start <= timestamp < end:
                totals[index] += count
                break
    return totals


def load_deterministic_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        collector = int(row.get("collector", -1) or -1)
        if row.get("reset_mode") == "full" and collector >= 0 and collector % 2 == 0:
            rows.append(row)
    return rows


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket = rows[-40:]
    scores = [int(row.get("scored", 0) or 0) for row in bucket]
    cycles = [int(row.get("cycles_completed", 0) or 0) for row in bucket]
    windows = [window_scores(row) for row in bucket]
    result = {
        "deterministic_episodes": len(rows),
        "bucket_size": len(bucket),
        "score_mean": statistics.fmean(scores),
        "score_median": statistics.median(scores),
        "score_max": max(scores),
        "cycles_mean": statistics.fmean(cycles),
        "window_means": [
            statistics.fmean(values[index] for values in windows) for index in range(4)
        ],
        "all_four_window_rate": statistics.fmean(
            float(all(value > 0 for value in values)) for values in windows
        ),
        "unhealthy": sum(row.get("terminal_reason") == "unhealthy" for row in bucket),
        "policy_train_steps": max(
            int(row.get("policy_train_steps", 0) or 0) for row in bucket
        ),
    }
    result["passed"] = bool(
        result["score_mean"] >= 115.0
        and result["cycles_mean"] >= 2.1
        and result["window_means"][3] >= 4.0
        and result["all_four_window_rate"] >= 0.15
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    telemetry = args.run_dir / "cycle_telemetry.jsonl"
    while True:
        rows = load_deterministic_rows(telemetry)
        if len(rows) >= 40:
            result = evaluate(rows)
            result["run_dir"] = str(args.run_dir.resolve())
            result["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0 if result["passed"] else 1
        print(f"waiting deterministic={len(rows)}/40", flush=True)
        time.sleep(max(10.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
