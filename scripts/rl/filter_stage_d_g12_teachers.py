#!/usr/bin/env python3
"""Select complete, efficient Stage-D deterministic teacher episodes.

The seed-mine evaluator captures every episode that completes any qualified
cycle.  That capture gate is intentionally broad.  This script applies the
much stricter Gen-12 curriculum gate before behavior-cloning data is exposed
to the learner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


WINDOWS = ((0.0, 30.0), (55.0, 80.0), (105.0, 130.0), (130.0, 160.1))
WINDOW_MINIMUMS = (50, 25, 20, 15)
REQUIRED_FIELDS = {"obs", "proprio", "privileged", "action", "reward", "done", "metadata"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_FIELDS.difference(archive.files)
        if missing:
            raise ValueError(f"missing capture fields: {sorted(missing)}")
        raw = np.asarray(archive["metadata"], dtype=np.uint8)
    metadata = json.loads(raw.tobytes().decode("utf-8"))
    if metadata.get("schema") != "stagec_training_episode_v1":
        raise ValueError(f"unexpected capture schema: {metadata.get('schema')!r}")
    return metadata


def score_windows(episode: dict[str, Any]) -> list[int]:
    totals = [0 for _ in WINDOWS]
    for event in episode.get("timeline") or []:
        if event.get("ev") != "score":
            continue
        timestamp = float(event.get("t", -1.0))
        count = int(event.get("q", 0) or 0) + int(event.get("u", 0) or 0)
        for index, (start, end) in enumerate(WINDOWS):
            if start <= timestamp < end:
                totals[index] += count
                break
    return totals


def inspect_capture(path: Path, source_sha256: str) -> dict[str, Any]:
    metadata = load_metadata(path)
    episode = metadata.get("episode") or {}
    cycle_times = [
        round(float(step) / 10.0, 3)
        for step in (episode.get("cycle_success_steps") or [])
    ]
    repeat_gaps = [
        round(cycle_times[index] - cycle_times[index - 1], 3)
        for index in range(1, len(cycle_times))
    ]
    windows = score_windows(episode)
    ramp_attempts = int(episode.get("ramp_out_attempts", 0) or 0)
    ramp_successes = int(episode.get("ramp_out_successes", 0) or 0)
    ramp_rate = ramp_successes / ramp_attempts if ramp_attempts else 0.0

    checks = {
        "source_checkpoint": episode.get("checkpoint_sha256") == source_sha256,
        "deterministic": episode.get("action_mode") == "deterministic",
        "full_match": episode.get("mode") == "full"
        and float(episode.get("episode_len_s", 0.0) or 0.0) >= 159.0,
        "healthy_horizon": episode.get("terminal_reason") == "horizon",
        "score": int(episode.get("scored", 0) or 0) >= 170,
        "collection": int(episode.get("collected", 0) or 0) >= 180,
        "four_cycles": int(episode.get("cycles_completed", 0) or 0) >= 4
        and len(cycle_times) >= 4,
        "all_windows": all(
            actual >= required
            for actual, required in zip(windows, WINDOW_MINIMUMS, strict=True)
        ),
        "final_cycle": bool(cycle_times) and cycle_times[-1] <= 145.0,
        "cycle_spacing": bool(repeat_gaps) and max(repeat_gaps) <= 37.0,
        "clean_dumps": (
            int(episode.get("dump_empty_completions", 0) or 0)
            >= int(episode.get("cycles_completed", 0) or 0) + 1
            and int(episode.get("partial_dumps", 0) or 0) <= 2
        ),
        "ramp_usage": ramp_successes >= 3 and ramp_rate >= 0.75,
    }
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "selected": all(checks.values()),
        "checks": checks,
        "score": int(episode.get("scored", 0) or 0),
        "collected": int(episode.get("collected", 0) or 0),
        "cycles": int(episode.get("cycles_completed", 0) or 0),
        "cycle_times_s": cycle_times,
        "repeat_gaps_s": repeat_gaps,
        "window_scores": windows,
        "outer_rail_fraction": float(episode.get("outer_rail_fraction", 0.0) or 0.0),
        "ramp_out": {"successes": ramp_successes, "attempts": ramp_attempts},
        "env_seed": int(episode.get("env_seed", 0) or 0),
        "env_index": int(episode.get("env_index", 0) or 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = sha256_file(source)

    candidates: list[Path] = []
    for directory in args.input_dir:
        if directory.exists():
            candidates.extend(directory.rglob("*.npz"))
    candidates = sorted(set(path.resolve() for path in candidates))

    rows: list[dict[str, Any]] = []
    for path in candidates:
        try:
            rows.append(inspect_capture(path, source_sha256))
        except Exception as exc:  # Keep one corrupt capture from aborting curation.
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "selected": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    selected = [row for row in rows if row.get("selected")]
    selected.sort(
        key=lambda row: (
            -int(row["score"]),
            -int(row["collected"]),
            float(row["outer_rail_fraction"]),
        )
    )

    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    seen_names: set[str] = set()
    for index, row in enumerate(selected):
        source_path = Path(row["path"])
        name = source_path.name
        if name in seen_names:
            name = f"{source_path.stem}_{index:03d}{source_path.suffix}"
        seen_names.add(name)
        shutil.copy2(source_path, staging / name)
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)

    rejection_counts: dict[str, int] = {}
    for row in rows:
        for key, passed in (row.get("checks") or {}).items():
            if not passed:
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
    report = {
        "schema": "stage_d_g12_teacher_filter_v1",
        "source_checkpoint": str(source),
        "source_sha256": source_sha256,
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "window_ranges_s": WINDOWS,
        "window_minimums": WINDOW_MINIMUMS,
        "rejection_counts": rejection_counts,
        "selected": selected,
        "all_candidates": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_count": len(rows),
                "selected_count": len(selected),
                "output_dir": str(output),
                "report": str(args.report.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
