"""Select a phase-complete exact-source bank for full-speed retention.

The seed-mine learner validates checkpoint provenance, so this selector only
copies clean captures produced by the requested source checkpoint.  Stage D
also needs examples in every legal scoring window; a high total score alone
is not sufficient because it can hide a missing late-match behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("opener", 0.0, 33.0),
    ("live1", 55.0, 83.0),
    ("live2", 105.0, 130.0),
    ("endgame", 130.0, 160.0),
)
DEFAULT_PHASE_MINIMUMS = {
    "opener": 50,
    "live1": 25,
    "live2": 10,
    "endgame": 10,
}
CAPTURE_SCHEMA = "stagec_training_episode_v1"
DETERMINISTIC_ACTION_MODE = "deterministic"
FULL_MATCH_MODE = "full"
MIN_FULL_MATCH_SECONDS = 159.0
AUXILIARY_COUNTER_FIELDS = (
    "ferried_balls",
    "owncourt_score_entries",
    "owncourt_shots",
    "owncourt_scored",
)
AUXILIARY_TIMELINE_EVENTS = ("ferry", "oc_entry")
NO_AUX_STAGE_D_CONTRACT = {
    "stage_d": True,
    "first_inactive": "blue",
    "ferry": False,
    "return_when_live": False,
    "owncourt_loop": False,
}
NO_AUX_POLICY_SPEED_SCALE = 1.0
DEFAULT_NO_AUX_PREFIX_RESCUE_S = 35.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_capture_episode(path: Path) -> dict[str, Any]:
    """Read the immutable episode record embedded in a training capture."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata" not in archive.files:
                raise ValueError("missing metadata")
            raw = np.asarray(archive["metadata"], dtype=np.uint8)
        metadata = json.loads(raw.tobytes().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read capture metadata: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("unexpected capture metadata schema")
    episode = metadata.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("capture metadata has no episode object")
    return episode


def auxiliary_usage(episode: dict[str, Any]) -> dict[str, Any]:
    """Summarize *trainable-suffix* ferry/own-court behavior.

    The frozen Stage-C opener owns every action through its first dump.  That
    legacy policy can emit a few ``ferry``-head presses before its first score
    even when the Stage-D suffix contract has ferry disabled.  Those presses
    are part of the immutable prefix, not evidence that a candidate teacher
    used the deferred Stage-D ferry mechanic.  Actual ferried balls remain
    forbidden by the counters below, and any ferry press at/after scoring
    begins is treated as suffix use.
    """

    counters: dict[str, int] = {}
    for field in AUXILIARY_COUNTER_FIELDS:
        try:
            counters[field] = int(episode.get(field, 0) or 0)
        except (TypeError, ValueError):
            # A malformed value cannot be treated as proof of zero usage.
            counters[field] = 1
    timeline = [
        event
        for event in (episode.get("timeline") or [])
        if isinstance(event, dict)
    ]
    score_times = [
        float(event.get("t"))
        for event in timeline
        if event.get("ev") == "score" and event.get("t") is not None
    ]
    first_score_t = min(score_times) if score_times else None
    timeline_events = Counter()
    for event in timeline:
        name = str(event.get("ev"))
        if name == "oc_entry":
            timeline_events[name] += 1
        elif name == "ferry" and first_score_t is not None:
            try:
                is_suffix = float(event.get("t", -1.0)) >= first_score_t
            except (TypeError, ValueError):
                # Malformed timing is not proof that the press belonged to the
                # frozen prefix, so fail closed.
                is_suffix = True
            if is_suffix:
                timeline_events[name] += 1
    return {
        **counters,
        "timeline_events": {
            name: int(timeline_events.get(name, 0))
            for name in AUXILIARY_TIMELINE_EVENTS
        },
        "used": any(value != 0 for value in counters.values())
        or any(timeline_events.values()),
    }


def stage_d_contract_failure(
    episode: dict[str, Any],
    *,
    require_no_ferry_owncourt: bool,
    expected_prefix_rescue_s: float | None,
) -> str | None:
    """Validate declared mechanics, not merely their observed episode usage."""

    if not require_no_ferry_owncourt and expected_prefix_rescue_s is None:
        return None
    contract = episode.get("stage_d_contract")
    if not isinstance(contract, dict):
        return "missing_stage_d_contract"
    if require_no_ferry_owncourt:
        for field, expected in NO_AUX_STAGE_D_CONTRACT.items():
            actual = contract.get(field)
            if isinstance(expected, bool):
                if type(actual) is not bool or actual is not expected:
                    return f"stage_d_contract_{field}"
            elif actual != expected:
                return f"stage_d_contract_{field}"
        try:
            speed_scale = float(contract.get("policy_speed_scale"))
        except (TypeError, ValueError):
            return "stage_d_contract_policy_speed_scale"
        if not math.isclose(
            speed_scale, NO_AUX_POLICY_SPEED_SCALE, rel_tol=0.0, abs_tol=1e-12
        ):
            return "stage_d_contract_policy_speed_scale"
    if expected_prefix_rescue_s is not None:
        try:
            prefix_rescue_s = float(contract.get("prefix_rescue_s"))
        except (TypeError, ValueError):
            return "stage_d_contract_prefix_rescue_s"
        if not math.isclose(
            prefix_rescue_s,
            expected_prefix_rescue_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return "stage_d_contract_prefix_rescue_s"
    return None


def capture_contract_failure(
    jsonl_episode: dict[str, Any],
    capture_episode: dict[str, Any],
    *,
    source_hash: str,
    require_no_ferry_owncourt: bool,
    expected_prefix_rescue_s: float | None,
) -> str | None:
    """Return a fail-closed reason when either record violates the contract."""

    records = (("jsonl", jsonl_episode), ("capture", capture_episode))
    for label, episode in records:
        if str(episode.get("checkpoint_sha256", "")).lower() != source_hash:
            return f"{label}_source_checkpoint"
        if episode.get("action_mode") != DETERMINISTIC_ACTION_MODE:
            return f"{label}_action_mode"
        if episode.get("mode") != FULL_MATCH_MODE:
            return f"{label}_reset_mode"
        if episode.get("terminal_reason") != "horizon":
            return f"{label}_terminal_reason"
        try:
            episode_len_s = float(episode.get("episode_len_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            return f"{label}_episode_len"
        if episode_len_s < MIN_FULL_MATCH_SECONDS:
            return f"{label}_episode_len"
        stage_d_failure = stage_d_contract_failure(
            episode,
            require_no_ferry_owncourt=require_no_ferry_owncourt,
            expected_prefix_rescue_s=expected_prefix_rescue_s,
        )
        if stage_d_failure is not None:
            return f"{label}_{stage_d_failure}"
        if require_no_ferry_owncourt and bool(auxiliary_usage(episode)["used"]):
            return f"{label}_auxiliary_behavior"
    return None


def phase_scores(timeline: Iterable[dict[str, Any]] | None) -> dict[str, int]:
    """Return qualified + unqualified score events in each legal Stage-D bin.

    All windows are half-open except the final one, which includes an event at
    exactly 160.0 seconds.  This keeps the 130.0 boundary unambiguous: it is an
    endgame score, not a live2 score.
    """

    totals = {name: 0 for name, _, _ in PHASE_WINDOWS}
    for event in timeline or ():
        if event.get("ev") != "score":
            continue
        try:
            timestamp = float(event.get("t", -1.0))
        except (TypeError, ValueError):
            continue
        count = int(event.get("q", 0) or 0) + int(event.get("u", 0) or 0)
        for index, (name, start, end) in enumerate(PHASE_WINDOWS):
            final_window = index == len(PHASE_WINDOWS) - 1
            if start <= timestamp < end or (
                final_window and start <= timestamp <= end
            ):
                totals[name] += count
                break
    return totals


def _value_key(value: object) -> str:
    """Return a stable, hashable key for JSON telemetry values."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _capture_key(row: dict[str, object]) -> Path:
    return Path(row["capture"]).resolve()


def _seed_key(row: dict[str, object]) -> str:
    return _value_key(row.get("seed"))


def _env_key(row: dict[str, object]) -> str:
    return _value_key(row.get("env_index"))


def _find_feasible_completion(
    selected: list[dict[str, object]],
    candidates: list[dict[str, object]],
    *,
    count: int,
    min_unique_seeds: int,
    max_per_seed: int,
    min_unique_envs: int,
) -> list[dict[str, object]] | None:
    """Find any completion satisfying the bank diversity constraints.

    Teacher banks are normally five rows, so a pruned include-first search is
    both clearer and safer than a greedy repair pass.  The latter can reject a
    valid harvest simply because the last required seed and env index occur in
    the same candidate.
    """

    selected_paths = {_capture_key(row) for row in selected}
    seed_counts = Counter(_seed_key(row) for row in selected)
    if len(selected_paths) != len(selected):
        return None
    if any(value > max_per_seed for value in seed_counts.values()):
        return None
    if len(selected) > count:
        return None

    remaining = [
        row for row in candidates if _capture_key(row) not in selected_paths
    ]

    def search(
        chosen: list[dict[str, object]],
        available: list[dict[str, object]],
        counts: Counter[str],
        envs: set[str],
    ) -> list[dict[str, object]] | None:
        slots = count - len(chosen)
        if slots == 0:
            if len(counts) < min_unique_seeds or len(envs) < min_unique_envs:
                return None
            return chosen

        allowed = [
            row
            for row in available
            if counts[_seed_key(row)] < max_per_seed
        ]
        if len(allowed) < slots:
            return None

        possible_new_seeds = {
            _seed_key(row) for row in allowed if _seed_key(row) not in counts
        }
        if len(counts) + min(slots, len(possible_new_seeds)) < min_unique_seeds:
            return None

        capacity_by_seed = Counter(_seed_key(row) for row in allowed)
        capacity = sum(
            min(max_per_seed - counts[seed], available_count)
            for seed, available_count in capacity_by_seed.items()
        )
        if capacity < slots:
            return None

        possible_envs = envs | {_env_key(row) for row in allowed}
        if min(len(possible_envs), len(envs) + slots) < min_unique_envs:
            return None

        for index, row in enumerate(allowed):
            seed = _seed_key(row)
            next_counts = counts.copy()
            next_counts[seed] += 1
            result = search(
                [*chosen, row],
                allowed[index + 1 :],
                next_counts,
                envs | {_env_key(row)},
            )
            if result is not None:
                return result
        return None

    return search(
        list(selected),
        remaining,
        seed_counts,
        {_env_key(row) for row in selected},
    )


def select_candidates(
    candidates: list[dict[str, object]],
    *,
    count: int,
    score_slots: int,
    min_score_cycles: int,
    min_unique_seeds: int,
    max_per_seed: int,
) -> tuple[list[dict[str, object]], int]:
    """Choose a high-score/multi-cycle bank without sacrificing diversity."""

    unique_envs = {_env_key(row) for row in candidates}
    min_unique_envs = min(count, 2, len(unique_envs))
    if (
        _find_feasible_completion(
            [],
            candidates,
            count=count,
            min_unique_seeds=min_unique_seeds,
            max_per_seed=max_per_seed,
            min_unique_envs=min_unique_envs,
        )
        is None
    ):
        raise ValueError(
            "qualifying captures cannot satisfy teacher-bank diversity: "
            f"count={count}, min_unique_seeds={min_unique_seeds}, "
            f"max_per_seed={max_per_seed}, min_unique_env_indices={min_unique_envs}"
        )

    path_tiebreaker = lambda row: str(_capture_key(row))
    cycle_ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["cycles"]),
            -int(row["scored"]),
            -int(row["collected"]),
            path_tiebreaker(row),
        ),
    )
    score_ranked = sorted(
        (row for row in candidates if int(row["cycles"]) >= min_score_cycles),
        key=lambda row: (
            -int(row["scored"]),
            -int(row["cycles"]),
            -int(row["collected"]),
            path_tiebreaker(row),
        ),
    )

    selected: list[dict[str, object]] = []
    selected_paths: set[Path] = set()

    def choose_one(ranked: list[dict[str, object]], reason: str) -> bool:
        for row in ranked:
            capture = _capture_key(row)
            if capture in selected_paths:
                continue
            proposal = [*selected, row]
            if (
                _find_feasible_completion(
                    proposal,
                    candidates,
                    count=count,
                    min_unique_seeds=min_unique_seeds,
                    max_per_seed=max_per_seed,
                    min_unique_envs=min_unique_envs,
                )
                is None
            ):
                continue
            selected_paths.add(capture)
            selected.append({**row, "selection_reason": reason})
            return True
        return False

    for _ in range(score_slots):
        if not choose_one(score_ranked, "score"):
            break
    while len(selected) < count:
        if not choose_one(cycle_ranked, "cycles"):
            raise ValueError("unable to complete a diverse teacher bank")
    return selected, min_unique_envs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--harvest-dir",
        dest="harvest_dirs",
        action="append",
        type=Path,
        required=True,
        help="harvest directory to scan; repeat to combine independent waves",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--score-slots", type=int, default=2)
    parser.add_argument("--min-score-cycles", type=int, default=2)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--min-collected", type=int, default=0)
    parser.add_argument("--min-cycles", type=int, default=0)
    parser.add_argument("--min-repeat-scored-load-max", type=int, default=0)
    parser.add_argument(
        "--min-opener-score", type=int, default=DEFAULT_PHASE_MINIMUMS["opener"]
    )
    parser.add_argument(
        "--min-live1-score", type=int, default=DEFAULT_PHASE_MINIMUMS["live1"]
    )
    parser.add_argument(
        "--min-live2-score", type=int, default=DEFAULT_PHASE_MINIMUMS["live2"]
    )
    parser.add_argument(
        "--min-endgame-score", type=int, default=DEFAULT_PHASE_MINIMUMS["endgame"]
    )
    parser.add_argument("--min-unique-seeds", type=int, default=4)
    parser.add_argument("--max-per-seed", type=int, default=2)
    parser.add_argument(
        "--require-no-ferry-owncourt",
        action="store_true",
        help=(
            "accept only captures with no ferry or own-court scoring usage "
            "in either JSONL telemetry or embedded NPZ metadata"
        ),
    )
    parser.add_argument(
        "--expected-prefix-rescue-s",
        type=float,
        default=None,
        help=(
            "require this stage_d_contract prefix_rescue_s value; defaults "
            "to 35.0 when --require-no-ferry-owncourt is enabled"
        ),
    )
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be positive")
    if not 0 <= args.score_slots <= args.count:
        raise SystemExit("--score-slots must be between zero and --count")
    if not 1 <= args.min_unique_seeds <= args.count:
        raise SystemExit("--min-unique-seeds must be between one and --count")
    if args.max_per_seed < 1:
        raise SystemExit("--max-per-seed must be positive")
    candidate_minimums = {
        "score": args.min_score,
        "collected": args.min_collected,
        "cycles": args.min_cycles,
        "repeat_scored_load_max": args.min_repeat_scored_load_max,
    }
    if any(value < 0 for value in candidate_minimums.values()):
        raise SystemExit("candidate minimums must be non-negative")
    expected_prefix_rescue_s = args.expected_prefix_rescue_s
    if args.require_no_ferry_owncourt and expected_prefix_rescue_s is None:
        expected_prefix_rescue_s = DEFAULT_NO_AUX_PREFIX_RESCUE_S
    if expected_prefix_rescue_s is not None and (
        not math.isfinite(expected_prefix_rescue_s) or expected_prefix_rescue_s < 0.0
    ):
        raise SystemExit("--expected-prefix-rescue-s must be finite and non-negative")

    phase_minimums = {
        "opener": args.min_opener_score,
        "live1": args.min_live1_score,
        "live2": args.min_live2_score,
        "endgame": args.min_endgame_score,
    }
    if any(value < 0 for value in phase_minimums.values()):
        raise SystemExit("phase score minimums must be non-negative")

    source = args.source_checkpoint.resolve()
    source_hash = sha256(source)
    harvest_dirs = list(
        dict.fromkeys(path.resolve() for path in args.harvest_dirs)
    )
    candidates: list[dict[str, object]] = []
    seen_captures: set[Path] = set()
    discovered_capture_rows = 0
    rejection_counts: Counter[str] = Counter()
    for harvest_dir in harvest_dirs:
        for jsonl in sorted(harvest_dir.glob("*.jsonl")):
            for line in jsonl.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rejection_counts["invalid_json"] += 1
                    continue
                if not isinstance(row, dict):
                    rejection_counts["invalid_json_record"] += 1
                    continue
                capture = row.get("capture_path")
                if not capture:
                    continue
                discovered_capture_rows += 1
                path = Path(str(capture)).resolve()
                if not path.is_file():
                    rejection_counts["missing_capture"] += 1
                    continue
                if path in seen_captures:
                    rejection_counts["duplicate_capture"] += 1
                    continue
                seen_captures.add(path)
                try:
                    episode = load_capture_episode(path)
                except ValueError:
                    rejection_counts["invalid_capture_metadata"] += 1
                    continue
                failure = capture_contract_failure(
                    row,
                    episode,
                    source_hash=source_hash,
                    require_no_ferry_owncourt=bool(args.require_no_ferry_owncourt),
                    expected_prefix_rescue_s=expected_prefix_rescue_s,
                )
                if failure is not None:
                    rejection_counts[failure] += 1
                    continue

                # All ranking, phase, and diversity fields come from the immutable
                # capture record.  JSONL is only the discovery index and must not
                # be able to overstate the captured trajectory.
                scores = phase_scores(episode.get("timeline"))
                candidates.append(
                    {
                        "capture": path,
                        "scored": int(episode.get("scored", 0) or 0),
                        "collected": int(episode.get("collected", 0) or 0),
                        "cycles": int(episode.get("cycles_completed", 0) or 0),
                        "repeat_scored_load_max": int(
                            episode.get("repeat_scored_load_max", 0) or 0
                        ),
                        "clean": True,
                        "seed": episode.get("env_seed"),
                        "env_index": episode.get("env_index"),
                        "phase_scores": scores,
                        "auxiliary_usage": auxiliary_usage(episode),
                    }
                )

    clean = [row for row in candidates if bool(row["clean"])]
    qualifying = [
        row
        for row in clean
        if int(row["scored"]) >= candidate_minimums["score"]
        and int(row["collected"]) >= candidate_minimums["collected"]
        and int(row["cycles"]) >= candidate_minimums["cycles"]
        and int(row["repeat_scored_load_max"])
        >= candidate_minimums["repeat_scored_load_max"]
        and all(
            int(row["phase_scores"][name]) >= minimum  # type: ignore[index]
            for name, minimum in phase_minimums.items()
        )
    ]
    if len(qualifying) < args.count:
        raise SystemExit(
            f"only {len(qualifying)} qualifying clean exact-source captures; "
            f"need {args.count}"
        )

    try:
        selected, min_unique_envs = select_candidates(
            qualifying,
            count=args.count,
            score_slots=args.score_slots,
            min_score_cycles=args.min_score_cycles,
            min_unique_seeds=args.min_unique_seeds,
            max_per_seed=args.max_per_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_rows: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        source_capture = Path(row["capture"])
        destination = args.output_dir / f"teacher_{index:02d}_{source_capture.name}"
        shutil.copy2(source_capture, destination)
        report_rows.append(
            {
                **{key: value for key, value in row.items() if key != "capture"},
                "capture": str(source_capture),
                "teacher": str(destination),
            }
        )

    selected_seeds = {_seed_key(row) for row in selected}
    selected_envs = {_env_key(row) for row in selected}
    report = {
        "source_checkpoint": str(source),
        "source_sha256": source_hash,
        "harvest_dirs": [str(path) for path in harvest_dirs],
        "discovered_capture_row_count": discovered_capture_rows,
        "candidate_count": len(candidates),
        "clean_candidate_count": len(clean),
        "qualifying_candidate_count": len(qualifying),
        "selected_count": len(selected),
        "score_slots": args.score_slots,
        "min_score_cycles": args.min_score_cycles,
        "candidate_minimums": candidate_minimums,
        "phase_windows_s": {
            name: [start, end] for name, start, end in PHASE_WINDOWS
        },
        "phase_minimums": phase_minimums,
        "min_unique_seeds": args.min_unique_seeds,
        "max_per_seed": args.max_per_seed,
        "selected_unique_seed_count": len(selected_seeds),
        "required_unique_env_index_count": min_unique_envs,
        "selected_unique_env_index_count": len(selected_envs),
        "require_no_ferry_owncourt": bool(args.require_no_ferry_owncourt),
        "expected_stage_d_contract": {
            **(
                {
                    **NO_AUX_STAGE_D_CONTRACT,
                    "policy_speed_scale": NO_AUX_POLICY_SPEED_SCALE,
                }
                if args.require_no_ferry_owncourt
                else {}
            ),
            **(
                {"prefix_rescue_s": expected_prefix_rescue_s}
                if expected_prefix_rescue_s is not None
                else {}
            ),
        },
        "auxiliary_behavior_contract": {
            "enabled": bool(args.require_no_ferry_owncourt),
            "counter_fields": list(AUXILIARY_COUNTER_FIELDS),
            "forbidden_timeline_events": list(AUXILIARY_TIMELINE_EVENTS),
            "accepted_candidate_usage_count": sum(
                bool(row["auxiliary_usage"]["used"])  # type: ignore[index]
                for row in candidates
            ),
            "selected_usage_count": sum(
                bool(row["auxiliary_usage"]["used"])  # type: ignore[index]
                for row in selected
            ),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected": report_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
