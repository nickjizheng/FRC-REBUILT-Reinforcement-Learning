#!/usr/bin/env python3
"""Promote an explicitly frozen checkpoint after a provenance-clean evaluation.

This utility intentionally does not import Torch or inspect checkpoint payloads.  It
hashes and copies the exact file named by ``--checkpoint`` only after every JSONL
row proves that it evaluated that same content in deterministic, 160-second full
match mode.  ``latest.pt`` is forbidden as either an input or an output because it
is mutable by design and cannot provide promotion custody.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "frozen_checkpoint_promotion_v1"
FULL_EPISODE_S = 160.0
ENDGAME_START_S = 130.0
LATEST_NAME = "latest.pt"

_INFRA_INVALID_TRUE_KEYS = (
    "infra_invalid",
    "infrastructure_invalid",
    "physics_invalid",
    "eval_invalid",
    "invalid_episode",
)
_INFRA_VALID_FALSE_KEYS = (
    "infra_valid",
    "infrastructure_valid",
    "physics_valid",
    "eval_valid",
    "healthy",
    "is_healthy",
)
_RESTART_TRUE_KEYS = (
    "restart_boundary",
    "restart_crossing",
    "crossed_restart",
    "collector_restart_boundary",
    "watchdog_restart_during_episode",
)
_RESTART_PAIR_KEYS = (
    ("collector_generation_start", "collector_generation_end"),
    ("restart_generation_start", "restart_generation_end"),
    ("process_start_id", "process_end_id"),
)


class PromotionError(ValueError):
    """Raised when promotion custody or a requested performance gate fails."""


@dataclass(frozen=True)
class Gates:
    min_episodes: int = 1
    min_mean_score: float = 0.0
    min_max_score: int = 0
    min_cycle2_rate: float = 0.0
    min_cycle3_rate: float = 0.0
    min_endgame_mean: float = 0.0
    min_endgame_rate: float = 0.0
    min_repeat_load_mean: float = 0.0

    def validate(self) -> None:
        if self.min_episodes < 1:
            raise PromotionError("min_episodes must be at least 1")
        for name in ("min_cycle2_rate", "min_cycle3_rate", "min_endgame_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise PromotionError(f"{name} must be between 0 and 1")
        for name in (
            "min_mean_score",
            "min_max_score",
            "min_endgame_mean",
            "min_repeat_load_mean",
        ):
            if float(getattr(self, name)) < 0.0:
                raise PromotionError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "min_episodes": self.min_episodes,
            "min_mean_score": self.min_mean_score,
            "min_max_score": self.min_max_score,
            "min_cycle2_rate": self.min_cycle2_rate,
            "min_cycle3_rate": self.min_cycle3_rate,
            "min_endgame_mean": self.min_endgame_mean,
            "min_endgame_rate": self.min_endgame_rate,
            "min_repeat_load_mean": self.min_repeat_load_mean,
        }


def _forbid_latest(path: Path, purpose: str) -> Path:
    # Check the literal path before resolving so this guard runs before any open,
    # stat, or hash of a file explicitly named latest.pt.
    if path.name.casefold() == LATEST_NAME:
        raise PromotionError(f"{purpose} must not be {LATEST_NAME}")
    resolved = path.expanduser().resolve()
    # Also reject a differently named symlink whose resolved target is latest.pt.
    if resolved.name.casefold() == LATEST_NAME:
        raise PromotionError(f"{purpose} resolves to forbidden {LATEST_NAME}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise PromotionError("evaluation JSONL has an incomplete trailing record")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromotionError(
                f"evaluation JSONL line {line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise PromotionError(
                f"evaluation JSONL line {line_number} is not an object"
            )
        rows.append(value)
    if not rows:
        raise PromotionError("evaluation JSONL contains no episode rows")
    return rows, hashlib.sha256(data).hexdigest()


def _as_finite_float(value: Any, field: str, row_index: int) -> float:
    if isinstance(value, bool):
        raise PromotionError(f"row {row_index} field {field!r} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PromotionError(
            f"row {row_index} field {field!r} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise PromotionError(f"row {row_index} field {field!r} must be finite")
    return result


def _as_nonnegative_int(value: Any, field: str, row_index: int) -> int:
    number = _as_finite_float(value, field, row_index)
    if number < 0 or not number.is_integer():
        raise PromotionError(
            f"row {row_index} field {field!r} must be a non-negative integer"
        )
    return int(number)


def _recorded_checkpoint_is_latest(row: dict[str, Any]) -> bool:
    recorded = row.get("checkpoint")
    if not isinstance(recorded, str) or not recorded.strip():
        return False
    # This is string inspection only; never resolve or open the recorded path.
    return Path(recorded).name.casefold() == LATEST_NAME


def validate_eval_contract(
    rows: Iterable[dict[str, Any]], checkpoint_sha256: str
) -> list[dict[str, Any]]:
    checked = list(rows)
    for index, row in enumerate(checked):
        recorded_sha = str(row.get("checkpoint_sha256", "")).strip().lower()
        if recorded_sha != checkpoint_sha256:
            raise PromotionError(
                f"row {index} checkpoint SHA does not match the frozen checkpoint"
            )
        if _recorded_checkpoint_is_latest(row):
            raise PromotionError(
                f"row {index} records mutable {LATEST_NAME}, not an isolated checkpoint"
            )
        if row.get("action_mode") != "deterministic":
            raise PromotionError(
                f"row {index} action_mode must be exactly 'deterministic'"
            )
        if row.get("mode") != "full":
            raise PromotionError(f"row {index} mode must be exactly 'full'")
        episode_len_s = _as_finite_float(
            row.get("episode_len_s"), "episode_len_s", index
        )
        if not math.isclose(episode_len_s, FULL_EPISODE_S, abs_tol=1e-9):
            raise PromotionError(
                f"row {index} episode_len_s must be exactly {FULL_EPISODE_S:g}"
            )
    return checked


def rejection_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(row.get("terminal_reason", "")).lower() == "unhealthy":
        reasons.append("unhealthy")
    if any(row.get(key) is True for key in _INFRA_INVALID_TRUE_KEYS) or any(
        key in row and row.get(key) is False for key in _INFRA_VALID_FALSE_KEYS
    ):
        reasons.append("infra_invalid")
    restart_crossing = any(row.get(key) is True for key in _RESTART_TRUE_KEYS)
    for start_key, end_key in _RESTART_PAIR_KEYS:
        if start_key in row and end_key in row and row[start_key] != row[end_key]:
            restart_crossing = True
    if restart_crossing:
        reasons.append("restart_crossing")
    # A 160-second full-match result must reach its horizon.  This also prevents
    # skill-lane early exits from being presented as full-match evaluations.
    terminal_reason = str(row.get("terminal_reason", "")).lower()
    if terminal_reason != "horizon" and terminal_reason != "unhealthy":
        reasons.append("non_horizon")
    return reasons


def _endgame_score(row: dict[str, Any], row_index: int) -> int:
    timeline = row.get("timeline")
    if not isinstance(timeline, list):
        raise PromotionError(f"row {row_index} has no usable timeline")
    result = 0
    for event_index, event in enumerate(timeline):
        if not isinstance(event, dict):
            raise PromotionError(
                f"row {row_index} timeline event {event_index} is not an object"
            )
        if event.get("ev") != "score":
            continue
        timestamp = _as_finite_float(
            event.get("t"), f"timeline[{event_index}].t", row_index
        )
        quantity = _as_nonnegative_int(
            event.get("q", 0), f"timeline[{event_index}].q", row_index
        ) + _as_nonnegative_int(
            event.get("u", 0), f"timeline[{event_index}].u", row_index
        )
        if ENDGAME_START_S <= timestamp <= FULL_EPISODE_S:
            result += quantity
    return result


def compute_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reasons = rejection_reasons(row)
        if reasons:
            rejected.append({"row_index": index, "reasons": reasons})
        else:
            eligible.append(row)
    if not eligible:
        raise PromotionError("evaluation contains no healthy, restart-clean horizon rows")

    scores: list[int] = []
    cycles: list[int] = []
    endgame_scores: list[int] = []
    repeat_sum = 0
    repeat_count = 0
    repeat_max = 0
    for index, row in enumerate(eligible):
        scores.append(_as_nonnegative_int(row.get("scored"), "scored", index))
        cycles.append(
            _as_nonnegative_int(row.get("cycles_completed"), "cycles_completed", index)
        )
        endgame_scores.append(_endgame_score(row, index))
        row_count = _as_nonnegative_int(
            row.get("repeat_scored_load_count", 0),
            "repeat_scored_load_count",
            index,
        )
        row_sum = _as_nonnegative_int(
            row.get("repeat_scored_load_sum", 0),
            "repeat_scored_load_sum",
            index,
        )
        row_max = _as_nonnegative_int(
            row.get("repeat_scored_load_max", 0),
            "repeat_scored_load_max",
            index,
        )
        if row_count == 0 and row_sum != 0:
            raise PromotionError(
                f"eligible row {index} has repeat load sum without repeat load events"
            )
        repeat_count += row_count
        repeat_sum += row_sum
        repeat_max = max(repeat_max, row_max)

    count = len(eligible)
    return {
        "input_rows": count + len(rejected),
        "eligible_episodes": count,
        "rejected_rows": len(rejected),
        "rejections": rejected,
        "score_mean": round(sum(scores) / count, 6),
        "score_max": max(scores),
        "score_min": min(scores),
        "score_total": sum(scores),
        # cycles_completed=1 means the first repeat trip (cycle 2) completed.
        "cycle2_rate": round(sum(value >= 1 for value in cycles) / count, 6),
        "cycle3_rate": round(sum(value >= 2 for value in cycles) / count, 6),
        "cycles_mean": round(sum(cycles) / count, 6),
        "endgame_score_mean": round(sum(endgame_scores) / count, 6),
        "endgame_score_max": max(endgame_scores),
        "endgame_score_rate": round(
            sum(value > 0 for value in endgame_scores) / count, 6
        ),
        "repeat_load_mean": round(
            repeat_sum / repeat_count, 6
        ) if repeat_count else 0.0,
        "repeat_load_max": repeat_max,
        "repeat_load_events": repeat_count,
    }


def require_gates(metrics: dict[str, Any], gates: Gates) -> dict[str, Any]:
    gates.validate()
    comparisons = {
        "min_episodes": (metrics["eligible_episodes"], gates.min_episodes),
        "min_mean_score": (metrics["score_mean"], gates.min_mean_score),
        "min_max_score": (metrics["score_max"], gates.min_max_score),
        "min_cycle2_rate": (metrics["cycle2_rate"], gates.min_cycle2_rate),
        "min_cycle3_rate": (metrics["cycle3_rate"], gates.min_cycle3_rate),
        "min_endgame_mean": (
            metrics["endgame_score_mean"],
            gates.min_endgame_mean,
        ),
        "min_endgame_rate": (
            metrics["endgame_score_rate"],
            gates.min_endgame_rate,
        ),
        "min_repeat_load_mean": (
            metrics["repeat_load_mean"],
            gates.min_repeat_load_mean,
        ),
    }
    result = {
        name: {
            "actual": actual,
            "required": required,
            "passed": actual >= required,
        }
        for name, (actual, required) in comparisons.items()
    }
    failures = [name for name, value in result.items() if not value["passed"]]
    if failures:
        detail = ", ".join(
            f"{name}={result[name]['actual']} < {result[name]['required']}"
            for name in failures
        )
        raise PromotionError(f"promotion gates failed: {detail}")
    return result


def _stage_checkpoint(source: Path, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as origin:
            for chunk in iter(lambda: origin.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
        if digest.hexdigest() != expected_sha256:
            raise PromotionError("checkpoint changed while it was being promoted")
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _stage_json(destination: Path, value: dict[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def promote(
    checkpoint: Path,
    evaluation_jsonl: Path,
    output_checkpoint: Path,
    sidecar: Path,
    gates: Gates,
) -> dict[str, Any]:
    source = _forbid_latest(Path(checkpoint), "checkpoint")
    output = _forbid_latest(Path(output_checkpoint), "output checkpoint")
    report_path = Path(sidecar).expanduser().resolve()
    evaluation = Path(evaluation_jsonl).expanduser().resolve()
    if not source.is_file():
        raise PromotionError(f"checkpoint does not exist: {source}")
    if not evaluation.is_file():
        raise PromotionError(f"evaluation JSONL does not exist: {evaluation}")
    if source == output:
        raise PromotionError("output checkpoint must differ from source checkpoint")
    if output == report_path:
        raise PromotionError("output checkpoint and sidecar must differ")
    if output.exists() or report_path.exists():
        raise PromotionError("promotion outputs already exist")

    checkpoint_sha256 = sha256_file(source)
    rows, evaluation_sha256 = read_jsonl(evaluation)
    validate_eval_contract(rows, checkpoint_sha256)
    metrics = compute_metrics(rows)
    gate_results = require_gates(metrics, gates)

    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "source": str(source),
            "source_sha256": checkpoint_sha256,
            "promoted": str(output),
            "promoted_sha256": checkpoint_sha256,
        },
        "source_evaluation": {
            "path": str(evaluation),
            "sha256": evaluation_sha256,
            "contract": {
                "checkpoint_sha256": checkpoint_sha256,
                "action_mode": "deterministic",
                "mode": "full",
                "episode_len_s": FULL_EPISODE_S,
            },
        },
        "metrics": metrics,
        "gates": gates.as_dict(),
        "gate_results": gate_results,
    }

    checkpoint_tmp: Path | None = None
    sidecar_tmp: Path | None = None
    try:
        checkpoint_tmp = _stage_checkpoint(source, output, checkpoint_sha256)
        sidecar_tmp = _stage_json(report_path, report)
        os.replace(checkpoint_tmp, output)
        checkpoint_tmp = None
        os.replace(sidecar_tmp, report_path)
        sidecar_tmp = None
    finally:
        for temporary in (checkpoint_tmp, sidecar_tmp):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-jsonl", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--min-mean-score", type=float, default=0.0)
    parser.add_argument("--min-max-score", type=int, default=0)
    parser.add_argument("--min-cycle2-rate", type=float, default=0.0)
    parser.add_argument("--min-cycle3-rate", type=float, default=0.0)
    parser.add_argument("--min-endgame-mean", type=float, default=0.0)
    parser.add_argument("--min-endgame-rate", type=float, default=0.0)
    parser.add_argument("--min-repeat-load-mean", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gates = Gates(
        min_episodes=args.min_episodes,
        min_mean_score=args.min_mean_score,
        min_max_score=args.min_max_score,
        min_cycle2_rate=args.min_cycle2_rate,
        min_cycle3_rate=args.min_cycle3_rate,
        min_endgame_mean=args.min_endgame_mean,
        min_endgame_rate=args.min_endgame_rate,
        min_repeat_load_mean=args.min_repeat_load_mean,
    )
    try:
        report = promote(
            args.checkpoint,
            args.evaluation_jsonl,
            args.output_checkpoint,
            args.sidecar,
            gates,
        )
    except (OSError, PromotionError) as exc:
        print(f"PROMOTION_REJECTED: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "promoted": report["checkpoint"]["promoted"],
                "sha256": report["checkpoint"]["promoted_sha256"],
                "sidecar": str(Path(args.sidecar).expanduser().resolve()),
                "metrics": report["metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
