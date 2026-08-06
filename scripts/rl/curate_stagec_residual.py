#!/usr/bin/env python3
"""Fail-closed pairing and curation for Stage C residual route training.

Control and exploratory evaluations must use the same immutable V4 wrapper,
field template, mechanics, horizon, environment seed, environment index, and
per-environment episode sequence.  A capture is admitted only when the
exploratory episode is non-inferior in score and completed cycles and improves
at least one useful route outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "stagec_residual_curation_v1"
PAIR_FIELDS = ("env_seed", "env_index", "env_episode_sequence")
CONTRACT_FIELDS = (
    "checkpoint_sha256",
    "prefix_sha256",
    "template_sha256",
    "mode",
    "episode_len_s",
    "num_envs",
    "stagec_v2_metadata",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                value["_record_source"] = str(path.resolve())
                records.append(value)
    return records


def _pair_key(record: dict[str, Any]) -> tuple[int, int, int]:
    missing = [name for name in PAIR_FIELDS if name not in record]
    if missing:
        raise ValueError(f"episode record is missing pairing fields: {missing}")
    return tuple(int(record[name]) for name in PAIR_FIELDS)


def _index_unique(records: Iterable[dict[str, Any]], label: str) -> dict[tuple[int, int, int], dict]:
    indexed: dict[tuple[int, int, int], dict] = {}
    for record in records:
        key = _pair_key(record)
        if key in indexed:
            raise ValueError(f"duplicate {label} episode pair key: {key}")
        indexed[key] = record
    return indexed


def _last_cycle_step(record: dict[str, Any]) -> int:
    values = record.get("cycle_success_steps") or []
    return int(values[-1]) if values else 1_000_000_000


def _strict_improvements(
    control: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    improvements: list[str] = []
    if int(candidate.get("scored", 0)) > int(control.get("scored", 0)):
        improvements.append("score")
    if int(candidate.get("cycles_completed", 0)) > int(
        control.get("cycles_completed", 0)
    ):
        improvements.append("cycles")
    if _last_cycle_step(candidate) + 10 <= _last_cycle_step(control):
        improvements.append("cycle_time")
    if int(candidate.get("repeat_scored_load_sum", 0)) >= int(
        control.get("repeat_scored_load_sum", 0)
    ) + 3:
        improvements.append("repeat_load")
    if float(candidate.get("outer_rail_fraction", 1.0)) <= float(
        control.get("outer_rail_fraction", 1.0)
    ) - 0.05:
        improvements.append("rail_fraction")
    if int(candidate.get("outer_rail_max_streak", 1_000_000)) <= int(
        control.get("outer_rail_max_streak", 1_000_000)
    ) - 20:
        improvements.append("rail_streak")
    return improvements


def _validate_contract(control: dict[str, Any], candidate: dict[str, Any], phase: str) -> None:
    for name in CONTRACT_FIELDS:
        if control.get(name) != candidate.get(name):
            raise ValueError(
                f"paired episode contract mismatch for {name}: "
                f"{control.get(name)!r} != {candidate.get(name)!r}"
            )
    if control.get("action_mode") != "deterministic":
        raise ValueError("control episode must use deterministic actions")
    if candidate.get("action_mode") != "smooth-drive":
        raise ValueError("candidate episode must use smooth-drive actions")
    if candidate.get("noise_phases") != [phase]:
        raise ValueError(
            f"candidate noise_phases must contain only {phase!r}, got "
            f"{candidate.get('noise_phases')!r}"
        )
    if float(candidate.get("noise_cap", 0.0)) > 0.05:
        raise ValueError("candidate smooth-drive cap exceeds the residual safety limit")
    if str(control.get("mode")) != "full":
        raise ValueError("residual route curation accepts full-match episodes only")


def curate(
    control_paths: list[Path],
    candidate_paths: list[Path],
    out_dir: Path,
    phase: str,
) -> dict[str, Any]:
    controls = _index_unique(_load_jsonl(control_paths), "control")
    candidates = _index_unique(_load_jsonl(candidate_paths), "candidate")
    out_dir = out_dir.resolve()
    captures_dir = out_dir / "captures"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty curation directory: {out_dir}")
    captures_dir.mkdir(parents=True, exist_ok=True)

    decisions: list[dict[str, Any]] = []
    copied: list[Path] = []
    for key in sorted(candidates):
        candidate = candidates[key]
        control = controls.get(key)
        decision: dict[str, Any] = {"pair_key": list(key), "accepted": False}
        if control is None:
            decision["reason"] = "missing_control"
            decisions.append(decision)
            continue
        _validate_contract(control, candidate, phase)
        if int(candidate.get("cycles_completed", 0)) < 1:
            decision["reason"] = "no_completed_cycle"
        elif int(candidate.get("scored", 0)) < int(control.get("scored", 0)):
            decision["reason"] = "score_regression"
        elif int(candidate.get("cycles_completed", 0)) < int(
            control.get("cycles_completed", 0)
        ):
            decision["reason"] = "cycle_regression"
        else:
            improvements = _strict_improvements(control, candidate)
            capture = candidate.get("capture_path")
            if not improvements:
                decision["reason"] = "no_meaningful_improvement"
            elif not capture:
                decision["reason"] = "missing_capture"
            else:
                source = Path(str(capture))
                if not source.is_file():
                    raise FileNotFoundError(f"candidate capture does not exist: {source}")
                destination = captures_dir / source.name
                if destination.exists():
                    raise FileExistsError(f"duplicate capture basename: {destination.name}")
                shutil.copy2(source, destination)
                copied.append(destination)
                decision.update(
                    {
                        "accepted": True,
                        "reason": "accepted",
                        "improvements": improvements,
                        "capture": str(destination),
                        "capture_sha256": _sha256(destination),
                        "control_score": int(control.get("scored", 0)),
                        "candidate_score": int(candidate.get("scored", 0)),
                        "control_cycles": int(control.get("cycles_completed", 0)),
                        "candidate_cycles": int(candidate.get("cycles_completed", 0)),
                    }
                )
        decisions.append(decision)

    result = {
        "schema": SCHEMA,
        "phase": phase,
        "controls": len(controls),
        "candidates": len(candidates),
        "paired": sum(tuple(item["pair_key"]) in controls for item in decisions),
        "accepted": sum(bool(item["accepted"]) for item in decisions),
        "control_jsonl": [str(path.resolve()) for path in control_paths],
        "candidate_jsonl": [str(path.resolve()) for path in candidate_paths],
        "decisions": decisions,
    }
    if not copied:
        result["status"] = "NO_QUALIFIED_DATA"
    else:
        result["status"] = "READY"
    tmp = out_dir / f".curation.json.{os.getpid()}.tmp"
    tmp.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, out_dir / "curation.json")
    manifest = "\n".join(
        f"{_sha256(path)}  captures/{path.name}" for path in sorted(copied)
    )
    (out_dir / "MANIFEST.sha256").write_text(
        manifest + ("\n" if manifest else ""), encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("leave", "collect", "return"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = curate(args.control, args.candidate, args.out_dir, args.phase)
    print("RESIDUAL_CURATION " + json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
