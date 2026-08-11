"""Pure custody and serialization helpers for verified two-pass HD replays.

This module deliberately imports neither Isaac Sim nor OpenCV.  The exact
evaluator records a compact, complete visual-state trace in pass one; a fresh
offline renderer validates and consumes that trace in pass two.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


TRACE_SCHEMA = "stage_d_verified_visual_state_trace_v1"
TRACE_PROVENANCE_SCHEMA = "stage_d_verified_visual_state_trace_provenance_v1"
RENDER_PROVENANCE_SCHEMA = "stage_d_verified_trace_topdown_render_v1"
STEPS = 1600
FUEL_COUNT = 456
MIN_LIVE_SCORE = 200
FPS = 10.0
CAMERA_SIZE = (1280, 720)
SIDEBAR_WIDTH = 360
CODEC = "MJPG"

REQUIRED_TRACE_FIELDS = {
    "robot_position",
    "robot_orientation_wxyz",
    "robot_joint_position",
    "robot_joint_velocity",
    "fuel_position",
    "fuel_orientation_wxyz",
    "mechanism",
    "clock_s",
    "score",
    "collected",
    "magazine",
    "phase",
    "hub_active",
    "cycles",
    "action",
    "proprio",
    "privileged",
    "reward",
    "done",
    "metadata",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be one complete SHA256")
    return normalized


def require_file_sha(path: Path, expected: str, label: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    expected = expected_sha(expected, f"expected {label} SHA256")
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def encode_metadata(value: Mapping[str, Any]) -> np.ndarray:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8).copy()


def decode_metadata(value: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype != np.dtype("uint8"):
        raise ValueError("trace metadata must be a one-dimensional uint8 array")
    parsed = json.loads(raw.tobytes().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("trace metadata must decode to an object")
    return parsed


def field_declarations(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for key, value in sorted(arrays.items())
        if key != "metadata"
    }


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"trace {label} shape mismatch: {array.shape} != {shape}")


def validate_trace_arrays(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    require_publication_gate: bool = True,
) -> None:
    """Fail closed on incomplete, non-finite, or non-publishable traces."""

    if metadata.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"unexpected trace schema: {metadata.get('schema')!r}")
    steps = int(metadata.get("steps", -1))
    fuel_count = int(metadata.get("fuel_count", -1))
    joint_count = int(metadata.get("joint_count", -1))
    env_index = int(metadata.get("env_index", -1))
    if steps != STEPS:
        raise ValueError(f"trace must contain exactly {STEPS} frames")
    if fuel_count != FUEL_COUNT:
        raise ValueError(f"trace must contain all {FUEL_COUNT} FUEL bodies")
    if joint_count <= 0:
        raise ValueError("trace joint_count must be positive")
    if env_index not in (0, 1):
        raise ValueError("trace env_index must be 0 or 1")
    missing = REQUIRED_TRACE_FIELDS - set(arrays)
    extra = set(arrays) - REQUIRED_TRACE_FIELDS
    if missing or extra:
        raise ValueError(f"trace fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    expected_shapes = {
        "robot_position": (steps, 3),
        "robot_orientation_wxyz": (steps, 4),
        "robot_joint_position": (steps, joint_count),
        "robot_joint_velocity": (steps, joint_count),
        "fuel_position": (steps, fuel_count, 3),
        "fuel_orientation_wxyz": (steps, fuel_count, 4),
        "mechanism": (steps, 3),
        "clock_s": (steps,),
        "score": (steps,),
        "collected": (steps,),
        "magazine": (steps,),
        "phase": (steps,),
        "hub_active": (steps,),
        "cycles": (steps,),
        "action": (steps, 2, 7),
        "proprio": (steps, 30),
        "privileged": (steps, 26),
        "reward": (steps,),
        "done": (steps,),
    }
    for key, shape in expected_shapes.items():
        _require_shape(np.asarray(arrays[key]), shape, key)
    numeric = (
        "robot_position",
        "robot_orientation_wxyz",
        "robot_joint_position",
        "robot_joint_velocity",
        "fuel_position",
        "fuel_orientation_wxyz",
        "mechanism",
        "clock_s",
        "score",
        "collected",
        "magazine",
        "cycles",
        "action",
        "proprio",
        "privileged",
        "reward",
    )
    for key in numeric:
        if not np.isfinite(np.asarray(arrays[key])).all():
            raise ValueError(f"trace {key} contains non-finite values")
    if np.asarray(arrays["phase"]).dtype.kind != "U":
        raise ValueError("trace phase must use a fixed-width Unicode dtype")
    if np.asarray(arrays["hub_active"]).dtype != np.dtype("bool"):
        raise ValueError("trace hub_active must use bool dtype")
    if np.asarray(arrays["done"]).dtype != np.dtype("bool"):
        raise ValueError("trace done must use bool dtype")
    if np.asarray(arrays["done"])[:-1].any() or not bool(np.asarray(arrays["done"])[-1]):
        raise ValueError("trace must terminate only at the final transition")
    if np.any(np.abs(np.asarray(arrays["action"])) > 1.000001):
        raise ValueError("trace action exceeds [-1, 1]")
    live_scores = np.asarray(arrays["score"], dtype=np.int64)
    if int(live_scores[0]) != 0:
        raise ValueError("trace must start with live score zero")
    terminal_score = int(metadata.get("live_terminal_score", -1))
    if terminal_score != int(live_scores[-1]):
        raise ValueError(
            "trace terminal score metadata differs from the final live score sample"
        )
    if np.any(np.diff(live_scores) < 0):
        raise ValueError("trace live score must be monotonic")
    gate = int(metadata.get("publication_min_score", -1))
    if gate != MIN_LIVE_SCORE:
        raise ValueError(f"trace publication gate must be {MIN_LIVE_SCORE}")
    if require_publication_gate and terminal_score < gate:
        raise ValueError(f"live terminal score {terminal_score} is below publication gate {gate}")
    declarations = metadata.get("fields")
    if declarations != field_declarations(arrays):
        raise ValueError("trace field declarations differ from stored arrays")
    for key in ("checkpoint_sha256", "prefix_checkpoint_sha256", "bundle_sha256"):
        expected_sha(str(metadata.get(key, "")), key)
    contract = metadata.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("trace is missing evaluator contract")
    if contract.get("policy_speed_scale") != 1.0:
        raise ValueError("trace policy speed must be exactly 1.0")
    if contract.get("ferry") is not False or contract.get("return_when_live") is not False:
        raise ValueError("trace contains forbidden auxiliary mechanics")


def atomic_save_trace(
    path: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> str:
    """Validate and atomically publish one compressed trace."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items() if key != "metadata"}
    complete_metadata = dict(metadata)
    complete_metadata["fields"] = field_declarations(payload)
    payload["metadata"] = encode_metadata(complete_metadata)
    validate_trace_arrays(complete_metadata, payload)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return sha256_file(path)


@dataclass(frozen=True)
class VerifiedTrace:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def load_verified_trace(
    path: Path,
    *,
    expected_trace_sha256: str,
    expected_checkpoint_sha256: str,
    require_publication_gate: bool = True,
) -> VerifiedTrace:
    digest = require_file_sha(path, expected_trace_sha256, "verified visual-state trace")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    metadata = decode_metadata(arrays["metadata"])
    validate_trace_arrays(
        metadata, arrays, require_publication_gate=require_publication_gate
    )
    expected_checkpoint_sha256 = expected_sha(
        expected_checkpoint_sha256, "expected checkpoint SHA256"
    )
    if metadata["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("trace checkpoint SHA256 differs from expected checkpoint")
    return VerifiedTrace(
        path=Path(path).resolve(), sha256=digest, metadata=metadata, arrays=arrays
    )


def frame_telemetry(trace: VerifiedTrace, index: int) -> dict[str, Any]:
    if not 0 <= int(index) < int(trace.metadata["steps"]):
        raise IndexError(index)
    arrays = trace.arrays
    elapsed = float(arrays["clock_s"][index])
    horizon = float(trace.metadata["episode_len_s"])
    return {
        "step": int(index),
        "elapsed_s": elapsed,
        "remaining_s": max(0.0, horizon - elapsed),
        "score": int(arrays["score"][index]),
        "collected": int(arrays["collected"][index]),
        "magazine": int(arrays["magazine"][index]),
        "phase": str(arrays["phase"][index]),
        "hub": "ACTIVE" if bool(arrays["hub_active"][index]) else "INACTIVE",
        "cycles": int(arrays["cycles"][index]),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def normalized_quaternion_error(quaternions: np.ndarray) -> float:
    norms = np.linalg.norm(np.asarray(quaternions, dtype=np.float64), axis=-1)
    if not np.isfinite(norms).all():
        return math.inf
    return float(np.max(np.abs(norms - 1.0)))
