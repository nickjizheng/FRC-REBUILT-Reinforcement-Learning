"""Curate immutable, checkpoint-specific Stage C seed-mining elites.

The seed miner intentionally writes captures from several candidate checkpoints
into one directory.  A focused learner must never ingest that directory
directly: a duplicate episode, a capture from the wrong checkpoint, or a
partial dump would silently change the replay distribution.

This utility validates every ``.npz`` in the source directory, selects only
clean cycle completions for one exact checkpoint/prefix pair, and atomically
publishes a new directory plus a custody manifest.  Inputs are never modified.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CAPTURE_SCHEMA = "stagec_training_episode_v1"
EVAL_SCHEMA = "stagec_seed_eval_v1"
CURATION_SCHEMA = "stagec_seedmine_curated_v1"
STAGEC_SCHEMA = "stagec_v2.3"
ACTION_POLICY = "frozen_prefix_exact_first_v2"
FIELD_STRATEGY = "native_field_return_preload_v1"
FIELD_KEYS = ("obs", "proprio", "privileged", "action", "reward", "done")
EXPECTED_DTYPES = {
    "obs": np.dtype(np.uint8),
    "proprio": np.dtype(np.float32),
    "privileged": np.dtype(np.float32),
    "action": np.dtype(np.float32),
    "reward": np.dtype(np.float32),
    "done": np.dtype(bool),
}


@dataclass(frozen=True)
class ValidatedCapture:
    path: Path
    sha256: str
    checkpoint_sha256: str
    prefix_sha256: str
    provenance: tuple[str, int, int, int, int]
    episode: dict[str, Any]
    capture_tier: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_sha(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return text


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    # Metadata came through JSON, so accepting strings or truncating floats
    # would only hide a malformed/miner-incompatible archive.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    result = value
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{label} is out of range")
    return result


def _decode_metadata(raw: np.ndarray, path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name}: metadata must be an object")
    return metadata


def validate_capture(path: Path, expected_prefix_sha256: str) -> ValidatedCapture:
    """Fully validate one miner archive before it can be classified."""

    expected_files = {*FIELD_KEYS, "metadata"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != expected_files:
                raise ValueError(
                    f"{path.name}: archive fields {sorted(data.files)} != "
                    f"{sorted(expected_files)}"
                )
            metadata = _decode_metadata(data["metadata"], path)
            arrays = {key: data[key] for key in FIELD_KEYS}

            if metadata.get("schema") != CAPTURE_SCHEMA:
                raise ValueError(f"{path.name}: wrong capture schema")
            if metadata.get("field_keys") != list(FIELD_KEYS):
                raise ValueError(f"{path.name}: replay field order mismatch")
            length = _integer(metadata.get("length"), "capture length", minimum=1)
            declared = metadata.get("fields")
            if not isinstance(declared, dict) or set(declared) != set(FIELD_KEYS):
                raise ValueError(f"{path.name}: incomplete field declarations")

            expected_shapes = {
                "obs": (length, 9, 90, 160),
                "proprio": (length, 30),
                "privileged": (length, 26),
                "action": (length, 7),
                "reward": (length,),
                "done": (length,),
            }
            for key, value in arrays.items():
                declaration = declared.get(key)
                if not isinstance(declaration, dict):
                    raise ValueError(f"{path.name}: invalid declaration for {key}")
                if declaration.get("shape") != list(value.shape):
                    raise ValueError(f"{path.name}: declared shape mismatch for {key}")
                if declaration.get("dtype") != str(value.dtype):
                    raise ValueError(f"{path.name}: declared dtype mismatch for {key}")
                if value.shape != expected_shapes[key]:
                    raise ValueError(
                        f"{path.name}: {key} shape {value.shape} != {expected_shapes[key]}"
                    )
                if value.dtype != EXPECTED_DTYPES[key]:
                    raise ValueError(
                        f"{path.name}: {key} dtype {value.dtype} != {EXPECTED_DTYPES[key]}"
                    )
                if key != "obs" and not bool(np.isfinite(value).all()):
                    raise ValueError(f"{path.name}: non-finite values in {key}")
            done = arrays["done"]
            if not bool(done[-1]) or bool(done[:-1].any()):
                raise ValueError(
                    f"{path.name}: capture must terminate exactly once on its final row"
                )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(path.name):
            raise
        raise ValueError(f"{path.name}: cannot validate archive: {exc}") from exc

    episode = metadata.get("episode")
    if not isinstance(episode, dict) or episode.get("schema") != EVAL_SCHEMA:
        raise ValueError(f"{path.name}: wrong or missing evaluation schema")
    if _integer(episode.get("episode_steps"), "episode_steps", minimum=1) != length:
        raise ValueError(f"{path.name}: episode_steps disagrees with capture length")

    checkpoint_sha = _hex_sha(episode.get("checkpoint_sha256"), "checkpoint hash")
    prefix_sha = _hex_sha(episode.get("prefix_sha256"), "prefix hash")
    if prefix_sha != expected_prefix_sha256:
        raise ValueError(f"{path.name}: capture came from a different frozen prefix")

    stage_meta = episode.get("stagec_v2_metadata")
    expected_stage_meta = {
        "schema_version": STAGEC_SCHEMA,
        "prefix_sha256": expected_prefix_sha256,
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "proprio_dim": 30,
    }
    if not isinstance(stage_meta, dict):
        raise ValueError(f"{path.name}: missing Stage C metadata")
    for key, wanted in expected_stage_meta.items():
        actual = stage_meta.get(key)
        if key == "prefix_sha256":
            actual = str(actual).lower()
        if actual != wanted:
            raise ValueError(
                f"{path.name}: Stage C metadata mismatch for {key}: "
                f"{actual!r} != {wanted!r}"
            )

    mode = episode.get("mode")
    if mode not in ("full", "return"):
        raise ValueError(f"{path.name}: unsupported reset mode {mode!r}")
    tier = metadata.get("capture_tier")
    if tier != episode.get("capture_tier") or tier not in ("cycle", "returned_home"):
        raise ValueError(f"{path.name}: inconsistent capture tier")
    milestones = episode.get("milestones")
    if not isinstance(milestones, dict):
        raise ValueError(f"{path.name}: milestones must be an object")
    cycle_complete = (
        _integer(episode.get("cycles_completed", 0), "cycles_completed") >= 1
        or _integer(milestones.get("cycle_scored", 0), "cycle_scored") >= 1
    )
    returned_home = _integer(milestones.get("returned_home", 0), "returned_home") >= 1
    if tier == "cycle" and not cycle_complete:
        raise ValueError(f"{path.name}: cycle tier has no completed cycle")
    if tier == "returned_home" and (cycle_complete or not returned_home):
        raise ValueError(f"{path.name}: returned-home tier is inconsistent")

    env_seed = _integer(episode.get("env_seed"), "env_seed", maximum=2**63 - 1)
    action_seed = _integer(
        episode.get("action_seed"), "action_seed", maximum=2**63 - 1
    )
    episode_index = _integer(episode.get("episode_index"), "episode_index")
    env_index = _integer(episode.get("env_index"), "env_index")
    num_envs = _integer(episode.get("num_envs"), "num_envs", minimum=1, maximum=2)
    if env_index >= num_envs:
        raise ValueError(f"{path.name}: env_index is outside num_envs")
    provenance = (
        checkpoint_sha,
        env_seed,
        action_seed,
        env_index,
        episode_index,
    )
    return ValidatedCapture(
        path=path,
        sha256=sha256_file(path),
        checkpoint_sha256=checkpoint_sha,
        prefix_sha256=prefix_sha,
        provenance=provenance,
        episode=episode,
        capture_tier=str(tier),
    )


def selection_reason(capture: ValidatedCapture, checkpoint_sha256: str) -> str | None:
    """Return ``None`` only for an eligible clean cycle capture."""

    episode = capture.episode
    if capture.checkpoint_sha256 != checkpoint_sha256:
        return "other_checkpoint"
    if capture.capture_tier != "cycle":
        return "not_cycle_tier"
    if _integer(episode.get("partial_dumps", 0), "partial_dumps") != 0:
        return "partial_dump"
    clean = _integer(
        episode.get("dump_empty_completions", 0), "dump_empty_completions"
    )
    required = 2 if episode.get("mode") == "full" else 1
    if clean < required:
        return "insufficient_clean_dumps"
    return None


def _transfer(source: Path, destination: Path, mode: str) -> str:
    if mode in ("auto", "hardlink"):
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError as exc:
            if mode == "hardlink" or exc.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                getattr(errno, "ENOTSUP", errno.EPERM),
            }:
                raise
    shutil.copy2(source, destination)
    return "copy"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def curate(
    capture_dir: Path,
    checkpoint: Path,
    prefix_checkpoint: Path,
    out_dir: Path,
    *,
    copy_mode: str = "auto",
    minimum: int = 1,
) -> dict[str, Any]:
    capture_dir = capture_dir.resolve()
    checkpoint = checkpoint.resolve()
    prefix_checkpoint = prefix_checkpoint.resolve()
    out_dir = out_dir.resolve()
    if not capture_dir.is_dir():
        raise ValueError(f"capture directory does not exist: {capture_dir}")
    for label, path in (("checkpoint", checkpoint), ("prefix checkpoint", prefix_checkpoint)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    if out_dir in (capture_dir, checkpoint, prefix_checkpoint):
        raise ValueError("output directory aliases an input")
    if copy_mode not in ("auto", "copy", "hardlink"):
        raise ValueError("copy mode must be auto, copy, or hardlink")
    if minimum < 1:
        raise ValueError("minimum must be positive")

    checkpoint_sha = sha256_file(checkpoint)
    prefix_sha = sha256_file(prefix_checkpoint)
    source_files = sorted(capture_dir.glob("*.npz"), key=lambda item: item.name)
    if not source_files:
        raise ValueError(f"capture directory contains no .npz files: {capture_dir}")

    validated: list[ValidatedCapture] = []
    provenance_seen: dict[tuple[str, int, int, int, int], Path] = {}
    for source in source_files:
        capture = validate_capture(source, prefix_sha)
        prior = provenance_seen.get(capture.provenance)
        if prior is not None:
            raise ValueError(
                "duplicate seed-mine provenance: "
                f"{prior.name} and {source.name} identify {capture.provenance}"
            )
        provenance_seen[capture.provenance] = source
        validated.append(capture)

    accepted = [
        capture
        for capture in validated
        if selection_reason(capture, checkpoint_sha) is None
    ]
    if len(accepted) < minimum:
        raise ValueError(
            f"only {len(accepted)} clean cycle capture(s) for checkpoint; "
            f"minimum is {minimum}"
        )

    # Mining must be complete before custody is published.  A newly appearing
    # atomic capture means this was not a stable source snapshot.
    if [item.name for item in source_files] != [
        item.name for item in sorted(capture_dir.glob("*.npz"), key=lambda item: item.name)
    ]:
        raise RuntimeError("capture directory changed during validation")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        accepted_manifest: list[dict[str, Any]] = []
        for capture in accepted:
            destination = staging / capture.path.name
            if destination.exists():
                raise ValueError(f"duplicate output filename: {destination.name}")
            method = _transfer(capture.path, destination, copy_mode)
            destination_sha = sha256_file(destination)
            source_sha_after = sha256_file(capture.path)
            if destination_sha != capture.sha256 or source_sha_after != capture.sha256:
                raise RuntimeError(f"capture changed while copying: {capture.path.name}")
            checkpoint_tag, env_seed, action_seed, env_index, episode_index = (
                capture.provenance
            )
            accepted_manifest.append(
                {
                    "filename": destination.name,
                    "sha256": capture.sha256,
                    "transfer": method,
                    "provenance": {
                        "checkpoint_sha256": checkpoint_tag,
                        "env_seed": env_seed,
                        "action_seed": action_seed,
                        "env_index": env_index,
                        "episode_index": episode_index,
                    },
                    "scored": _integer(capture.episode.get("scored", 0), "scored"),
                    "dump_empty_completions": _integer(
                        capture.episode.get("dump_empty_completions", 0),
                        "dump_empty_completions",
                    ),
                    "partial_dumps": 0,
                    "mode": capture.episode.get("mode"),
                }
            )

        skipped = [
            {
                "filename": capture.path.name,
                "checkpoint_sha256": capture.checkpoint_sha256,
                "reason": selection_reason(capture, checkpoint_sha),
            }
            for capture in validated
            if selection_reason(capture, checkpoint_sha) is not None
        ]
        manifest = {
            "schema": CURATION_SCHEMA,
            "source_capture_dir": str(capture_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "prefix_checkpoint": str(prefix_checkpoint),
            "prefix_sha256": prefix_sha,
            "selection": {
                "capture_tier": "cycle",
                "partial_dumps": 0,
                "minimum_full_clean_dumps": 2,
                "minimum_return_clean_dumps": 1,
            },
            "source_archives": len(validated),
            "accepted_count": len(accepted_manifest),
            "accepted": accepted_manifest,
            "skipped": skipped,
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        os.replace(staging, out_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=("auto", "copy", "hardlink"),
        default="auto",
        help="auto tries a hardlink and safely falls back to a byte copy",
    )
    parser.add_argument("--minimum", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = curate(
            args.capture_dir,
            args.checkpoint,
            args.prefix_checkpoint,
            args.out_dir,
            copy_mode=args.copy_mode,
            minimum=args.minimum,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"CURATE_STAGEC_FAILED {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        "CURATE_STAGEC_DONE "
        + json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "accepted_count": manifest["accepted_count"],
                "source_archives": manifest["source_archives"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
