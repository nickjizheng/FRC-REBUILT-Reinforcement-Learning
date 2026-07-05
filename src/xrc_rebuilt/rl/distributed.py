"""RAM-backed transport for distributed DrQ-v2 collection.

Design (deliberately decoupled + debuggable, since it must run untested first):

- N *collector* processes each render one env-set on their own GPU and drop
  transition chunks into a tmpfs directory (``/dev/shm/xrc_dist/collector_<id>``).
- One *learner* process drains every collector's chunks into a per-stream replay,
  trains, and publishes fresh actor+encoder weights back to a shared directory.
- Collectors reload the newest weights every few steps.

Everything moves through **atomically-published files on tmpfs** (write to a
``.tmp`` then ``os.replace``), so:
- a half-written chunk is never read (only the renamed final name is globbed);
- a dead collector just reduces throughput; a restarted learner reconnects;
- off-policy RL tolerates the few-second chunk latency and mild weight staleness.

The chunk layout keeps each env's trajectory contiguous and ordered so the
learner can preserve per-stream n-step returns:

    field arrays shaped (num_envs, chunk_steps, *field_shape)

The learner appends collector ``c`` env ``e`` to global replay stream
``c * num_envs + e`` in chunk-sequence order.

This module has **no Isaac / torch-CUDA dependency at import** beyond torch for
the weight blobs, so the transport is unit-testable on any machine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CHUNK_PREFIX = "chunk_"
WEIGHTS_PREFIX = "weights_"
DEFAULT_ROOT = "/dev/shm/xrc_dist"


# --------------------------------------------------------------------------- #
# policy-frame packing (shared by collector + learner so they never disagree)
# --------------------------------------------------------------------------- #
def to_policy_frames(rgb: np.ndarray) -> np.ndarray:
    """(N, C_cam, 360, 640, 3) uint8 -> (N, 9, 90, 160) uint8 (4x downsample)."""
    small = rgb[:, :, ::4, ::4, :]
    n, cams, h, w, c = small.shape
    return small.transpose(0, 1, 4, 2, 3).reshape(n, cams * c, h, w).copy()


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def collector_dir(root: str | Path, collector_id: int) -> Path:
    return Path(root) / f"collector_{collector_id}"


def weights_dir(root: str | Path) -> Path:
    return Path(root) / "weights"


# --------------------------------------------------------------------------- #
# transition chunks (collector -> learner)
# --------------------------------------------------------------------------- #
FIELD_KEYS = ("obs", "proprio", "privileged", "action", "reward", "done")


def write_chunk(
    cdir: Path,
    seq: int,
    arrays: dict[str, np.ndarray],
    episodes: list[dict[str, Any]],
) -> Path:
    """Atomically publish one chunk. ``arrays[k]`` shaped (num_envs, steps, ...)."""
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / f".{CHUNK_PREFIX}{seq:09d}.tmp"
    final = cdir / f"{CHUNK_PREFIX}{seq:09d}.npz"
    payload = dict(arrays)
    payload["episodes"] = np.frombuffer(
        json.dumps(episodes).encode("utf-8"), dtype=np.uint8
    )
    # write to a real file handle so np.savez doesn't mangle the extension,
    # then atomically publish under the final .npz name
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp, final)
    return final


@dataclass
class Chunk:
    collector_id: int
    seq: int
    arrays: dict[str, np.ndarray]
    episodes: list[dict[str, Any]]


def _parse_seq(path: Path) -> int:
    try:
        return int(path.stem.replace(CHUNK_PREFIX, ""))
    except ValueError:
        return -1


def drain_chunks(
    root: str | Path,
    num_collectors: int,
    consumed: set[str],
    max_chunks: int = 128,
) -> list[Chunk]:
    """Load + delete every not-yet-seen chunk, in per-collector seq order.

    Returns chunks grouped so that, within a collector, sequence order is
    preserved (required for per-stream trajectory continuity). A chunk that
    fails to load (still being written / torn) is left for the next tick.
    """
    out: list[Chunk] = []
    for cid in range(num_collectors):
        cdir = collector_dir(root, cid)
        if not cdir.exists():
            continue
        files = sorted(cdir.glob(f"{CHUNK_PREFIX}*.npz"), key=_parse_seq)
        for f in files:
            key = str(f)
            if key in consumed:
                continue
            try:
                with np.load(f, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in FIELD_KEYS if k in data.files}
                    raw = data["episodes"] if "episodes" in data.files else None
                episodes = (
                    json.loads(bytes(raw).decode("utf-8")) if raw is not None and raw.size else []
                )
            except Exception:
                break  # this + later files for this collector aren't ready yet
            out.append(Chunk(cid, _parse_seq(f), arrays, episodes))
            consumed.add(key)
            try:
                f.unlink()
            except OSError:
                pass
            if len(out) >= max_chunks:
                return out
    return out


# --------------------------------------------------------------------------- #
# weight sync (learner -> collectors)
# --------------------------------------------------------------------------- #
def publish_weights(wdir: Path, blob: dict[str, Any], step: int, keep: int = 2) -> Path:
    """Atomically publish {encoder, actor, train_steps}. Keeps the newest ``keep``."""
    import torch

    wdir.mkdir(parents=True, exist_ok=True)
    tmp = wdir / f".{WEIGHTS_PREFIX}{step:09d}.pt.tmp"
    final = wdir / f"{WEIGHTS_PREFIX}{step:09d}.pt"
    torch.save(blob, tmp)
    os.replace(tmp, final)
    files = sorted(wdir.glob(f"{WEIGHTS_PREFIX}*.pt"), key=_parse_weight_step)
    for old in files[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass
    return final


def _parse_weight_step(path: Path) -> int:
    try:
        return int(path.stem.replace(WEIGHTS_PREFIX, ""))
    except ValueError:
        return -1


def latest_weights(wdir: str | Path) -> tuple[Path, int] | None:
    files = sorted(Path(wdir).glob(f"{WEIGHTS_PREFIX}*.pt"), key=_parse_weight_step)
    if not files:
        return None
    return files[-1], _parse_weight_step(files[-1])
