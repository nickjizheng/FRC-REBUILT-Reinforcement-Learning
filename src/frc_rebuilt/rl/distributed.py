"""RAM-backed transport for distributed DrQ-v2 collection.

Design (deliberately decoupled + debuggable, since it must run untested first):

- N *collector* processes each render one env-set on their own GPU and drop
  transition chunks into a tmpfs directory (``/dev/shm/frc_dist/collector_<id>``).
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
DEFAULT_ROOT = "/dev/shm/frc_dist"


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
# Prefix-takeover (Stage-E) suffix transport: a FLAT per-stream layout instead of
# the uniform (num_envs, steps) grid, because different envs hand off at different
# times so their suffix runs have different lengths.  ``stream`` tags each row with
# the local env index; the collector only ever writes SUFFIX transitions here, so no
# champion-prefix frame/reward is present in the chunk (spec property #2).
SUFFIX_FIELD_KEYS = ("obs", "proprio", "privileged", "action", "reward", "done", "stream")


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


def write_suffix_chunk(
    cdir: Path,
    seq: int,
    arrays: dict[str, np.ndarray],
    episodes: list[dict[str, Any]],
) -> Path:
    """Atomically publish one FLAT suffix chunk (prefix-takeover).

    ``arrays`` carries SUFFIX_FIELD_KEYS, each shaped (T, ...) and grouped by
    ``stream`` in temporal order within a stream.  Same atomic-publish discipline
    as ``write_chunk`` (write ``.tmp`` -> ``os.replace``); reuses CHUNK_PREFIX so
    ``drain_suffix_chunks`` globs the same names.
    """
    return write_chunk(cdir, seq, arrays, episodes)


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


def _chunk_identity(path: Path) -> str:
    """Return an identity that changes when a collector reuses a filename.

    Collectors intentionally restart their sequence counter at zero.  A path by
    itself therefore is not a durable identity: after a collector restart,
    ``chunk_000000000.npz`` is a new chunk even though its pathname is old.
    Including inode, size and mtime keeps the duplicate guard useful when an
    unlink temporarily fails without permanently black-holing restarted data.
    """
    stat = path.stat()
    return f"{path}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"


def drain_chunks(
    root: str | Path,
    num_collectors: int,
    consumed: set[str],
    max_chunks: int = 128,
    field_keys: tuple[str, ...] = FIELD_KEYS,
) -> list[Chunk]:
    """Load + delete every not-yet-seen chunk, in per-collector seq order.

    Returns chunks grouped so that, within a collector, sequence order is
    preserved (required for per-stream trajectory continuity). A chunk that
    fails to load (still being written / torn) is left for the next tick.

    ``field_keys`` selects the schema: the default uniform (num_envs, steps) grid,
    or ``SUFFIX_FIELD_KEYS`` for the flat prefix-takeover suffix layout.
    """
    out: list[Chunk] = []
    for cid in range(num_collectors):
        cdir = collector_dir(root, cid)
        if not cdir.exists():
            continue
        files = sorted(cdir.glob(f"{CHUNK_PREFIX}*.npz"), key=_parse_seq)
        for f in files:
            try:
                key = _chunk_identity(f)
            except OSError:
                continue
            if key in consumed:
                continue
            try:
                with np.load(f, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in field_keys if k in data.files}
                    raw = data["episodes"] if "episodes" in data.files else None
                episodes = (
                    json.loads(bytes(raw).decode("utf-8")) if raw is not None and raw.size else []
                )
            except Exception:
                break  # this + later files for this collector aren't ready yet
            out.append(Chunk(cid, _parse_seq(f), arrays, episodes))
            consumed.add(key)
            try:
                # Do not unlink a replacement that appeared at the same path
                # while the old file was being loaded.
                if _chunk_identity(f) == key:
                    f.unlink()
                    consumed.discard(key)
            except OSError:
                pass
            if len(out) >= max_chunks:
                return out
    return out


def drain_suffix_chunks(
    root: str | Path,
    num_collectors: int,
    consumed: set[str],
    max_chunks: int = 128,
) -> list[Chunk]:
    """``drain_chunks`` specialized to the flat suffix schema (prefix-takeover)."""
    return drain_chunks(root, num_collectors, consumed, max_chunks, field_keys=SUFFIX_FIELD_KEYS)


def ingest_suffix_chunk(replay, collector_id: int, collector_envs: int, chunk: "Chunk") -> dict:
    """Append a flat suffix chunk to ``replay`` (a PerEnvReplay), preserving each
    stream's order and terminal boundaries.

    The chunk holds ONLY suffix transitions (the collector never wrote prefix), so
    the first transition of a stream is the candidate's H+1 action and no
    champion-prefix frame/reward is present.  A non-finite transition is dropped and
    its stream terminal-boundaried (identical to the learner's uniform ingestion), so
    an n-step return can never bridge the gap it leaves.  ``done`` rows -- natural
    episode end OR forced-reset truncation -- terminal-boundary the stream so an
    n-step return cannot cross into the next episode's suffix through the (unwritten)
    prefix gap.  Returns {added, rejected}.
    """
    a = chunk.arrays
    stream_local = a["stream"]
    T = int(stream_local.shape[0])
    # Validate the local stream ids: an out-of-range value (a resized / misconfigured
    # collector) would silently index the WRONG ring or raise. Reject the whole chunk and
    # terminal-boundary this collector's streams, mirroring the uniform learner's
    # ``chunk_envs > configured`` rejection (adversarial finding wf_778e3bb8, ingest-map).
    if T and (int(stream_local.min()) < 0 or int(stream_local.max()) >= collector_envs):
        for e in range(collector_envs):
            replay.mark_boundary(collector_id * collector_envs + e)
        return {"added": 0, "rejected": T, "invalid_stream": True}
    added = rejected = 0
    for t in range(T):
        stream = collector_id * collector_envs + int(stream_local[t])
        if not (
            np.isfinite(a["proprio"][t]).all()
            and np.isfinite(a["privileged"][t]).all()
            and np.isfinite(a["action"][t]).all()
            and np.isfinite(a["reward"][t])
        ):
            rejected += 1
            replay.mark_boundary(stream)
            continue
        replay.add(
            stream,
            a["obs"][t],
            a["proprio"][t],
            a["privileged"][t],
            a["action"][t],
            float(a["reward"][t]),
            bool(a["done"][t]),
        )
        added += 1
    return {"added": added, "rejected": rejected}


class SuffixIngestor:
    """Restart-safe suffix ingestion for the prefix-takeover learner.

    A collector restart (routine PhysX crash or the STREAM_STALL watchdog) resets its
    chunk sequence to 0 and abandons any in-flight suffix, whose last stored row is
    NON-terminal (a mid-episode flush).  The learner's replay is persistent and
    survives the restart, so the restarted collector's fresh episode would otherwise be
    appended to the SAME ring immediately after that abandoned non-terminal tail -- and
    an n-step return could bridge the two episodes across the unwritten prefix gap
    (adversarial finding wf_778e3bb8, cross-chunk/restart; contract property 5).

    This ingestor tracks per-collector chunk-sequence continuity and, on any seq reset
    or gap (``seq != last + 1``), terminal-boundaries every stream that collector owns
    BEFORE ingesting the restarted chunk.  For an off-policy n-step replay the boundary
    is the whole guarantee -- episode ORDER within a ring is irrelevant because every
    sampled n-step chain is intra-episode -- so this fully restores isolation without an
    epoch-tagged filename scheme.
    """

    def __init__(self, replay, collector_envs: int):
        self.replay = replay
        self.collector_envs = int(collector_envs)
        self._last_seq: dict[int, int] = {}

    def ingest(self, chunk: "Chunk") -> dict:
        cid = chunk.collector_id
        last = self._last_seq.get(cid)
        restart = last is not None and chunk.seq != last + 1
        if restart:                       # close every open tail this collector owns
            for e in range(self.collector_envs):
                self.replay.mark_boundary(cid * self.collector_envs + e)
        self._last_seq[cid] = chunk.seq
        res = ingest_suffix_chunk(self.replay, cid, self.collector_envs, chunk)
        res["restart_boundary"] = bool(restart)
        return res


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
