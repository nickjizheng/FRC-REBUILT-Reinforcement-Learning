"""Stage-C v2 replay composition and restart-safe uniform ingestion.

This module intentionally leaves the original replay and distributed transport
untouched.  ``GroupedPerEnvReplay`` owns a normal :class:`PerEnvReplay`, but
draws an exact quota per curriculum group before delegating each draw to the
individual trajectory rings.  A ring is still the smallest sampling unit, so
n-step returns can never cross collector environments or curriculum streams.

``UniformChunkIngestor`` is the uniform-grid counterpart of the suffix
ingestor in ``distributed.py``.  It validates a chunk before indexing it and
closes every ring owned by a collector whenever that collector's sequence
counter resets or skips.  This is important because Isaac collectors restart
frequently while the learner/replay process remains alive.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .replay import Batch, PerEnvReplay


DEFAULT_STAGE_C_STREAM_GROUPS = (
    "full",
    "full",
    "postdump",
    "collect",
    "return",
    "return",
)


@dataclass(frozen=True)
class CapturedEpisode:
    """One complete, continuity-checked episode retained by the ingestor.

    Capture is deliberately separate from the live replay rings.  Replay rows
    arrive in short chunks, while the terminal statistics that tell us whether
    an episode was useful arrive only with its final chunk.  Keeping a small
    per-stream cache lets the learner archive a rare successful trajectory
    without changing the default replay or transport schemas.
    """

    stream_index: int
    group: str
    arrays: dict[str, np.ndarray]
    stats: dict[str, Any]


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _exact_quotas(total: int, names: Sequence[str], weights: np.ndarray) -> dict[str, int]:
    """Largest-remainder allocation whose result always sums to ``total``."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not names:
        return {}
    raw = total * weights / weights.sum()
    base = np.floor(raw).astype(np.int64)
    remaining = total - int(base.sum())
    # Stable group-order tie breaking makes configured quotas reproducible.
    order = sorted(range(len(names)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:remaining]:
        base[i] += 1
    return {name: int(count) for name, count in zip(names, base)}


class GroupedPerEnvReplay:
    """A grouped sampler backed by one :class:`PerEnvReplay` ring per stream.

    Parameters
    ----------
    stream_groups:
        One group name per replay stream.  Repeated names (for example two
        ``"return"`` streams) share one group quota.
    group_weights:
        Optional relative weights by group.  Missing names default to 1.0.
        The default therefore gives every *available group* equal weight, not
        every stream.  A group with no valid n-step anchors is omitted and its
        quota is automatically redistributed across the remaining groups.

    All other keyword arguments are passed to ``PerEnvReplay``/``ReplayRing``.
    ``sample`` is an alias for ``sample_grouped`` so this class can be dropped
    into the existing learner with minimal wiring.
    """

    def __init__(
        self,
        stream_groups: Sequence[str] = DEFAULT_STAGE_C_STREAM_GROUPS,
        capacity_per_env: int = 100_000,
        *,
        group_weights: Mapping[str, float] | None = None,
        seed: int = 0,
        **ring_kwargs: Any,
    ):
        groups = tuple(str(group).strip() for group in stream_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("stream_groups must contain one non-empty name per stream")

        self.stream_groups = groups
        self.group_names = _ordered_unique(groups)
        configured = dict(group_weights or {})
        unknown = set(configured).difference(self.group_names)
        if unknown:
            raise ValueError(f"group_weights contains unknown groups: {sorted(unknown)}")
        self.group_weights = {
            name: float(configured.get(name, 1.0)) for name in self.group_names
        }
        if any(not np.isfinite(weight) or weight < 0.0 for weight in self.group_weights.values()):
            raise ValueError("group weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in self.group_weights.values()):
            raise ValueError("at least one group weight must be positive")

        self.replay = PerEnvReplay(
            num_envs=len(groups),
            capacity_per_env=capacity_per_env,
            seed=seed,
            **ring_kwargs,
        )
        self.rng = np.random.default_rng(seed + 104_729)
        self.last_group_counts: dict[str, int] = {}
        self.last_ring_counts: tuple[int, ...] = (0,) * len(groups)

    @property
    def rings(self):
        """Expose rings for diagnostics without changing their ownership."""
        return self.replay.rings

    def add(self, stream_index: int, *args, **kwargs) -> None:
        self.replay.add(stream_index, *args, **kwargs)

    def mark_boundary(self, stream_index: int) -> None:
        self.replay.mark_boundary(stream_index)

    def __len__(self) -> int:
        return len(self.replay)

    def ready(self, min_total: int, min_live_fraction: float = 1.0) -> bool:
        return self.replay.ready(min_total, min_live_fraction)

    def valid_fill(self) -> tuple[int, ...]:
        """Number of legal n-step anchors in each stream ring."""
        return tuple(max(0, len(ring) - ring.n_step - 1) for ring in self.rings)

    def _weights_for_call(
        self, override: Mapping[str, float] | None
    ) -> dict[str, float]:
        weights = dict(self.group_weights)
        if override is not None:
            unknown = set(override).difference(self.group_names)
            if unknown:
                raise ValueError(f"group_weights contains unknown groups: {sorted(unknown)}")
            weights.update({name: float(value) for name, value in override.items()})
        if any(not np.isfinite(weight) or weight < 0.0 for weight in weights.values()):
            raise ValueError("group weights must be finite and non-negative")
        return weights

    def sample_grouped(
        self,
        batch_size: int,
        group_weights: Mapping[str, float] | None = None,
    ) -> Batch:
        """Sample an exact weighted group mixture, then fill it by ring size.

        Within a group, ring choice follows the same proportional-to-valid-fill
        multinomial rule as ``PerEnvReplay.sample``.  Unavailable groups never
        receive a quota; normalization over the remaining weights performs the
        requested redistribution.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        fills = np.asarray(self.valid_fill(), dtype=np.float64)
        weights = self._weights_for_call(group_weights)
        available = [
            name
            for name in self.group_names
            if weights[name] > 0.0
            and any(fills[i] > 0.0 for i, group in enumerate(self.stream_groups) if group == name)
        ]
        if not available:
            raise RuntimeError("no replay group has a valid n-step anchor")

        group_quota = _exact_quotas(
            batch_size,
            available,
            np.asarray([weights[name] for name in available], dtype=np.float64),
        )
        ring_counts = np.zeros(len(self.rings), dtype=np.int64)
        for name in available:
            indices = np.asarray(
                [i for i, group in enumerate(self.stream_groups) if group == name and fills[i] > 0],
                dtype=np.int64,
            )
            probability = fills[indices] / fills[indices].sum()
            ring_counts[indices] = self.rng.multinomial(group_quota[name], probability)

        parts = [
            ring.sample(int(count))
            for ring, count in zip(self.rings, ring_counts)
            if count > 0
        ]
        # available always implies at least one positive group quota for B > 0.
        batch = Batch(
            **{
                name: np.concatenate([getattr(part, name) for part in parts])
                for name in Batch.__dataclass_fields__
            }
        )
        self.last_group_counts = {
            name: int(sum(ring_counts[i] for i, group in enumerate(self.stream_groups) if group == name))
            for name in self.group_names
        }
        self.last_ring_counts = tuple(int(count) for count in ring_counts)
        return batch

    sample = sample_grouped


_UNIFORM_FIELDS = ("obs", "proprio", "privileged", "action", "reward", "done")


class UniformChunkIngestor:
    """Validate and ingest uniform ``(env, time, ...)`` collector chunks.

    The supplied replay only needs ``rings``, ``add`` and ``mark_boundary``;
    both ``PerEnvReplay`` and ``GroupedPerEnvReplay`` satisfy that interface.
    Malformed chunks are rejected as a unit and every owned stream is closed.
    A non-finite row is rejected individually and closes just that stream at
    the gap.  ``ingest`` returns counters plus ``restart_boundary`` and, for a
    malformed chunk, a human-readable ``schema_errors`` list.
    """

    def __init__(
        self,
        replay,
        collector_envs: int,
        *,
        capture_groups: Sequence[str] = (),
    ):
        self.replay = replay
        self.collector_envs = int(collector_envs)
        if self.collector_envs <= 0:
            raise ValueError("collector_envs must be positive")
        if not hasattr(replay, "rings") or not replay.rings:
            raise ValueError("replay must expose at least one trajectory ring")
        self._last_seq: dict[int, int] = {}
        requested = frozenset(str(group) for group in capture_groups)
        stream_groups = tuple(getattr(replay, "stream_groups", ()))
        if requested and len(stream_groups) != len(replay.rings):
            raise ValueError(
                "capture_groups requires replay.stream_groups to identify every ring"
            )
        unknown = requested.difference(stream_groups)
        if unknown:
            raise ValueError(f"capture_groups contains unknown groups: {sorted(unknown)}")
        self.capture_groups = requested
        self._stream_groups = stream_groups
        self._episode_rows: dict[int, dict[str, list[np.ndarray]]] = {}
        self._capture_tainted: set[int] = set()

    def _owned_streams(self, collector_id: int) -> range:
        start = collector_id * self.collector_envs
        stop = start + self.collector_envs
        if collector_id < 0 or stop > len(self.replay.rings):
            raise ValueError(
                f"collector {collector_id} owns [{start}, {stop}), but replay has "
                f"{len(self.replay.rings)} streams"
            )
        return range(start, stop)

    def _clear_capture(self, stream_index: int, *, tainted: bool = False) -> None:
        stream_index = int(stream_index)
        self._episode_rows.pop(stream_index, None)
        if tainted:
            self._capture_tainted.add(stream_index)
        else:
            self._capture_tainted.discard(stream_index)

    def _close_owned(self, collector_id: int, *, tainted: bool = False) -> None:
        for stream in self._owned_streams(collector_id):
            self.replay.mark_boundary(stream)
            self._clear_capture(stream, tainted=tainted)

    def _captures_stream(self, stream_index: int) -> bool:
        return bool(
            self.capture_groups
            and self._stream_groups[int(stream_index)] in self.capture_groups
        )

    def _capture_row(
        self,
        stream_index: int,
        arrays: Mapping[str, np.ndarray],
        env: int,
        step: int,
    ) -> None:
        if (
            not self._captures_stream(stream_index)
            or int(stream_index) in self._capture_tainted
        ):
            return
        rows = self._episode_rows.setdefault(
            int(stream_index), {key: [] for key in _UNIFORM_FIELDS}
        )
        for key in _UNIFORM_FIELDS:
            rows[key].append(np.asarray(arrays[key][env, step]).copy())

    def _finish_capture(
        self,
        stream_index: int,
        stats: Mapping[str, Any] | None,
    ) -> CapturedEpisode | None:
        rows = self._episode_rows.pop(int(stream_index), None)
        if rows is None or not rows["reward"]:
            return None
        arrays = {
            key: np.stack(values, axis=0)
            for key, values in rows.items()
        }
        return CapturedEpisode(
            stream_index=int(stream_index),
            group=self._stream_groups[int(stream_index)],
            arrays=arrays,
            stats=dict(stats or {}),
        )

    @staticmethod
    def _terminal_stats_by_env(
        episodes: Sequence[Mapping[str, Any]], collector_envs: int
    ) -> dict[int, deque[dict[str, Any]]]:
        """Index terminal records in the same per-env order as ``done`` rows.

        Older collectors do not emit ``env_index``.  They remain fully
        compatible for replay ingestion, but their episodes cannot be safely
        associated with one of two same-mode streams and therefore carry empty
        capture metadata.
        """

        indexed: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
        for episode in episodes:
            if not isinstance(episode, Mapping) or "env_index" not in episode:
                continue
            try:
                env = int(episode["env_index"])
            except (TypeError, ValueError):
                continue
            if 0 <= env < int(collector_envs):
                indexed[env].append(dict(episode))
        return dict(indexed)

    def _schema_errors(self, arrays: Mapping[str, np.ndarray]) -> list[str]:
        errors: list[str] = []
        missing = [key for key in _UNIFORM_FIELDS if key not in arrays]
        if missing:
            return [f"missing fields: {', '.join(missing)}"]
        if any(not isinstance(arrays[key], np.ndarray) for key in _UNIFORM_FIELDS):
            return ["all fields must be numpy arrays"]

        first = arrays["reward"].shape[:2]
        if arrays["reward"].ndim != 2:
            errors.append(f"reward must have shape (env,time), got {arrays['reward'].shape}")
            first = ()
        elif first[0] != self.collector_envs:
            errors.append(
                f"chunk has {first[0]} envs, configured collector_envs={self.collector_envs}"
            )

        ring = self.replay.rings[0]
        expected_tails = {
            "obs": ring.obs.shape[1:],
            "proprio": ring.proprio.shape[1:],
            "privileged": ring.privileged.shape[1:],
            "action": ring.action.shape[1:],
            "reward": (),
            "done": (),
        }
        if first:
            envs, steps = first
            for key, tail in expected_tails.items():
                expected = (envs, steps, *tail)
                if arrays[key].shape != expected:
                    errors.append(f"{key} must have shape {expected}, got {arrays[key].shape}")

        for key in _UNIFORM_FIELDS:
            dtype = arrays[key].dtype
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                errors.append(f"{key} must have a numeric/bool dtype, got {dtype}")
        if arrays["obs"].dtype != np.uint8:
            errors.append(f"obs must be uint8, got {arrays['obs'].dtype}")
        return errors

    @staticmethod
    def _row_count_hint(arrays: Mapping[str, np.ndarray]) -> int:
        reward = arrays.get("reward")
        return int(reward.size) if isinstance(reward, np.ndarray) else 0

    @staticmethod
    def _row_is_valid(arrays: Mapping[str, np.ndarray], env: int, step: int) -> bool:
        for key in _UNIFORM_FIELDS:
            try:
                if not np.isfinite(arrays[key][env, step]).all():
                    return False
            except TypeError:
                return False
        done = arrays["done"][env, step]
        return bool(done == 0 or done == 1)

    def ingest(self, chunk) -> dict[str, Any]:
        """Ingest a ``distributed.Chunk`` (or equivalent duck-typed object)."""
        collector_id = int(chunk.collector_id)
        self._owned_streams(collector_id)  # validate mapping before mutating state
        seq = int(chunk.seq)
        last = self._last_seq.get(collector_id)
        discontinuity = last is not None and seq != last + 1
        if discontinuity:
            # A sequence rewind is a collector restart whose first chunk begins
            # at a fresh environment reset.  A forward gap lost part of the
            # current episode, which must stay ineligible until its next done.
            self._close_owned(collector_id, tainted=bool(seq > int(last)))
        self._last_seq[collector_id] = seq

        arrays = chunk.arrays
        if not isinstance(arrays, Mapping):
            self._close_owned(collector_id, tainted=True)
            return {
                "added": 0,
                "rejected": 0,
                "invalid_chunk": True,
                "schema_errors": ["chunk.arrays must be a mapping"],
                "restart_boundary": bool(discontinuity),
            }
        errors = self._schema_errors(arrays)
        if errors:
            self._close_owned(collector_id, tainted=True)
            return {
                "added": 0,
                "rejected": self._row_count_hint(arrays),
                "invalid_chunk": True,
                "schema_errors": errors,
                "restart_boundary": bool(discontinuity),
            }

        envs, steps = arrays["reward"].shape
        terminal_stats = self._terminal_stats_by_env(
            getattr(chunk, "episodes", ()), self.collector_envs
        )
        completed: list[CapturedEpisode] = []
        added = rejected = 0
        for env in range(envs):
            stream = collector_id * self.collector_envs + env
            for step in range(steps):
                if not self._row_is_valid(arrays, env, step):
                    rejected += 1
                    self.replay.mark_boundary(stream)
                    self._clear_capture(stream, tainted=True)
                    continue
                self.replay.add(
                    stream,
                    arrays["obs"][env, step],
                    arrays["proprio"][env, step],
                    arrays["privileged"][env, step],
                    arrays["action"][env, step],
                    float(arrays["reward"][env, step]),
                    bool(arrays["done"][env, step]),
                )
                self._capture_row(stream, arrays, env, step)
                added += 1
                if bool(arrays["done"][env, step]) and self._captures_stream(stream):
                    queue = terminal_stats.get(env)
                    stats = queue.popleft() if queue else None
                    if stream in self._capture_tainted:
                        self._clear_capture(stream)
                    else:
                        episode = self._finish_capture(stream, stats)
                        if episode is not None:
                            completed.append(episode)
        result = {
            "added": added,
            "rejected": rejected,
            "invalid_chunk": False,
            "restart_boundary": bool(discontinuity),
        }
        # Preserve the exact legacy return shape unless capture was explicitly
        # enabled, keeping existing users and equality-based tests compatible.
        if self.capture_groups:
            result["completed_episodes"] = completed
        return result


__all__ = [
    "CapturedEpisode",
    "DEFAULT_STAGE_C_STREAM_GROUPS",
    "GroupedPerEnvReplay",
    "UniformChunkIngestor",
]
