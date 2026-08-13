"""Phase-balanced Stage C v2.3 learner with safe legacy warm start."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from frc_rebuilt.competition_robot import CAMERA_RIG_REVISION
from frc_rebuilt.rl.cycle_v2 import ROUTE_EFFICIENCY_REVISION, STAGE_D_REVISIONS
from frc_rebuilt.rl.policy_v2 import (
    ACTION_POLICY,
    FIELD_STRATEGY,
    LEGACY_PROPRIO_DIM,
    RETURN_SKILL_PRELOAD,
    SCHEMA_VERSION,
    V2_PROPRIO_DIM,
)
from frc_rebuilt.rl.replay import Batch, ReplayRing
from frc_rebuilt.rl.replay_v2 import CapturedEpisode

FIRST_PHASE_INDEX = LEGACY_PROPRIO_DIM
V2_FEATURE_NAMES = (
    "phase_first_cycle",
    "phase_leave",
    "phase_collect",
    "phase_return",
    "phase_score",
    "qualified_load_over_60",
    "target_load_over_60",
    "time_remaining",
)
ELITE_FIELDS = ("obs", "proprio", "privileged", "action", "reward", "done")
ELITE_ARCHIVE_SCHEMA = "stagec_elite_source_v7"
ELITE_POOLS = (
    "full_multi",
    "full_cycle",
    "full_return",
    "postdump_cycle",
    "return",
)
# Capacity and sampling deliberately use the same fixed 30/20/10/30/10 mix.
# Missing pools do not donate their share to an easier source.
ELITE_POOL_WEIGHTS = {
    "full_multi": 3,
    "full_cycle": 2,
    "full_return": 1,
    "postdump_cycle": 3,
    "return": 1,
}
# Ninety-six retained files become 24/24/8/32/8.  Each source owns its quota,
# so high-throughput RETURN episodes cannot evict rare FULL or POSTDUMP wins.
ELITE_ARCHIVE_POOL_WEIGHTS = {
    "full_multi": 3,
    "full_cycle": 3,
    "full_return": 1,
    "postdump_cycle": 4,
    "return": 1,
}
SEEDMINE_CAPTURE_SCHEMA = "stagec_training_episode_v1"
SEEDMINE_EVAL_SCHEMA = "stagec_seed_eval_v1"
ROUTE_EFFICIENCY_V3_KEYS = (
    "reward_revision",
    "refresh_ramp_side_on_dump",
    "ramp_side_deadband_x",
    "outer_rail_enter_x",
    "outer_rail_exit_x",
    "outer_rail_max_x",
    "outer_rail_grace_steps",
    "outer_rail_penalty_per_step",
    "outer_rail_penalty_cap",
    "outer_rail_min_scale",
    "outer_rail_escalation_steps",
    "outer_rail_max_multiplier",
    "intake_substeps",
)
ROUTE_EFFICIENCY_V4_KEYS = ROUTE_EFFICIENCY_V3_KEYS + (
    "require_ramp_out",
    "ramp_out_half_width",
    "ramp_out_bonus",
    "off_ramp_exit_penalty",
)
ROUTE_EFFICIENCY_V5_KEYS = ROUTE_EFFICIENCY_V4_KEYS + (
    "postdump_require_target_load",
)
ROUTE_EFFICIENCY_V6_KEYS = ROUTE_EFFICIENCY_V5_KEYS + (
    "postdump_complete_cycle",
    "postdump_depleted_count",
    "postdump_depleted_prob",
)
ROUTE_EFFICIENCY_V8_KEYS = ROUTE_EFFICIENCY_V6_KEYS + (
    "preferred_repeat_load",
    "repeat_load_return_bonus",
    "repeat_load_score_bonus",
)
ROUTE_EFFICIENCY_V9_KEYS = ROUTE_EFFICIENCY_V8_KEYS + (
    "collect_stall_steps",
    "return_time_guard",
)
ROUTE_EFFICIENCY_KEYS = ROUTE_EFFICIENCY_V9_KEYS + (
    "intake_during_return",
)
ROUTE_EFFICIENCY_V1_KEYS = ROUTE_EFFICIENCY_V3_KEYS[:9]
ROUTE_EFFICIENCY_MIGRATIONS = {
    ("outer_rail_v1", "outer_rail_v2"),
    ("outer_rail_v2", "outer_rail_v3"),
    ("outer_rail_v3", "outer_rail_v4_ramp_out"),
    ("outer_rail_v4_ramp_out", "cycle_efficiency_v5"),
    ("cycle_efficiency_v5", "cycle_bridge_v6"),
    ("outer_rail_v4_ramp_out", "score_efficiency_v8"),
    ("outer_rail_v4_ramp_out", "score_efficiency_v9"),
    ("score_efficiency_v8", "score_efficiency_v9"),
    ("outer_rail_v4_ramp_out", "score_efficiency_v10_return_intake"),
    ("score_efficiency_v9", "score_efficiency_v10_return_intake"),
    # v15 ramp-free (score_efficiency_v11_rampfree).  The v9 pair is the one
    # exercised here: v14 checkpoints carry reward_revision=score_efficiency_v9.
    # The v8 and outer_rail_v4 pairs mirror the v11 plumbing so an older
    # champion could also be migrated onto the ramp-free contract.
    ("outer_rail_v4_ramp_out", "score_efficiency_v11_rampfree"),
    ("score_efficiency_v8", "score_efficiency_v11_rampfree"),
    ("score_efficiency_v9", "score_efficiency_v11_rampfree"),
    # Stage D official-match bring-up (docs/STAGE_D_DESIGN.md).  D0 keeps
    # score_efficiency_v11_rampfree (no reward-contract change), so the first
    # rename is D1's hub-gated revision; the v11 pair also allows a direct
    # jump if D1 ever resumes straight from a Stage C champion.
    ("score_efficiency_v11_rampfree", "stage_d_v1"),
    ("stage_d_v1", "stage_d_v2"),
    ("score_efficiency_v11_rampfree", "stage_d_v2"),
}
# The v11/v15 ramp-free experiment intentionally retires the ramp-out route
# contract inherited from the V4 champion, so these four keys -- and ONLY these
# four -- are allowed to differ across that one migration.  Every other key in
# the parent reward contract is still enforced exactly as before.
RAMPFREE_REVISION = "score_efficiency_v11_rampfree"
RAMPFREE_RELAXED_KEYS = frozenset(
    {
        "require_ramp_out",
        "ramp_out_half_width",
        "ramp_out_bonus",
        "off_ramp_exit_penalty",
    }
)
# Stage D migrations (v11_rampfree -> stage_d_v1 -> stage_d_v2) intentionally
# retune the keys below; every other parent-contract key is enforced exactly
# as before.
#   * return_time_guard is a FRACTION of the episode, so the 90 s -> 160 s
#     horizon change alters its absolute meaning (0.20 = force-return at 18 s
#     remaining on 90 s, 32 s remaining on 160 s).
#   * preferred_repeat_load: the 160 s ferry-first strategy fills the chamber
#     toward capacity during blackouts instead of stopping at the 90 s-era 30.
#   * outer_rail_penalty_per_step/_cap + off_ramp_exit_penalty: the Stage C
#     caps made a botched crossing cost up to 110+20 while LEAVE camping
#     capped at 5 -- deterministic policies learned that NOT trying is the
#     cheap action (leaveCamp 30-56%).  stage_d_v1 rebalances so an attempt
#     is always cheaper than a freeze.
STAGE_D_RELAXED_KEYS = frozenset(
    {
        "postdump_require_target_load",
        "return_time_guard",
        "preferred_repeat_load",
        "outer_rail_penalty_per_step",
        "outer_rail_penalty_cap",
        "off_ramp_exit_penalty",
    }
)

# Mirrors --require-ramp-out for this process.  When the ramp-out contract is
# off, elite capture must not reject or down-credit ramp-free episodes: those
# are precisely the behaviour the run exists to learn.  Set in main().
_ROUTE_GATE_ENABLED = True


@dataclass(frozen=True)
class EliteBehaviorBatch:
    """Minimal successful-behavior batch used only by the actor anchor."""

    obs: np.ndarray
    proprio: np.ndarray
    action: np.ndarray


_ELITE_BEHAVIOR_WINDOWS = (
    ("opener", 0.0, 33.0),
    ("live1", 55.0, 83.0),
    ("live2", 105.0, 130.0),
    ("endgame", 130.0, 160.000001),
)


class EliteScoreBehaviorPool:
    """Bounded suffix-actor examples from integrated cycle successes.

    The previous implementation retained only SCORE-phase rows.  That protected
    the fire trigger but allowed LEAVE/COLLECT/RETURN driving to collapse within
    a few actor updates.  Retain every suffix-controlled non-trigger row from a
    successful integrated episode, while still keeping positive fire triggers
    in a separate pool so they cannot disappear inside hundreds of drive rows.
    """

    def __init__(
        self,
        *,
        score_capacity: int,
        trigger_capacity: int,
        seed: int,
    ) -> None:
        if int(score_capacity) <= 0 or int(trigger_capacity) <= 0:
            raise ValueError("elite behavior capacities must be positive")
        self.score_capacity = int(score_capacity)
        self.trigger_capacity = int(trigger_capacity)
        self.rng = np.random.default_rng(seed)
        self._score: EliteBehaviorBatch | None = None
        self._trigger: EliteBehaviorBatch | None = None
        self._score_seen = 0
        self._trigger_seen = 0

    @staticmethod
    def _take(
        batch: EliteBehaviorBatch | None, indices: np.ndarray
    ) -> EliteBehaviorBatch | None:
        if not len(indices):
            return None
        return EliteBehaviorBatch(
            obs=np.asarray(batch.obs[indices]).copy(),
            proprio=np.asarray(batch.proprio[indices]).copy(),
            action=np.asarray(batch.action[indices]).copy(),
        )

    def _append_bounded(
        self,
        existing: EliteBehaviorBatch | None,
        incoming: EliteBehaviorBatch | None,
        capacity: int,
        seen: int,
    ) -> tuple[EliteBehaviorBatch | None, int]:
        """Add rows with seeded reservoir sampling instead of tail eviction.

        Keeping the last ``capacity`` rows made custody depend on archive file
        order: a late mediocre teacher could completely evict an early elite
        teacher.  A reservoir gives every row seen so far the same probability
        of surviving, while remaining deterministic for a fixed learner seed.
        """

        if incoming is None:
            return existing, int(seen)

        capacity = int(capacity)
        seen = int(seen)
        incoming_rows = int(len(incoming.proprio))
        if incoming_rows == 0:
            return existing, seen

        retained = 0 if existing is None else int(len(existing.proprio))
        fill = min(capacity - retained, incoming_rows)
        if fill > 0:
            prefix = self._take(incoming, np.arange(fill, dtype=np.int64))
            assert prefix is not None
            if existing is None:
                existing = prefix
            else:
                existing = EliteBehaviorBatch(
                    obs=np.concatenate((existing.obs, prefix.obs), axis=0),
                    proprio=np.concatenate((existing.proprio, prefix.proprio), axis=0),
                    action=np.concatenate((existing.action, prefix.action), axis=0),
                )
            retained += fill
            seen += fill

        assert existing is not None
        # Once full, apply Algorithm R one row at a time.  ``seen`` counts the
        # population represented by the reservoir, not merely retained rows.
        for row in range(fill, incoming_rows):
            seen += 1
            slot = int(self.rng.integers(0, seen))
            if slot < capacity:
                existing.obs[slot] = incoming.obs[row]
                existing.proprio[slot] = incoming.proprio[row]
                existing.action[slot] = incoming.action[row]
        return existing, seen

    def add(self, arrays: dict[str, np.ndarray]) -> None:
        obs = np.asarray(arrays["obs"])
        proprio = np.asarray(arrays["proprio"])
        action = np.asarray(arrays["action"])
        if (
            proprio.ndim != 2
            or proprio.shape[1] != V2_PROPRIO_DIM
            or action.ndim != 2
            or action.shape[0] != proprio.shape[0]
            or action.shape[1] <= 5
            or obs.shape[0] != proprio.shape[0]
        ):
            raise ValueError("elite behavior arrays do not match Stage C v2")
        phases = np.argmax(
            proprio[:, FIRST_PHASE_INDEX : FIRST_PHASE_INDEX + 5], axis=1
        )
        # A qualified FULL archive may continue for hundreds of steps after its
        # last completed repeat cycle.  Those unfinished LEAVE/COLLECT tails
        # are failure data, not behavior custody.  The verified cycle boundary
        # is visible as SCORE -> LEAVE in the stored phase stream.
        completed_cycle_ends = np.flatnonzero(
            (phases[:-1] == 4) & (phases[1:] == 1)
        )
        if not len(completed_cycle_ends):
            raise ValueError(
                "elite behavior episode has no verified SCORE-to-LEAVE cycle boundary"
            )
        successful_end = int(completed_cycle_ends[-1])
        suffix_controlled = (
            (proprio[:, FIRST_PHASE_INDEX] < 0.5)
            & (np.arange(len(proprio)) <= successful_end)
        )
        trigger = suffix_controlled & (action[:, 5] > 0.0)
        score_wait = suffix_controlled & ~trigger
        source = EliteBehaviorBatch(obs=obs, proprio=proprio, action=action)
        self._score, self._score_seen = self._append_bounded(
            self._score,
            self._take(source, np.flatnonzero(score_wait)),
            self.score_capacity,
            self._score_seen,
        )
        self._trigger, self._trigger_seen = self._append_bounded(
            self._trigger,
            self._take(source, np.flatnonzero(trigger)),
            self.trigger_capacity,
            self._trigger_seen,
        )

    @property
    def score_rows(self) -> int:
        return 0 if self._score is None else int(len(self._score.proprio))

    @property
    def trigger_rows(self) -> int:
        return 0 if self._trigger is None else int(len(self._trigger.proprio))

    @staticmethod
    def _window_indices(
        batch: EliteBehaviorBatch | None,
        start_s: float,
        end_s: float,
        full_episode_s: float,
    ) -> np.ndarray:
        if batch is None:
            return np.empty(0, dtype=np.int64)
        clock_s = np.clip(np.asarray(batch.proprio[:, 7]), 0.0, 1.0) * float(
            full_episode_s
        )
        return np.flatnonzero((clock_s >= float(start_s)) & (clock_s < float(end_s)))

    def window_rows(self, full_episode_s: float) -> dict[str, dict[str, int]]:
        if not np.isfinite(full_episode_s) or float(full_episode_s) <= 0.0:
            raise ValueError("full_episode_s must be finite and positive")
        rows: dict[str, dict[str, int]] = {}
        for name, start_s, end_s in _ELITE_BEHAVIOR_WINDOWS:
            score = int(
                len(self._window_indices(self._score, start_s, end_s, full_episode_s))
            )
            trigger = int(
                len(self._window_indices(self._trigger, start_s, end_s, full_episode_s))
            )
            rows[name] = {"score": score, "trigger": trigger, "total": score + trigger}
        return rows

    def sample(
        self,
        batch_size: int,
        trigger_fraction: float,
        *,
        window_balanced: bool = False,
        full_episode_s: float = 160.0,
    ) -> EliteBehaviorBatch | None:
        batch_size = int(batch_size)
        if batch_size <= 0 or (self.score_rows == 0 and self.trigger_rows == 0):
            return None
        trigger_fraction = float(np.clip(trigger_fraction, 0.0, 1.0))
        if window_balanced:
            if not np.isfinite(full_episode_s) or float(full_episode_s) <= 0.0:
                raise ValueError("full_episode_s must be finite and positive")
            window_count = len(_ELITE_BEHAVIOR_WINDOWS)
            quotas = np.full(window_count, batch_size // window_count, dtype=np.int64)
            quotas[: batch_size % window_count] += 1
            candidates: list[tuple[np.ndarray, np.ndarray]] = []
            for name, start_s, end_s in _ELITE_BEHAVIOR_WINDOWS:
                score_indices = self._window_indices(
                    self._score, start_s, end_s, full_episode_s
                )
                trigger_indices = self._window_indices(
                    self._trigger, start_s, end_s, full_episode_s
                )
                if not len(score_indices) and not len(trigger_indices):
                    raise ValueError(
                        f"elite behavior window {name} contains no validated rows"
                    )
                candidates.append((score_indices, trigger_indices))

            trigger_target = int(round(batch_size * trigger_fraction))
            if self.trigger_rows and self.score_rows:
                trigger_target = min(batch_size, max(1, trigger_target))
            elif self.trigger_rows:
                trigger_target = batch_size
            else:
                trigger_target = 0
            trigger_quotas = np.zeros(window_count, dtype=np.int64)
            eligible = [
                index for index, (_, trigger_indices) in enumerate(candidates)
                if len(trigger_indices) and quotas[index] > 0
            ]
            while int(trigger_quotas.sum()) < trigger_target:
                progressed = False
                for index in eligible:
                    if trigger_quotas[index] < quotas[index]:
                        trigger_quotas[index] += 1
                        progressed = True
                        if int(trigger_quotas.sum()) == trigger_target:
                            break
                if not progressed:
                    raise ValueError(
                        "window-balanced elite batch cannot preserve requested trigger coverage"
                    )

            parts: list[EliteBehaviorBatch] = []
            for index, (score_indices, trigger_indices) in enumerate(candidates):
                trigger_count = int(trigger_quotas[index])
                score_count = int(quotas[index]) - trigger_count
                if score_count and not len(score_indices):
                    if not len(trigger_indices):
                        raise ValueError("elite behavior window has no sampleable rows")
                    trigger_count += score_count
                    score_count = 0
                for source, source_indices, count in (
                    (self._score, score_indices, score_count),
                    (self._trigger, trigger_indices, trigger_count),
                ):
                    if count:
                        selected = source_indices[
                            self.rng.integers(0, len(source_indices), size=count)
                        ]
                        part = self._take(source, selected)
                        assert part is not None
                        parts.append(part)
            obs = np.concatenate([part.obs for part in parts], axis=0)
            proprio = np.concatenate([part.proprio for part in parts], axis=0)
            action = np.concatenate([part.action for part in parts], axis=0)
            order = self.rng.permutation(batch_size)
            return EliteBehaviorBatch(
                obs=obs[order], proprio=proprio[order], action=action[order]
            )

        if self.trigger_rows and self.score_rows:
            trigger_count = int(round(batch_size * trigger_fraction))
            trigger_count = min(batch_size, max(1, trigger_count))
        elif self.trigger_rows:
            trigger_count = batch_size
        else:
            trigger_count = 0
        score_count = batch_size - trigger_count
        parts: list[EliteBehaviorBatch] = []
        for source, count, rows in (
            (self._score, score_count, self.score_rows),
            (self._trigger, trigger_count, self.trigger_rows),
        ):
            if count:
                indices = self.rng.integers(0, rows, size=count)
                part = self._take(source, indices)
                assert part is not None
                parts.append(part)
        obs = np.concatenate([part.obs for part in parts], axis=0)
        proprio = np.concatenate([part.proprio for part in parts], axis=0)
        action = np.concatenate([part.action for part in parts], axis=0)
        order = self.rng.permutation(batch_size)
        return EliteBehaviorBatch(obs=obs[order], proprio=proprio[order], action=action[order])


def _parse_groups(text: str, expected: int) -> tuple[str, ...]:
    groups = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(groups) != int(expected):
        raise ValueError(f"expected {expected} stream groups, got {len(groups)}")
    return groups


def _parse_group_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in (part.strip() for part in text.split(",") if part.strip()):
        name, value = item.split("=", 1)
        out[name.strip()] = float(value)
    return out


def _elite_capture_groups(
    stream_groups: tuple[str, ...], *, enabled: bool
) -> tuple[str, ...]:
    """Capture only elite-capable groups that actually exist in this run."""

    if not enabled:
        return ()
    configured = set(stream_groups)
    return tuple(
        group
        for group in ("full", "postdump", "return")
        if group in configured
    )


def _anchor_beta(update: int, critic_only: int, start: float, floor: float, decay: int) -> float:
    if update <= critic_only:
        return float(start)
    mix = min(1.0, max(0.0, (update - critic_only) / max(1, decay)))
    return float(start) + (float(floor) - float(start)) * mix


def _elite_behavior_weight(
    update: int,
    critic_only: int,
    start: float,
    end: float,
    decay: int,
) -> float:
    """Linearly anneal imitation pressure after critic-only warm-up."""

    if update <= critic_only or decay <= 0:
        return float(start)
    mix = min(1.0, max(0.0, (update - critic_only) / decay))
    return float(start) + (float(end) - float(start)) * mix


def _schedule_origin_updates(
    resume_updates: int, metadata: object, *, reset: bool
) -> int:
    """Return the lifetime-update origin for branch-relative schedules.

    ``v2_updates`` is a lifetime lineage counter.  Warm-up, exploration, anchor
    decay, and elite consolidation are run-local schedules and must not inherit
    an already-exhausted 350k counter when a checkpoint starts a new branch.
    """

    resume_updates = max(0, int(resume_updates))
    if reset:
        return resume_updates
    if isinstance(metadata, dict):
        try:
            origin = int(metadata.get("schedule_origin_updates", 0))
        except (TypeError, ValueError):
            origin = 0
        if 0 <= origin <= resume_updates:
            return origin
    return 0


def _initial_explore_offset(
    train_steps: int,
    *,
    stddev_start: float,
    stddev_end: float,
    stddev_steps: int,
    initial_stddev: float,
) -> int:
    """Choose ``explore_offset`` so ``agent.stddev()`` starts as requested."""

    span = max(1e-9, float(stddev_start) - float(stddev_end))
    mix = (float(stddev_start) - float(initial_stddev)) / span
    warm_steps = int(round(np.clip(mix, 0.0, 1.0) * int(stddev_steps)))
    return int(train_steps) - warm_steps


def _expand_anchor_proprio(proprio: np.ndarray, target_load: int) -> np.ndarray:
    proprio = np.asarray(proprio, np.float32)
    if proprio.ndim != 2:
        raise ValueError(f"anchor proprio must be a matrix, got {proprio.shape}")
    if proprio.shape[1] == V2_PROPRIO_DIM:
        return proprio
    if proprio.shape[1] != LEGACY_PROPRIO_DIM:
        raise ValueError(f"anchor proprio width {proprio.shape[1]} is not 22 or 30")
    features = np.zeros((len(proprio), 8), np.float32)
    features[:, 0] = 1.0
    features[:, 6] = float(target_load) / 60.0
    features[:, 7] = 1.0 - np.clip(proprio[:, 7], 0.0, 1.0)
    return np.concatenate([proprio, features], axis=1)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suffix_actor_mask(proprio: np.ndarray) -> np.ndarray:
    """Select rows controlled by the suffix actor for actor-Q and critic learning."""
    proprio = np.asarray(proprio)
    if proprio.ndim != 2 or proprio.shape[1] != V2_PROPRIO_DIM:
        raise ValueError(f"suffix proprio must have shape (N, 30), got {proprio.shape}")
    return np.equal(proprio[:, FIRST_PHASE_INDEX], 0.0)


_PHASE_FEATURE_INDEX = {
    "first": FIRST_PHASE_INDEX,
    "leave": FIRST_PHASE_INDEX + 1,
    "collect": FIRST_PHASE_INDEX + 2,
    "return": FIRST_PHASE_INDEX + 3,
    "score": FIRST_PHASE_INDEX + 4,
}


def _parse_actor_phases(text: str) -> tuple[str, ...]:
    phases = tuple(part.strip().lower() for part in str(text).split(",") if part.strip())
    if not phases:
        raise ValueError("--actor-phases must name at least one suffix phase")
    unknown = sorted(set(phases) - (set(_PHASE_FEATURE_INDEX) - {"first"}))
    if unknown:
        raise ValueError(f"unknown --actor-phases values: {unknown}")
    return phases


def _actor_phase_mask(proprio: np.ndarray, phases: tuple[str, ...]) -> np.ndarray:
    """Select only explicitly trainable suffix phases for actor-Q gradients."""

    proprio = np.asarray(proprio)
    if proprio.ndim != 2 or proprio.shape[1] != V2_PROPRIO_DIM:
        raise ValueError(f"suffix proprio must have shape (N, 30), got {proprio.shape}")
    selected = np.zeros(len(proprio), dtype=bool)
    for phase in phases:
        selected |= np.asarray(
            proprio[:, _PHASE_FEATURE_INDEX[phase]] > 0.5, dtype=bool
        )
    return selected & _suffix_actor_mask(proprio)


_ACTOR_INTERVAL_MEAN_KEYS = (
    "actor_loss",
    "bc_anchor",
    "anchor_weight",
    "elite_behavior_bc",
    "elite_behavior_rows",
    "elite_behavior_weight",
    "elite_behavior_bc_opener",
    "elite_behavior_bc_live1",
    "elite_behavior_bc_live2",
    "elite_behavior_bc_endgame",
    "elite_behavior_rows_opener",
    "elite_behavior_rows_live1",
    "elite_behavior_rows_live2",
    "elite_behavior_rows_endgame",
    "lambda_q",
    "q_pi",
    "q_pi_noisy",
    "q_pi_center",
    "q_pi_center_minus_noisy",
    "actor_rows",
)


def _summarize_actor_interval_metrics(
    records: list[dict[str, float]],
) -> dict[str, float | int]:
    """Aggregate actor-bearing learner steps without leaking NaNs into JSON."""

    actor_records: list[dict[str, float]] = []
    for record in records:
        try:
            actor_rows = float(record.get("actor_rows", 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(actor_rows) and actor_rows > 0.0:
            actor_records.append(record)

    summary: dict[str, float | int] = {
        "actor_interval_updates": len(actor_records),
        "actor_interval_applied": sum(
            1
            for record in actor_records
            if np.isfinite(float(record.get("actor_applied", 0.0)))
            and float(record.get("actor_applied", 0.0)) > 0.5
        ),
    }
    for key in _ACTOR_INTERVAL_MEAN_KEYS:
        values: list[float] = []
        for record in actor_records:
            try:
                value = float(record[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    return summary


def _validate_resume_metadata(metadata: object, expected: dict[str, object]) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("30-wide resume is missing Stage C v2 metadata")
    # 2026-07-28: postdump_require_target_load selects a lane SUCCESS MILESTONE
    # (termination), not a reward magnitude.  Exempt at the validator so every
    # call site (resume, weights transport, seedmine source) treats it alike.
    expected = {k: v for k, v in expected.items() if k != "postdump_require_target_load"}
    for key, wanted in expected.items():
        actual = metadata.get(key)
        if isinstance(wanted, float):
            try:
                matches = actual is not None and abs(float(actual) - wanted) <= 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == wanted
        if not matches:
            raise ValueError(
                f"v2 resume metadata mismatch for {key}: {actual!r} != {wanted!r}"
            )


_SEEDMINE_SCHEDULE_KEYS = frozenset(
    {
        "schedule_origin_updates",
        "stddev_start",
        "stddev_end",
        "stddev_steps",
        "actor_q_center_fraction",
    }
)


def _validate_seedmine_source_metadata(
    metadata: object, expected: dict[str, object]
) -> None:
    """Validate deterministic teacher compatibility, excluding run schedules.

    Seed-mine archives contain deterministic observations and executed actions.
    Critic/anchor schedule origins and exploration-noise schedules do not alter
    those records, and a deliberate new-branch reset must not invalidate
    teachers captured from the exact parent checkpoint.  All environment,
    reward, observation, action-policy, and prefix fields remain pinned.
    """

    if not isinstance(metadata, dict):
        raise ValueError("30-wide seed-mine source is missing Stage C v2 metadata")
    behavioral_expected = {
        key: value
        for key, value in expected.items()
        if key not in _SEEDMINE_SCHEDULE_KEYS
    }
    _validate_resume_metadata(metadata, behavioral_expected)


def _route_efficiency_metadata(args) -> dict[str, object]:
    if not bool(args.route_efficiency_revision):
        return {}
    return {
        "reward_revision": ROUTE_EFFICIENCY_REVISION,
        "refresh_ramp_side_on_dump": bool(args.refresh_ramp_side_on_dump),
        "ramp_side_deadband_x": float(args.ramp_side_deadband_x),
        "outer_rail_enter_x": float(args.outer_rail_enter_x),
        "outer_rail_exit_x": float(args.outer_rail_exit_x),
        "outer_rail_max_x": float(args.outer_rail_max_x),
        "outer_rail_grace_steps": int(args.outer_rail_grace_steps),
        "outer_rail_penalty_per_step": float(
            args.outer_rail_penalty_per_step
        ),
        "outer_rail_penalty_cap": float(args.outer_rail_penalty_cap),
        "outer_rail_min_scale": float(args.outer_rail_min_scale),
        "outer_rail_escalation_steps": int(
            args.outer_rail_escalation_steps
        ),
        "outer_rail_max_multiplier": float(
            args.outer_rail_max_multiplier
        ),
        "intake_substeps": int(args.intake_substeps),
        "require_ramp_out": bool(args.require_ramp_out),
        "ramp_out_half_width": float(args.ramp_out_half_width),
        "ramp_out_bonus": float(args.ramp_out_bonus),
        "off_ramp_exit_penalty": float(args.off_ramp_exit_penalty),
        "postdump_require_target_load": bool(
            args.postdump_require_target_load
        ),
        "postdump_complete_cycle": bool(args.postdump_complete_cycle),
        "postdump_depleted_count": int(args.postdump_depleted_count),
        "postdump_depleted_prob": float(args.postdump_depleted_prob),
        "preferred_repeat_load": int(args.preferred_repeat_load),
        "collect_stall_steps": int(args.collect_stall_steps),
        "return_time_guard": float(args.return_time_guard),
        "intake_during_return": bool(args.intake_during_return),
        "repeat_load_return_bonus": float(args.repeat_load_return_bonus),
        "repeat_load_score_bonus": float(args.repeat_load_score_bonus),
    }


def _validate_route_efficiency_resume(
    metadata: object,
    expected: dict[str, object],
    *,
    allow_legacy_missing: bool,
    allow_revision_migration: bool = False,
    extra_relaxed_keys: frozenset[str] = frozenset(),
) -> bool:
    """Validate the opt-in reward revision; return true for one-time migration.

    ``extra_relaxed_keys`` (stage_d_v1 wave-4c) removes explicitly opted-in
    keys from the comparison for THIS resume only -- the mechanism behind
    ``--allow-collect-stall-migration``.  The new value is stored in the
    checkpoint and later resumes validate it strictly again.
    """

    if not expected:
        return False
    if extra_relaxed_keys:
        expected = {
            key: value
            for key, value in expected.items()
            if key not in extra_relaxed_keys
        }
    if not isinstance(metadata, dict):
        raise ValueError("30-wide resume is missing Stage C v2 metadata")
    actual_revision = metadata.get("reward_revision")
    expected_revision = expected.get("reward_revision")
    if (
        actual_revision is not None
        and actual_revision != expected_revision
    ):
        if not (
            allow_revision_migration
            and (actual_revision, expected_revision)
            in ROUTE_EFFICIENCY_MIGRATIONS
        ):
            raise ValueError(
                "route-efficiency reward revision mismatch: "
                f"{actual_revision!r} != {expected_revision!r}"
            )
        if actual_revision == "outer_rail_v1":
            required_parent_keys = ROUTE_EFFICIENCY_V1_KEYS
        elif actual_revision == "outer_rail_v4_ramp_out":
            required_parent_keys = ROUTE_EFFICIENCY_V4_KEYS
        elif actual_revision == "cycle_efficiency_v5":
            required_parent_keys = ROUTE_EFFICIENCY_V5_KEYS
        elif actual_revision == "cycle_bridge_v6":
            required_parent_keys = ROUTE_EFFICIENCY_V6_KEYS
        elif actual_revision == "score_efficiency_v8":
            required_parent_keys = ROUTE_EFFICIENCY_V8_KEYS
        elif actual_revision == "score_efficiency_v9":
            required_parent_keys = ROUTE_EFFICIENCY_V9_KEYS
        elif actual_revision in (RAMPFREE_REVISION,) + STAGE_D_REVISIONS:
            # v11_rampfree and the Stage D revisions all carry the full
            # current key set (v9 keys + intake_during_return).
            required_parent_keys = ROUTE_EFFICIENCY_KEYS
        else:
            required_parent_keys = ROUTE_EFFICIENCY_V3_KEYS
        missing_parent = [
            key for key in required_parent_keys if key not in metadata
        ]
        if missing_parent:
            raise ValueError(
                f"resume has an incomplete {actual_revision} reward contract: "
                f"{missing_parent}"
            )
        if actual_revision in (
            "outer_rail_v4_ramp_out",
            "cycle_efficiency_v5",
            "score_efficiency_v8",
            "score_efficiency_v9",
            RAMPFREE_REVISION,
        ) + STAGE_D_REVISIONS:
            intentionally_changed = (
                {"collect_stall_steps", "return_time_guard"}
                if (
                    actual_revision == "score_efficiency_v9"
                    and expected_revision
                    == "score_efficiency_v10_return_intake"
                )
                else set()
            )
            if expected_revision == RAMPFREE_REVISION:
                # Ramp-free retires the ramp-out route contract inherited from
                # the V4 champion; relax ONLY these four keys.  Every other
                # parent-contract key is still enforced exactly as before.
                intentionally_changed = (
                    intentionally_changed | RAMPFREE_RELAXED_KEYS
                )
            if expected_revision in STAGE_D_REVISIONS:
                # Stage D retunes only the clock-fraction key for the 160 s
                # horizon (see STAGE_D_RELAXED_KEYS); everything else in the
                # parent contract is still enforced exactly as before.
                intentionally_changed = (
                    intentionally_changed | STAGE_D_RELAXED_KEYS
                )
            shared_expected = {
                key: expected[key]
                for key in required_parent_keys
                if (
                    key != "reward_revision"
                    and key not in intentionally_changed
                    and key in expected
                )
            }
            _validate_resume_metadata(metadata, shared_expected)
        return True

    present = [key for key in expected if key in metadata]
    if not present:
        if allow_legacy_missing:
            return True
        raise ValueError(
            "resume predates the route-efficiency reward contract; pass "
            "--allow-route-efficiency-revision-from-legacy only for the "
            "intentional first branch"
        )
    if len(present) != len(expected):
        missing = sorted(set(expected) - set(present))
        raise ValueError(
            f"resume has a partial route-efficiency reward contract: {missing}"
        )
    # 2026-07-27: postdump_require_target_load selects the postdump lane's
    # SUCCESS MILESTONE (termination), not a reward magnitude; it is already in
    # STAGE_D_RELAXED_KEYS for migrations, so exempt it here for same-revision
    # resumes too -- otherwise the drill gate can never be changed mid-lineage.
    expected = {k: v for k, v in expected.items() if k != "postdump_require_target_load"}
    _validate_resume_metadata(metadata, expected)
    return False


def _allow_target_load_mismatch(
    metadata: object,
    wanted: int,
    *,
    route_revision_migrated: bool,
    explicitly_allowed: bool,
) -> bool:
    """Authorize exactly one target-load change alongside a route revision.

    Changing this constant alters an observed feature and the COLLECT->RETURN
    transition, so a generic resume flag must never silently permit it.
    """

    if not isinstance(metadata, dict):
        raise ValueError("30-wide resume is missing Stage C v2 metadata")
    try:
        actual = int(metadata.get("target_load"))
    except (TypeError, ValueError) as exc:
        raise ValueError("resume target_load is missing or invalid") from exc
    if explicitly_allowed and not route_revision_migrated:
        raise ValueError(
            "--allow-target-load-migration requires an active supported "
            "route-efficiency revision migration"
        )
    if actual == int(wanted):
        return False
    if route_revision_migrated and explicitly_allowed:
        return True
    raise ValueError(
        f"v2 resume metadata mismatch for target_load: {actual!r} != {int(wanted)!r}"
    )


def _allow_suffix_alpha_mismatch(
    metadata: object,
    wanted: float,
    *,
    route_revision_migrated: bool,
    explicitly_allowed: bool,
) -> bool:
    """Authorize one conservative actor-objective change on a new revision."""

    if not isinstance(metadata, dict):
        raise ValueError("30-wide resume is missing Stage C v2 metadata")
    try:
        actual = float(metadata.get("suffix_alpha"))
    except (TypeError, ValueError) as exc:
        raise ValueError("resume suffix_alpha is missing or invalid") from exc
    if explicitly_allowed and not route_revision_migrated:
        raise ValueError(
            "--allow-suffix-alpha-migration requires an active supported "
            "route-efficiency revision migration"
        )
    if abs(actual - float(wanted)) <= 1e-9:
        return False
    if route_revision_migrated and explicitly_allowed:
        return True
    raise ValueError(
        f"v2 resume metadata mismatch for suffix_alpha: "
        f"{actual!r} != {float(wanted)!r}"
    )


def _allow_actor_q_center_fraction_mismatch(
    metadata: object,
    wanted: float,
    *,
    explicitly_allowed: bool,
) -> bool:
    """Gate one actor-Q objective migration; missing legacy metadata means 0."""

    if not isinstance(metadata, dict):
        raise ValueError("30-wide resume is missing Stage C v2 metadata")
    missing = "actor_q_center_fraction" not in metadata
    try:
        actual = float(metadata.get("actor_q_center_fraction", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "resume actor_q_center_fraction is missing or invalid"
        ) from exc
    if not np.isfinite(actual) or not 0.0 <= actual <= 1.0:
        raise ValueError("resume actor_q_center_fraction is invalid")
    if abs(actual - float(wanted)) <= 1e-9:
        # Strict metadata validation cannot compare a missing legacy key even
        # though its defined compatibility value is zero.
        return missing
    if explicitly_allowed:
        return True
    raise ValueError(
        f"v2 resume metadata mismatch for actor_q_center_fraction: "
        f"{actual!r} != {float(wanted)!r}"
    )


def _elite_tier(stats: object) -> str | None:
    """Protect multi-cycle completions above cycle-2 and return near-successes."""

    if not isinstance(stats, dict):
        return None
    route_gate_active = _ROUTE_GATE_ENABLED and "ramp_out_attempts" in stats
    reset_mode = _episode_reset_mode(stats)
    clean_ramp_cycles: int | None = None
    if route_gate_active and reset_mode == "full":
        try:
            if (
                int(stats.get("ramp_out_attempts", 0)) > 0
                and int(stats.get("ramp_out_successes", 0)) <= 0
            ):
                return None
            # Aggregate successes can incorrectly let a later clean departure
            # certify an earlier off-ramp cycle 2.  Custody for the first
            # repeat cycle must reject that trajectory outright.
            if int(stats.get("cycle2_off_ramp_outs", 0)) > 0:
                return None
        except (TypeError, ValueError):
            return None
    try:
        cycles_completed = int(stats.get("cycles_completed", 0))
        clean_ramp_cycles = int(stats.get("ramp_out_successes", 0))
        if (
            route_gate_active
            and reset_mode == "full"
            and clean_ramp_cycles < cycles_completed
        ):
            cycles_completed = clean_ramp_cycles
        if cycles_completed >= 2:
            return "multi_cycle"
        if cycles_completed >= 1:
            return "cycle"
    except (TypeError, ValueError):
        pass
    milestones = stats.get("milestones", {})
    if isinstance(milestones, dict):
        try:
            cycle_scored = int(milestones.get("cycle_scored", 0))
            if (
                route_gate_active
                and reset_mode == "full"
                and clean_ramp_cycles is not None
            ):
                cycle_scored = min(cycle_scored, clean_ramp_cycles)
            if cycle_scored >= 2:
                return "multi_cycle"
            if cycle_scored >= 1:
                return "cycle"
            if int(milestones.get("returned_home", 0)) >= 1:
                return "return"
        except (TypeError, ValueError):
            pass
    return None


def _episode_reset_mode(stats: object) -> str:
    """Return the declared reset source, including seed-miner ``mode``."""

    if not isinstance(stats, dict):
        return ""
    for key in ("reset_mode", "stream_mode", "mode"):
        value = stats.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _exact_pool_quotas(
    total: int, weights: dict[str, int]
) -> dict[str, int]:
    """Allocate an integer total by largest remainder without renormalizing."""

    total = int(total)
    if total < 0:
        raise ValueError("elite pool quota total must be non-negative")
    keys = tuple(weights)
    if not keys or any(int(weights[key]) < 0 for key in keys):
        raise ValueError("elite pool weights must be non-negative and non-empty")
    weight_total = sum(int(weights[key]) for key in keys)
    if weight_total <= 0:
        raise ValueError("elite pool weights must have positive mass")
    raw = {
        key: float(total) * float(int(weights[key])) / float(weight_total)
        for key in keys
    }
    quotas = {key: int(np.floor(raw[key])) for key in keys}
    remaining = total - sum(quotas.values())
    order = sorted(
        keys,
        key=lambda key: (
            raw[key] - quotas[key],
            int(weights[key]),
            -keys.index(key),
        ),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _elite_classification(episode: CapturedEpisode) -> tuple[str, str] | None:
    """Return independent outcome tier and source-owned replay pool.

    The stream group is trusted only when terminal statistics declare the same
    reset mode.  POSTDUMP is intentionally stricter than FULL: it must finish
    the entire ramp-out, target-load, return, and score chain before it can
    enter protected replay.
    """

    group = str(episode.group)
    if group not in ("full", "postdump", "return"):
        raise ValueError(f"elite capture group is not supported: {group!r}")
    mode = _episode_reset_mode(episode.stats)
    if not mode:
        raise ValueError("elite capture statistics are missing reset mode")
    if mode != group:
        raise ValueError(
            f"elite capture group/reset mode mismatch: {group!r} != {mode!r}"
        )

    if group == "postdump":
        stats = episode.stats
        milestones = stats.get("milestones", {})
        if not isinstance(milestones, dict):
            return None
        try:
            qualifies = (
                stats.get("terminal_reason") == "skill_success"
                and int(stats.get("cycles_completed", 0)) >= 1
                and int(stats.get("ramp_out_successes", 0)) >= 1
                and int(milestones.get("target_load", 0)) >= 1
                and int(milestones.get("returned_home", 0)) >= 1
                and int(milestones.get("cycle_scored", 0)) >= 1
            )
        except (TypeError, ValueError):
            return None
        return ("cycle", "postdump_cycle") if qualifies else None

    tier = _elite_tier(episode.stats)
    if group == "full":
        pool = {
            "multi_cycle": "full_multi",
            "cycle": "full_cycle",
            "return": "full_return",
        }.get(tier)
        return (tier, pool) if tier is not None and pool is not None else None
    if tier in ("cycle", "return"):
        return tier, "return"
    return None


def _prepare_elite_episode_record(
    episode: CapturedEpisode,
) -> tuple[str, str, dict[str, np.ndarray]] | None:
    """Validate an elite episode and retain its trainable suffix and source."""

    classification = _elite_classification(episode)
    if classification is None:
        return None
    tier, pool = classification
    if any(key not in episode.arrays for key in ELITE_FIELDS):
        raise ValueError("captured elite episode is missing replay fields")
    arrays = {key: np.asarray(episode.arrays[key]) for key in ELITE_FIELDS}
    lengths = {int(value.shape[0]) for value in arrays.values() if value.ndim >= 1}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("captured elite episode fields have inconsistent lengths")
    if arrays["proprio"].ndim != 2 or arrays["proprio"].shape[1] != V2_PROPRIO_DIM:
        raise ValueError("captured elite proprio is not Stage C v2 width 30")
    if not bool(np.asarray(arrays["done"], dtype=bool)[-1]):
        raise ValueError("captured elite episode is missing its terminal boundary")
    for key, value in arrays.items():
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"captured elite episode contains non-finite {key}")

    # FULL capture includes the immutable champion opening.  Keep the first
    # non-FIRST row onward so elite replay can only consolidate trainable
    # suffix behavior.  POSTDUMP and RETURN naturally begin in the suffix.
    suffix = np.equal(arrays["proprio"][:, FIRST_PHASE_INDEX], 0.0)
    indices = np.flatnonzero(suffix)
    if not len(indices):
        raise ValueError("elite episode contains no trainable suffix rows")
    start = int(indices[0])
    if not bool(suffix[start:].all()):
        raise ValueError("elite episode re-entered FIRST after suffix takeover")
    prepared = {key: value[start:].copy() for key, value in arrays.items()}
    return tier, pool, prepared


def _prepare_elite_episode(
    episode: CapturedEpisode,
) -> tuple[str, dict[str, np.ndarray]] | None:
    """Compatibility wrapper returning outcome tier plus prepared arrays."""

    prepared = _prepare_elite_episode_record(episode)
    if prepared is None:
        return None
    tier, _, arrays = prepared
    return tier, arrays


def _elite_contract_matches(actual: object, expected: dict[str, object]) -> None:
    if not isinstance(actual, dict):
        raise ValueError("elite archive is missing its contract")
    for key, wanted in expected.items():
        if actual.get(key) != wanted:
            raise ValueError(
                f"elite archive contract mismatch for {key}: "
                f"{actual.get(key)!r} != {wanted!r}"
            )


def _archive_elite_episode(
    directory: Path,
    episode: CapturedEpisode,
    contract: dict[str, object],
) -> tuple[Path, str, str, dict[str, np.ndarray]] | None:
    prepared = _prepare_elite_episode_record(episode)
    if prepared is None:
        return None
    tier, pool, arrays = prepared
    stats = dict(episode.stats)
    collector = int(stats.get("collector", -1))
    env_index = int(stats.get("env_index", -1))
    episode_seq = int(stats.get("episode_seq", -1))
    policy_steps = int(stats.get("policy_train_steps", -1))
    archive_meta = {
        "schema": ELITE_ARCHIVE_SCHEMA,
        "contract": dict(contract),
        "outcome_tier": tier,
        "pool": pool,
        "stream_index": int(episode.stream_index),
        "source_group": str(episode.group),
        "stats": stats,
    }
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"elite_{pool}_{tier}_c{collector}_e{env_index}_n{episode_seq}_"
        f"p{policy_steps}_{time.time_ns()}.npz"
    )
    path = directory / name
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = dict(arrays)
    payload["metadata"] = np.frombuffer(
        json.dumps(archive_meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return path, tier, pool, arrays


def _elite_archive_pool(path: Path) -> str | None:
    """Read only the source-pool identity needed for safe pruning."""

    try:
        with np.load(path, allow_pickle=False) as data:
            if "metadata" not in data.files:
                return None
            meta = json.loads(bytes(data["metadata"]).decode("utf-8"))
    except (OSError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or meta.get("schema") != ELITE_ARCHIVE_SCHEMA:
        return None
    pool = str(meta.get("pool", ""))
    return pool if pool in ELITE_POOLS else None


def _prune_elite_archives(directory: Path, max_files: int) -> list[Path]:
    """Prune oldest v7 archives independently inside fixed source quotas.

    Unrecognized or legacy files are deliberately left untouched.  This makes
    the new retention pass non-destructive if an operator points a fresh v7 run
    at a directory containing older archives.
    """

    max_files = int(max_files)
    if max_files <= 0:
        raise ValueError("elite archive max files must be positive")
    quotas = _exact_pool_quotas(max_files, ELITE_ARCHIVE_POOL_WEIGHTS)
    by_pool: dict[str, list[Path]] = {pool: [] for pool in ELITE_POOLS}
    for path in directory.glob("elite_*.npz"):
        pool = _elite_archive_pool(path)
        if pool is not None:
            by_pool[pool].append(path)
    removed: list[Path] = []
    for pool in ELITE_POOLS:
        archives = sorted(
            by_pool[pool],
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        pool_removed = archives[: max(0, len(archives) - quotas[pool])]
        for path in pool_removed:
            path.unlink()
        removed.extend(pool_removed)
    return removed


def _load_elite_archive(
    path: Path, expected_contract: dict[str, object]
) -> tuple[str, dict[str, np.ndarray]]:
    tier, _, prepared_arrays, _ = _load_elite_archive_record(
        path, expected_contract
    )
    return tier, prepared_arrays


def _load_elite_archive_record(
    path: Path, expected_contract: dict[str, object]
) -> tuple[str, str, dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise ValueError("elite archive is missing metadata")
        meta = json.loads(bytes(data["metadata"]).decode("utf-8"))
        arrays = {key: data[key].copy() for key in ELITE_FIELDS if key in data.files}
    if not isinstance(meta, dict) or meta.get("schema") != ELITE_ARCHIVE_SCHEMA:
        raise ValueError("elite archive has the wrong source-aware schema")
    _elite_contract_matches(meta.get("contract"), expected_contract)
    captured = CapturedEpisode(
        stream_index=int(meta.get("stream_index", -1)),
        group=str(meta.get("source_group", "")),
        arrays=arrays,
        stats=dict(meta.get("stats", {})),
    )
    prepared = _prepare_elite_episode_record(captured)
    if prepared is None:
        raise ValueError("elite archive metadata no longer qualifies")
    tier, pool, prepared_arrays = prepared
    if tier != meta.get("outcome_tier"):
        raise ValueError("elite archive tier disagrees with terminal statistics")
    if pool != meta.get("pool"):
        raise ValueError("elite archive pool disagrees with source statistics")
    return tier, pool, prepared_arrays, str(meta.get("source_group", ""))


def _load_seedmine_archive_record(
    path: Path,
    expected_contract: dict[str, object],
    source_checkpoint_sha256: str,
) -> tuple[str, str, dict[str, np.ndarray]]:
    """Validate and adapt one standalone seed-miner episode archive.

    Seed mining is a separate, read-only process and intentionally uses its own
    schema.  This bridge requires an explicit source checkpoint hash, validates
    every array and provenance field, then passes the episode through the same
    suffix-only preparation used by live elite capture.
    """

    with np.load(path, allow_pickle=False) as data:
        expected_files = {*ELITE_FIELDS, "metadata"}
        if set(data.files) != expected_files:
            raise ValueError(
                f"seed-mine archive fields {sorted(data.files)} != "
                f"{sorted(expected_files)}"
            )
        try:
            meta = json.loads(bytes(data["metadata"]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("seed-mine archive metadata is not valid JSON") from exc
        arrays = {key: data[key].copy() for key in ELITE_FIELDS}

    if not isinstance(meta, dict) or meta.get("schema") != SEEDMINE_CAPTURE_SCHEMA:
        raise ValueError("seed-mine archive has the wrong capture schema")
    if meta.get("field_keys") != list(ELITE_FIELDS):
        raise ValueError("seed-mine archive field order does not match replay")
    episode = meta.get("episode")
    if not isinstance(episode, dict) or episode.get("schema") != SEEDMINE_EVAL_SCHEMA:
        raise ValueError("seed-mine archive has the wrong evaluation schema")

    try:
        length = int(meta.get("length", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed-mine archive length is invalid") from exc
    if length <= 0 or any(value.ndim < 1 or value.shape[0] != length for value in arrays.values()):
        raise ValueError("seed-mine archive arrays disagree with metadata length")

    declared_fields = meta.get("fields")
    if not isinstance(declared_fields, dict) or set(declared_fields) != set(ELITE_FIELDS):
        raise ValueError("seed-mine archive field declarations are incomplete")
    for key, value in arrays.items():
        declaration = declared_fields.get(key)
        if not isinstance(declaration, dict):
            raise ValueError(f"seed-mine archive declaration for {key} is invalid")
        if declaration.get("shape") != list(value.shape):
            raise ValueError(f"seed-mine archive declared shape mismatch for {key}")
        if declaration.get("dtype") != str(value.dtype):
            raise ValueError(f"seed-mine archive declared dtype mismatch for {key}")

    expected_shapes = {
        "obs": (length, *tuple(int(v) for v in expected_contract["obs_shape"])),
        "proprio": (length, int(expected_contract["proprio_dim"])),
        "privileged": (length, int(expected_contract["privileged_dim"])),
        "action": (length, int(expected_contract["action_dim"])),
        "reward": (length,),
        "done": (length,),
    }
    expected_dtypes = {
        "obs": np.dtype(np.uint8),
        "proprio": np.dtype(np.float32),
        "privileged": np.dtype(np.float32),
        "action": np.dtype(np.float32),
        "reward": np.dtype(np.float32),
        "done": np.dtype(bool),
    }
    for key, value in arrays.items():
        if value.shape != expected_shapes[key]:
            raise ValueError(
                f"seed-mine archive {key} shape {value.shape} != {expected_shapes[key]}"
            )
        if value.dtype != expected_dtypes[key]:
            raise ValueError(
                f"seed-mine archive {key} dtype {value.dtype} != {expected_dtypes[key]}"
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"seed-mine archive contains non-finite {key}")
    if not bool(arrays["done"][-1]) or bool(arrays["done"][:-1].any()):
        raise ValueError("seed-mine archive must terminate exactly once on its final row")

    source_sha = str(source_checkpoint_sha256).lower()
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("seed-mine source checkpoint SHA-256 is invalid")
    if str(episode.get("checkpoint_sha256", "")).lower() != source_sha:
        raise ValueError("seed-mine archive came from a different candidate checkpoint")
    if str(episode.get("prefix_sha256", "")).lower() != str(
        expected_contract["prefix_sha256"]
    ).lower():
        raise ValueError("seed-mine archive came from a different frozen prefix")
    try:
        episode_steps = int(episode.get("episode_steps", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed-mine episode_steps is invalid") from exc
    if episode_steps != length:
        raise ValueError("seed-mine episode_steps disagrees with captured length")
    if str(episode.get("action_mode", "")) != "deterministic":
        raise ValueError(
            "seed-mine actor custody requires deterministic action_mode"
        )
    if episode.get("mode") not in ("full", "return"):
        raise ValueError("seed-mine archive reset mode is not supported")
    if episode.get("mode") == "full" and "full_episode_s" in expected_contract:
        try:
            episode_len_s = float(episode.get("episode_len_s"))
            expected_episode_len_s = float(expected_contract["full_episode_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError("seed-mine episode horizon is missing or invalid") from exc
        if abs(episode_len_s - expected_episode_len_s) > 1e-9:
            raise ValueError(
                "seed-mine full episode horizon mismatch: "
                f"{episode_len_s} != {expected_episode_len_s}"
            )

    stage_meta = episode.get("stagec_v2_metadata")
    if not isinstance(stage_meta, dict):
        raise ValueError("seed-mine archive is missing Stage C metadata")
    provenance = {
        "schema_version": expected_contract["schema_version"],
        "prefix_sha256": expected_contract["prefix_sha256"],
        "action_policy": expected_contract["action_policy"],
        "field_strategy": expected_contract["field_strategy"],
        "proprio_dim": expected_contract["proprio_dim"],
    }
    for key, wanted in provenance.items():
        actual = stage_meta.get(key)
        if key == "prefix_sha256":
            actual = str(actual).lower()
            wanted = str(wanted).lower()
        if actual != wanted:
            raise ValueError(
                f"seed-mine Stage C metadata mismatch for {key}: {actual!r} != {wanted!r}"
            )

    declared_tier = meta.get("capture_tier")
    episode_tier = episode.get("capture_tier")
    if declared_tier != episode_tier or declared_tier not in ("cycle", "returned_home"):
        raise ValueError("seed-mine capture tier is missing or inconsistent")
    success_end: int | None = None
    if declared_tier == "cycle":
        raw_success_steps = episode.get("cycle_success_steps")
        if not isinstance(raw_success_steps, list) or not raw_success_steps:
            raise ValueError(
                "seed-mine cycle capture is missing exact cycle success steps"
            )
        try:
            success_steps = [int(value) for value in raw_success_steps]
        except (TypeError, ValueError) as exc:
            raise ValueError("seed-mine cycle success steps are invalid") from exc
        if (
            success_steps != sorted(set(success_steps))
            or success_steps[0] < 0
            or success_steps[-1] >= length
        ):
            raise ValueError("seed-mine cycle success steps are out of range")
        # Discard any later unsuccessful attempt/tail before critic replay or
        # actor custody. The state snapshot that first reports CYCLE_SCORED is
        # the final admitted transition.
        success_end = int(success_steps[-1]) + 1
    try:
        env_index = int(episode.get("env_index", -1))
        num_envs = int(episode.get("num_envs", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed-mine environment provenance is invalid") from exc
    if not (1 <= num_envs <= 2 and 0 <= env_index < num_envs):
        raise ValueError("seed-mine environment provenance is out of range")
    captured = CapturedEpisode(
        stream_index=env_index,
        group=str(episode["mode"]),
        arrays=(
            {
                **{
                    key: value[:success_end].copy()
                    for key, value in arrays.items()
                    if key != "done"
                },
                "done": np.concatenate(
                    (
                        np.zeros(max(0, success_end - 1), dtype=bool),
                        np.ones(1, dtype=bool),
                    )
                ),
            }
            if success_end is not None
            else arrays
        ),
        stats=episode,
    )
    prepared = _prepare_elite_episode_record(captured)
    if prepared is None:
        raise ValueError("seed-mine episode does not contain an elite milestone")
    tier, pool, prepared_arrays = prepared
    wanted_tiers = (
        {"cycle", "multi_cycle"} if declared_tier == "cycle" else {"return"}
    )
    if tier not in wanted_tiers:
        raise ValueError("seed-mine capture tier disagrees with episode milestones")
    return tier, pool, prepared_arrays


def _load_seedmine_archive(
    path: Path,
    expected_contract: dict[str, object],
    source_checkpoint_sha256: str,
) -> tuple[str, dict[str, np.ndarray]]:
    """Compatibility wrapper returning seed-mine outcome tier plus arrays."""

    tier, _, arrays = _load_seedmine_archive_record(
        path, expected_contract, source_checkpoint_sha256
    )
    return tier, arrays


def _add_episode_to_ring(ring: ReplayRing, arrays: dict[str, np.ndarray]) -> None:
    for row in range(int(arrays["reward"].shape[0])):
        ring.add(*(arrays[key][row] for key in ELITE_FIELDS))


def _add_seedmine_episode_to_replay(
    rings: dict[str, ReplayRing],
    pool: str,
    arrays: dict[str, np.ndarray],
    *,
    behavior_only: bool,
) -> bool:
    """Route a validated seed-mine episode to critic replay when permitted."""

    if bool(behavior_only):
        return False
    _add_episode_to_ring(rings[pool], arrays)
    return True


def _concat_batches(parts: list[Batch], rng: np.random.Generator | None = None) -> Batch:
    if not parts:
        raise ValueError("cannot concatenate an empty batch list")
    values = {
        name: np.concatenate([getattr(part, name) for part in parts], axis=0)
        for name in Batch.__dataclass_fields__
    }
    if rng is not None and len(values["reward"]) > 1:
        order = rng.permutation(len(values["reward"]))
        values = {name: value[order] for name, value in values.items()}
    return Batch(**values)


def _sample_elite(
    rings: dict[str, ReplayRing],
    batch_size: int,
    rng: np.random.Generator,
) -> Batch | None:
    unknown = set(rings).difference(ELITE_POOLS)
    if unknown:
        raise ValueError(f"unknown elite replay pools: {sorted(unknown)}")
    available = {
        pool: ring
        for pool, ring in rings.items()
        if len(ring) > ring.n_step + 1
    }
    if batch_size <= 0 or not available:
        return None
    counts = _exact_pool_quotas(int(batch_size), ELITE_POOL_WEIGHTS)
    parts = [
        available[pool].sample(counts[pool])
        for pool in ELITE_POOLS
        if pool in available and counts[pool] > 0
    ]
    return _concat_batches(parts, rng) if parts else None


def _validate_seedmine_options(
    seedmine_dir: Path | None,
    source_checkpoint: Path | None,
    *,
    behavior_only: bool,
) -> None:
    """Fail closed when seed-mine custody lacks its exact source contract."""

    if (seedmine_dir is None) != (source_checkpoint is None):
        raise ValueError(
            "--seedmine-elite-dir and --seedmine-source-checkpoint must be used together"
        )
    if bool(behavior_only) and seedmine_dir is None:
        raise ValueError(
            "--seedmine-behavior-only requires --seedmine-elite-dir and "
            "--seedmine-source-checkpoint"
        )


def _elite_replay_rings_for_update(
    *,
    critic_only: bool,
    elite_replays: dict[str, ReplayRing],
    seedmine_warmup_replays: dict[str, ReplayRing],
) -> dict[str, ReplayRing]:
    """Expose only provenance-validated seed-mine replay during warm-up."""

    return seedmine_warmup_replays if bool(critic_only) else elite_replays


def _elite_behavior_eligible(pool: str) -> bool:
    """RETURN curricula may train value estimates, never the actor anchor."""

    return pool in ("full_multi", "full_cycle", "postdump_cycle")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/dev/shm/frc_stagec_v2")
    ap.add_argument("--num-collectors", type=int, default=3)
    ap.add_argument("--collector-envs", type=int, default=2)
    ap.add_argument("--stream-groups", default="full,full,postdump,collect,return,return")
    ap.add_argument(
        "--group-weights", default="full=.25,postdump=.25,collect=.25,return=.25"
    )
    ap.add_argument("--resume", required=True)
    ap.add_argument("--camera-rig-revision", required=True)
    ap.add_argument("--template-sha256", required=True)
    ap.add_argument(
        "--train-encoder",
        action="store_true",
        help="adapt the visual encoder to an explicitly migrated camera contract",
    )
    ap.add_argument(
        "--allow-camera-rig-migration",
        action="store_true",
        help=(
            "allow one fresh-replay branch from a checkpoint whose camera "
            "contract predates --camera-rig-revision/--template-sha256"
        ),
    )
    ap.add_argument(
        "--camera-rig-parent-revision",
        default="unversioned_front_center",
        help="honest label for the parent checkpoint's legacy camera contract",
    )
    ap.add_argument(
        "--camera-rig-parent-template-sha256",
        default="",
        help="SHA-256 of the parent checkpoint's environment template",
    )
    ap.add_argument(
        "--prefix-checkpoint",
        type=Path,
        required=True,
        help="immutable champion checkpoint whose first-cycle policy is protected",
    )
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=600.0)
    ap.add_argument(
        "--full-episode-s",
        type=float,
        default=120.0,
        help="required horizon provenance for FULL seed-mine captures",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--updates-per-tx", type=float, default=1.0)
    ap.add_argument("--replay-capacity", type=int, default=400_000)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--seed-transitions", type=int, default=3_000)
    ap.add_argument("--min-live-fraction", type=float, default=0.5)
    ap.add_argument("--max-updates-per-tick", type=int, default=100)
    ap.add_argument("--weight-publish-updates", type=int, default=400)
    ap.add_argument(
        "--freeze-collector-weights",
        action="store_true",
        help=(
            "publish the validated resume once at startup, but never publish "
            "learned candidates during or at the end of this run; candidates "
            "must be promoted by an external deterministic gate"
        ),
    )
    ap.add_argument("--eval-snapshot-updates", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--stddev-start", type=float, default=1.0)
    ap.add_argument("--stddev-end", type=float, default=0.30)
    ap.add_argument("--stddev-steps", type=int, default=150_000)
    ap.add_argument("--initial-stddev", type=float, default=0.50)
    ap.add_argument(
        "--reset-schedules-on-resume",
        action="store_true",
        help=(
            "start critic warm-up, exploration, anchor decay, and elite "
            "consolidation from zero while retaining lifetime checkpoint updates"
        ),
    )
    ap.add_argument("--critic-only-updates", type=int, default=5_000)
    ap.add_argument(
        "--actor-update-interval",
        type=int,
        default=1,
        help="apply one actor step for every N suffix critic steps after warm-up",
    )
    ap.add_argument(
        "--actor-q-center-fraction",
        type=float,
        default=0.0,
        help=(
            "blend deterministic-center Q into the suffix actor objective; "
            "0 preserves the historical noisy-Q objective"
        ),
    )
    ap.add_argument(
        "--actor-phases",
        default="leave,collect,return,score",
        help="comma-separated suffix phases allowed to receive actor-Q gradients",
    )
    ap.add_argument(
        "--reset-optimizer-state-on-resume",
        action="store_true",
        help="discard inherited Adam moments without changing model tensors",
    )
    ap.add_argument("--anchor-beta-start", type=float, default=0.30)
    ap.add_argument("--anchor-beta-floor", type=float, default=0.08)
    ap.add_argument("--anchor-decay-updates", type=int, default=60_000)
    ap.add_argument("--anchor-batch", type=int, default=128)
    ap.add_argument("--suffix-alpha", type=float, default=1.0)
    ap.add_argument("--target-load", type=int, default=15)
    ap.add_argument("--reserve-count", type=int, default=18)
    ap.add_argument("--reserve-batches", type=int, default=3)
    ap.add_argument("--max-dump-ticks", type=int, default=180)
    ap.add_argument("--cycle-score-fraction", type=float, default=0.75)
    ap.add_argument("--cycle-score-floor", type=int, default=6)
    ap.add_argument("--collect-weight", type=float, default=0.3)
    ap.add_argument("--progress-per-m", type=float, default=5.0)
    ap.add_argument("--progress-step-cap", type=float, default=0.75)
    ap.add_argument("--ramp-bonus", type=float, default=6.0)
    ap.add_argument("--route-efficiency-revision", action="store_true")
    ap.add_argument("--refresh-ramp-side-on-dump", action="store_true")
    ap.add_argument("--ramp-side-deadband-x", type=float, default=0.25)
    ap.add_argument("--require-ramp-out", action="store_true")
    ap.add_argument("--ramp-out-half-width", type=float, default=0.90)
    ap.add_argument("--ramp-out-bonus", type=float, default=0.0)
    ap.add_argument("--off-ramp-exit-penalty", type=float, default=0.0)
    ap.add_argument("--postdump-require-target-load", action="store_true")
    ap.add_argument("--postdump-complete-cycle", action="store_true")
    ap.add_argument("--postdump-depleted-count", type=int, default=0)
    ap.add_argument("--postdump-depleted-prob", type=float, default=0.0)
    ap.add_argument("--preferred-repeat-load", type=int, default=0)
    ap.add_argument("--collect-stall-steps", type=int, default=0)
    ap.add_argument("--return-time-guard", type=float, default=0.0)
    ap.add_argument("--intake-during-return", action="store_true")
    ap.add_argument("--stage-d-ferry", action="store_true")  # STAGE-D1B: learner torch-mask must match collector
    # STAGE-D1C: accepted for wrapper symmetry only.  The own-court short loop is
    # env-side (vec_env reward machine + cycle_v2 phase transition); its effect
    # reaches the learner entirely through collected rewards and the SCORE phase
    # feature already stored in the replay proprio, so apply_executed_action_policy
    # (which reads phase from that proprio) stays in sync WITHOUT this flag.  No
    # learner action-mask change is required.
    ap.add_argument("--stage-d-owncourt-loop", action="store_true")  # STAGE-D1C: no-op (env-side)
    ap.add_argument("--stage-d-owncourt-min-balls", type=int, default=2)  # STAGE-D1C: no-op (env-side)
    ap.add_argument("--repeat-load-return-bonus", type=float, default=0.0)
    ap.add_argument("--repeat-load-score-bonus", type=float, default=0.0)
    ap.add_argument("--outer-rail-enter-x", type=float, default=2.85)
    ap.add_argument("--outer-rail-exit-x", type=float, default=2.55)
    ap.add_argument("--outer-rail-max-x", type=float, default=3.60)
    ap.add_argument("--outer-rail-grace-steps", type=int, default=20)
    ap.add_argument("--outer-rail-penalty-per-step", type=float, default=0.0)
    ap.add_argument("--outer-rail-penalty-cap", type=float, default=8.0)
    ap.add_argument("--outer-rail-min-scale", type=float, default=0.0)
    ap.add_argument("--outer-rail-escalation-steps", type=int, default=0)
    ap.add_argument("--outer-rail-max-multiplier", type=float, default=1.0)
    ap.add_argument("--intake-substeps", type=int, default=1)
    ap.add_argument(
        "--allow-route-efficiency-revision-from-legacy",
        action="store_true",
        help=(
            "allow exactly one intentional branch from a v2.3 checkpoint that "
            "predates outer-rail metadata; subsequent resumes validate strictly"
        ),
    )
    ap.add_argument(
        "--allow-route-efficiency-revision-migration",
        action="store_true",
        help=(
            "allow one intentional supported outer-rail mechanics migration; "
            "subsequent resumes validate the new contract strictly"
        ),
    )
    ap.add_argument(
        "--allow-target-load-migration",
        action="store_true",
        help=(
            "allow target_load to change only while performing an explicit "
            "supported route-efficiency revision migration"
        ),
    )
    ap.add_argument(
        "--allow-suffix-alpha-migration",
        action="store_true",
        help=(
            "allow suffix_alpha to change only while performing an explicit "
            "supported reward-revision migration"
        ),
    )
    ap.add_argument(
        "--allow-actor-q-center-migration",
        action="store_true",
        help=(
            "allow one explicit actor_q_center_fraction change while resuming; "
            "this does not relax reward or environment metadata"
        ),
    )
    ap.add_argument(
        "--allow-collect-stall-migration",
        action="store_true",
        help=(
            "allow collect_stall_steps to differ from the resume checkpoint "
            "on a SAME-revision resume; explicit opt-in for the wave-4c "
            "collection-depth retune (45 -> 90)."
        ),
    )
    ap.add_argument(
        "--allow-delay-penalty-migration",
        action="store_true",
        help=(
            "allow the leave/return phase-delay penalty knobs to differ from "
            "the resume checkpoint on a SAME-revision resume; explicit opt-in "
            "for an intentional anti-camping retune (stage_d_v1 wave-2). The "
            "new values are stored in the checkpoint and later resumes "
            "validate them strictly again."
        ),
    )
    ap.add_argument(
        "--allow-stddev-schedule-migration",
        action="store_true",
        help=(
            "allow the exploration stddev schedule (stddev_start/end/steps) to "
            "differ from the resume checkpoint on a SAME-revision resume; explicit "
            "opt-in for an intentional exploration-floor change (ceiling_v14). The "
            "reward contract and observation space are unaffected."
        ),
    )
    ap.add_argument("--leave-grace-steps", type=int, default=5)
    ap.add_argument("--leave-penalty-per-step", type=float, default=0.03)
    ap.add_argument("--leave-penalty-cap", type=float, default=5.0)
    ap.add_argument("--return-grace-steps", type=int, default=10)
    ap.add_argument("--return-penalty-per-step", type=float, default=0.02)
    ap.add_argument("--return-penalty-cap", type=float, default=5.0)
    ap.add_argument("--shoot-grace-s", type=float, default=2.0)
    ap.add_argument("--shoot-penalty-per-step", type=float, default=0.05)
    ap.add_argument("--shoot-penalty-cap", type=float, default=5.0)
    ap.add_argument("--dump-lost-aim-grace-ticks", type=int, default=15)
    ap.add_argument("--partial-dump-penalty-per-ball", type=float, default=0.5)
    ap.add_argument("--partial-dump-penalty-cap", type=float, default=15.0)
    ap.add_argument(
        "--elite-dir",
        type=Path,
        default=None,
        help="opt-in directory for continuity-checked cycle/return episode archives",
    )
    ap.add_argument(
        "--elite-replay-fraction",
        type=float,
        default=0.0,
        help="fraction of each batch drawn from archived elite suffix episodes",
    )
    ap.add_argument("--elite-replay-capacity", type=int, default=40_000)
    ap.add_argument("--elite-consolidation-updates", type=int, default=10_000)
    ap.add_argument(
        "--elite-archive-max-files",
        type=int,
        default=128,
        help="maximum recent successful episode archives retained in the active run",
    )
    ap.add_argument(
        "--elite-behavior-weight",
        type=float,
        default=0.0,
        help="initial direct actor behavior-cloning weight on successful cycle suffixes",
    )
    ap.add_argument(
        "--elite-behavior-weight-end",
        type=float,
        default=None,
        help="final behavior-cloning weight after post-warm-up annealing",
    )
    ap.add_argument(
        "--elite-behavior-decay-updates",
        type=int,
        default=0,
        help="post-warm-up learner updates used to anneal behavior-cloning weight",
    )
    ap.add_argument("--elite-behavior-batch-size", type=int, default=32)
    ap.add_argument("--elite-behavior-score-capacity", type=int, default=1800)
    ap.add_argument("--elite-behavior-trigger-capacity", type=int, default=200)
    ap.add_argument("--elite-behavior-trigger-fraction", type=float, default=0.25)
    ap.add_argument(
        "--elite-behavior-window-balanced",
        action="store_true",
        help=(
            "sample equal actor-custody rows from opener/live1/live2/endgame "
            "while preserving the configured trigger fraction"
        ),
    )
    ap.add_argument(
        "--elite-behavior-seedmine-only",
        action="store_true",
        help=(
            "pin actor behavior custody to validated deterministic seed-mine "
            "episodes; live successes remain critic replay only"
        ),
    )
    ap.add_argument(
        "--seedmine-elite-dir",
        type=Path,
        default=None,
        help="opt-in directory of validated stagec_training_episode_v1 captures",
    )
    ap.add_argument(
        "--seedmine-source-checkpoint",
        type=Path,
        default=None,
        help="exact candidate checkpoint used to create --seedmine-elite-dir",
    )
    ap.add_argument(
        "--seedmine-behavior-only",
        action="store_true",
        help=(
            "use validated seed-mine episodes only for actor behavior custody; "
            "never insert them into critic elite replay"
        ),
    )
    args = ap.parse_args()
    if str(args.camera_rig_revision) != CAMERA_RIG_REVISION:
        raise ValueError(
            "camera rig revision does not match this code tree: "
            f"{args.camera_rig_revision!r} != {CAMERA_RIG_REVISION!r}"
        )
    if len(str(args.template_sha256)) != 64 or any(
        char not in "0123456789abcdef" for char in str(args.template_sha256)
    ):
        raise ValueError("--template-sha256 must be a lowercase SHA-256 digest")
    if bool(args.allow_camera_rig_migration):
        if not bool(args.train_encoder):
            raise ValueError("camera-rig migration requires --train-encoder")
        if not bool(args.reset_schedules_on_resume):
            raise ValueError(
                "camera-rig migration requires --reset-schedules-on-resume"
            )
        if not bool(args.reset_optimizer_state_on_resume):
            raise ValueError(
                "camera-rig migration requires --reset-optimizer-state-on-resume"
            )
        if len(str(args.camera_rig_parent_template_sha256)) != 64 or any(
            char not in "0123456789abcdef"
            for char in str(args.camera_rig_parent_template_sha256)
        ):
            raise ValueError(
                "camera-rig migration requires a lowercase parent template SHA-256"
            )
    global _ROUTE_GATE_ENABLED
    _ROUTE_GATE_ENABLED = bool(args.require_ramp_out)

    if not np.isfinite(args.suffix_alpha) or args.suffix_alpha <= 0.0:
        raise ValueError("--suffix-alpha must be finite and positive")
    if (
        not np.isfinite(args.actor_q_center_fraction)
        or not 0.0 <= float(args.actor_q_center_fraction) <= 1.0
    ):
        raise ValueError("--actor-q-center-fraction must be finite and in [0, 1]")
    if int(args.actor_update_interval) <= 0:
        raise ValueError("--actor-update-interval must be positive")
    if not np.isfinite(args.full_episode_s) or float(args.full_episode_s) <= 0.0:
        raise ValueError("--full-episode-s must be finite and positive")
    actor_phases = _parse_actor_phases(args.actor_phases)
    if not (0.0 < float(args.cycle_score_fraction) <= 1.0):
        raise ValueError("--cycle-score-fraction must be in (0, 1]")
    if int(args.cycle_score_floor) < 1:
        raise ValueError("--cycle-score-floor must be positive")
    if args.route_efficiency_revision:
        if not args.refresh_ramp_side_on_dump:
            raise ValueError(
                "--route-efficiency-revision requires "
                "--refresh-ramp-side-on-dump"
            )
        if float(args.outer_rail_penalty_per_step) <= 0.0:
            raise ValueError(
                "--route-efficiency-revision requires a positive "
                "--outer-rail-penalty-per-step"
            )
        if args.require_ramp_out and float(args.ramp_out_bonus) <= 0.0:
            raise ValueError(
                "--require-ramp-out requires a positive --ramp-out-bonus"
            )
        if args.postdump_complete_cycle and (
            not args.postdump_require_target_load or not args.require_ramp_out
        ):
            raise ValueError(
                "--postdump-complete-cycle requires target-load and ramp-out gates"
            )
        if int(args.postdump_depleted_count) < 0:
            raise ValueError("--postdump-depleted-count cannot be negative")
        if not (0.0 <= float(args.postdump_depleted_prob) <= 1.0):
            raise ValueError("--postdump-depleted-prob must be in [0, 1]")
        if int(args.preferred_repeat_load) and not (
            int(args.target_load) < int(args.preferred_repeat_load) <= 60
        ):
            raise ValueError(
                "--preferred-repeat-load must exceed --target-load and be <= 60"
            )
        if (
            ROUTE_EFFICIENCY_REVISION
            in ("score_efficiency_v9", RAMPFREE_REVISION) + STAGE_D_REVISIONS
            and int(args.preferred_repeat_load)
            and (
                int(args.collect_stall_steps) <= 0
                and float(args.return_time_guard) <= 0.0
            )
        ):
            raise ValueError(
                "--preferred-repeat-load requires --collect-stall-steps or "
                "--return-time-guard"
            )
        if (
            ROUTE_EFFICIENCY_REVISION == "score_efficiency_v10_return_intake"
            and not bool(args.intake_during_return)
        ):
            raise ValueError(
                "score_efficiency_v10_return_intake requires "
                "--intake-during-return"
            )
        if (
            float(args.repeat_load_return_bonus) > 0.0
            or float(args.repeat_load_score_bonus) > 0.0
        ) and not int(args.preferred_repeat_load):
            raise ValueError(
                "repeat-load bonuses require --preferred-repeat-load"
            )
    elif (
        args.refresh_ramp_side_on_dump
        or args.require_ramp_out
        or float(args.ramp_out_bonus) != 0.0
        or float(args.off_ramp_exit_penalty) != 0.0
        or args.postdump_require_target_load
        or args.postdump_complete_cycle
        or int(args.postdump_depleted_count) != 0
        or float(args.postdump_depleted_prob) != 0.0
        or int(args.preferred_repeat_load) != 0
        or int(args.collect_stall_steps) != 0
        or float(args.return_time_guard) != 0.0
        or args.intake_during_return
        or float(args.repeat_load_return_bonus) != 0.0
        or float(args.repeat_load_score_bonus) != 0.0
        or float(args.outer_rail_penalty_per_step) != 0.0
        or args.allow_route_efficiency_revision_from_legacy
        or args.allow_route_efficiency_revision_migration
        or args.allow_target_load_migration
        or args.allow_suffix_alpha_migration
        or int(args.intake_substeps) != 1
    ):
        raise ValueError(
            "route-efficiency settings require --route-efficiency-revision"
        )
    if not (
        0.0
        <= float(args.outer_rail_exit_x)
        < float(args.outer_rail_enter_x)
        < float(args.outer_rail_max_x)
    ):
        raise ValueError(
            "outer-rail geometry must satisfy 0 <= exit < enter < max"
        )
    for name in (
        "leave_grace_steps",
        "return_grace_steps",
        "dump_lost_aim_grace_ticks",
        "outer_rail_grace_steps",
        "outer_rail_escalation_steps",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not np.isfinite(args.shoot_grace_s) or float(args.shoot_grace_s) < 0.0:
        raise ValueError("--shoot-grace-s must be finite and non-negative")
    for name in (
        "collect_weight",
        "progress_per_m",
        "progress_step_cap",
        "ramp_bonus",
        "leave_penalty_per_step",
        "leave_penalty_cap",
        "return_penalty_per_step",
        "return_penalty_cap",
        "shoot_penalty_per_step",
        "shoot_penalty_cap",
        "partial_dump_penalty_per_ball",
        "partial_dump_penalty_cap",
        "ramp_side_deadband_x",
        "ramp_out_bonus",
        "off_ramp_exit_penalty",
        "repeat_load_return_bonus",
        "repeat_load_score_bonus",
        "outer_rail_penalty_per_step",
        "outer_rail_penalty_cap",
        "outer_rail_min_scale",
        "outer_rail_max_multiplier",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not 0.0 <= float(args.outer_rail_min_scale) <= 1.0:
        raise ValueError("--outer-rail-min-scale must be in [0, 1]")
    if float(args.outer_rail_max_multiplier) < 1.0:
        raise ValueError("--outer-rail-max-multiplier must be at least 1")
    if (
        not np.isfinite(args.ramp_out_half_width)
        or float(args.ramp_out_half_width) <= 0.0
    ):
        raise ValueError("--ramp-out-half-width must be finite and positive")
    if not 1 <= int(args.intake_substeps) <= 3:
        raise ValueError("--intake-substeps must be in [1, 3]")
    if int(args.collect_stall_steps) < 0:
        raise ValueError("--collect-stall-steps cannot be negative")
    if not 0.0 <= float(args.return_time_guard) <= 1.0:
        raise ValueError("--return-time-guard must be in [0, 1]")
    if not np.isfinite(args.elite_replay_fraction) or not (
        0.0 <= float(args.elite_replay_fraction) < 0.5
    ):
        raise ValueError("--elite-replay-fraction must be in [0, 0.5)")
    if int(args.elite_replay_capacity) < 100:
        raise ValueError("--elite-replay-capacity must be at least 100")
    if int(args.elite_consolidation_updates) < 0:
        raise ValueError("--elite-consolidation-updates must be non-negative")
    if int(args.elite_archive_max_files) <= 0:
        raise ValueError("--elite-archive-max-files must be positive")
    if not np.isfinite(args.elite_behavior_weight) or float(args.elite_behavior_weight) < 0.0:
        raise ValueError("--elite-behavior-weight must be finite and non-negative")
    if args.elite_behavior_weight_end is None:
        args.elite_behavior_weight_end = float(args.elite_behavior_weight)
    if (
        not np.isfinite(args.elite_behavior_weight_end)
        or float(args.elite_behavior_weight_end) < 0.0
    ):
        raise ValueError(
            "--elite-behavior-weight-end must be finite and non-negative"
        )
    if float(args.elite_behavior_weight_end) > float(args.elite_behavior_weight):
        raise ValueError(
            "--elite-behavior-weight-end cannot exceed --elite-behavior-weight"
        )
    if int(args.elite_behavior_decay_updates) < 0:
        raise ValueError("--elite-behavior-decay-updates must be non-negative")
    if (
        float(args.elite_behavior_weight_end) != float(args.elite_behavior_weight)
        and int(args.elite_behavior_decay_updates) <= 0
    ):
        raise ValueError(
            "changing elite behavior weight requires positive decay updates"
        )
    if int(args.elite_behavior_batch_size) <= 0:
        raise ValueError("--elite-behavior-batch-size must be positive")
    if int(args.elite_behavior_score_capacity) <= 0:
        raise ValueError("--elite-behavior-score-capacity must be positive")
    if int(args.elite_behavior_trigger_capacity) <= 0:
        raise ValueError("--elite-behavior-trigger-capacity must be positive")
    if not 0.0 <= float(args.elite_behavior_trigger_fraction) <= 1.0:
        raise ValueError("--elite-behavior-trigger-fraction must be in [0, 1]")
    _validate_seedmine_options(
        args.seedmine_elite_dir,
        args.seedmine_source_checkpoint,
        behavior_only=bool(args.seedmine_behavior_only),
    )
    if (
        bool(args.elite_behavior_seedmine_only)
        and float(args.elite_behavior_weight) > 0.0
        and args.seedmine_elite_dir is None
    ):
        raise ValueError(
            "--elite-behavior-seedmine-only with positive behavior weight "
            "requires deterministic seed-mine custody"
        )
    if (
        bool(args.route_efficiency_revision)
        and float(args.elite_behavior_weight) > 0.0
        and not bool(args.elite_behavior_seedmine_only)
    ):
        raise ValueError(
            f"{ROUTE_EFFICIENCY_REVISION} actor custody must be seed-mine-only"
        )

    import torch

    from frc_rebuilt.rl import distributed as D
    from frc_rebuilt.rl.checkpoint_v2 import load_legacy_checkpoint_into_v2
    from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
    from frc_rebuilt.rl.prefix_takeover import (
        AnchorSampler,
        FROZEN_HOLDOUT_EPISODES,
    )
    from frc_rebuilt.rl.replay_v2 import GroupedPerEnvReplay, UniformChunkIngestor

    args.out.mkdir(parents=True, exist_ok=True)
    eval_queue = args.out / "eval_queue"
    eval_queue.mkdir(parents=True, exist_ok=True)
    streams = int(args.num_collectors) * int(args.collector_envs)
    stream_groups = _parse_groups(args.stream_groups, streams)
    group_weights = _parse_group_weights(args.group_weights)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prefix_sha256 = _sha256_file(args.prefix_checkpoint)

    cfg = DrQConfig(
        proprio_dim=V2_PROPRIO_DIM,
        privileged_dim=26,
        lr=args.learning_rate,
        stddev_start=args.stddev_start,
        stddev_end=args.stddev_end,
        stddev_steps=args.stddev_steps,
    )
    agent = DrQV2Agent(cfg)
    try:
        resume_payload = torch.load(args.resume, map_location=agent.device, weights_only=True)
    except TypeError:
        resume_payload = torch.load(args.resume, map_location=agent.device)
    resume_cols = int(resume_payload["actor"]["trunk.0.weight"].shape[1])
    new_cols = int(agent.feat_dim + V2_PROPRIO_DIM)
    legacy_cols = int(agent.feat_dim + LEGACY_PROPRIO_DIM)
    route_efficiency_expected = _route_efficiency_metadata(args)
    route_efficiency_migrated = False
    camera_rig_migrated = False
    camera_rig_parent_checkpoint_sha256: str | None = None
    camera_rig_parent_revision: str | None = None
    camera_rig_parent_template_sha256: str | None = None
    reward_revision_parent_sha256: str | None = None
    if resume_cols == new_cols:
        old_meta = resume_payload.get("stagec_v2")
        expected_resume = {
            "schema_version": SCHEMA_VERSION,
            "proprio_dim": V2_PROPRIO_DIM,
            "dump_on_press": True,
            "prefix_sha256": prefix_sha256,
            "action_policy": ACTION_POLICY,
            "field_strategy": FIELD_STRATEGY,
            "return_skill_preload": RETURN_SKILL_PRELOAD,
            "suffix_alpha": float(args.suffix_alpha),
            "actor_q_center_fraction": float(args.actor_q_center_fraction),
            "encoder_frozen": not bool(args.train_encoder),
            "camera_rig_revision": str(args.camera_rig_revision),
            "template_sha256": str(args.template_sha256),
            "stddev_start": float(args.stddev_start),
            "stddev_end": float(args.stddev_end),
            "stddev_steps": int(args.stddev_steps),
            "target_load": int(args.target_load),
            "reserve_count": int(args.reserve_count),
            "reserve_batches": int(args.reserve_batches),
            "max_dump_ticks": int(args.max_dump_ticks),
            "cycle_score_fraction": float(args.cycle_score_fraction),
            "cycle_score_floor": int(args.cycle_score_floor),
            "collect_weight": float(args.collect_weight),
            "progress_per_m": float(args.progress_per_m),
            "progress_step_cap": float(args.progress_step_cap),
            "ramp_bonus": float(args.ramp_bonus),
            "leave_grace_steps": int(args.leave_grace_steps),
            "leave_penalty_per_step": float(args.leave_penalty_per_step),
            "leave_penalty_cap": float(args.leave_penalty_cap),
            "return_grace_steps": int(args.return_grace_steps),
            "return_penalty_per_step": float(args.return_penalty_per_step),
            "return_penalty_cap": float(args.return_penalty_cap),
            "shoot_grace_s": float(args.shoot_grace_s),
            "shoot_penalty_per_step": float(args.shoot_penalty_per_step),
            "shoot_penalty_cap": float(args.shoot_penalty_cap),
            "dump_lost_aim_grace_ticks": int(args.dump_lost_aim_grace_ticks),
            "partial_dump_penalty_per_ball": float(
                args.partial_dump_penalty_per_ball
            ),
            "partial_dump_penalty_cap": float(args.partial_dump_penalty_cap),
        }
        route_efficiency_migrated = _validate_route_efficiency_resume(
            old_meta,
            route_efficiency_expected,
            allow_legacy_missing=bool(
                args.allow_route_efficiency_revision_from_legacy
            ),
            allow_revision_migration=bool(
                args.allow_route_efficiency_revision_migration
            ),
            extra_relaxed_keys=(
                frozenset({"collect_stall_steps"})
                if bool(args.allow_collect_stall_migration)
                else frozenset()
            ),
        )
        if (
            route_efficiency_migrated
            and ROUTE_EFFICIENCY_REVISION
            in (
                "score_efficiency_v9",
                "score_efficiency_v10_return_intake",
                RAMPFREE_REVISION,
            )
            + STAGE_D_REVISIONS
            and not bool(args.reset_optimizer_state_on_resume)
        ):
            raise ValueError(
                f"{ROUTE_EFFICIENCY_REVISION} migration requires "
                "--reset-optimizer-state-on-resume"
            )
        if _allow_target_load_mismatch(
            old_meta,
            int(args.target_load),
            route_revision_migrated=route_efficiency_migrated,
            explicitly_allowed=bool(args.allow_target_load_migration),
        ):
            expected_resume.pop("target_load")
        if _allow_suffix_alpha_mismatch(
            old_meta,
            float(args.suffix_alpha),
            route_revision_migrated=route_efficiency_migrated,
            explicitly_allowed=bool(args.allow_suffix_alpha_migration),
        ):
            expected_resume.pop("suffix_alpha")
        if _allow_actor_q_center_fraction_mismatch(
            old_meta,
            float(args.actor_q_center_fraction),
            explicitly_allowed=bool(args.allow_actor_q_center_migration),
        ):
            expected_resume.pop("actor_q_center_fraction")
        if bool(args.allow_stddev_schedule_migration):
            # Exploration-schedule knobs are not part of the reward contract or
            # observation space; an explicit opt-in relaxes only these three keys
            # so a same-revision resume can raise the sustained stddev floor.
            for _stddev_key in ("stddev_start", "stddev_end", "stddev_steps"):
                expected_resume.pop(_stddev_key, None)
        if route_efficiency_migrated or bool(args.allow_delay_penalty_migration):
            # stage_d_v1 ferry-first: the one-time route-revision migration also
            # retunes the phase-delay economics (LEAVE camping must never be
            # cheaper than an imperfect crossing; see STAGE_D_RELAXED_KEYS).
            # These base-contract knobs are relaxed ONLY across the explicit
            # migration launch (or the explicit --allow-delay-penalty-migration
            # opt-in for a same-revision anti-camping retune); the new
            # checkpoint stores the new values and every subsequent
            # same-revision resume validates them strictly.
            for _delay_key in (
                "leave_penalty_per_step",
                "leave_penalty_cap",
                "return_penalty_per_step",
                "return_penalty_cap",
            ):
                expected_resume.pop(_delay_key, None)
        if bool(args.allow_camera_rig_migration):
            for _camera_key in (
                "encoder_frozen",
                "camera_rig_revision",
                "template_sha256",
            ):
                expected_resume.pop(_camera_key, None)
            camera_rig_migrated = True
            camera_rig_parent_checkpoint_sha256 = _sha256_file(args.resume)
            camera_rig_parent_revision = str(args.camera_rig_parent_revision)
            camera_rig_parent_template_sha256 = str(
                args.camera_rig_parent_template_sha256
            )
        _validate_resume_metadata(old_meta, expected_resume)
        if not camera_rig_migrated:
            camera_rig_migrated = bool(old_meta.get("camera_rig_migrated", False))
            camera_rig_parent_checkpoint_sha256 = old_meta.get(
                "camera_rig_parent_checkpoint_sha256"
            )
            camera_rig_parent_revision = old_meta.get("camera_rig_parent_revision")
            camera_rig_parent_template_sha256 = old_meta.get(
                "camera_rig_parent_template_sha256"
            )
        if route_efficiency_expected:
            if route_efficiency_migrated:
                reward_revision_parent_sha256 = _sha256_file(args.resume)
            else:
                reward_revision_parent_sha256 = str(
                    old_meta.get("reward_revision_parent_sha256")
                    or _sha256_file(args.resume)
                )
        agent.load(args.resume)
        expanded_legacy = False
        run_updates = int(resume_payload.get("v2_updates", 0))
        schedule_origin = _schedule_origin_updates(
            run_updates, old_meta, reset=bool(args.reset_schedules_on_resume)
        )
        elite_updates = (
            0
            if args.reset_schedules_on_resume
            else int(resume_payload.get("elite_updates", 0))
        )
        if args.reset_schedules_on_resume:
            agent.explore_offset = _initial_explore_offset(
                agent.train_steps,
                stddev_start=args.stddev_start,
                stddev_end=args.stddev_end,
                stddev_steps=args.stddev_steps,
                initial_stddev=args.initial_stddev,
            )
        restored_actor_updates = (
            0
            if args.reset_schedules_on_resume
            else int(resume_payload.get("actor_updates", 0))
        )
    elif resume_cols == legacy_cols:
        resume_sha256 = _sha256_file(args.resume)
        if resume_sha256 != prefix_sha256:
            raise ValueError(
                "legacy --resume must exactly match --prefix-checkpoint: "
                f"{resume_sha256} != {prefix_sha256}"
            )
        load_legacy_checkpoint_into_v2(agent, resume_payload)
        expanded_legacy = True
        run_updates = 0
        schedule_origin = 0
        elite_updates = 0
        restored_actor_updates = 0
        if route_efficiency_expected:
            reward_revision_parent_sha256 = resume_sha256
        agent.explore_offset = _initial_explore_offset(
            agent.train_steps,
            stddev_start=args.stddev_start,
            stddev_end=args.stddev_end,
            stddev_steps=args.stddev_steps,
            initial_stddev=args.initial_stddev,
        )
    else:
        raise ValueError(
            f"resume actor has {resume_cols} inputs; expected legacy {legacy_cols} or v2 {new_cols}"
        )
    if bool(args.reset_optimizer_state_on_resume):
        agent.reset_optimizers(float(args.learning_rate))
    else:
        for optimizer in (agent.encoder_opt, agent.actor_opt, agent.critic_opt):
            for group in optimizer.param_groups:
                group["lr"] = float(args.learning_rate)

    sampler = AnchorSampler(args.anchor_dir, FROZEN_HOLDOUT_EPISODES, seed=args.seed + 7)
    wdir = D.weights_dir(args.root)
    wdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "proprio_dim": V2_PROPRIO_DIM,
        "legacy_proprio_dim": LEGACY_PROPRIO_DIM,
        "proprio_feature_names": list(V2_FEATURE_NAMES),
        "dump_on_press": True,
        "prefix_sha256": prefix_sha256,
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "return_skill_preload": RETURN_SKILL_PRELOAD,
        "suffix_alpha": float(args.suffix_alpha),
        "actor_q_center_fraction": float(args.actor_q_center_fraction),
        "encoder_frozen": not bool(args.train_encoder),
        "camera_rig_revision": str(args.camera_rig_revision),
        "template_sha256": str(args.template_sha256),
        "camera_rig_migrated": bool(camera_rig_migrated),
        "camera_rig_parent_checkpoint_sha256": camera_rig_parent_checkpoint_sha256,
        "camera_rig_parent_revision": camera_rig_parent_revision,
        "camera_rig_parent_template_sha256": camera_rig_parent_template_sha256,
        "stddev_start": float(args.stddev_start),
        "stddev_end": float(args.stddev_end),
        "stddev_steps": int(args.stddev_steps),
        "target_load": int(args.target_load),
        "reserve_count": int(args.reserve_count),
        "reserve_batches": int(args.reserve_batches),
        "max_dump_ticks": int(args.max_dump_ticks),
        "cycle_score_fraction": float(args.cycle_score_fraction),
        "cycle_score_floor": int(args.cycle_score_floor),
        "collect_weight": float(args.collect_weight),
        "progress_per_m": float(args.progress_per_m),
        "progress_step_cap": float(args.progress_step_cap),
        "ramp_bonus": float(args.ramp_bonus),
        "leave_grace_steps": int(args.leave_grace_steps),
        "leave_penalty_per_step": float(args.leave_penalty_per_step),
        "leave_penalty_cap": float(args.leave_penalty_cap),
        "return_grace_steps": int(args.return_grace_steps),
        "return_penalty_per_step": float(args.return_penalty_per_step),
        "return_penalty_cap": float(args.return_penalty_cap),
        "shoot_grace_s": float(args.shoot_grace_s),
        "shoot_penalty_per_step": float(args.shoot_penalty_per_step),
        "shoot_penalty_cap": float(args.shoot_penalty_cap),
        "dump_lost_aim_grace_ticks": int(args.dump_lost_aim_grace_ticks),
        "partial_dump_penalty_per_ball": float(args.partial_dump_penalty_per_ball),
        "partial_dump_penalty_cap": float(args.partial_dump_penalty_cap),
        "schedule_origin_updates": int(schedule_origin),
    }
    metadata.update(route_efficiency_expected)
    if route_efficiency_expected:
        metadata["reward_revision_parent_sha256"] = str(
            reward_revision_parent_sha256
        )

    def publish() -> None:
        if not agent.weights_finite():
            raise RuntimeError("refusing to publish non-finite Stage C v2 weights")
        D.publish_weights(
            wdir,
            {
                "encoder": agent.encoder.state_dict(),
                "actor": agent.actor.state_dict(),
                "train_steps": int(agent.train_steps),
                "explore_offset": int(agent.explore_offset),
                "stagec_v2": metadata,
            },
            int(agent.train_steps),
        )

    def checkpoint_payload() -> dict:
        return {
            "encoder": agent.encoder.state_dict(),
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
            "encoder_opt": agent.encoder_opt.state_dict(),
            "actor_opt": agent.actor_opt.state_dict(),
            "critic_opt": agent.critic_opt.state_dict(),
            "train_steps": int(agent.train_steps),
            "skipped_updates": int(agent.skipped_updates),
            "explore_offset": int(agent.explore_offset),
            "v2_updates": int(run_updates),
            "elite_updates": int(elite_updates),
            "actor_updates": int(actor_updates),
            "stagec_v2": metadata,
        }

    def save_atomic(path: Path) -> None:
        if not agent.weights_finite():
            raise RuntimeError("refusing to save non-finite Stage C v2 checkpoint")
        # BUGFIX 2026-07-25: recreate the parent directory before writing.  The
        # rolling disk janitor can remove an eval_queue/ between snapshots (it
        # did: 11 consecutive "Parent directory does not exist" crashes in
        # stage_blue2_20260724_191059), which killed the learner every snapshot
        # interval.  Snapshot saving must never depend on housekeeping order.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot create checkpoint directory {path.parent}: {exc}")
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(checkpoint_payload(), tmp)
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    replay = GroupedPerEnvReplay(
        stream_groups=stream_groups,
        group_weights=group_weights,
        capacity_per_env=max(1000, int(args.replay_capacity) // streams),
        seed=args.seed + 5,
        obs_shape=(cfg.frame_channels, cfg.frame_h, cfg.frame_w),
        proprio_dim=cfg.proprio_dim,
        privileged_dim=cfg.privileged_dim,
        action_dim=cfg.action_dim,
        n_step=args.n_step,
        gamma=args.gamma,
    )
    seedmine_enabled = args.seedmine_elite_dir is not None
    seedmine_replay_enabled = (
        seedmine_enabled and not bool(args.seedmine_behavior_only)
    )
    elite_replay_enabled = (
        args.elite_dir is not None
        or float(args.elite_replay_fraction) > 0.0
        or seedmine_replay_enabled
    )
    elite_enabled = elite_replay_enabled or seedmine_enabled
    elite_dir = args.elite_dir or (args.out / "elite_episodes")
    ingestor = UniformChunkIngestor(
        replay,
        collector_envs=args.collector_envs,
        capture_groups=_elite_capture_groups(
            stream_groups, enabled=elite_replay_enabled
        ),
    )
    elite_contract = {
        "schema_version": SCHEMA_VERSION,
        "prefix_sha256": prefix_sha256,
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "proprio_dim": int(cfg.proprio_dim),
        "privileged_dim": int(cfg.privileged_dim),
        "action_dim": int(cfg.action_dim),
        "obs_shape": [int(cfg.frame_channels), int(cfg.frame_h), int(cfg.frame_w)],
        "n_step": int(args.n_step),
        "gamma": float(args.gamma),
        "full_episode_s": float(args.full_episode_s),
        "camera_rig_revision": str(args.camera_rig_revision),
        "template_sha256": str(args.template_sha256),
        "encoder_frozen": not bool(args.train_encoder),
    }
    elite_contract.update(route_efficiency_expected)
    seedmine_source_sha256: str | None = None
    if seedmine_enabled:
        if not args.seedmine_elite_dir.is_dir():
            raise ValueError(
                f"seed-mine elite directory does not exist: {args.seedmine_elite_dir}"
            )
        if not args.seedmine_source_checkpoint.is_file():
            raise ValueError(
                "seed-mine source checkpoint does not exist: "
                f"{args.seedmine_source_checkpoint}"
            )
        seedmine_source_sha256 = _sha256_file(args.seedmine_source_checkpoint)
        try:
            seedmine_source = torch.load(
                args.seedmine_source_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            seedmine_source = torch.load(
                args.seedmine_source_checkpoint, map_location="cpu"
            )
        source_cols = int(seedmine_source["actor"]["trunk.0.weight"].shape[1])
        if source_cols != new_cols:
            raise ValueError("seed-mine source checkpoint is not Stage C v2 width 30")
        _validate_seedmine_source_metadata(
            seedmine_source.get("stagec_v2"), metadata
        )
    elite_pool_capacities = _exact_pool_quotas(
        int(args.elite_replay_capacity), ELITE_POOL_WEIGHTS
    )
    ring_kwargs = {
        "obs_shape": tuple(elite_contract["obs_shape"]),
        "proprio_dim": int(cfg.proprio_dim),
        "privileged_dim": int(cfg.privileged_dim),
        "action_dim": int(cfg.action_dim),
        "n_step": int(args.n_step),
        "gamma": float(args.gamma),
    }
    elite_replays = (
        {
            pool: ReplayRing(
                elite_pool_capacities[pool],
                seed=args.seed + 10_991 + 13 * index,
                **ring_kwargs,
            )
            for index, pool in enumerate(ELITE_POOLS)
        }
        if elite_replay_enabled
        else {}
    )
    seedmine_warmup_replays: dict[str, ReplayRing] = {}
    elite_archive_counts = {pool: 0 for pool in ELITE_POOLS}
    seedmine_archive_counts = {pool: 0 for pool in ELITE_POOLS}
    elite_invalid_archives = 0
    seedmine_invalid_archives = 0
    elite_behavior = EliteScoreBehaviorPool(
        score_capacity=int(args.elite_behavior_score_capacity),
        trigger_capacity=int(args.elite_behavior_trigger_capacity),
        seed=args.seed + 11_031,
    )
    if elite_replay_enabled and elite_dir.exists():
        pruned = _prune_elite_archives(elite_dir, int(args.elite_archive_max_files))
        if pruned:
            print(
                f"ELITE_V2_PRUNE removed={len(pruned)} "
                f"quota_total={int(args.elite_archive_max_files)}",
                flush=True,
            )
        for archive in sorted(elite_dir.glob("elite_*.npz")):
            try:
                tier, pool, arrays, group = _load_elite_archive_record(
                    archive, elite_contract
                )
                _add_episode_to_ring(elite_replays[pool], arrays)
                if (
                    _elite_behavior_eligible(pool)
                    and not bool(args.elite_behavior_seedmine_only)
                ):
                    elite_behavior.add(arrays)
                elite_archive_counts[pool] += 1
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                elite_invalid_archives += 1
                print(f"ELITE_V2_REJECT path={archive} error={exc}", flush=True)
    if seedmine_enabled:
        seedmine_archives = [
            archive
            for archive in sorted(args.seedmine_elite_dir.glob("*.npz"))
            if not archive.name.startswith("elite_")
        ]
        if not seedmine_archives:
            raise ValueError(
                f"seed-mine elite directory contains no episode archives: "
                f"{args.seedmine_elite_dir}"
            )
        seedmine_records: list[tuple[str, dict[str, np.ndarray]]] = []
        for archive in seedmine_archives:
            try:
                tier, pool, arrays = _load_seedmine_archive_record(
                    archive, elite_contract, seedmine_source_sha256
                )
                if _elite_behavior_eligible(pool):
                    elite_behavior.add(arrays)
                seedmine_records.append((pool, arrays))
                seedmine_archive_counts[pool] += 1
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                seedmine_invalid_archives += 1
                print(f"SEEDMINE_ELITE_REJECT path={archive} error={exc}", flush=True)
        if seedmine_invalid_archives:
            raise ValueError(
                f"refusing seed-mine consolidation: {seedmine_invalid_archives} "
                "archive(s) failed validation"
            )
        if seedmine_replay_enabled:
            # The normal elite rings preserve existing post-warm-up behavior.
            # A compact, seed-mine-only shadow ring allows critic consolidation
            # during warm-up without leaking live exploratory elites into it.
            for pool, arrays in seedmine_records:
                _add_seedmine_episode_to_replay(
                    elite_replays,
                    pool,
                    arrays,
                    behavior_only=bool(args.seedmine_behavior_only),
                )
            if int(args.critic_only_updates) > 0:
                seedmine_rows = {
                    pool: sum(
                        int(len(arrays["reward"]))
                        for record_pool, arrays in seedmine_records
                        if record_pool == pool
                    )
                    for pool in ELITE_POOLS
                }
                seedmine_warmup_replays = {
                    pool: ReplayRing(
                        max(
                            int(args.n_step) + 2,
                            min(elite_pool_capacities[pool], seedmine_rows[pool]),
                        ),
                        seed=args.seed + 12_991 + 13 * index,
                        **ring_kwargs,
                    )
                    for index, pool in enumerate(ELITE_POOLS)
                    if seedmine_rows[pool] > 0
                }
                for pool, arrays in seedmine_records:
                    _add_episode_to_ring(seedmine_warmup_replays[pool], arrays)
        seedmine_records.clear()
    if bool(args.elite_behavior_window_balanced):
        elite_behavior_window_rows = elite_behavior.window_rows(
            float(args.full_episode_s)
        )
        missing_windows = [
            name
            for name, counts in elite_behavior_window_rows.items()
            if int(counts["total"]) == 0
        ]
        if missing_windows:
            raise ValueError(
                "window-balanced elite behavior lacks validated rows for "
                + ",".join(missing_windows)
            )
        print(
            "ELITE_BEHAVIOR_WINDOW_COVERAGE "
            + json.dumps(elite_behavior_window_rows, sort_keys=True),
            flush=True,
        )
    elite_rng = np.random.default_rng(args.seed + 11_029)
    # Publish only after every opt-in archive and provenance check has passed.
    # A bad consolidation directory must fail closed before collectors can load
    # weights from a learner that is about to exit.
    publish()
    print(
        f"LEARNER_V2_READY resume={args.resume} expanded_legacy={expanded_legacy} "
        f"route_revision={metadata.get('reward_revision', 'legacy')} "
        f"route_migrated={route_efficiency_migrated} "
        f"camera_rig={metadata['camera_rig_revision']} "
        f"camera_migrated={camera_rig_migrated} "
        f"encoder_frozen={metadata['encoder_frozen']} "
        f"intake_substeps={metadata.get('intake_substeps', 1)} "
        f"steps={agent.train_steps} updates={run_updates} "
        f"schedule_updates={run_updates - schedule_origin} "
        f"stddev={agent.stddev():.3f} "
        f"actor_q_center_fraction={float(args.actor_q_center_fraction):.3f} "
        f"groups={stream_groups}",
        flush=True,
    )
    run_started = time.time()
    config = {
        **{key: str(value) for key, value in vars(args).items()},
        "streams": streams,
        "stream_groups_parsed": list(stream_groups),
        "group_weights_parsed": group_weights,
        "expanded_legacy": expanded_legacy,
        "elite_contract": elite_contract,
        "elite_enabled": elite_enabled,
        "elite_replay_enabled": elite_replay_enabled,
        "elite_archive_schema": ELITE_ARCHIVE_SCHEMA,
        "elite_pool_capacities": (
            dict(elite_pool_capacities) if elite_replay_enabled else {}
        ),
        "elite_archive_pool_quotas": _exact_pool_quotas(
            int(args.elite_archive_max_files), ELITE_ARCHIVE_POOL_WEIGHTS
        ),
        "seedmine_source_sha256": seedmine_source_sha256,
        "started_at_unix": run_started,
        "stagec_v2": metadata,
    }
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2))

    consumed: set[str] = set()
    transitions = 0
    update_debt = 0.0
    elite_new_counts = {pool: 0 for pool in ELITE_POOLS}
    elite_rows_last = 0
    elite_behavior_rows_last = 0
    actor_updates = int(restored_actor_updates)
    episodes: list[dict] = []
    train_metrics: dict[str, float] = {}
    train_metric_interval: list[dict[str, float]] = []
    restarts_seen = rejected = 0
    last_report = time.time()
    deadline = time.time() + args.minutes * 60.0
    metrics_path = args.out / "metrics.jsonl"

    while time.time() < deadline:
        chunks = D.drain_chunks(args.root, args.num_collectors, consumed)
        new_tx = 0
        for chunk in chunks:
            result = ingestor.ingest(chunk)
            new_tx += int(result["added"])
            rejected += int(result["rejected"])
            restarts_seen += int(bool(result["restart_boundary"]))
            for captured in result.get("completed_episodes", ()):
                try:
                    archived = _archive_elite_episode(
                        elite_dir, captured, elite_contract
                    )
                    if archived is not None:
                        archive, tier, pool, arrays = archived
                        _add_episode_to_ring(elite_replays[pool], arrays)
                        if (
                            _elite_behavior_eligible(pool)
                            and not bool(args.elite_behavior_seedmine_only)
                        ):
                            elite_behavior.add(arrays)
                        elite_new_counts[pool] += 1
                        pruned = _prune_elite_archives(
                            elite_dir, int(args.elite_archive_max_files)
                        )
                        print(
                            f"ELITE_V2_CAPTURE group={captured.group} tier={tier} "
                            f"pool={pool} rows={len(arrays['reward'])} "
                            f"path={archive} pruned={len(pruned)}",
                            flush=True,
                        )
                except (OSError, ValueError, TypeError) as exc:
                    print(
                        f"ELITE_V2_REJECT stream={captured.stream_index} error={exc}",
                        flush=True,
                    )
            episodes.extend(chunk.episodes)
        transitions += new_tx

        if replay.ready(max(args.batch_size, args.seed_transitions), args.min_live_fraction):
            update_debt += float(args.updates_per_tx) * new_tx
            tick_updates = 0
            while update_debt >= 1.0 and tick_updates < int(args.max_updates_per_tick):
                schedule_updates = run_updates - schedule_origin
                critic_only = schedule_updates < int(args.critic_only_updates)
                post_warmup_updates = max(
                    0, schedule_updates - int(args.critic_only_updates)
                )
                actor_due = (
                    not critic_only
                    and post_warmup_updates % int(args.actor_update_interval) == 0
                )
                beta = _anchor_beta(
                    schedule_updates,
                    args.critic_only_updates,
                    args.anchor_beta_start,
                    args.anchor_beta_floor,
                    args.anchor_decay_updates,
                )
                anchor_obs, anchor_pro, anchor_act = sampler.sample(args.anchor_batch)
                anchor_pro = _expand_anchor_proprio(anchor_pro, args.target_load)
                elite_batch = None
                requested_elite = 0
                update_elite_replays = _elite_replay_rings_for_update(
                    critic_only=critic_only,
                    elite_replays=elite_replays,
                    seedmine_warmup_replays=seedmine_warmup_replays,
                )
                if (
                    float(args.elite_replay_fraction) > 0.0
                    and elite_updates < int(args.elite_consolidation_updates)
                ):
                    requested_elite = min(
                        int(args.batch_size) - 1,
                        int(round(int(args.batch_size) * float(args.elite_replay_fraction))),
                    )
                    elite_batch = _sample_elite(
                        update_elite_replays, requested_elite, elite_rng
                    )
                elite_rows_last = (
                    int(len(elite_batch.reward)) if elite_batch is not None else 0
                )
                batch = replay.sample_grouped(int(args.batch_size) - elite_rows_last)
                if elite_batch is not None:
                    batch = _concat_batches([batch, elite_batch], elite_rng)
                    elite_updates += 1
                # Warm-up and interleaved critic steps now use the exact Stage-C
                # legal-action transform and exclude FIRST rows.  The previous
                # update_finetune warm-up bootstrapped from illegal raw actions
                # and the suffix actor optimized that distorted critic.
                elite_behavior_weight = _elite_behavior_weight(
                    schedule_updates,
                    int(args.critic_only_updates),
                    float(args.elite_behavior_weight),
                    float(args.elite_behavior_weight_end),
                    int(args.elite_behavior_decay_updates),
                )
                elite_behavior_batch = None
                if actor_due and elite_behavior_weight > 0.0:
                    elite_behavior_batch = elite_behavior.sample(
                        int(args.elite_behavior_batch_size),
                        float(args.elite_behavior_trigger_fraction),
                        window_balanced=bool(args.elite_behavior_window_balanced),
                        full_episode_s=float(args.full_episode_s),
                    )
                elite_behavior_rows_last = (
                    int(len(elite_behavior_batch.proprio))
                    if elite_behavior_batch is not None
                    else 0
                )
                suffix_mask = _suffix_actor_mask(batch.proprio)
                train_metrics = agent.update_suffix(
                    batch,
                    anchor_obs,
                    anchor_pro,
                    anchor_act,
                    alpha=float(args.suffix_alpha),
                    anchor_weight=float(beta),
                    freeze_encoder=not bool(args.train_encoder),
                    actor_mask=_actor_phase_mask(batch.proprio, actor_phases),
                    critic_mask=suffix_mask,
                    actor_update=bool(actor_due),
                    elite_behavior_batch=elite_behavior_batch,
                    elite_behavior_weight=elite_behavior_weight,
                    elite_behavior_full_episode_s=float(args.full_episode_s),
                    intake_during_return=bool(args.intake_during_return),
                    stage_d_ferry=bool(args.stage_d_ferry),  # STAGE-D1B
                    actor_q_center_fraction=float(args.actor_q_center_fraction),
                )
                # A newly started FULL-only run can temporarily contain no
                # suffix rows at all.  Preserve update debt and wait for the
                # first verified takeover instead of fitting the critic to the
                # immutable FIRST prefix or counting a fictitious update.
                if int(train_metrics.get("critic_rows", 0)) == 0:
                    train_metrics["waiting_for_suffix"] = 1.0
                    time.sleep(0.1)
                    break
                train_metric_interval.append(dict(train_metrics))
                if actor_due and int(train_metrics.get("actor_rows", 0)) > 0:
                    actor_updates += 1
                run_updates += 1
                tick_updates += 1
                update_debt -= 1.0
                if (
                    not bool(args.freeze_collector_weights)
                    and run_updates % int(args.weight_publish_updates) == 0
                ):
                    publish()
                if args.eval_snapshot_updates > 0 and run_updates % args.eval_snapshot_updates == 0:
                    save_atomic(eval_queue / f"v2_{run_updates:09d}.pt")
        elif new_tx == 0:
            time.sleep(0.25)

        if time.time() - last_report >= 60.0:
            last_report = time.time()
            actor_interval_summary = _summarize_actor_interval_metrics(
                train_metric_interval
            )
            recent = episodes[-80:]
            by_mode: dict[str, dict] = {}
            # BUGFIX 2026-07-25: "bank" was missing, so the time-sliced
            # blackout curriculum was invisible in by_mode summaries and on the
            # dashboard even while it was 20-45% of replay.
            for mode in ("full", "postdump", "collect", "return", "bank"):
                subset = [ep for ep in recent if ep.get("reset_mode", ep.get("stream_mode")) == mode]
                if subset:
                    by_mode[mode] = {
                        "episodes": len(subset),
                        "success_rate": round(
                            float(np.mean([ep.get("terminal_reason") == "skill_success" for ep in subset])), 3
                        ),
                        "score_mean": round(float(np.mean([ep.get("scored", 0) for ep in subset])), 2),
                        "cycles_mean": round(
                            float(np.mean([ep.get("cycles_completed", 0) for ep in subset])), 3
                        ),
                    }
            recent_full = [ep for ep in recent if ep.get("reset_mode") == "full"]
            line = {
                "wall_time": datetime.now().astimezone().isoformat(),
                "elapsed_s": round(time.time() - run_started, 1),
                "transitions": transitions,
                "updates": run_updates,
                "actor_updates": int(actor_updates),
                "schedule_updates": run_updates - schedule_origin,
                "replay": len(replay),
                "stddev": round(float(agent.stddev()), 4),
                "phase": (
                    "critic_only"
                    if run_updates - schedule_origin < args.critic_only_updates
                    else "suffix_sparse"
                ),
                "suffix_alpha": float(args.suffix_alpha),
                "elite_behavior_weight_schedule": round(
                    _elite_behavior_weight(
                        run_updates - schedule_origin,
                        int(args.critic_only_updates),
                        float(args.elite_behavior_weight),
                        float(args.elite_behavior_weight_end),
                        int(args.elite_behavior_decay_updates),
                    ),
                    6,
                ),
                "beta": round(
                    _anchor_beta(
                        run_updates - schedule_origin,
                        args.critic_only_updates,
                        args.anchor_beta_start,
                        args.anchor_beta_floor,
                        args.anchor_decay_updates,
                    ),
                    4,
                ),
                "collector_restart_boundaries": restarts_seen,
                "rejected_transitions": rejected,
                "elite_rows": int(elite_rows_last),
                "elite_behavior_rows": int(elite_behavior_rows_last),
                "elite_behavior_pool": {
                    "score": int(elite_behavior.score_rows),
                    "trigger": int(elite_behavior.trigger_rows),
                    "windows": elite_behavior.window_rows(float(args.full_episode_s)),
                },
                "elite_behavior_window_balanced": bool(
                    args.elite_behavior_window_balanced
                ),
                "elite_updates": int(elite_updates),
                "elite_loaded": dict(elite_archive_counts),
                "seedmine_elite_loaded": dict(seedmine_archive_counts),
                "elite_captured": dict(elite_new_counts),
                "elite_invalid_archives": int(elite_invalid_archives),
                "seedmine_invalid_archives": int(seedmine_invalid_archives),
                "seedmine_behavior_only": bool(args.seedmine_behavior_only),
                "elite_replay_rows": {
                    tier: int(len(ring)) for tier, ring in elite_replays.items()
                },
                "seedmine_warmup_replay_rows": {
                    tier: int(len(ring))
                    for tier, ring in seedmine_warmup_replays.items()
                },
                "episodes": len(episodes),
                "recent_full_score_mean": (
                    round(float(np.mean([ep.get("scored", 0) for ep in recent_full])), 2)
                    if recent_full
                    else None
                ),
                "recent_full_score_max": (
                    int(max(ep.get("scored", 0) for ep in recent_full)) if recent_full else None
                ),
                "recent_full_cycle2_rate": (
                    round(
                        float(np.mean([ep.get("cycles_completed", 0) >= 1 for ep in recent_full])), 3
                    )
                    if recent_full
                    else None
                ),
                "recent_full_cycle3_rate": (
                    round(
                        float(
                            np.mean(
                                [
                                    ep.get("cycles_completed", 0) >= 2
                                    for ep in recent_full
                                ]
                            )
                        ),
                        3,
                    )
                    if recent_full
                    else None
                ),
                "recent_full_repeat_return_load_mean": (
                    round(
                        float(
                            sum(
                                int(ep.get("repeat_return_load_sum", 0) or 0)
                                for ep in recent_full
                            )
                        )
                        / max(
                            1,
                            sum(
                                int(
                                    ep.get("repeat_return_load_count", 0) or 0
                                )
                                for ep in recent_full
                            ),
                        ),
                        2,
                    )
                    if recent_full
                    and any(
                        int(ep.get("repeat_return_load_count", 0) or 0) > 0
                        for ep in recent_full
                    )
                    else None
                ),
                "recent_full_repeat_scored_load_mean": (
                    round(
                        float(
                            sum(
                                int(ep.get("repeat_scored_load_sum", 0) or 0)
                                for ep in recent_full
                            )
                        )
                        / max(
                            1,
                            sum(
                                int(
                                    ep.get("repeat_scored_load_count", 0) or 0
                                )
                                for ep in recent_full
                            ),
                        ),
                        2,
                    )
                    if recent_full
                    and any(
                        int(ep.get("repeat_scored_load_count", 0) or 0) > 0
                        for ep in recent_full
                    )
                    else None
                ),
                "recent_full_ramp_out_rate": (
                    round(
                        float(
                            sum(
                                int(ep.get("ramp_out_successes", 0) or 0)
                                for ep in recent_full
                            )
                        )
                        / max(
                            1.0,
                            float(
                                sum(
                                    int(ep.get("ramp_out_attempts", 0) or 0)
                                    for ep in recent_full
                                )
                            ),
                        ),
                        3,
                    )
                    if recent_full
                    and any(
                        int(ep.get("ramp_out_attempts", 0) or 0) > 0
                        for ep in recent_full
                    )
                    else None
                ),
                "recent_full_cycle2_ramp_out_rate": (
                    round(
                        float(
                            sum(
                                int(ep.get("cycle2_ramp_out_successes", 0) or 0)
                                for ep in recent_full
                            )
                        )
                        / max(
                            1.0,
                            float(
                                sum(
                                    int(ep.get("cycle2_ramp_out_attempts", 0) or 0)
                                    for ep in recent_full
                                )
                            ),
                        ),
                        3,
                    )
                    if recent_full
                    and any(
                        int(ep.get("cycle2_ramp_out_attempts", 0) or 0) > 0
                        for ep in recent_full
                    )
                    else None
                ),
                "recent_full_cycle3plus_ramp_out_rate": (
                    round(
                        float(
                            sum(
                                int(ep.get("cycle3plus_ramp_out_successes", 0) or 0)
                                for ep in recent_full
                            )
                        )
                        / max(
                            1.0,
                            float(
                                sum(
                                    int(ep.get("cycle3plus_ramp_out_attempts", 0) or 0)
                                    for ep in recent_full
                                )
                            ),
                        ),
                        3,
                    )
                    if recent_full
                    and any(
                        int(ep.get("cycle3plus_ramp_out_attempts", 0) or 0) > 0
                        for ep in recent_full
                    )
                    else None
                ),
                "recent_full_outer_rail_fraction_raw": (
                    round(
                        float(
                            sum(
                                int(ep.get("outer_rail_steps", 0) or 0)
                                for ep in recent_full
                            )
                        )
                        / max(
                            1.0,
                            float(
                                sum(
                                    int(ep.get("outer_rail_active_steps", 0) or 0)
                                    for ep in recent_full
                                )
                            ),
                        ),
                        3,
                    )
                    if recent_full
                    else None
                ),
                "recent_full_outer_rail_fraction_mean": (
                    round(
                        float(
                            np.mean(
                                [
                                    float(ep.get("outer_rail_fraction", 0.0))
                                    for ep in recent_full
                                ]
                            )
                        ),
                        3,
                    )
                    if recent_full
                    else None
                ),
                "recent_full_outer_rail_max_streak_p90": (
                    round(
                        float(
                            np.percentile(
                                [
                                    int(ep.get("outer_rail_max_streak", 0))
                                    for ep in recent_full
                                ],
                                90,
                            )
                        ),
                        1,
                    )
                    if recent_full
                    else None
                ),
                "by_mode": by_mode,
                "sample_group_counts": replay.last_group_counts,
                **{key: round(float(value), 5) for key, value in train_metrics.items()},
                "actor_rows": int(train_metrics.get("actor_rows", 0)),
                "critic_rows": int(train_metrics.get("critic_rows", 0)),
                "actor_update_interval": int(args.actor_update_interval),
                "actor_q_center_fraction": float(args.actor_q_center_fraction),
                "actor_phases": list(actor_phases),
                **{
                    key: round(value, 5) if isinstance(value, float) else value
                    for key, value in actor_interval_summary.items()
                },
            }
            print("TRAIN_V2 " + json.dumps(line, sort_keys=True), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, sort_keys=True) + "\n")
            save_atomic(args.out / "latest.pt")
            train_metric_interval.clear()

    if not bool(args.freeze_collector_weights):
        publish()
    else:
        print(
            "LEARNER_V2_COLLECTOR_WEIGHTS_FROZEN "
            "final candidate saved but not published",
            flush=True,
        )
    save_atomic(args.out / "final.pt")
    print(
        "LEARNER_V2_DONE "
        + json.dumps({"transitions": transitions, "updates": run_updates}),
        flush=True,
    )


if __name__ == "__main__":
    main()
