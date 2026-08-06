"""Pure-Python Stage C v2 repeat-cycle state machine.

This module deliberately has no Isaac, NumPy, or Torch dependencies.  One
``CycleV2State`` belongs to one environment slot.  It turns the hidden history
of a repeat trip into an explicit phase, keeps ball-level route qualification,
and emits *successful* milestone events.  It never applies a continuous time
penalty.

The caller supplies only events from the current simulator step:

* ``magazine_ids`` is the complete current chamber contents;
* ``score_event_ids`` contains each newly scored ball event (duplicates are
  meaningful and are not de-duplicated);
* ``position`` is the robot position (the y coordinate is used by default);
* ``score`` is the cumulative episode score; and
* ``done`` closes and resets the per-environment state after the result is
  captured.

The first verified score-mode unload is protected at the full score value.
Once its chamber-empty completion edge fires, only balls collected on the away
side of the hysteretic boundary carry full score value.  Scoring consumes that
per-ball entitlement, so a ball cannot keep earning full credit merely by
recycling through the hub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, exp
from typing import Iterable, Sequence


ROUTE_EFFICIENCY_REVISION = "stage_d_v1"
# Stage D official-match revisions. D0 keeps
# score_efficiency_v11_rampfree (no reward-contract change); stage_d_v1 adds
# hub-gated scoring + the ineligible-fire mask + the shoot-penalty pause;
# stage_d_v2 adds the 8-ball preload and "rules" shift parity.  Flipping
# ROUTE_EFFICIENCY_REVISION to one of these at launch REQUIRES updating every
# hardcoded-literal site covered by the revision audit.
STAGE_D_REVISIONS = ("stage_d_v1", "stage_d_v2")
SUPPORTED_ROUTE_EFFICIENCY_REVISIONS = (
    "outer_rail_v1",
    "outer_rail_v2",
    "outer_rail_v3",
    "outer_rail_v4_ramp_out",
    "cycle_efficiency_v5",
    "cycle_bridge_v6",
    "score_efficiency_v8",
    "score_efficiency_v9",
    ROUTE_EFFICIENCY_REVISION,
) + STAGE_D_REVISIONS
RAMP_OUT_REVISIONS = (
    "outer_rail_v4_ramp_out",
    "cycle_efficiency_v5",
    "cycle_bridge_v6",
    "score_efficiency_v8",
    "score_efficiency_v9",
    ROUTE_EFFICIENCY_REVISION,
) + STAGE_D_REVISIONS  # mirror v11_rampfree: the ramp keys exist (launchers set them 0)
POSTDUMP_TARGET_REVISIONS = (
    "cycle_efficiency_v5",
    "cycle_bridge_v6",
)
POSTDUMP_COMPLETE_CYCLE_REVISIONS = ("cycle_bridge_v6",)
SCORE_EFFICIENCY_REVISIONS = (
    "score_efficiency_v8",
    "score_efficiency_v9",
    ROUTE_EFFICIENCY_REVISION,
) + STAGE_D_REVISIONS
# v15 ramp-free reuses v9's collect-until-preferred gate.  The active
# revision (score_efficiency_v11_rampfree, via ROUTE_EFFICIENCY_REVISION) is
# listed so eval/GUI config reconstruction matches the training env, which
# drives the gate from its own config flags, not this tuple.  The Stage D
# revisions inherit the same gate (v9-literal lesson: EVERY membership tuple
# that keys mechanics must name them BEFORE the revision flip).
COLLECT_UNTIL_PREFERRED_REVISIONS = (
    "score_efficiency_v9",
    ROUTE_EFFICIENCY_REVISION,
) + STAGE_D_REVISIONS
RETURN_INTAKE_REVISIONS = ("score_efficiency_v10_return_intake",)  # decoupled from ROUTE_EFFICIENCY_REVISION so a v9 ckpt is collect-until-preferred only (eval fix)


class CyclePhase(str, Enum):
    """Ordered skills used by the Stage C v2 policy."""

    FIRST_CYCLE = "first_cycle"
    LEAVE = "leave"
    COLLECT = "collect"
    RETURN = "return"
    SCORE = "score"


PHASE_ORDER = (
    CyclePhase.FIRST_CYCLE,
    CyclePhase.LEAVE,
    CyclePhase.COLLECT,
    CyclePhase.RETURN,
    CyclePhase.SCORE,
)


class FieldRegion(str, Enum):
    """Hysteretic home/away classification."""

    HOME = "home"
    AWAY = "away"


class Milestone(str, Enum):
    """Discrete successes; all except ``LATCHED`` re-arm every cycle."""

    LATCHED = "latched"
    LEFT_HOME = "left_home"
    TARGET_LOAD = "target_load"
    RETURNED_HOME = "returned_home"
    CYCLE_DUMPED = "cycle_dumped"
    CYCLE_SCORED = "cycle_scored"


@dataclass(frozen=True)
class CycleV2Config:
    """Reward and geometry settings for a Stage C v2 state machine.

    ``home_enter`` and ``away_enter`` define a dead band.  With the default
    ``home_is_lower=True``, y <= ``home_enter`` is home, y >= ``away_enter`` is
    away, and the previous classification is retained between them.
    """

    target_load: int = 15
    # V9 separates "a useful load may return" from "prefer collecting a fuller
    # load."  Once target_load is reached, COLLECT remains active until the
    # preferred load is reached, collection stalls, or the match clock forces
    # a return.  A zero preferred_load preserves the historical immediate
    # COLLECT -> RETURN transition exactly.
    preferred_load: int = 0
    collect_stall_steps: int = 0
    return_time_guard: float = 0.0
    # STAGE-D (ferry-first): when the caller reports the hub LIVE, COLLECT
    # exits to RETURN only once the qualified load reaches this threshold
    # instead of immediately at target_load.  Zero preserves the historical
    # exit-at-target behavior exactly (the D1E return-when-live semantics).
    live_return_load: int = 0
    chamber_capacity: int = 60
    # Retained for checkpoint/config compatibility.  Handoff is deliberately
    # driven by ``score_dump_completed`` rather than this legacy threshold.
    first_unload_score_floor: int = 25
    full_score_reward: float = 10.0
    # R5: was 2.0 — rewarding un-route-qualified scores created a re-score farm and
    # added reward variance/spikes with no task value (only own-court qualified scores
    # count). Zeroed so the only positive score signal is the qualified +10/ball.
    unqualified_score_reward: float = 0.0
    qualified_collect_reward: float = 1.5
    base_collect_reward: float = 0.3
    # A repeat unload is counted as a completed cycle only after this fraction
    # of its snapshotted route-qualified load scores.  The absolute floor is
    # capped by the actual load, so a legitimate small return can still pass.
    cycle_score_fraction: float = 0.75
    cycle_score_floor: int = 6
    position_axis: int = 1
    home_enter: float = -3.05
    away_enter: float = -2.50
    home_is_lower: bool = True

    def __post_init__(self) -> None:
        if self.target_load <= 0:
            raise ValueError("target_load must be positive")
        if self.chamber_capacity <= 0:
            raise ValueError("chamber_capacity must be positive")
        if self.target_load > self.chamber_capacity:
            raise ValueError("target_load cannot exceed chamber_capacity")
        if self.preferred_load and not (
            self.target_load < self.preferred_load <= self.chamber_capacity
        ):
            raise ValueError(
                "preferred_load must exceed target_load and not exceed chamber_capacity"
            )
        if self.collect_stall_steps < 0:
            raise ValueError("collect_stall_steps cannot be negative")
        if self.live_return_load < 0:
            raise ValueError("live_return_load cannot be negative")
        if self.live_return_load > self.chamber_capacity:
            raise ValueError("live_return_load cannot exceed chamber_capacity")
        if not 0.0 <= self.return_time_guard <= 1.0:
            raise ValueError("return_time_guard must be in [0, 1]")
        if self.preferred_load and (
            self.collect_stall_steps <= 0 and self.return_time_guard <= 0.0
        ):
            raise ValueError(
                "preferred_load requires a collection-stall or time fallback"
            )
        if self.first_unload_score_floor <= 0:
            raise ValueError("first_unload_score_floor must be positive")
        if not (0.0 < self.cycle_score_fraction <= 1.0):
            raise ValueError("cycle_score_fraction must be in (0, 1]")
        if self.cycle_score_floor <= 0:
            raise ValueError("cycle_score_floor must be positive")
        if self.home_enter >= self.away_enter:
            raise ValueError("home_enter must be below away_enter")
        for name in (
            "full_score_reward",
            "unqualified_score_reward",
            "qualified_collect_reward",
            "base_collect_reward",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MilestoneEvent:
    """A one-shot success emitted by ``CycleV2State.update``."""

    name: Milestone
    cycle_index: int
    elapsed_steps: int


@dataclass(frozen=True)
class CycleV2Step:
    """Immutable output for one update, captured before any ``done`` reset."""

    reward: float
    score_reward: float
    collect_reward: float
    phase: CyclePhase
    region: FieldRegion
    latched: bool
    cycle_index: int
    cycles_attempted: int
    cycles_completed: int
    phase_elapsed_steps: int
    qualified_load: int
    qualified_ids: frozenset[int]
    collected_ids: tuple[int, ...]
    score_event_ids: tuple[int, ...]
    full_score_events: int
    unqualified_score_events: int
    collect_exit_reason: str | None
    milestones: tuple[MilestoneEvent, ...]
    features: tuple[float, ...]
    done: bool


def time_decayed_success_bonus(
    base_reward: float,
    elapsed_steps: int,
    decay_steps: float,
) -> float:
    """Return a non-negative bonus paid only when a success event occurs.

    The exponential schedule rewards faster successful legs without charging
    the policy every step while it is travelling.  This helper is intentionally
    not called automatically: the trainer can assign different ``base_reward``
    and ``decay_steps`` values to each emitted milestone.
    """

    if base_reward < 0.0:
        raise ValueError("base_reward must be non-negative")
    if elapsed_steps < 0:
        raise ValueError("elapsed_steps must be non-negative")
    if decay_steps <= 0.0:
        raise ValueError("decay_steps must be positive")
    return float(base_reward) * exp(-float(elapsed_steps) / float(decay_steps))


def soft_repeat_load_bonus(
    *,
    qualified_load: int,
    minimum_load: int,
    preferred_load: int,
    max_bonus: float,
) -> float:
    """Reward fuller repeat trips without blocking a useful early return.

    ``minimum_load`` remains the state-machine gate that makes a return legal.
    Loads at or below it receive no extra shaping, loads between it and
    ``preferred_load`` receive a linear bonus, and larger loads are capped.
    This is deliberately a bonus rather than a penalty: a policy can still
    return and score a smaller load when time or field geometry makes that the
    efficient choice.
    """

    if qualified_load < 0:
        raise ValueError("qualified_load must be non-negative")
    if minimum_load <= 0:
        raise ValueError("minimum_load must be positive")
    if preferred_load <= minimum_load:
        raise ValueError("preferred_load must exceed minimum_load")
    if max_bonus < 0.0:
        raise ValueError("max_bonus must be non-negative")
    progress = (
        float(qualified_load) - float(minimum_load)
    ) / float(preferred_load - minimum_load)
    return float(max_bonus) * min(1.0, max(0.0, progress))


def capped_phase_delay_penalty(
    *,
    elapsed_steps: int,
    grace_steps: int,
    penalty_per_step: float,
    spent: float,
    cap: float,
) -> float:
    """Return this step's non-positive delay charge for one phase.

    ``spent`` is the non-negative magnitude already charged in this phase.
    Calling this once per phase step yields no charge through ``grace_steps``,
    then charges ``penalty_per_step`` without letting total spend exceed
    ``cap``.  Returning the incremental (rather than cumulative) charge makes
    the helper safe to add directly to a LEAVE, RETURN, or SCORE step reward.
    """

    if elapsed_steps < 0:
        raise ValueError("elapsed_steps must be non-negative")
    if grace_steps < 0:
        raise ValueError("grace_steps must be non-negative")
    if penalty_per_step < 0.0:
        raise ValueError("penalty_per_step must be non-negative")
    if spent < 0.0:
        raise ValueError("spent must be non-negative")
    if cap < 0.0:
        raise ValueError("cap must be non-negative")
    if int(elapsed_steps) <= int(grace_steps):
        return 0.0
    remaining = max(0.0, float(cap) - float(spent))
    return -min(float(penalty_per_step), remaining)


@dataclass
class _ScoreDumpLedger:
    """Qualified-ball quality ledger that survives the chamber emptying."""

    cycle_index: int
    qualified_ids: frozenset[int]
    required_scores: int
    cycle_started_step: int
    scored_ids: set[int] = field(default_factory=set)
    completed: bool = False
    success_emitted: bool = False


@dataclass
class CycleV2State:
    """Per-environment Stage C v2 state and reward ledger."""

    config: CycleV2Config = field(default_factory=CycleV2Config)
    phase: CyclePhase = field(init=False, default=CyclePhase.FIRST_CYCLE)
    region: FieldRegion = field(init=False, default=FieldRegion.HOME)
    latched: bool = field(init=False, default=False)
    cycle_index: int = field(init=False, default=1)
    qualified_ids: set[int] = field(init=False, default_factory=set)
    # Balls acquired before the first productive unload retain the protected
    # +10 entitlement until their first score event, even if projectile flight
    # lag makes them land after the repeat-cycle latch flips.
    protected_first_ids: set[int] = field(init=False, default_factory=set)
    prev_magazine: frozenset[int] = field(init=False, default_factory=frozenset)
    step_count: int = field(init=False, default=0)
    phase_started_step: int = field(init=False, default=0)
    cycle_started_step: int = field(init=False, default=0)
    last_qualified_collect_step: int = field(init=False, default=0)
    last_score: int = field(init=False, default=0)
    cycles_attempted: int = field(init=False, default=0)
    _cycles_completed: int = field(init=False, default=0, repr=False)
    _score_dump_active: bool = field(init=False, default=False, repr=False)
    _score_dump_ledger: _ScoreDumpLedger | None = field(
        init=False, default=None, repr=False
    )
    _milestones_this_cycle: set[Milestone] = field(
        init=False, default_factory=set, repr=False
    )

    def __post_init__(self) -> None:
        self.reset()

    def reset(
        self,
        initial_magazine_ids: Iterable[int] = (),
        position: Sequence[float] | float | None = None,
        score: int = 0,
        *,
        phase: CyclePhase = CyclePhase.FIRST_CYCLE,
        qualified_ids: Iterable[int] = (),
        cycle_index: int | None = None,
    ) -> None:
        """Reset one environment slot and seed its initial chamber contents.

        Seeding ``prev_magazine`` prevents a preload from being mistaken for a
        collection on the first simulator step.
        """

        phase = CyclePhase(phase)
        initial_magazine = frozenset(int(i) for i in initial_magazine_ids)
        qualified = set(int(i) for i in qualified_ids)
        if phase is CyclePhase.FIRST_CYCLE and qualified:
            raise ValueError("first-cycle reset cannot seed qualified balls")
        if not qualified.issubset(initial_magazine):
            raise ValueError("qualified_ids must be present in the initial magazine")

        self.phase = phase
        self.region = FieldRegion.HOME
        if position is not None:
            self.region = self._classify_position(position, FieldRegion.HOME)
        self.latched = phase is not CyclePhase.FIRST_CYCLE
        self.cycle_index = int(
            cycle_index if cycle_index is not None else (2 if self.latched else 1)
        )
        if self.cycle_index < (2 if self.latched else 1):
            raise ValueError("cycle_index is inconsistent with the reset phase")
        self.qualified_ids = qualified
        self.protected_first_ids = (
            set(initial_magazine) if phase is CyclePhase.FIRST_CYCLE else set()
        )
        self.prev_magazine = initial_magazine
        self.step_count = 0
        self.phase_started_step = 0
        self.cycle_started_step = 0
        self.last_qualified_collect_step = 0
        self.last_score = int(score)
        self.cycles_attempted = 0
        self._cycles_completed = 0
        self._score_dump_active = False
        self._score_dump_ledger = None
        self._milestones_this_cycle = set()

    @property
    def cycles_completed(self) -> int:
        """Number of post-first-unload cycles completed in this episode."""

        return int(self._cycles_completed)

    @property
    def phase_elapsed_steps(self) -> int:
        """Steps elapsed in the current phase, including the current step."""

        return max(0, int(self.step_count) - int(self.phase_started_step))

    def feature_vector(
        self,
        magazine_ids: Iterable[int] | None = None,
        *,
        time_remaining: float = 1.0,
    ) -> tuple[float, ...]:
        """Actor-ready phase features appended to the legacy proprio vector.

        The result has eight elements: five phase bits in ``PHASE_ORDER``, then
        qualified load / chamber capacity, target load / chamber capacity, and
        normalized episode time remaining.  The five-way one-hot makes the
        protected first cycle explicit rather than encoding it as all-zero.
        """

        if magazine_ids is None:
            magazine = self.prev_magazine
        else:
            magazine = frozenset(int(i) for i in magazine_ids)
        qualified_load = len(self.qualified_ids.intersection(magazine))
        one_hot = tuple(1.0 if self.phase is phase else 0.0 for phase in PHASE_ORDER)
        capacity = float(self.config.chamber_capacity)
        load_ratio = min(float(qualified_load) / capacity, 1.0)
        target_ratio = min(float(self.config.target_load) / capacity, 1.0)
        remaining = min(1.0, max(0.0, float(time_remaining)))
        return one_hot + (load_ratio, target_ratio, remaining)

    def update(
        self,
        magazine_ids: Iterable[int],
        score_event_ids: Iterable[int],
        position: Sequence[float] | float,
        score: int,
        done: bool = False,
        time_remaining: float = 1.0,
        *,
        score_dump_started: bool = False,
        score_dump_completed: bool = False,
        owncourt_score_ready: bool = False,
        hub_live: bool = False,
    ) -> CycleV2Step:
        """Advance one environment by one simulator step.

        ``score_event_ids`` must contain only events newly emitted on this step;
        repeated values are valid repeat scores.  ``score_dump_started`` and
        ``score_dump_completed`` must describe a verified score-mode dump, not
        merely an action request or an empty chamber.  Completion is what
        releases SCORE/FIRST_CYCLE into LEAVE; projectile landings can arrive
        later and are audited independently.  When ``done`` is true, events are
        processed and returned normally, then internal state is reset for the
        next episode.  ``owncourt_score_ready`` (Stage-D1C, default False) lets
        a home load re-enter SCORE from LEAVE/COLLECT without a cross-field trip;
        the caller is responsible for its flag and hub-eligibility gating.
        """

        magazine_ordered = tuple(int(i) for i in magazine_ids)
        magazine = frozenset(magazine_ordered)
        score_events = tuple(int(i) for i in score_event_ids)
        collected = self._new_magazine_ids(magazine_ordered)
        self.step_count += 1
        self.region = self._classify_position(position, self.region)
        milestones: list[MilestoneEvent] = []

        # A collection only becomes route-qualified after the first unload and
        # on the away side of the hysteretic boundary.  Re-collecting an ID that
        # still owns an unused qualification does not mint another entitlement.
        collect_reward = 0.0
        for ball_id in collected:
            can_qualify = self.latched and self.region is FieldRegion.AWAY
            if can_qualify and ball_id not in self.qualified_ids:
                # A missed first-volley ball legitimately collected on a new
                # neutral excursion is now route-qualified, not merely trailing.
                self.protected_first_ids.discard(ball_id)
                self.qualified_ids.add(ball_id)
                collect_reward += self.config.qualified_collect_reward
                self.last_qualified_collect_step = self.step_count
            else:
                if can_qualify and ball_id in self.qualified_ids:
                    # Re-acquiring a previously qualified dropped ball is real
                    # intake progress even though it must not mint reward twice.
                    self.last_qualified_collect_step = self.step_count
                if not self.latched:
                    self.protected_first_ids.add(ball_id)
                collect_reward += self.config.base_collect_reward

        # Advance navigation phases before applying score events.  This handles
        # a simulator step that crosses home while still holding the load.
        collect_exit_reason: str | None = None
        if self.latched:
            if self.phase is CyclePhase.LEAVE and self.region is FieldRegion.AWAY:
                self._emit_once(Milestone.LEFT_HOME, milestones)
                self._set_phase(CyclePhase.COLLECT)

            qualified_load = len(self.qualified_ids.intersection(magazine))
            if self.phase is CyclePhase.COLLECT and qualified_load >= self.config.target_load:
                self._emit_once(Milestone.TARGET_LOAD, milestones)
                preferred_reached = (
                    self.config.preferred_load <= self.config.target_load
                    or qualified_load >= self.config.preferred_load
                    # STAGE-D1E: when the caller says the blue hub is LIVE, return
                    # to score instead of dwelling in COLLECT chasing the (often
                    # unreachable) preferred load while a live scoring window
                    # burns.  live_return_load (stage_d_v1 ferry-first) sets how
                    # full the chamber must be before a live hub pulls the robot
                    # home; zero keeps the original exit-at-target behavior.
                    # hub_live defaults False => byte-identical for every
                    # non-stage-D caller and whenever the hub is dark.
                    or (hub_live and qualified_load >= self.config.live_return_load)
                )
                collection_stalled = (
                    self.config.collect_stall_steps > 0
                    and self.step_count - self.last_qualified_collect_step
                    >= self.config.collect_stall_steps
                )
                clock_forces_return = (
                    self.config.return_time_guard > 0.0
                    and float(time_remaining) <= self.config.return_time_guard
                )
                if preferred_reached or collection_stalled or clock_forces_return:
                    if self.config.preferred_load <= self.config.target_load:
                        collect_exit_reason = "target"
                    elif qualified_load >= self.config.preferred_load:
                        collect_exit_reason = "preferred"
                    elif collection_stalled:
                        collect_exit_reason = "stall"
                    elif clock_forces_return:
                        collect_exit_reason = "clock"
                    else:
                        collect_exit_reason = "hub_live"
                    self._set_phase(CyclePhase.RETURN)

            if (
                self.phase is CyclePhase.RETURN
                and self.region is FieldRegion.HOME
                and qualified_load > 0
            ):
                self._emit_once(Milestone.RETURNED_HOME, milestones)
                self._set_phase(CyclePhase.SCORE)

            # STAGE-D1C own-court short loop: when the caller asserts the payoff
            # is live (blue hub active in the latched suffix), a load intaked in
            # our own court from ferried stock may re-enter SCORE WITHOUT a fresh
            # cross-field AWAY trip.  The caller owns the flag/eligibility gate;
            # here we only require a home load that carries a downstream +10
            # entitlement.  Byte-identical when ``owncourt_score_ready`` is False
            # (the default), so the ordinary cross-field cycle is untouched.
            if (
                owncourt_score_ready
                and self.phase in (CyclePhase.LEAVE, CyclePhase.COLLECT)
                and self.region is FieldRegion.HOME
                and qualified_load > 0
            ):
                self._emit_once(Milestone.RETURNED_HOME, milestones)
                self._set_phase(CyclePhase.SCORE)

        # Snapshot the route-qualified load at the verified start of a score
        # dump.  ``prev_magazine`` covers the common case where the first ball
        # has already left the chamber on the same simulator step as the edge.
        if (
            bool(score_dump_started)
            and self.phase in (CyclePhase.FIRST_CYCLE, CyclePhase.SCORE)
        ):
            self._begin_score_dump(magazine)

        # A completion edge is authoritative even if the caller was attached
        # one tick too late to observe its start edge.  Build the best possible
        # snapshot before consuming this step's score events.
        if (
            bool(score_dump_completed)
            and not self._score_dump_active
            and self.phase in (CyclePhase.FIRST_CYCLE, CyclePhase.SCORE)
        ):
            self._begin_score_dump(magazine)

        # Score credit is ball-specific after the latch.  Membership is removed
        # immediately, so another score of the same ID is worth the lower value
        # unless the ball is legitimately collected away again.
        score_reward = 0.0
        full_score_events = 0
        unqualified_score_events = 0
        was_latched = self.latched
        for ball_id in score_events:
            if not was_latched:
                score_reward += self.config.full_score_reward
                full_score_events += 1
                self.protected_first_ids.discard(ball_id)
            elif ball_id in self.protected_first_ids:
                # The chamber can empty before a long volley finishes flying.
                # Protect every ball from that first load through its landing;
                # it does not count as a qualified repeat-cycle score.
                self.protected_first_ids.remove(ball_id)
                score_reward += self.config.full_score_reward
                full_score_events += 1
            elif ball_id in self.qualified_ids:
                self.qualified_ids.remove(ball_id)
                score_reward += self.config.full_score_reward
                full_score_events += 1
            else:
                score_reward += self.config.unqualified_score_reward
                unqualified_score_events += 1

            ledger = self._score_dump_ledger
            if ledger is not None and ball_id in ledger.qualified_ids:
                ledger.scored_ids.add(ball_id)

        # Navigation is released by the verified empty-dump edge, never by a
        # cumulative score threshold.  Any first-load projectiles still in
        # flight remain in ``protected_first_ids`` and retain their +10 landing.
        if (
            not self.latched
            and bool(score_dump_completed)
            and not magazine
        ):
            self._score_dump_active = False
            self._score_dump_ledger = None
            self.latched = True
            self._set_phase(CyclePhase.LEAVE)
            self.cycle_index = 2
            self.cycle_started_step = self.step_count
            self._milestones_this_cycle = set()
            milestones.append(MilestoneEvent(Milestone.LATCHED, 2, 0))

        # A repeat dump is an *attempt* as soon as the verified score-mode dump
        # empties.  Start the next outbound leg immediately; success remains
        # pending until enough members of the snapshotted volley actually land.
        elif (
            self.latched
            and self.phase is CyclePhase.SCORE
            and bool(score_dump_completed)
            and not magazine
        ):
            ledger = self._score_dump_ledger
            if ledger is None:
                self._begin_score_dump(magazine)
                ledger = self._score_dump_ledger
            if ledger is not None:
                ledger.completed = True
            self._score_dump_active = False
            self._emit_once(Milestone.CYCLE_DUMPED, milestones)
            self.cycles_attempted += 1
            self.cycle_index += 1
            self._set_phase(CyclePhase.LEAVE)
            self.cycle_started_step = self.step_count
            self._milestones_this_cycle = set()

        self._maybe_emit_dump_success(milestones)

        self.prev_magazine = magazine
        self.last_score = int(score)
        qualified_load = len(self.qualified_ids.intersection(magazine))
        features = self.feature_vector(magazine, time_remaining=time_remaining)
        result = CycleV2Step(
            reward=score_reward + collect_reward,
            score_reward=score_reward,
            collect_reward=collect_reward,
            phase=self.phase,
            region=self.region,
            latched=self.latched,
            cycle_index=self.cycle_index,
            cycles_attempted=self.cycles_attempted,
            cycles_completed=self.cycles_completed,
            phase_elapsed_steps=self.phase_elapsed_steps,
            qualified_load=qualified_load,
            qualified_ids=frozenset(self.qualified_ids),
            collected_ids=collected,
            score_event_ids=score_events,
            full_score_events=full_score_events,
            unqualified_score_events=unqualified_score_events,
            collect_exit_reason=collect_exit_reason,
            milestones=tuple(milestones),
            features=features,
            done=bool(done),
        )
        if done:
            self.reset()
        return result

    def _set_phase(self, phase: CyclePhase) -> None:
        """Enter ``phase`` and restart its elapsed-step clock."""

        phase = CyclePhase(phase)
        if phase is self.phase:
            return
        self.phase = phase
        self.phase_started_step = self.step_count

    def rearm_owncourt_collect(self) -> None:
        """STAGE-D1F: re-open the own-court short loop after a completed dump by
        dropping the latched suffix from SCORE back to COLLECT, so the next
        re-intake of a loose entitled ball re-triggers the SCORE re-entry and the
        rest of the own-court stockpile converts.  Caller gates on the flag +
        hub-live + stockpile remaining; no-op unless latched and in SCORE."""

        if self.latched and self.phase is CyclePhase.SCORE:
            self._set_phase(CyclePhase.COLLECT)

    def _begin_score_dump(self, magazine: frozenset[int]) -> None:
        """Start a verified score-mode dump and snapshot its qualified load."""

        self._score_dump_active = True
        if not self.latched or self.phase is not CyclePhase.SCORE:
            self._score_dump_ledger = None
            return

        # A new dump closes any still-unresolved old volley: later recycling of
        # those IDs must not retroactively turn the old attempt into a success.
        source_ids = self.prev_magazine.union(magazine)
        qualified = frozenset(self.qualified_ids.intersection(source_ids))
        start_load = len(qualified)
        required = (
            min(
                start_load,
                max(
                    int(self.config.cycle_score_floor),
                    int(ceil(self.config.cycle_score_fraction * float(start_load))),
                ),
            )
            if start_load > 0
            else 0
        )
        self._score_dump_ledger = _ScoreDumpLedger(
            cycle_index=self.cycle_index,
            qualified_ids=qualified,
            required_scores=required,
            cycle_started_step=self.cycle_started_step,
        )

    def _maybe_emit_dump_success(
        self,
        output: list[MilestoneEvent],
    ) -> None:
        """Credit one completed cycle once its dumped volley clears quality."""

        ledger = self._score_dump_ledger
        if (
            ledger is None
            or not ledger.completed
            or ledger.success_emitted
            or ledger.required_scores <= 0
            or len(ledger.scored_ids) < ledger.required_scores
        ):
            return
        ledger.success_emitted = True
        self._cycles_completed += 1
        output.append(
            MilestoneEvent(
                name=Milestone.CYCLE_SCORED,
                cycle_index=ledger.cycle_index,
                elapsed_steps=max(0, self.step_count - ledger.cycle_started_step),
            )
        )

    def _emit_once(
        self,
        name: Milestone,
        output: list[MilestoneEvent],
    ) -> None:
        if name in self._milestones_this_cycle:
            return
        self._milestones_this_cycle.add(name)
        output.append(
            MilestoneEvent(
                name=name,
                cycle_index=self.cycle_index,
                elapsed_steps=max(0, self.step_count - self.cycle_started_step),
            )
        )

    def _new_magazine_ids(self, magazine_ordered: tuple[int, ...]) -> tuple[int, ...]:
        """Set difference while retaining stable caller order and unique IDs."""

        seen: set[int] = set()
        return tuple(
            ball_id
            for ball_id in magazine_ordered
            if ball_id not in self.prev_magazine
            and ball_id not in seen
            and not seen.add(ball_id)
        )

    def _coordinate(self, position: Sequence[float] | float) -> float:
        if isinstance(position, (int, float)):
            return float(position)
        try:
            return float(position[self.config.position_axis])
        except (IndexError, TypeError) as exc:
            raise ValueError(
                f"position must contain axis {self.config.position_axis}"
            ) from exc

    def _classify_position(
        self,
        position: Sequence[float] | float,
        previous: FieldRegion,
    ) -> FieldRegion:
        coordinate = self._coordinate(position)
        if self.config.home_is_lower:
            if coordinate <= self.config.home_enter:
                return FieldRegion.HOME
            if coordinate >= self.config.away_enter:
                return FieldRegion.AWAY
        else:
            # Mirrored field: thresholds remain ordered, but the upper side is
            # home and the lower side is away.
            if coordinate >= self.config.away_enter:
                return FieldRegion.HOME
            if coordinate <= self.config.home_enter:
                return FieldRegion.AWAY
        return previous


__all__ = [
    "ROUTE_EFFICIENCY_REVISION",
    "RAMP_OUT_REVISIONS",
    "POSTDUMP_TARGET_REVISIONS",
    "POSTDUMP_COMPLETE_CYCLE_REVISIONS",
    "SCORE_EFFICIENCY_REVISIONS",
    "COLLECT_UNTIL_PREFERRED_REVISIONS",
    "RETURN_INTAKE_REVISIONS",
    "SUPPORTED_ROUTE_EFFICIENCY_REVISIONS",
    "CyclePhase",
    "CycleV2Config",
    "CycleV2State",
    "CycleV2Step",
    "FieldRegion",
    "Milestone",
    "MilestoneEvent",
    "PHASE_ORDER",
    "capped_phase_delay_penalty",
    "time_decayed_success_bonus",
]
