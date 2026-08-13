"""Vectorized full-physics competition environment (Isaac Sim, N cloned fields).

The environment provides:

- N complete fields (robot + FUEL + hub) cloned from the exported USD template
  onto one shared GPU PhysX scene (throughput is FUEL-count-bound; stages A-C
  use small FUEL subsets and clear the 8 policy-tx/s gate with headroom).
- The existing, well-tested single-robot stack (``CompetitionRobotController``,
  ``HubRouter``) runs per env behind thin adapters that translate between the
  batched views and the single-env numpy interface, shifting every world
  position by the env origin so all field math stays in field coordinates.
- Actions are the frozen 7-D contract (``frc_rebuilt.rl.spec``); the policy
  runs at 10 Hz (action repeat 6 over 60 Hz physics; controller cadence 30 Hz
  exactly like the interactive GUI).
- Observations: the 3-camera 640x360 rig (rendered once per policy step) plus
  a non-privileged proprio vector; a privileged vector is provided separately
  for the asymmetric critic (training-time only, never an actor input).
- Reward: legal blue score under the match scoring rules (router-confirmed),
  plus curriculum shaping terms logged separately.

This module must be imported only after ``SimulationApp`` is created.
"""
from __future__ import annotations

import math
import os as _os_speed

# Opt-in: drop the keyboard ergonomics damper from the policy drive path
# (2.25 -> 3.21 m/s).  Off by default: checkpoints trained before
# 2026-07-31 learned their motion timing under the damped mapping.
_POLICY_FULL_SPEED = _os_speed.environ.get("FRC_POLICY_FULL_SPEED") == "1"
_SPEED_RAMP = _os_speed.environ.get("FRC_POLICY_SPEED_RAMP")
_SPEED_FIXED = _os_speed.environ.get("FRC_POLICY_SPEED_SCALE")
_DAMPED_SCALE = 0.70  # KEYBOARD_TRANSLATION_SCALE
_SPEED_PROGRESS = {"steps": 0.0}
_SPEED_FILE = _os_speed.environ.get("FRC_POLICY_SPEED_FILE")
_SPEED_FILE_CACHE = {"t": 0.0, "scale": None}


def set_policy_train_steps(value) -> None:
    """Collectors publish the loaded checkpoint step so the curriculum is
    tied to LEARNING progress, not wall clock -- a respawned collector must
    not reset the ramp."""
    _SPEED_PROGRESS["steps"] = float(value)


def _policy_speed_scale():
    """Suffix translation scale: live file, fixed override, ramp, or None."""
    if _SPEED_FILE:
        import time as _t
        now = _t.time()
        if now - _SPEED_FILE_CACHE["t"] > 5.0:
            _SPEED_FILE_CACHE["t"] = now
            try:
                with open(_SPEED_FILE, "r", encoding="utf-8") as fh:
                    value = float(fh.read().strip())
                if 0.3 <= value <= 1.0:
                    _SPEED_FILE_CACHE["scale"] = value
            except (OSError, ValueError):
                pass
        if _SPEED_FILE_CACHE["scale"] is not None:
            return _SPEED_FILE_CACHE["scale"]
    if _SPEED_FIXED:
        return float(_SPEED_FIXED)
    if _SPEED_RAMP:
        a, b = (float(x) for x in _SPEED_RAMP.split(","))
        if b <= a:
            return 1.0
        f = (_SPEED_PROGRESS["steps"] - a) / (b - a)
        f = max(0.0, min(1.0, f))
        return _DAMPED_SCALE + f * (1.0 - _DAMPED_SCALE)
    return None
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from frc_rebuilt.rl.spec import CompetitionRLSpec, decode_policy_actions
from frc_rebuilt.rl.policy_v2 import RETURN_SKILL_PRELOAD
from frc_rebuilt.rl import stage_d as _stage_d

FUEL_RADIUS_M = 0.076


class SimulationUnstable(RuntimeError):
    """Raised when the shared PhysX scene is unrecoverably non-finite/exploded.

    The trainer lets this propagate so the process exits and the supervisor
    restarts it from the latest checkpoint with a freshly built simulation - a
    globally NaN GPU PhysX solver cannot be repaired by per-env resets.
    """


def _effective_fire_mode(
    proposed_mode: str,
    *,
    dumping: bool,
    dump_mode: str | None,
) -> str:
    """Keep an active score volley authoritative over later policy heads."""

    if dumping and dump_mode is not None:
        return str(dump_mode)
    return str(proposed_mode)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class VecEnvCfg:
    num_envs: int = 2
    template_usd: str = ""            # exported env template (required)
    spacing_m: float = 24.0
    episode_len_s: float = 20.0       # stage A: short acquisition episodes
    cameras: bool = True
    camera_names: tuple[str, ...] = ()  # default: rig baseline from the robot module
    # stage A settings
    lock_storage_extended: bool = True
    # stage C: begin each episode compact and fully beneath the blue TRENCH (the
    # source-faithful match start; see competition_robot.BLUE_TRENCH_START_*).
    # Forces the escape -> extend -> collect -> score chain, and only makes sense
    # with storage unlocked (lock_storage_extended=False) so the policy can
    # extend once it clears the 0.565 m trench opening.
    spawn_under_trench: bool = False
    sandbox_scoring: bool = True      # both hubs always eligible (stages A/B)
    spawn_xy_range: tuple[float, float, float, float] = (-2.4, 2.4, -1.8, 1.2)
    spawn_yaw_range_deg: tuple[float, float] = (-180.0, 180.0)
    proprio_noise_std: float = 0.02
    seed: int = 2026
    # reward weights (collect weight is annealed live by the trainer via
    # env.collect_weight; score weight stays fixed per the converged plan)
    score_reward_weight: float = 10.0
    collect_reward_weight: float = 1.5
    # Custody-weighted reward: a ball's FIRST legal score/collect
    # earns full weight; re-scoring or re-collecting the SAME ball (the recycling exploit)
    # earns rho x weight. rho_*=1.0 reproduces the original raw-score reward exactly, so
    # existing runs are unaffected unless the trainer sets these below 1.0.
    rho_score: float = 1.0
    rho_collect: float = 1.0
    # small per-step penalty for sitting empty-handed in own court: pushes the
    # robot to LEAVE and go collect again once it has scored its load (teaches the
    # 2nd collect->score cycle; also nudges it out of the empty trench spawn). Only
    # fires when magazine==0 AND y<-2.775, so it never touches the productive cycle.
    empty_own_court_penalty: float = 0.02
    # one legal fire press -> freeze in place, auto-aim, and empty the WHOLE
    # magazine before resuming policy control. False = the default one-click single
    # (one ball per press; holding gives cooldown-rate rapid fire).
    dump_on_press: bool = False
    # dump safety cap: force-release the dump movement-lock after this many control
    # ticks even if the chamber never empties, so a stuck dump (blocked shooter,
    # lost aim) can never deadlock the chassis. The primary release is loss of a
    # legal shot; this is the belt-and-suspenders hard cap (audit).
    max_dump_ticks: int = 90
    # stage B: acquire-and-score.  With probability preload_prob an episode
    # starts ALREADY holding preload_count FUEL at a valid shooting pose
    # (learn to convert inventory); otherwise it must collect first.
    preload_prob: float = 0.0
    preload_count_range: tuple[int, int] = (4, 8)
    shoot_spawn_xy_range: tuple[float, float, float, float] = (-2.0, 2.0, -2.2, -0.9)
    # stage B ramp-teaching: with this probability a COLD (non-preloaded) episode
    # instead starts in the neutral zone aligned to a hub RAMP lane (|x|~0.9-2.2),
    # facing the hub, and preloaded - so the shortest path to a legal shot runs
    # straight south OVER the drive-over ramp (which the locked-extended robot
    # clears fine) instead of drifting out to the compact-only trench lanes where
    # an extended robot jams against the gate arm.  0.0 = off (pre-ramp behaviour).
    spawn_ramp_prob: float = 0.0
    # 2nd-cycle CURRICULUM (2026-07-14): with probability neutral_refill_prob, RELOCATE
    # neutral_refill_count existing FUEL to a shallow-neutral cluster near the collect
    # point at reset. After cycle 1 the near balls are gone and the leftovers are deep,
    # so the 2nd-cycle round-trip is too long to finish in-episode; guaranteeing a NEAR
    # cluster makes the leave->collect->return->score loop short + completable so the
    # policy can actually learn it. No new prims (existing balls are moved -> physics-safe);
    # fade it out by lowering the prob once cycle2_rate lifts. 0.0 = off.
    neutral_refill_count: int = 0
    neutral_refill_prob: float = 0.0
    neutral_refill_center: tuple[float, float] = (0.0, 1.6)   # field-local (x, y), just past the board
    neutral_refill_spread: tuple[float, float] = (2.2, 1.1)   # half-extent of the cluster grid (x, y)
    # 2nd-cycle RETURN curriculum (2026-07-14): on neutral_loaded_prob of episodes (Stage C
    # only), OVERRIDE the under-trench start with a NEUTRAL, LOADED start facing the hub, so
    # the episode is just RETURN -> shoot. Exploration alone never cracked the back half of
    # the 2nd cycle (bot collects but never comes home to score); starting mid-cycle makes
    # return+shoot succeed often -> the raw score reward teaches the leg, which transfers to
    # the full cycle. 0.0 = off. These episodes are tagged so the funnel can exclude them.
    neutral_loaded_prob: float = 0.0
    neutral_loaded_count: tuple[int, int] = (8, 14)          # FUEL preloaded at the neutral start
    neutral_loaded_y: tuple[float, float] = (0.5, 2.5)       # field-local y band (deep-neutral, past board)
    # DEPRECATED / no-op: legal-fire gating is now UNCONDITIONAL (see the fire
    # block in step()). An illegal fire press is always a no-op that keeps the
    # chassis driving, in every stage, so pressing fire outside the alliance zone
    # can never freeze the robot. Retained only so existing --mask-illegal-fire
    # command lines keep parsing; its value no longer changes behavior.
    mask_illegal_fire: bool = True
    # physics hygiene: an env whose robot/fuel exceeds these bounds (or goes
    # non-finite) is force-reset that step BEFORE it poisons the shared GPU
    # PhysX scene.  Bounds are ~10x any legal gameplay speed (robot <5 m/s,
    # shots <~20 m/s) so they never fire during normal play - only on blow-ups.
    max_robot_linear_speed: float = 50.0     # m/s
    max_robot_angular_speed: float = 200.0   # rad/s
    max_fuel_speed: float = 120.0            # m/s (shooter launches are well under)
    max_position_abs: float = 100.0          # m from field origin
    max_unhealthy_trips: int = 30            # total force-resets before aborting
    # small energy penalty on drive; MUST stay << collect reward so collecting a
    # distant ball is always net-positive (audit: the old constant -0.01 could
    # exceed the annealed collect reward and invert the collection incentive)
    action_penalty: float = 0.001
    # Stage C v2 is an opt-in observation/reward/curriculum contract.  Every
    # legacy default above remains unchanged when this is false.
    stagec_v2: bool = False
    # One fixed reset mode per env slot.  A single value is broadcast; an empty
    # tuple means ``full``.  Fixed streams make grouped replay honest.
    cycle_v2_reset_modes: tuple[str, ...] = ()
    cycle_v2_target_load: int = 15
    cycle_v2_chamber_capacity: int = 60
    # Park one batch only for the physical RETURN preload.  FULL, postdump, and
    # collect streams keep the native 200-ball field: their target ramp areas
    # already contain enough FUEL, and adding a blind grid can overlap bodies.
    cycle_v2_reserve_count: int = 18
    cycle_v2_reserve_batches: int = 3
    cycle_v2_cluster_x: float = 1.55
    cycle_v2_cluster_y: float = -0.85
    cycle_v2_cluster_spread: tuple[float, float] = (0.72, 0.62)
    # Success-only shaping.  Progress is paid only for a new best distance in
    # the current phase, so back-and-forth motion cannot farm it.
    cycle_v2_progress_per_m: float = 5.0
    cycle_v2_progress_step_cap: float = 0.75
    cycle_v2_ramp_bonus: float = 6.0
    # V4 outbound-route curriculum.  Historical checkpoints leave this off.
    # When enabled, LEAVE progress is ordered (move inward into a ramp corridor,
    # then cross toward neutral), postdump succeeds only through a physical
    # ramp, and a trench/outer exit receives an explicit one-shot deduction.
    cycle_v2_require_ramp_out: bool = False
    cycle_v2_ramp_out_half_width: float = 0.90
    cycle_v2_ramp_out_bonus: float = 0.0
    cycle_v2_off_ramp_exit_penalty: float = 0.0
    # V5 closes the ramp-crest distribution gap: a POSTDUMP drill continues
    # through COLLECT and terminates only after a real ramp exit plus target
    # load.  V4 stopped at LEFT_HOME, leaving the deterministic actor with no
    # training data for clearing the ramp.
    cycle_v2_postdump_require_target_load: bool = False
    # Bridge curriculum: POSTDUMP remains active through collect, return, and
    # the next qualified dump instead of terminating at the collection gate.
    cycle_v2_postdump_complete_cycle: bool = False
    # Optional realistic suffix reset: hide part of the neutral field to mimic
    # balls consumed by the frozen first trip.  A probability below one keeps
    # easier bridge examples in the same curriculum.
    cycle_v2_postdump_depleted_count: int = 0
    cycle_v2_postdump_depleted_prob: float = 0.0
    # V8 score-efficiency shaping.  ``cycle_v2_target_load`` remains the small
    # minimum that unlocks RETURN, while this separate target rewards bringing
    # back a fuller load.  Neither bonus can prevent an early return or score.
    cycle_v2_preferred_repeat_load: int = 0
    # V9 opt-in state-machine behavior. Historical V8 checkpoints used the
    # preferred load for reward shaping only and must still return immediately
    # at target_load when replayed.
    cycle_v2_collect_until_preferred: bool = False
    # When the V9 switch is active, COLLECT/intake stays active after the
    # minimum until the preferred load, a no-pickup fallback, or clock guard.
    cycle_v2_collect_stall_steps: int = 0
    cycle_v2_return_time_guard: float = 0.0
    # V10 preserves the champion's immediate COLLECT->RETURN transition at the
    # minimum target, but keeps the intake running while the unchanged RETURN
    # route drives through loose balls.  This is deliberately independent of
    # ``cycle_v2_collect_until_preferred`` so V8/V9 replay stays exact.
    cycle_v2_intake_during_return: bool = False
    cycle_v2_repeat_load_return_bonus: float = 0.0
    cycle_v2_repeat_load_score_bonus: float = 0.0
    cycle_v2_skill_terminate: bool = True
    # A repeat dump is promoted only when most of the qualified load lands.
    # This prevents a 2/15 spray from being labelled a completed cycle.
    cycle_v2_score_fraction: float = 0.75
    cycle_v2_score_floor: int = 6
    # State-specific delay costs.  One policy step is 0.1 s.  These budgets
    # reset on each scoring dump; unlike the old blanket "loafing" nag, every
    # cost switches off at its own milestone.
    cycle_v2_leave_grace_steps: int = 5
    cycle_v2_leave_penalty_per_step: float = 0.03
    cycle_v2_leave_penalty_cap: float = 5.0
    cycle_v2_return_grace_steps: int = 10
    cycle_v2_return_penalty_per_step: float = 0.02
    cycle_v2_return_penalty_cap: float = 5.0
    cycle_v2_shoot_grace_steps: int = 20
    cycle_v2_shoot_penalty_per_step: float = 0.05
    cycle_v2_shoot_penalty_cap: float = 5.0
    # A legal press owns the whole volley.  Brief aim loss is tolerated while
    # auto-align recovers; a timed-out/aborted partial dump is explicitly bad.
    cycle_v2_dump_lost_aim_grace_ticks: int = 15
    cycle_v2_partial_dump_penalty_per_ball: float = 0.5
    cycle_v2_partial_dump_penalty_cap: float = 15.0
    # Anti-camping.  Penalize CONSECUTIVE dwelling within a band of the home/away
    # borderline (y ~ midpoint of home_enter/away_enter, -2.775).  A normal
    # crossing passes through in well under the grace window and pays nothing; only
    # *staying* at the edge instead of committing to deep-collect or deep-score is
    # charged.  per_step=0 disables it.  Budget resets on each scoring dump.
    cycle_v2_border_y: float = -2.65
    cycle_v2_border_band: float = 0.35
    cycle_v2_border_grace_steps: int = 20
    cycle_v2_border_penalty_per_step: float = 0.0
    cycle_v2_border_penalty_cap: float = 12.0
    # Later-cycle route efficiency.  The chosen ramp side is refreshed after
    # every verified repeat dump so cycle 3 is not pulled toward the lane that
    # happened to be nearest after cycle 1.  This is opt-in for checkpoint
    # reproducibility: historical Stage C v2 checkpoints keep the old frozen
    # side unless their metadata explicitly enables the revision.
    cycle_v2_refresh_ramp_side_on_dump: bool = False
    cycle_v2_ramp_side_deadband_x: float = 0.25
    # Penalize only sustained dwelling at either outer rail during the
    # LEAVE/COLLECT/RETURN legs of cycles 2+.  A clean trench crossing is
    # protected by the grace window, SCORE and FIRST_CYCLE are never charged,
    # and the bounded depth-scaled cost cannot dominate one scored ball.
    cycle_v2_outer_rail_enter_x: float = 2.85
    cycle_v2_outer_rail_exit_x: float = 2.55
    cycle_v2_outer_rail_max_x: float = 3.60
    cycle_v2_outer_rail_grace_steps: int = 20
    cycle_v2_outer_rail_penalty_per_step: float = 0.0
    cycle_v2_outer_rail_penalty_cap: float = 8.0
    # V2 closes the old threshold loophole: once the grace window expires,
    # even a robot barely outside enter_x pays a meaningful minimum charge.
    # Continued dwelling then ramps the charge up to max_multiplier.
    cycle_v2_outer_rail_min_scale: float = 0.0
    cycle_v2_outer_rail_escalation_steps: int = 0
    cycle_v2_outer_rail_max_multiplier: float = 1.0
    # Repeat the unchanged 30 Hz intake/controller capture path within each
    # control tick.  One preserves all historical environments; two is the
    # opt-in fast-intake mechanics used by the new training branch.
    cycle_v2_intake_substeps: int = 1
    # ---- Stage D: official 160 s match rules (opt-in; OFF = Stage C exact) ----
    # When stage_d is False every branch below is dead code and Stage C behavior
    # is byte-identical.  When True the HubRouter leaves sandbox mode, the
    # official shift schedule gates score eligibility at sensor time
    # (rules.fuel_score_is_eligible incl. the 3 s assessment grace, wired via
    # HubRouter._score_eligible), and proprio idx 12 reports real blue-hub
    # eligibility instead of the Stage A/B/C constant 1.0.  Pair with
    # episode_len_s=160 and the 456-ball template for the full Stage D contract.
    stage_d: bool = False
    # SHIFT 1 inactive alliance: "red"/"blue" fixed parity (curriculum D1a;
    # "red" keeps the blue hub active 0-55 s, matching the champion's first-dump
    # timing), "random" seeded 50/50 per episode (D1b), "rules" = official
    # select_first_inactive_alliance from router AUTO counts plus a synthetic
    # red AUTO draw (D2/eval).
    stage_d_first_inactive: str = "red"
    # Inclusive range for the synthetic red-alliance AUTO fuel count used by
    # the "rules" mode (the single-robot sim has no red robot; (0, 0) means a
    # 0-0 AUTO tie decided by the official seeded coin).
    stage_d_synthetic_red_auto: tuple[int, int] = (0, 0)
    # D2: official 8-ball robot preload at the accepted trench start.
    stage_d_preload: bool = False
    # Mask the SUFFIX's fire press while the blue hub cannot score (Stage-B
    # illegal-fire-mask precedent: an ineligible press is a no-op that keeps
    # the chassis driving, so wasting a full load into a dead hub is physically
    # impossible).  The FROZEN PREFIX (pre-latch FIRST_CYCLE) is exempt:
    # masking would deadlock its baked dump timing and the episode would never
    # hand over to the suffix.
    stage_d_mask_ineligible_fire: bool = True
    # Do not charge the SCORE-phase delay penalty while the hub is ineligible:
    # holding a full load at the hub waiting for reactivation is correct Stage D
    # play, not loafing.
    stage_d_pause_shoot_penalty_when_ineligible: bool = True
    # STAGE-D1B: FERRY repatriation.  Both default OFF so stage_d (D1) WITHOUT
    # --stage-d-ferry stays byte-identical to the current run; only D1b flips them.
    stage_d_ferry: bool = False
    stage_d_ferry_reward: float = 1.0
    # STAGE-D1F: FERRY-DUMP-ON-PRESS -- one blackout ferry press commits the
    # robot to empty the whole magazine home in one action (reuses the score
    # dump machinery). Default OFF (byte-identical).
    stage_d_ferry_dump_on_press: bool = False
    # STAGE-D1F: credit/count ONLY entitled (qualified_ids) balls on a ferry so a
    # full-magazine ferry-dump does not pay for unqualified clutter. OFF => the
    # old qualification-blind credit (byte-identical).
    stage_d_ferry_entitled_only: bool = False
    # STAGE-D1G: ferry is BLACKOUT-ONLY. A ferry press while the blue hub is
    # LIVE is a no-op that keeps the chassis driving (the exact mirror of
    # stage_d_mask_ineligible_fire, which no-ops SHOOT while the hub is dark).
    # Together they force the strategy right-side-up by mechanics: live window
    # -> the only fire is a scoring shot; blackout -> the only fire is a ferry.
    # An already-committed ferry dump rides to completion. OFF = byte-identical.
    stage_d_ferry_blackout_only: bool = False
    # stage_d_v1 wave-2: a suffix ferry press with fewer than this many balls in
    # the chamber is a no-op (chassis keeps driving).  Blue2 telemetry showed
    # 4.4 balls per committed ferry volley -- spam-pressing an almost-empty
    # chamber wastes the blackout on aim/commit overhead instead of banking
    # 30-40-ball volleys.  0 = off (byte-identical).  Suffix-only; the frozen
    # prefix's baked flings stay exempt (D1G lesson: masking them diverges
    # FIRST_CYCLE into zero-score episodes).
    stage_d_ferry_min_load: int = 0
    # STAGE-D1E: penalty per ACTIVE-window ferry press (a fire that leaked out
    # as a ferry while the blue hub was LIVE -- the robot should shoot or
    # reposition, not fling fuel one-way for nothing).  0.0 = OFF (byte-identical).
    stage_d_active_ferry_penalty: float = 0.0
    # STAGE-D2 (2026-07-26): heavy charge for sitting in RED's OWN court, i.e.
    # past the red trench/ramp ring.  Blue's home begins at y <= -3.05 (its own
    # ring sits at y=-3.269); the mirror y >= +3.05 is past red's ring.  The
    # whole neutral collecting band (-2.50 .. +3.05) is deliberately left free,
    # so the COLLECT phase is untouched and only over-extension is charged.
    stage_d_deep_red_penalty: float = 0.0
    stage_d_deep_red_y: float = 3.05
    stage_d_deep_red_grace_steps: int = 5
    stage_d_deep_red_penalty_cap: float = 60.0
    # STAGE-D2: per-step charge for standing still -- the freeze/camp failure
    # mode.  Suppressed in SCORE because a dump legitimately stops the chassis.
    stage_d_idle_penalty: float = 0.0
    stage_d_idle_speed_mps: float = 0.15
    stage_d_idle_grace_steps: int = 20
    stage_d_idle_penalty_cap: float = 40.0
    # STAGE-D1E: return-to-score as soon as target_load is met while the blue
    # hub is LIVE (breaks the collect-and-ferry dwell that idled the back-half
    # active windows A2/A3).  Default OFF (byte-identical).
    stage_d_return_when_live: bool = False
    # stage_d_v1 ferry-first: minimum qualified load before a LIVE hub pulls
    # COLLECT home (passed to CycleV2Config.live_return_load).  0 keeps the
    # original return-when-live exit-at-target behavior.  Only meaningful with
    # stage_d_return_when_live.
    stage_d_live_return_load: int = 0
    # stage_d_v1 wave-4: schedule-aware RETURN LEAD.  During a blackout, once
    # reactivation is within this many seconds AND the qualified load has
    # reached live_return_load, COLLECT ends and the robot returns home -- so
    # a nearly-full chamber is AT the hub when the lights come back instead of
    # starting its commute after the edge.  Implemented as "hub eligible at
    # (clock + lead)"; 0 = off (byte-identical).  Requires return_when_live.
    stage_d_return_lead_s: float = 0.0
    # STAGE-D1C: OWN-COURT SHORT LOOP.  When the blue hub reactivates and
    # ferried fuel is sitting in our own court, the efficient play is to intake
    # those local balls and shoot them WITHOUT a cross-field leave.  This flag
    # (1) lets a home load re-enter SCORE from LEAVE/COLLECT (cycle_v2's
    # ``owncourt_score_ready`` gate) and (2) suppresses the LEAVE delay penalty
    # while >= ``stage_d_owncourt_min_balls`` collectable balls remain in own
    # court and the hub is live.  Default OFF => byte-identical to D1/D1b; only
    # D1c flips it.  Requires stage_d and stage_d_ferry.
    stage_d_owncourt_loop: bool = False
    stage_d_owncourt_min_balls: int = 2
    # STAGE-D1F: reward pulling loose ENTITLED balls back into the magazine while
    # the own-court loop is ready (so qualified_load>0 triggers the SCORE
    # re-entry). Per entitled ball newly in-magazine. 0.0 = OFF.
    stage_d_owncourt_intake_reward: float = 0.0
    # STAGE-D1F: after an own-court dump completes with stockpile remaining and
    # the hub live, drop back to COLLECT to re-arm the short loop (convert the
    # whole stockpile, not ~1.4 balls). Default OFF (byte-identical).
    stage_d_owncourt_rearm: bool = False
    # Own-court boundary on the position axis: balls with y <= this value are
    # past the |y|=2.775 scoring board, i.e. inside the blue robot's own court
    # (matches the empty_own_court_penalty y<-2.775 test).
    stage_d_owncourt_board_y: float = -2.775
    # stage_d_v1 ferry-first: keep the own-court loop armed during a BLACKOUT
    # so a home robot can pre-load the ferried stockpile and hold at the hub
    # for the reactivation edge (the ineligible-fire mask still prevents a
    # dark dump; the SCORE delay penalty is paused while ineligible).  Also
    # extends the LEAVE-penalty stockpile suppression to blackouts.  Default
    # OFF (byte-identical).  Requires stage_d_owncourt_loop.
    stage_d_owncourt_blackout_intake: bool = False
    # stage_d_v1 wave-3 "bank" TIME-SLICED lane: episodes START mid-match just
    # before a blue blackout -- latched, post-dump pose in own court, match
    # clock preset -- so collectors drill the blackout->bank->reactivation-
    # convert arc every ~58 s instead of once per 160 s full episode.  The obs
    # keep FULL-match normalization (episode_len_s stays the real 160); the
    # lane terminates stage_d_bank_span_s after its start.  The clock start
    # draws 50/50 from (a, b); the (b) start also seeds an entitled own-court
    # stockpile so conversion is practiced from step 0.  The parity decision is
    # pre-made at reset (the t=23 write-once site skips itself).
    stage_d_bank_clock_a: float = 30.0
    stage_d_bank_clock_b: float = 80.0
    stage_d_bank_span_s: float = 52.0
    stage_d_bank_stockpile: int = 8
    stage_d_bank_fuel_jitter_m: float = 0.0
    # Fraction of neutral-zone FUEL relocated into messy edge/corner clumps.
    stage_d_bank_fuel_scatter: float = 0.0
    stage_d_bank_fuel_clumps: int = 6
    # Success gate for the blackout lane: come home with a FULL chamber.
    stage_d_bank_success_chamber: int = 0
    # SECTION 1 (user plan 2026-07-26): the opener lane.  A REAL match start
    # (t=0, normal spawn, protected first cycle) simply cut short -- nothing
    # about the opening is synthetic, only the horizon changes.  Succeeds on
    # stage_d_opener_success_cycles complete cycles inside the window.
    stage_d_opener_span_s: float = 30.0
    stage_d_opener_success_cycles: int = 2
    # SECTION 3 (user plan 2026-07-26): the live-window lane.  Starts at a hub
    # reactivation (55 s / 105 s) HOME + CHAMBERED with the ferried stockpile
    # already on the floor, so the drill is: dump now -> collect local -> dump
    # again -> only then go back out for a full cycle.
    stage_d_live_clock_a: float = 55.0
    stage_d_live_clock_b: float = 105.0
    stage_d_live_clock_c: float = 130.0
    stage_d_live_span_s: float = 40.0
    stage_d_live_stockpile: int = 16
    stage_d_live_success_conversions: int = 6
    # Hypothetical chamber load the live window opens with (section 2 is meant
    # to deliver this for real once trained; until then it is assumed).
    stage_d_live_chamber: int = 30
    # One-shot reward for starting a score dump with a real load while the hub
    # is live (section 3: makes the press discoverable).
    stage_d_live_dump_reward: float = 0.0
    stage_d_live_dump_min_load: int = 8
    # Per-step time cost inside the opener window (section 1 time pressure).
    stage_d_opener_time_penalty: float = 0.0
    # Section 2's objective inside the full match: pay for arriving HOME with a
    # real load while our hub is dark, scaled by the load actually carried.
    # Paid at most once per blackout so it cannot be farmed by hovering.
    stage_d_home_arrival_reward: float = 0.0
    stage_d_home_arrival_min_load: int = 6
    # Postdump lane clock starts under stage_d: the two hub reactivations.
    # Stall-rescue: hand control to the suffix if the frozen prefix has not
    # finished its first dump by this clock (0 disables).
    stage_d_prefix_rescue_s: float = 0.0
    stage_d_postdump_clock_a: float = 57.0
    stage_d_postdump_clock_b: float = 110.0
    # Bank lane success = this many ENTITLED own-court balls converted to legal
    # score (0 = off, span timer only -- the old behavior that taught ferrying
    # and nothing after it).  Termination/success only; no scripted actions.
    stage_d_bank_success_conversions: int = 0
    # stage_d_v1 wave-5 "COMMAND THE STRATEGY" (audit 2026-07-24 §4): the FSM
    # already commands returns (return_when_live/lead), dumps (dump_on_press)
    # and legality (masks); these commands close the asymmetry for the two
    # behaviors reward-discovery never scaled -- the blackout ferry press and
    # the own-court conversion.  All default OFF (byte-identical).
    #
    # auto-ferry: latched suffix, during a blackout, NOT inside the pre-edge
    # hold window, >= auto_ferry_load balls, valid ferry solution -> commit a
    # full ferry volley as if the policy pressed.
    stage_d_auto_ferry_load: int = 0
    stage_d_auto_ferry_hold_s: float = 12.0
    # auto own-court intake: while the own-court loop is ready and the robot
    # is HOME, force intake ON so the stockpile actually enters the chamber.
    stage_d_auto_oc_intake: bool = False
    # auto score press: in SCORE at an ELIGIBLE hub with valid aim and a
    # non-empty chamber, commit the dump without requiring the policy press.
    stage_d_auto_score_press: bool = False
    # "LAND THE BANK ON THE RETURN PATH": aim ferry volleys at the ramp/hub
    # approach corridor (|x| = lane, |y| = target row) instead of the far
    # corners, so returning trips drive through the banked balls with the
    # auto own-court intake running.  0 = historical solver behavior.
    stage_d_ferry_target_y: float = 0.0
    stage_d_ferry_lane_x: float = 0.0


@dataclass
class EnvSlot:
    """Everything owned by one cloned environment."""

    index: int
    origin: np.ndarray
    controller: Any
    router: Any
    articulation: Any
    fuel: Any
    clock_s: float = 0.0
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(7, np.float32))
    score_seen: int = 0
    collected_seen: int = 0
    custody: Any = None    # CustodyState (set in _reset_slot); per-ball score/collect ledger
    episode_neutral_loaded: bool = False   # this episode used the neutral-loaded return curriculum
    reward_components: dict[str, float] = field(default_factory=dict)
    cycle_v2: Any = None
    cycle_v2_mode: str = "legacy"
    cycle_v2_ramp_side: float = 1.0
    cycle_v2_reserved_ids: set[int] = field(default_factory=set)
    cycle_v2_reserved_batches: list[list[int]] = field(default_factory=list)
    cycle_v2_return_preload_count: int = 0
    cycle_v2_progress_phase: Any = None
    cycle_v2_best_distance: float | None = None
    cycle_v2_leave_corridor_entered: bool = False
    cycle_v2_stats: dict[str, Any] = field(default_factory=dict)
    cycle_v2_postdump_depleted_count: int = 0
    cycle_v2_return_loads: dict[int, int] = field(default_factory=dict)
    cycle_v2_terminal_reason: str = ""
    cycle_v2_penalty_spent: dict[str, float] = field(default_factory=dict)
    cycle_v2_phase_enter_step: int = 0
    cycle_v2_border_steps: int = 0
    cycle_v2_border_spent: float = 0.0
    cycle_v2_outer_rail_active: bool = False
    cycle_v2_outer_rail_streak: int = 0
    cycle_v2_outer_rail_spent: float = 0.0
    stage_d_deep_red_spent: float = 0.0
    stage_d_deep_red_streak: int = 0
    stage_d_idle_spent: float = 0.0
    stage_d_idle_streak: int = 0
    stage_d_live_dump_paid: bool = False
    stage_d_rescued: bool = False
    stage_d_arrival_paid: bool = False
    dump_started_this_step: bool = False
    dump_completed_this_step: bool = False
    dump_aborted_this_step: bool = False
    dump_start_ids: tuple[int, ...] = ()
    dump_start_mode: str | None = None
    dump_remaining_count: int = 0
    dump_lost_aim_ticks: int = 0
    # Stage D per-episode match context (None until the t=23 s AUTO decision).
    stage_d_first_inactive: Any = None
    stage_d_episode_seed: int = 0
    stage_d_masked_fires: int = 0


# --------------------------------------------------------------------------- #
# single-env adapters over the batched views
# --------------------------------------------------------------------------- #
class _GainShim:
    """`get_articulation_controller()` facade: set_gains / set_max_efforts."""

    def __init__(self, batched: Any, env_index: int):
        self._b = batched
        self._idx = [env_index]

    def set_gains(self, kps=None, kds=None) -> None:
        kps = None if kps is None else np.asarray(kps, np.float32)[None, :]
        kds = None if kds is None else np.asarray(kds, np.float32)[None, :]
        self._b.set_gains(kps=kps, kds=kds, indices=self._idx)

    def set_max_efforts(self, efforts) -> None:
        self._b.set_max_efforts(
            np.asarray(efforts, np.float32)[None, :], indices=self._idx
        )


class EnvArticulationAdapter:
    """Single-robot facade over the batched Articulation for one env.

    World poses are shifted by the env origin so the controller keeps working
    in field coordinates (hub targets, zones, trench boxes).
    """

    def __init__(self, batched: Any, env_index: int, origin: np.ndarray):
        self._b = batched
        self._i = env_index
        self._idx = [env_index]
        self._origin = origin.astype(np.float32)

    # -- identity ---------------------------------------------------------
    @property
    def dof_names(self):
        return self._b.dof_names

    def get_articulation_controller(self) -> _GainShim:
        return _GainShim(self._b, self._i)

    def set_max_efforts(self, efforts) -> None:
        self._b.set_max_efforts(
            np.asarray(efforts, np.float32)[None, :], indices=self._idx
        )

    # -- joints -----------------------------------------------------------
    @staticmethod
    def _np(value: Any) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    def get_joint_positions(self) -> np.ndarray:
        return self._np(self._b.get_joint_positions(indices=self._idx))[0]

    def set_joint_positions(self, positions) -> None:
        self._b.set_joint_positions(
            np.asarray(positions, np.float32)[None, :], indices=self._idx
        )

    def get_joint_velocities(self) -> np.ndarray:
        return self._np(self._b.get_joint_velocities(indices=self._idx))[0]

    def set_joint_velocities(self, velocities) -> None:
        self._b.set_joint_velocities(
            np.asarray(velocities, np.float32)[None, :], indices=self._idx
        )

    def apply_action(self, action: Any) -> None:
        """NaN-skip semantics identical to SingleArticulation.apply_action."""
        jp = getattr(action, "joint_positions", None)
        jv = getattr(action, "joint_velocities", None)
        if jp is not None:
            jp = np.asarray(jp, np.float32)
            live = np.flatnonzero(~np.isnan(jp))
            if live.size:
                self._b.set_joint_position_targets(
                    jp[live][None, :], indices=self._idx, joint_indices=live
                )
        if jv is not None:
            jv = np.asarray(jv, np.float32)
            live = np.flatnonzero(~np.isnan(jv))
            if live.size:
                self._b.set_joint_velocity_targets(
                    jv[live][None, :], indices=self._idx, joint_indices=live
                )

    # -- chassis ----------------------------------------------------------
    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self._b.get_world_poses(indices=self._idx)
        position = self._np(positions)[0].astype(np.float32) - self._origin
        return position, self._np(orientations)[0].astype(np.float32)

    def set_world_pose(self, position, orientation) -> None:
        self._b.set_world_poses(
            positions=(np.asarray(position, np.float32) + self._origin)[None, :],
            orientations=np.asarray(orientation, np.float32)[None, :],
            indices=self._idx,
        )

    def get_linear_velocity(self) -> np.ndarray:
        return self._np(self._b.get_linear_velocities(indices=self._idx))[0]

    def get_angular_velocity(self) -> np.ndarray:
        return self._np(self._b.get_angular_velocities(indices=self._idx))[0]

    def set_velocities_zero(self) -> None:
        zeros = np.zeros((1, 6), np.float32)
        try:
            self._b.set_velocities(zeros, indices=self._idx)
        except (AttributeError, TypeError):
            self._b.set_linear_velocities(zeros[:, :3], indices=self._idx)
            self._b.set_angular_velocities(zeros[:, 3:], indices=self._idx)


class EnvFuelAdapter:
    """Per-env slice of the global FUEL RigidPrim, origin-shifted.

    Exposes exactly the surface the controller + HubRouter consume:
    count, get/set_world_poses, get/set linear/angular velocities, with local
    ball indices 0..count-1.
    """

    def __init__(self, batched: Any, env_index: int, fuel_per_env: int, origin: np.ndarray):
        self._b = batched
        self.count = int(fuel_per_env)
        self._base = env_index * self.count
        self._all = np.arange(self._base, self._base + self.count, dtype=np.int32)
        self._origin = origin.astype(np.float32)

    @staticmethod
    def _np(value: Any) -> np.ndarray:
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    def _global(self, indices) -> np.ndarray:
        return np.asarray(indices, dtype=np.int32) + self._base

    def get_world_poses(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self._b.get_world_poses(indices=self._all)
        return (
            self._np(positions).astype(np.float32) - self._origin[None, :],
            self._np(orientations).astype(np.float32),
        )

    def set_world_poses(self, positions=None, indices=None, orientations=None) -> None:
        shifted = np.asarray(positions, np.float32) + self._origin[None, :]
        self._b.set_world_poses(
            positions=shifted,
            orientations=orientations,
            indices=self._global(indices if indices is not None else range(self.count)),
        )

    def get_linear_velocities(self) -> np.ndarray:
        return self._np(self._b.get_linear_velocities(indices=self._all)).astype(
            np.float32
        )

    def set_linear_velocities(self, velocities, indices=None) -> None:
        self._b.set_linear_velocities(
            np.asarray(velocities, np.float32),
            indices=self._global(indices if indices is not None else range(self.count)),
        )

    def set_angular_velocities(self, velocities, indices=None) -> None:
        self._b.set_angular_velocities(
            np.asarray(velocities, np.float32),
            indices=self._global(indices if indices is not None else range(self.count)),
        )


_TRENCH_CLEAR_MARGIN_M = 0.60  # body buffer so the full 0.74 m extended envelope clears the roof


def _under_trench_roof(controller) -> bool:
    """True while the chassis is still beneath the blue trench roof, where the
    0.74 m extended storage cannot clear the 0.565 m opening. The storage action
    is untrained in stages A/B (locked), so its Stage-C output is arbitrary; this
    mirrors the physical trench roof, which cannot be deployed under (audit)."""
    from frc_rebuilt.competition_robot import (
        BLUE_TRENCH_NEUTRAL_EDGE_Y_M,
        BLUE_TRENCH_CLEAR_X_MIN_M,
        BLUE_TRENCH_CLEAR_X_MAX_M,
    )
    pos, _ = controller.chassis_pose()
    x, y = float(pos[0]), float(pos[1])
    return (
        y <= BLUE_TRENCH_NEUTRAL_EDGE_Y_M + _TRENCH_CLEAR_MARGIN_M
        and BLUE_TRENCH_CLEAR_X_MIN_M - 0.30 <= x <= BLUE_TRENCH_CLEAR_X_MAX_M + 0.30
    )


# --------------------------------------------------------------------------- #
# the vectorized environment
# --------------------------------------------------------------------------- #
class VecCompetitionEnv:
    """N cloned full-physics fields with the real robot stack per env."""

    def __init__(self, cfg: VecEnvCfg):
        self.cfg = cfg
        self._validate_cycle_v2_cfg()
        self.spec = CompetitionRLSpec()
        self.spec.validate()
        self.rng = np.random.default_rng(cfg.seed)
        # live-annealable collection weight (trainer lowers it during stage B)
        self.collect_weight = float(cfg.collect_reward_weight)
        self._build_scene()
        self._build_views()
        self._build_cameras()
        self.reset_all()

    def _validate_cycle_v2_cfg(self) -> None:
        """Fail fast on mixed legacy/v2 contracts before Isaac is constructed."""

        cfg = self.cfg
        if cfg.stage_d:
            if not cfg.stagec_v2:
                raise ValueError("stage_d requires stagec_v2")
            if str(cfg.stage_d_first_inactive) not in _stage_d.FIRST_INACTIVE_MODES:
                raise ValueError(
                    "stage_d_first_inactive must be one of "
                    f"{_stage_d.FIRST_INACTIVE_MODES}, got "
                    f"{cfg.stage_d_first_inactive!r}"
                )
            lo, hi = cfg.stage_d_synthetic_red_auto
            if int(lo) < 0 or int(hi) < int(lo):
                raise ValueError(
                    f"invalid stage_d_synthetic_red_auto range: {(lo, hi)!r}"
                )
            for _n in (
                "stage_d_deep_red_penalty",
                "stage_d_deep_red_penalty_cap",
                "stage_d_idle_penalty",
                "stage_d_idle_penalty_cap",
                "stage_d_idle_speed_mps",
            ):
                if float(getattr(cfg, _n)) < 0.0:
                    raise ValueError(f"{_n} must be >= 0, got {getattr(cfg, _n)!r}")
            for _n in ("stage_d_deep_red_grace_steps", "stage_d_idle_grace_steps"):
                if int(getattr(cfg, _n)) < 0:
                    raise ValueError(f"{_n} must be non-negative")
            if float(cfg.stage_d_ferry_reward) < 0.0:  # STAGE-D1B
                raise ValueError(
                    "stage_d_ferry_reward must be >= 0, got "
                    f"{cfg.stage_d_ferry_reward!r}"
                )
            if float(cfg.stage_d_active_ferry_penalty) < 0.0:  # STAGE-D1E
                raise ValueError(
                    "stage_d_active_ferry_penalty must be >= 0, got "
                    f"{cfg.stage_d_active_ferry_penalty!r}"
                )
            if cfg.stage_d_owncourt_loop:  # STAGE-D1C
                if not cfg.stage_d_ferry:
                    raise ValueError(
                        "stage_d_owncourt_loop requires stage_d_ferry"
                    )
                if int(cfg.stage_d_owncourt_min_balls) < 1:
                    raise ValueError(
                        "stage_d_owncourt_min_balls must be >= 1, got "
                        f"{cfg.stage_d_owncourt_min_balls!r}"
                    )
                if float(cfg.stage_d_owncourt_intake_reward) < 0.0:  # STAGE-D1F
                    raise ValueError(
                        "stage_d_owncourt_intake_reward must be >= 0"
                    )
            if (  # STAGE-D1F / STAGE-D1G
                cfg.stage_d_ferry_dump_on_press
                or cfg.stage_d_ferry_entitled_only
                or cfg.stage_d_ferry_blackout_only
            ) and not cfg.stage_d_ferry:
                raise ValueError(
                    "stage_d_ferry_dump_on_press/entitled_only/blackout_only "
                    "require stage_d_ferry"
                )
            if int(cfg.stage_d_ferry_min_load) < 0:  # stage_d_v1 wave-2
                raise ValueError(
                    "stage_d_ferry_min_load must be >= 0, got "
                    f"{cfg.stage_d_ferry_min_load!r}"
                )
            if int(cfg.stage_d_ferry_min_load) > 0 and not cfg.stage_d_ferry:
                raise ValueError("stage_d_ferry_min_load requires stage_d_ferry")
            if int(cfg.stage_d_auto_ferry_load) < 0:  # wave-5 commands
                raise ValueError("stage_d_auto_ferry_load must be >= 0")
            if int(cfg.stage_d_auto_ferry_load) > 0 and not (
                cfg.stage_d_ferry and cfg.stage_d_ferry_blackout_only
            ):
                raise ValueError(
                    "stage_d_auto_ferry_load requires stage_d_ferry + "
                    "stage_d_ferry_blackout_only"
                )
            if float(cfg.stage_d_auto_ferry_hold_s) < 0.0:
                raise ValueError("stage_d_auto_ferry_hold_s must be >= 0")
            if cfg.stage_d_auto_oc_intake and not cfg.stage_d_owncourt_loop:
                raise ValueError(
                    "stage_d_auto_oc_intake requires stage_d_owncourt_loop"
                )
            if float(cfg.stage_d_ferry_target_y) < 0.0 or float(
                cfg.stage_d_ferry_lane_x
            ) < 0.0:
                raise ValueError(
                    "stage_d_ferry_target_y/lane_x must be >= 0"
                )
            if (
                float(cfg.stage_d_ferry_target_y) > 0.0
                or float(cfg.stage_d_ferry_lane_x) > 0.0
            ) and not cfg.stage_d_ferry:
                raise ValueError(
                    "stage_d ferry targeting requires stage_d_ferry"
                )
            if cfg.stage_d_owncourt_rearm and not cfg.stage_d_owncourt_loop:  # STAGE-D1F
                raise ValueError(
                    "stage_d_owncourt_rearm requires stage_d_owncourt_loop"
                )
            if (  # stage_d_v1 ferry-first
                cfg.stage_d_owncourt_blackout_intake
                and not cfg.stage_d_owncourt_loop
            ):
                raise ValueError(
                    "stage_d_owncourt_blackout_intake requires stage_d_owncourt_loop"
                )
            if int(cfg.stage_d_live_return_load) < 0:  # stage_d_v1 ferry-first
                raise ValueError(
                    "stage_d_live_return_load must be >= 0, got "
                    f"{cfg.stage_d_live_return_load!r}"
                )
            if int(cfg.stage_d_live_return_load) > 0 and not (
                cfg.stage_d_return_when_live
            ):
                raise ValueError(
                    "stage_d_live_return_load requires stage_d_return_when_live"
                )
            if float(cfg.stage_d_return_lead_s) < 0.0 or float(
                cfg.stage_d_return_lead_s
            ) > 30.0:  # stage_d_v1 wave-4
                raise ValueError(
                    "stage_d_return_lead_s must be in [0, 30], got "
                    f"{cfg.stage_d_return_lead_s!r}"
                )
            if float(cfg.stage_d_return_lead_s) > 0.0 and not (
                cfg.stage_d_return_when_live
            ):
                raise ValueError(
                    "stage_d_return_lead_s requires stage_d_return_when_live"
                )
        elif cfg.stage_d_ferry:  # STAGE-D1B
            raise ValueError("stage_d_ferry requires stage_d")
        elif cfg.stage_d_owncourt_loop:  # STAGE-D1C
            raise ValueError("stage_d_owncourt_loop requires stage_d")
        if not cfg.stagec_v2:
            return
        modes = tuple(cfg.cycle_v2_reset_modes) or ("full",)
        allowed = {"full", "postdump", "collect", "return", "bank", "opener", "live"}
        if "bank" in set(cfg.cycle_v2_reset_modes or ()):  # stage_d_v1 wave-3
            if not cfg.stage_d:
                raise ValueError("bank reset mode requires stage_d")
            if str(cfg.stage_d_first_inactive) not in ("red", "blue"):
                raise ValueError(
                    "bank reset mode requires a fixed stage_d_first_inactive "
                    f"parity (red|blue), got {cfg.stage_d_first_inactive!r}"
                )
            if not (0.0 < float(cfg.stage_d_bank_span_s) <= 160.0):
                raise ValueError("stage_d_bank_span_s must be in (0, 160]")
            for _bc in (cfg.stage_d_bank_clock_a, cfg.stage_d_bank_clock_b):
                if not (23.0 <= float(_bc) < 160.0):
                    raise ValueError(
                        "bank clock starts must be in [23, 160) (post-AUTO "
                        f"decision), got {_bc!r}"
                    )
            if int(cfg.stage_d_bank_stockpile) < 0:
                raise ValueError("stage_d_bank_stockpile must be >= 0")
        bad = sorted(set(modes) - allowed)
        if bad:
            raise ValueError(f"unknown Stage C v2 reset modes: {bad}")
        if len(modes) not in (1, int(cfg.num_envs)):
            raise ValueError(
                "cycle_v2_reset_modes must contain one mode or one per env slot"
            )
        if not (0 < int(cfg.cycle_v2_target_load) <= int(cfg.cycle_v2_chamber_capacity)):
            raise ValueError("invalid Stage C v2 target load/capacity")
        preferred_load = int(cfg.cycle_v2_preferred_repeat_load)
        if preferred_load:
            if not (
                int(cfg.cycle_v2_target_load)
                < preferred_load
                <= int(cfg.cycle_v2_chamber_capacity)
            ):
                raise ValueError(
                    "cycle_v2_preferred_repeat_load must be above target_load "
                    "and no greater than chamber capacity"
                )
            if bool(cfg.cycle_v2_collect_until_preferred) and (
                int(cfg.cycle_v2_collect_stall_steps) <= 0
                and float(cfg.cycle_v2_return_time_guard) <= 0.0
            ):
                raise ValueError(
                    "preferred repeat load requires a collection-stall or "
                    "time-to-return fallback"
                )
        if bool(cfg.cycle_v2_collect_until_preferred) and not preferred_load:
            raise ValueError(
                "cycle_v2_collect_until_preferred requires "
                "cycle_v2_preferred_repeat_load"
            )
        if (
            bool(cfg.cycle_v2_intake_during_return)
            and bool(cfg.cycle_v2_collect_until_preferred)
        ):
            raise ValueError(
                "return-intake and collect-until-preferred are mutually exclusive"
            )
        if not preferred_load and (
            float(cfg.cycle_v2_repeat_load_return_bonus) > 0.0
            or float(cfg.cycle_v2_repeat_load_score_bonus) > 0.0
        ):
            raise ValueError(
                "repeat-load bonuses require cycle_v2_preferred_repeat_load"
            )
        if int(cfg.cycle_v2_collect_stall_steps) < 0:
            raise ValueError("cycle_v2_collect_stall_steps cannot be negative")
        if not 0.0 <= float(cfg.cycle_v2_return_time_guard) <= 1.0:
            raise ValueError("cycle_v2_return_time_guard must be in [0, 1]")
        required_return_preload = min(
            int(cfg.cycle_v2_target_load), int(RETURN_SKILL_PRELOAD)
        )
        if int(cfg.cycle_v2_reserve_count) < required_return_preload:
            raise ValueError(
                "each reserve batch must cover the physical return-skill preload"
            )
        if int(cfg.cycle_v2_reserve_batches) < 1:
            raise ValueError("Stage C v2 needs at least one reserve batch")
        if int(cfg.cycle_v2_postdump_depleted_count) < 0:
            raise ValueError("postdump depleted count cannot be negative")
        if not (0.0 <= float(cfg.cycle_v2_postdump_depleted_prob) <= 1.0):
            raise ValueError("postdump depleted probability must be in [0, 1]")
        if (
            bool(cfg.cycle_v2_postdump_complete_cycle)
            and (
                not bool(cfg.cycle_v2_postdump_require_target_load)
                or not bool(cfg.cycle_v2_require_ramp_out)
            )
        ):
            raise ValueError(
                "postdump complete-cycle curriculum requires target-load and ramp-out gates"
            )
        if not (0.0 < float(cfg.cycle_v2_score_fraction) <= 1.0):
            raise ValueError("cycle_v2_score_fraction must be in (0, 1]")
        if int(cfg.cycle_v2_score_floor) < 1:
            raise ValueError("cycle_v2_score_floor must be positive")
        for name in (
            "cycle_v2_leave_grace_steps",
            "cycle_v2_return_grace_steps",
            "cycle_v2_shoot_grace_steps",
            "cycle_v2_dump_lost_aim_grace_ticks",
            "cycle_v2_outer_rail_grace_steps",
            "cycle_v2_outer_rail_escalation_steps",
        ):
            if int(getattr(cfg, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "cycle_v2_leave_penalty_per_step",
            "cycle_v2_leave_penalty_cap",
            "cycle_v2_return_penalty_per_step",
            "cycle_v2_return_penalty_cap",
            "cycle_v2_shoot_penalty_per_step",
            "cycle_v2_shoot_penalty_cap",
            "cycle_v2_partial_dump_penalty_per_ball",
            "cycle_v2_partial_dump_penalty_cap",
            "cycle_v2_outer_rail_penalty_per_step",
            "cycle_v2_outer_rail_penalty_cap",
            "cycle_v2_outer_rail_min_scale",
            "cycle_v2_outer_rail_max_multiplier",
            "cycle_v2_ramp_side_deadband_x",
            "cycle_v2_ramp_out_half_width",
            "cycle_v2_ramp_out_bonus",
            "cycle_v2_off_ramp_exit_penalty",
            "cycle_v2_repeat_load_return_bonus",
            "cycle_v2_repeat_load_score_bonus",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        outer_exit = float(cfg.cycle_v2_outer_rail_exit_x)
        outer_enter = float(cfg.cycle_v2_outer_rail_enter_x)
        outer_max = float(cfg.cycle_v2_outer_rail_max_x)
        if not (
            math.isfinite(outer_exit)
            and math.isfinite(outer_enter)
            and math.isfinite(outer_max)
            and 0.0 <= outer_exit < outer_enter < outer_max
        ):
            raise ValueError(
                "cycle_v2 outer-rail geometry must satisfy "
                "0 <= exit_x < enter_x < max_x"
            )
        if not 0.0 <= float(cfg.cycle_v2_outer_rail_min_scale) <= 1.0:
            raise ValueError("cycle_v2_outer_rail_min_scale must be in [0, 1]")
        if float(cfg.cycle_v2_outer_rail_max_multiplier) < 1.0:
            raise ValueError(
                "cycle_v2_outer_rail_max_multiplier must be at least 1"
            )
        if float(cfg.cycle_v2_ramp_out_half_width) <= 0.0:
            raise ValueError("cycle_v2_ramp_out_half_width must be positive")
        if not 1 <= int(cfg.cycle_v2_intake_substeps) <= 3:
            raise ValueError("cycle_v2_intake_substeps must be in [1, 3]")
        if cfg.neutral_refill_count or cfg.neutral_refill_prob or cfg.neutral_loaded_prob:
            raise ValueError("legacy neutral refill/loaded curricula cannot be mixed with Stage C v2")

    def _cycle_v2_mode(self, env_index: int) -> str:
        modes = tuple(self.cfg.cycle_v2_reset_modes) or ("full",)
        return modes[0] if len(modes) == 1 else modes[int(env_index)]

    # -- construction ------------------------------------------------------
    def _build_scene(self) -> None:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.cloner import GridCloner

        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        self.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateEnableGPUDynamicsAttr(True)
        physx.CreateBroadphaseTypeAttr("GPU")
        physx.CreateTimeStepsPerSecondAttr(60)
        physx.CreateGpuFoundLostPairsCapacityAttr(2097152)
        physx.CreateGpuTotalAggregatePairsCapacityAttr(2097152)
        physx.CreateGpuMaxRigidContactCountAttr(2097152)
        physx.CreateGpuMaxRigidPatchCountAttr(327680)
        UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(850)

        envs_root = "/World/envs"
        UsdGeom.Xform.Define(stage, envs_root)
        env0 = f"{envs_root}/env_0"
        UsdGeom.Xform.Define(stage, env0)
        stage.GetPrimAtPath(env0).GetReferences().AddReference(self.cfg.template_usd)

        cloner = GridCloner(spacing=self.cfg.spacing_m)
        cloner.define_base_env(envs_root)
        paths = cloner.generate_paths(f"{envs_root}/env", self.cfg.num_envs)
        positions = cloner.clone(
            source_prim_path=env0,
            prim_paths=paths,
            replicate_physics=True,
            base_env_path=envs_root,
        )
        self.env_origins = np.asarray(positions, dtype=np.float32)

        self.sim = SimulationContext(
            physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.0
        )
        self.sim.reset()

    def _build_views(self) -> None:
        from pxr import UsdPhysics
        from isaacsim.core.prims import Articulation, RigidPrim
        from frc_rebuilt.competition_robot import (
            CompetitionRobotController,
        )
        from frc_rebuilt.isaac_scene import HubRouter

        art_roots = [
            p.GetPath().pathString
            for p in self.stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        fuel_prims = sorted(
            p.GetPath().pathString
            for p in self.stage.Traverse()
            if p.GetName().startswith("Fuel_") and p.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        env0_fuel = [p for p in fuel_prims if "/env_0/" in p]
        self.fuel_per_env = len(env0_fuel)
        robot_expr = art_roots[0].replace("/env_0/", "/env_.*/")
        fuel_expr = env0_fuel[0].rsplit("/", 1)[0].replace("/env_0/", "/env_.*/") + "/Fuel_.*"

        self.robots = Articulation(robot_expr)
        self.robots.initialize()
        self.fuel = RigidPrim(fuel_expr)
        self.fuel.initialize()
        assert int(self.robots.count) == self.cfg.num_envs
        assert int(self.fuel.count) == self.cfg.num_envs * self.fuel_per_env

        # remember the template FUEL layout (field frame) for resets
        first = EnvFuelAdapter(self.fuel, 0, self.fuel_per_env, self.env_origins[0])
        self._fuel_home, self._fuel_home_quat = first.get_world_poses()

        env0_robot_root = art_roots[0]
        self.slots: list[EnvSlot] = []
        for i in range(self.cfg.num_envs):
            origin = self.env_origins[i]
            articulation = EnvArticulationAdapter(self.robots, i, origin)
            fuel_view = EnvFuelAdapter(self.fuel, i, self.fuel_per_env, origin)
            controller = CompetitionRobotController(
                alliance_lock="blue",
                usd_root_path=env0_robot_root.replace("/env_0/", f"/env_{i}/"),
            )
            controller.initialize(articulation)
            router = HubRouter(fuel_view, self.fuel_per_env, seed=self.cfg.seed + i)
            router.sandbox = bool(self.cfg.sandbox_scoring)
            if self.cfg.stage_d:
                # Official scoring: the router leaves sandbox mode.  Eligibility
                # then comes from rules.fuel_score_is_eligible (active phases +
                # the 3 s assessment grace) once the SHIFT 1 decision is written
                # to router.match_first_inactive in the step loop; before that
                # decision (AUTO/TRANSITION) both hubs are active.
                router.sandbox = False
            self.slots.append(
                EnvSlot(
                    index=i,
                    origin=origin,
                    controller=controller,
                    router=router,
                    articulation=articulation,
                    fuel=fuel_view,
                )
            )

    def _build_cameras(self) -> None:
        self.cameras: dict[tuple[int, str], Any] = {}
        if not self.cfg.cameras:
            return
        from isaacsim.sensors.camera import Camera
        from frc_rebuilt.competition_robot import (
            CAMERA_BASELINE_NAMES,
            CAMERA_PRIM_PATHS,
            CAMERA_RESOLUTION,
            ROBOT_ROOT_PATH,
        )

        names = self.cfg.camera_names or CAMERA_BASELINE_NAMES
        self.camera_names = tuple(names)
        self.camera_resolution = CAMERA_RESOLUTION
        for slot in self.slots:
            root = slot.controller.usd_root_path
            for name in names:
                path = CAMERA_PRIM_PATHS[name].replace(ROBOT_ROOT_PATH, root, 1)
                camera = Camera(prim_path=path, resolution=CAMERA_RESOLUTION)
                camera.initialize()
                self.cameras[(slot.index, name)] = camera
        # RTX render products publish allocated all-black buffers while they
        # start up asynchronously (same race the camera-preview tool fixed).
        # Gate on real content once, so the first policy observation is valid.
        # NOTE: headless Camera annotators are only fed by full Kit updates
        # (sim.step(render=True)); a bare sim.render() leaves them black.
        self._camera_ready = False
        stds: dict[tuple[int, str], float] = {}
        for _ in range(300):
            self.sim.step(render=True)
            stds = {
                key: (
                    float(np.asarray(camera.get_rgba())[..., :3].std())
                    if np.asarray(camera.get_rgba()).size
                    else 0.0
                )
                for key, camera in self.cameras.items()
            }
            if all(value > 1.0 for value in stds.values()):
                self._camera_ready = True
                break
        laggards = sorted(
            (key for key, value in stds.items() if value <= 1.0), key=str
        )
        print(
            f"VECENV_CAMERAS_READY {self._camera_ready}"
            + (f" laggards={laggards}" if laggards else ""),
            flush=True,
        )

    # -- physics health ----------------------------------------------------
    def _detect_unhealthy(self) -> np.ndarray:
        """Per-env: True where the robot/fuel state is non-finite or exploded.

        Uses only the controller/fuel APIs already exercised by _observe, so it
        adds no new Isaac coupling. Catches a solver blow-up while velocities are
        merely huge (hundreds of m/s), before they reach inf/NaN and segfault.
        """
        cfg = self.cfg
        bad = np.zeros(cfg.num_envs, dtype=bool)
        for slot in self.slots:
            i = slot.index
            try:
                pos, _ = slot.controller.chassis_pose()
                lin, yaw_rate = slot.controller.chassis_velocity()
                fv = np.asarray(slot.fuel.get_linear_velocities(), np.float32)
            except Exception:
                bad[i] = True
                continue
            pos = np.asarray(pos, np.float32).ravel()
            lin = np.asarray(lin, np.float32).ravel()
            yaw_rate = float(yaw_rate)
            finite = (
                np.isfinite(pos).all() and np.isfinite(lin).all()
                and math.isfinite(yaw_rate) and np.isfinite(fv).all()
            )
            if not finite:
                bad[i] = True
                continue
            if (
                float(np.abs(lin).max(initial=0.0)) > cfg.max_robot_linear_speed
                or abs(yaw_rate) > cfg.max_robot_angular_speed
                or float(np.abs(pos).max(initial=0.0)) > cfg.max_position_abs
                or (fv.size and float(np.abs(fv).max()) > cfg.max_fuel_speed)
            ):
                bad[i] = True
        return bad

    # -- reset -------------------------------------------------------------
    @staticmethod
    def _refill_grid(n: int, center, spread, z: float) -> np.ndarray:
        """Field-local (x,y,z) for n balls in an even grid around `center`, spanning
        center +/- spread. Grid spacing stays well above the 0.076 m ball radius so
        the relocated balls never overlap. Pure/deterministic -> unit-testable."""
        import math as _m
        cx, cy = float(center[0]), float(center[1])
        sx, sy = float(spread[0]), float(spread[1])
        cols = int(_m.ceil(_m.sqrt(max(1, n))))
        rows = int(_m.ceil(n / cols))
        out = np.zeros((int(n), 3), np.float32)
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= n:
                    break
                gx = cx + (sx * (2.0 * c / (cols - 1) - 1.0) if cols > 1 else 0.0)
                gy = cy + (sy * (2.0 * r / (rows - 1) - 1.0) if rows > 1 else 0.0)
                out[k] = (gx, gy, z)
                k += 1
        return out

    def _neutral_refill(self, positions: np.ndarray) -> np.ndarray:
        """Relocate the LAST neutral_refill_count balls to the shallow-neutral curriculum
        cluster (small per-ball jitter), leaving the rest at their template positions."""
        cfg = self.cfg
        n = min(int(cfg.neutral_refill_count), int(positions.shape[0]))
        if n <= 0:
            return positions
        positions = positions.copy()
        z = float(np.median(self._fuel_home[:, 2]))          # template rest height (floor)
        grid = self._refill_grid(n, cfg.neutral_refill_center, cfg.neutral_refill_spread, z)
        grid[:, :2] += self.rng.uniform(-0.12, 0.12, size=(n, 2)).astype(np.float32)
        positions[positions.shape[0] - n:] = grid
        return positions

    @staticmethod
    def _cycle_v2_holding_positions(indices: list[int]) -> np.ndarray:
        """Stable out-of-play parking slots, matching the hub-router hygiene pen."""

        out = np.zeros((len(indices), 3), np.float32)
        for row, ball_id in enumerate(indices):
            out[row] = (9.0 + float(ball_id) * 0.002, 0.0, -2.0)
        return out

    def _configure_cycle_v2_reserve(
        self, slot: EnvSlot, positions: np.ndarray
    ) -> np.ndarray:
        """Park suffix-skill reserves without changing a FULL prefix field.

        The frozen champion was trained on the template's complete 200-ball
        layout.  Removing even one nominal reserve batch before FIRST starves
        its learned neutral-zone route.  The native field also already supplies
        postdump/collect skill starts, so only RETURN parks a single batch from
        which its physical preload is drawn.
        """

        slot.cycle_v2_reserved_ids.clear()
        slot.cycle_v2_reserved_batches.clear()
        slot.cycle_v2_postdump_depleted_count = 0
        if not self.cfg.stagec_v2:
            return positions
        count = int(self.cfg.cycle_v2_reserve_count)
        batches = int(self.cfg.cycle_v2_reserve_batches)
        if (
            slot.cycle_v2_mode == "postdump"
            and int(self.cfg.cycle_v2_postdump_depleted_count) > 0
            and self.rng.random()
            < float(self.cfg.cycle_v2_postdump_depleted_prob)
        ):
            depleted = int(self.cfg.cycle_v2_postdump_depleted_count)
            home = np.asarray(self._fuel_home, np.float32)
            candidates = np.flatnonzero(
                (home[:, 1] >= -2.55)
                & (np.abs(home[:, 0]) <= 4.5)
                & (home[:, 2] >= -0.25)
            )
            if candidates.size < depleted:
                raise ValueError(
                    "postdump depletion exceeds available neutral-field FUEL"
                )
            selected = np.asarray(
                self.rng.choice(candidates, size=depleted, replace=False),
                dtype=np.int32,
            )
            ids = selected.tolist()
            slot.cycle_v2_reserved_ids.update(ids)
            slot.cycle_v2_postdump_depleted_count = len(ids)
            positions = positions.copy()
            positions[selected] = self._cycle_v2_holding_positions(ids)
            return positions
        if slot.cycle_v2_mode != "return":
            return positions
        # RETURN terminates after its score skill succeeds; it needs exactly one
        # preload source, never three parked/refill batches.
        batches = 1
        total = count
        if total > int(slot.fuel.count):
            raise ValueError(
                f"Stage C v2 reserve needs {total} FUEL, template has {slot.fuel.count}"
            )
        first = int(slot.fuel.count) - total
        all_ids = list(range(first, int(slot.fuel.count)))
        slot.cycle_v2_reserved_ids.update(all_ids)
        slot.cycle_v2_reserved_batches.extend(
            [all_ids[i : i + count] for i in range(0, len(all_ids), count)]
        )
        positions = positions.copy()
        positions[all_ids] = self._cycle_v2_holding_positions(all_ids)
        return positions

    def _pin_cycle_v2_reserve(self, slot: EnvSlot) -> None:
        """Keep parked dynamic bodies fixed and harmless between releases."""

        ids = sorted(slot.cycle_v2_reserved_ids)
        if not ids:
            return
        indices = np.asarray(ids, dtype=np.int32)
        positions = self._cycle_v2_holding_positions(ids)
        zeros = np.zeros((len(ids), 3), np.float32)
        slot.fuel.set_world_poses(positions=positions, indices=indices)
        slot.fuel.set_linear_velocities(zeros, indices=indices)
        slot.fuel.set_angular_velocities(zeros, indices=indices)

    def release_cycle_v2_reserve(self, env_index: int) -> tuple[int, ...]:
        """Restore a parked RETURN batch to its exact vacant template slots.

        RETURN can consume its one pre-parked batch.  Other modes use native
        field FUEL and therefore have no reserve batch to release.
        """

        slot = self.slots[int(env_index)]
        if slot.cycle_v2_reserved_batches:
            batch = list(slot.cycle_v2_reserved_batches.pop(0))
            if not set(batch).issubset(slot.cycle_v2_reserved_ids):
                raise RuntimeError("Stage C v2 reserve ownership is inconsistent")
            occupied = set(slot.controller.magazine) | set(slot.router.pending)
            if occupied.intersection(batch):
                raise RuntimeError("reserved FUEL overlaps magazine/router ownership")
            slot.cycle_v2_reserved_ids.difference_update(batch)
        else:
            return ()
        indices = np.asarray(batch, dtype=np.int32)
        zeros = np.zeros((len(batch), 3), np.float32)
        slot.fuel.set_world_poses(
            positions=self._fuel_home[indices].copy(), indices=indices
        )
        slot.fuel.set_linear_velocities(zeros, indices=indices)
        slot.fuel.set_angular_velocities(zeros, indices=indices)
        return tuple(batch)

    def _reset_slot(self, slot: EnvSlot) -> None:
        cfg = self.cfg
        v2_mode = self._cycle_v2_mode(slot.index) if cfg.stagec_v2 else "legacy"
        slot.cycle_v2_mode = v2_mode
        slot.cycle_v2_return_preload_count = 0
        # Do not consume a new environment RNG draw before the protected FULL
        # prefix (or in legacy mode).  Suffix-only skill streams can randomize
        # their lane immediately; FULL chooses the nearest ramp at first latch.
        slot.cycle_v2_ramp_side = 1.0
        if cfg.stagec_v2 and v2_mode != "full":
            slot.cycle_v2_ramp_side = -1.0 if self.rng.random() < 0.5 else 1.0
        neutral_loaded = bool(
            cfg.spawn_under_trench and cfg.neutral_loaded_prob > 0.0
            and self.rng.random() < cfg.neutral_loaded_prob
        )
        slot.episode_neutral_loaded = neutral_loaded
        if cfg.stagec_v2 and v2_mode == "live":
            hx, hy = -0.0199, -3.6874
            preloaded = True          # chamber filled below, as `return` does
            x, y = hx, hy + 2.2       # fallback if rejection sampling fails
            for _ in range(25):
                ang = float(self.rng.uniform(-math.pi, math.pi))
                rad = float(self.rng.uniform(1.8, 4.2))
                cx = hx + rad * math.cos(ang)
                cy = hy + rad * math.sin(ang)
                if cy < -2.95 and abs(cx) < 3.6:   # our court, past the board
                    x, y = cx, cy
                    break
            yaw = math.atan2(hy - y, hx - x) + math.radians(
                float(self.rng.uniform(-15.0, 15.0))
            )
            z = 0.02
        elif cfg.stagec_v2 and v2_mode == "return":
            # Return-skill stream: already across the ramp with a qualified
            # target load.  The reserved batch is preloaded below after the
            # FUEL bodies and controller have been reset.
            hx, hy = -0.0199, -3.6874
            preloaded = True
            x = slot.cycle_v2_ramp_side * float(self.rng.uniform(1.25, 1.85))
            y = float(self.rng.uniform(-1.35, -0.75))
            bearing = math.atan2(hy - y, hx - x)
            yaw = bearing + math.radians(float(self.rng.uniform(-12.0, 12.0)))
            z = 0.02
        elif cfg.stagec_v2 and v2_mode == "collect":
            # Collection-skill stream: just beyond the preferred ramp, empty,
            # extended, and facing the event-released shallow-neutral cluster.
            preloaded = False
            x = slot.cycle_v2_ramp_side * float(self.rng.uniform(1.30, 1.80))
            y = float(self.rng.uniform(-1.75, -1.30))
            tx = slot.cycle_v2_ramp_side * cfg.cycle_v2_cluster_x
            ty = cfg.cycle_v2_cluster_y
            yaw = math.atan2(ty - y, tx - x) + math.radians(
                float(self.rng.uniform(-15.0, 15.0))
            )
            z = 0.02
        elif cfg.stagec_v2 and v2_mode == "postdump":
            # Leave-skill stream: a plausible empty post-dump pose in home
            # court, aligned with the chosen ramp rather than the trench.
            preloaded = False
            x = slot.cycle_v2_ramp_side * float(self.rng.uniform(1.15, 2.05))
            y = float(self.rng.uniform(-5.35, -4.55))
            tx = slot.cycle_v2_ramp_side * 1.55
            ty = -2.35
            yaw = math.atan2(ty - y, tx - x) + math.radians(
                float(self.rng.uniform(-15.0, 15.0))
            )
            z = 0.02
        elif cfg.stagec_v2 and v2_mode == "bank":
            # stage_d_v1 wave-4 time-sliced lane (v2 REDESIGN): the wave-3
            # cold LEAVE-at-home teleport produced 100% null episodes (the
            # suffix froze in a pose it never reaches without an approach
            # trajectory).  v2 starts IN-DISTRIBUTION: mid-blackout, already
            # across in shallow neutral facing the cluster -- byte-for-byte
            # the working `collect` lane spawn family -- in COLLECT phase.
            # The episode drills: fill chamber under a dark hub -> big ferry
            # and/or hold -> schedule-aware return -> edge dump -> convert
            # the (b-variant seeded) own-court stockpile.
            preloaded = False
            slot.stage_d_bank_t0 = float(
                cfg.stage_d_bank_clock_a
                if self.rng.random() < 0.5
                else cfg.stage_d_bank_clock_b
            )
            x = slot.cycle_v2_ramp_side * float(self.rng.uniform(1.30, 1.80))
            y = float(self.rng.uniform(-1.75, -1.30))
            tx = slot.cycle_v2_ramp_side * cfg.cycle_v2_cluster_x
            ty = cfg.cycle_v2_cluster_y
            yaw = math.atan2(ty - y, tx - x) + math.radians(
                float(self.rng.uniform(-15.0, 15.0))
            )
            z = 0.02
        elif neutral_loaded:
            # 2nd-cycle RETURN curriculum: spawn in DEEP neutral, LOADED, extended, nose at
            # the hub, so the only task is cross back over the board -> shoot -> score. Trains
            # the back half of the 2nd cycle that exploration never reached from the trench.
            hx, hy = -0.0199, -3.6874
            preloaded = True
            side = -1.0 if self.rng.random() < 0.5 else 1.0
            x = side * float(self.rng.uniform(0.9, 2.2))          # aligned to a hub ramp lane
            y = float(self.rng.uniform(cfg.neutral_loaded_y[0], cfg.neutral_loaded_y[1]))
            bearing = math.atan2(hy - y, hx - x)                   # nose pointed south at the hub
            yaw = bearing + math.radians(float(self.rng.uniform(-25.0, 25.0)))
            z = 0.02
        elif cfg.spawn_under_trench:
            # source-faithful match start: compact, fully beneath the blue trench.
            # Field-local coords (the adapter adds the per-env origin). No preload
            # - the robot must escape the 0.565 m opening, extend, then collect.
            from frc_rebuilt.competition_robot import (
                BLUE_TRENCH_START_TRANSLATION as _TRENCH_XYZ,
                BLUE_TRENCH_START_YAW_DEG as _TRENCH_YAW,
            )
            preloaded = False
            x, y, z = (float(v) for v in _TRENCH_XYZ)
            yaw = math.radians(float(_TRENCH_YAW))
            # per-env pose jitter: the trench start is otherwise BIT-IDENTICAL and
            # resets in LOCKSTEP across envs, so one env's settle-step physics trip
            # hits all envs on the same step and the unhealthy.all() heuristic misreads
            # it as global-solver corruption -> whole-process exit(4). Small jitter
            # decorrelates them (stays well inside the clear trench opening x 2.72-4.00).
            x += float(self.rng.uniform(-0.04, 0.04))
            y += float(self.rng.uniform(-0.04, 0.04))
            yaw += math.radians(float(self.rng.uniform(-6.0, 6.0)))
        else:
            # stage B mix: preloaded episodes start HOLDING fuel at a LEGAL shooting
            # pose - a ring around the blue hub, inside our own court (past the board
            # at y=-2.775) and within auto-aim range (1.8-4.8 m) - so the robot learns
            # to fire and score from a legal spot. Cold episodes spawn in the neutral
            # middle and must collect + drive across the board to score.
            hx, hy = -0.0199, -3.6874
            preloaded = bool(cfg.preload_prob > 0.0 and self.rng.random() < cfg.preload_prob)
            # ramp-teaching: a slice of the COLD episodes start on a hub RAMP lane in
            # the neutral zone, facing the hub, preloaded - so the shortest path to a
            # legal shot runs straight south OVER the drive-over ramp (extended-legal)
            # rather than drifting to the compact-only outer trench lanes.
            ramp_start = bool(
                not preloaded
                and cfg.spawn_ramp_prob > 0.0
                and self.rng.random() < cfg.spawn_ramp_prob
            )
            if ramp_start:
                preloaded = True  # inventory in hand: cross the ramp -> shoot -> score
                side = -1.0 if self.rng.random() < 0.5 else 1.0
                x = side * float(self.rng.uniform(0.9, 2.2))    # aligned to a ramp lane
                y = float(self.rng.uniform(-1.6, 0.2))          # neutral zone, north of the line
                bearing = math.atan2(hy - y, hx - x)            # nose pointed at the hub
                yaw = bearing + math.radians(float(self.rng.uniform(-25.0, 25.0)))
            elif preloaded:
                x, y = hx, hy - 2.5  # fallback: 2.5 m straight south of the hub
                for _ in range(25):
                    ang = float(self.rng.uniform(-math.pi, math.pi))
                    rad = float(self.rng.uniform(1.8, 4.8))
                    cx, cy = hx + rad * math.cos(ang), hy + rad * math.sin(ang)
                    if cy < -2.85 and abs(cx) < 3.9:   # own court + inside field width
                        x, y = cx, cy
                        break
                yaw = math.radians(
                    float(self.rng.uniform(cfg.spawn_yaw_range_deg[0], cfg.spawn_yaw_range_deg[1]))
                )
            else:
                x = float(self.rng.uniform(cfg.spawn_xy_range[0], cfg.spawn_xy_range[1]))
                y = float(self.rng.uniform(cfg.spawn_xy_range[2], cfg.spawn_xy_range[3]))
                yaw = math.radians(
                    float(self.rng.uniform(cfg.spawn_yaw_range_deg[0], cfg.spawn_yaw_range_deg[1]))
                )
            z = 0.02
        # STAGE-D2 (user 2026-07-26): the blackout drill must start where the
        # robot ACTUALLY is when a blackout hits mid-cycle -- on a hub RAMP lane
        # in the neutral zone, nose toward our hub -- not at a synthetic pose.
        # Reuses the validated ramp-lane geometry from the stage-B ramp
        # curriculum above so the drive-over ramp (extended-legal) is the
        # shortest way home, not the compact-only outer trench lanes.
        if cfg.stagec_v2 and v2_mode == "bank":
            _side = -1.0 if self.rng.random() < 0.5 else 1.0
            x = _side * float(self.rng.uniform(0.9, 2.2))
            y = float(self.rng.uniform(-1.6, 0.2))
            _hx, _hy = -0.0199, -3.6874
            yaw = math.atan2(_hy - y, _hx - x) + math.radians(
                float(self.rng.uniform(-25.0, 25.0))
            )
            z = 0.02
        if cfg.stage_d and cfg.stagec_v2 and v2_mode == "postdump":
            # Warm-start (see module note): shallow neutral, nose at the near
            # cluster -- the proven collect/bank spawn family.  The home-edge
            # start was unlearnable in isolation (0/243 eps at stddev 0.45).
            x = slot.cycle_v2_ramp_side * float(self.rng.uniform(1.30, 1.80))
            y = float(self.rng.uniform(-1.75, -1.30))
            _tx = slot.cycle_v2_ramp_side * float(cfg.cycle_v2_cluster_x)
            _ty = -2.35
            yaw = math.atan2(_ty - y, _tx - x) + math.radians(
                float(self.rng.uniform(-15.0, 15.0))
            )
            z = 0.02
        half = yaw * 0.5
        slot.articulation.set_world_pose(
            np.asarray([x, y, z], np.float32),
            np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], np.float32),
        )
        slot.articulation.set_velocities_zero()

        # FUEL back to the template layout, at rest. CURRICULUM: with neutral_refill_prob,
        # relocate a subset to the shallow-neutral cluster so a 2nd cycle has near balls.
        fuel_pos = self._fuel_home.copy()
        if cfg.neutral_refill_count > 0 and self.rng.random() < cfg.neutral_refill_prob:
            fuel_pos = self._neutral_refill(fuel_pos)
        # STAGE-D2: scatter the neutral/away FUEL for the blackout drill so the
        # policy cannot memorise one template layout.  Our own court is left
        # alone -- the seeded stockpile below depends on its exact grid.
        _bj = float(getattr(cfg, "stage_d_bank_fuel_jitter_m", 0.0))
        _bs = float(getattr(cfg, "stage_d_bank_fuel_scatter", 0.0))
        if cfg.stagec_v2 and v2_mode == "bank" and (_bj > 0.0 or _bs > 0.0):
            # The neutral zone in a real match is NOT the tidy template grid: it
            # is messy, with FUEL piled along the side walls and in the corners
            # where robots shove it.  Relocate a fraction of the neutral balls
            # into a few edge/corner-biased clumps, then jitter the rest, so the
            # blackout drill cannot memorise one layout.  Our own court is left
            # untouched -- the seeded stockpile below needs its exact grid.
            _m = fuel_pos[:, 1] > -2.775
            _idx = np.flatnonzero(_m)
            if _idx.size:
                _NX, _NY = 3.90, 2.775   # neutral-zone half-width / half-depth
                _nmove = int(_idx.size * min(1.0, max(0.0, _bs)))
                if _nmove > 0:
                    _pick = self.rng.choice(_idx, size=_nmove, replace=False)
                    _nc = max(1, int(cfg.stage_d_bank_fuel_clumps))
                    # clump centres pushed toward the perimeter: u**0.3 crowds
                    # the sample against the wall, so corners get real weight.
                    _cx = np.sign(self.rng.uniform(-1.0, 1.0, _nc)) * (
                        _NX * self.rng.uniform(0.0, 1.0, _nc) ** 0.30
                    )
                    _cy = np.sign(self.rng.uniform(-1.0, 1.0, _nc)) * (
                        _NY * self.rng.uniform(0.0, 1.0, _nc) ** 0.30
                    )
                    _own = self.rng.integers(0, _nc, size=_nmove)
                    _rad = max(0.25, _bj if _bj > 0.0 else 0.45)
                    fuel_pos[_pick, 0] = (
                        _cx[_own] + self.rng.normal(0.0, _rad, _nmove)
                    ).astype(fuel_pos.dtype)
                    fuel_pos[_pick, 1] = (
                        _cy[_own] + self.rng.normal(0.0, _rad, _nmove)
                    ).astype(fuel_pos.dtype)
                    fuel_pos[_pick, 2] = 0.075
                if _bj > 0.0:
                    _rest = np.setdiff1d(_idx, _pick if _nmove > 0 else np.array([], dtype=_idx.dtype))
                    if _rest.size:
                        fuel_pos[_rest, 0] += self.rng.uniform(-_bj, _bj, _rest.size).astype(fuel_pos.dtype)
                        fuel_pos[_rest, 1] += self.rng.uniform(-_bj, _bj, _rest.size).astype(fuel_pos.dtype)
                # keep everything inside the field and out of our own court
                fuel_pos[_idx, 0] = np.clip(fuel_pos[_idx, 0], -_NX, _NX)
                fuel_pos[_idx, 1] = np.clip(fuel_pos[_idx, 1], -2.70, _NY)
        slot.stage_d_bank_seed_ids = ()
        if (
            cfg.stagec_v2
            and v2_mode == "live"
            and int(cfg.stage_d_live_stockpile) > 0
        ):
            _n_seed = int(cfg.stage_d_live_stockpile)
            _order = np.argsort(-fuel_pos[:, 1])
            _seed_idx = [int(i) for i in _order[:_n_seed]]
            # Six rows x ~10 columns = ~60 distinct spots, so a 60-ball floor
            # pile does not stack balls on top of each other.  The hub approach
            # lane (|x| < 0.85) stays clear so the robot can still reach it.
            _spots = []
            for _sy in (-4.10, -4.40, -4.70, -5.00, -5.30, -5.60):
                for _k in range(11):
                    _sx = -2.25 + 0.45 * _k
                    if abs(_sx) < 0.85:
                        continue
                    _spots.append((_sx, _sy))
            for _j, _fi in enumerate(_seed_idx):
                _sx, _sy = _spots[_j % len(_spots)]
                fuel_pos[_fi, 0] = _sx + float(self.rng.uniform(-0.10, 0.10))
                fuel_pos[_fi, 1] = _sy + float(self.rng.uniform(-0.10, 0.10))
                fuel_pos[_fi, 2] = 0.075
            slot.stage_d_bank_seed_ids = tuple(_seed_idx)
        if (
            cfg.stagec_v2
            and v2_mode == "bank"
            and float(getattr(slot, "stage_d_bank_t0", 0.0))
            == float(cfg.stage_d_bank_clock_b)
            and int(cfg.stage_d_bank_stockpile) > 0
        ):
            # Seed an ENTITLED own-court stockpile for the late-start variant:
            # teleport the deepest-neutral template balls onto the own-court
            # floor behind the hub (postdump-depleted teleport precedent).
            # They are marked qualified + already-ferried below, so they carry
            # the +10 entitlement for conversion but can never re-mint ferry
            # reward.  Grid keeps 0.45 m spacing clear of the hub footprint.
            n_seed = min(int(cfg.stage_d_bank_stockpile), 16)
            order = np.argsort(-fuel_pos[:, 1])  # farthest north first
            seed_idx = [int(i) for i in order[:n_seed]]
            spots = []
            for row, sy in ((0, -4.35), (1, -4.80)):
                for k in range(8):
                    sx = -1.75 + 0.5 * k
                    if abs(sx) < 0.85:
                        continue  # keep the hub approach lane clear
                    spots.append((sx, sy))
            for j, fi in enumerate(seed_idx):
                sx, sy = spots[j % len(spots)]
                fuel_pos[fi, 0] = sx + float(self.rng.uniform(-0.06, 0.06))
                fuel_pos[fi, 1] = sy + float(self.rng.uniform(-0.06, 0.06))
                fuel_pos[fi, 2] = 0.075
            slot.stage_d_bank_seed_ids = tuple(seed_idx)
        fuel_pos = self._configure_cycle_v2_reserve(slot, fuel_pos)
        slot.fuel.set_world_poses(
            positions=fuel_pos,
            orientations=self._fuel_home_quat.copy(),
            indices=np.arange(slot.fuel.count),
        )
        zeros = np.zeros((slot.fuel.count, 3), np.float32)
        slot.fuel.set_linear_velocities(zeros, indices=np.arange(slot.fuel.count))
        slot.fuel.set_angular_velocities(zeros, indices=np.arange(slot.fuel.count))

        # fresh robot logic state: EVERYTHING per-episode is cleared, including
        # drive slew memory and the shooter FSM (reviewer finding: FSM/motion
        # state used to leak across episodes)
        controller = slot.controller
        controller.reset_match_state()
        if cfg.stage_d and cfg.stage_d_ferry:
            # "Land the bank on the return path": steer the ferry solver's
            # landing target (see competition_robot.solve_ferry overrides).
            if float(cfg.stage_d_ferry_target_y) > 0.0:
                controller.ferry_target_y_m = float(cfg.stage_d_ferry_target_y)
            if float(cfg.stage_d_ferry_lane_x) > 0.0:
                controller.ferry_lane_x_m = float(cfg.stage_d_ferry_lane_x)
        # start compact beneath the trench; otherwise honor the locked posture. The
        # neutral-loaded curriculum starts EXTENDED (already collected, ready to shoot).
        start_extended = (
            neutral_loaded
            or (cfg.stagec_v2 and v2_mode in ("postdump", "collect", "return", "bank"))
            or (bool(cfg.lock_storage_extended) and not cfg.spawn_under_trench)
        )
        controller.snap_storage_state(start_extended)
        controller.intake_on = False
        v2_preload_ids: list[int] = []
        if preloaded:
            if cfg.stagec_v2 and v2_mode in ("return", "live"):
                # Consume a whole reserved batch.  The competition reset helper
                # has eight collision-safe physical preload slots (the chamber
                # can grow to 60 through the real intake during play), so seed
                # only what preload() actually accepted.  The true COLLECT ->
                # The RETURN gate follows target_load; this skill start is already
                # in RETURN and intentionally practices its back half safely.
                # `live` (section 3) has no reserved batches -- those are built
                # for the return micro-skill -- so fall back to the general pool
                # rather than popping an empty list.
                batch = (
                    slot.cycle_v2_reserved_batches.pop(0)
                    if slot.cycle_v2_reserved_batches
                    else None
                )
                # The source robot exposes eight collision-safe physical
                # preload anchors.  This stream is explicitly an eight-ball
                # return/score micro-skill; FULL uses the configured collection gate.
                count = min(
                    int(cfg.cycle_v2_target_load), int(RETURN_SKILL_PRELOAD)
                )
                if batch is None:
                    controller.preload(slot.fuel, count=count)
                else:
                    controller.preload(
                        slot.fuel, count=count, indices=list(batch[:count])
                    )
                if v2_mode == "live" and int(cfg.stage_d_live_chamber) > len(
                    controller.magazine
                ):
                    # Fill the chamber the way play does: drop the balls INSIDE
                    # the hopper envelope and let the per-step geometric capture
                    # detect them.  HOPPER_MIN/MAX_LOCAL are robot-local, so
                    # convert through the chassis pose.  Inset by a ball radius
                    # so nothing spawns intersecting a collider wall.
                    from frc_rebuilt.competition_robot import (
                        HOPPER_MAX_LOCAL,
                        HOPPER_MIN_LOCAL,
                        quat_wxyz_to_matrix,
                    )

                    _want = int(cfg.stage_d_live_chamber)
                    _used = set(controller.magazine) | set(
                        getattr(slot, "stage_d_bank_seed_ids", ()) or ()
                    )
                    _extra = [
                        int(i)
                        for i in range(int(slot.fuel.count) - 1, -1, -1)
                        if int(i) not in _used
                    ][: max(0, _want - len(controller.magazine))]
                    if _extra:
                        _pos, _quat = controller.chassis_pose()
                        _rot = quat_wxyz_to_matrix(_quat)
                        _lo = np.asarray(HOPPER_MIN_LOCAL, np.float32) + 0.075
                        _hi = np.asarray(HOPPER_MAX_LOCAL, np.float32) - 0.075
                        # FUEL is ~0.13 m across; use a 0.155 m pitch in all
                        # three axes and derive how many actually fit.  Seeding
                        # more than capacity used to clamp z and stack balls
                        # inside one another, which NaN'd the sim.
                        _pitch = 0.155
                        _nx = max(1, int((_hi[0] - _lo[0]) / _pitch) + 1)
                        _ny = max(1, int((_hi[1] - _lo[1]) / _pitch) + 1)
                        _nz = max(1, int((_hi[2] - _lo[2]) / _pitch) + 1)
                        _cap = _nx * _ny * _nz
                        if len(_extra) > _cap:
                            _extra = _extra[:_cap]
                        _pts = []
                        for _k in range(len(_extra)):
                            _ix = _k % _nx
                            _iy = (_k // _nx) % _ny
                            _iz = _k // (_nx * _ny)
                            _lx = _lo[0] + _pitch * _ix
                            _ly = _lo[1] + _pitch * _iy
                            _lz = _lo[2] + _pitch * _iz
                            _pts.append([_lx, _ly, _lz])
                        _local = np.asarray(_pts, np.float32)
                        _world = _pos + _local @ _rot.T
                        _ei = np.asarray(_extra, dtype=np.int32)
                        slot.fuel.set_world_poses(
                            positions=_world.astype(np.float32), indices=_ei
                        )
                        slot.fuel.set_linear_velocities(
                            np.zeros((len(_extra), 3), dtype=np.float32), indices=_ei
                        )
                # NOTE 2026-07-26: do NOT extend the chamber past the 8
                # physical preload slots.  Adding magazine entries whose bodies
                # are stowed off-field creates PHANTOM ammunition: the shooter
                # feeds a ball that is not really there, nothing reaches the hub,
                # and the real 8 never fire either (measured: scored 8 -> 0 and
                # chamber_end stuck at 30).  A genuinely full chamber has to come
                # from section 2 delivering it through real intake.
                v2_preload_ids = list(controller.magazine)
                slot.cycle_v2_return_preload_count = len(v2_preload_ids)
                slot.cycle_v2_reserved_ids.difference_update(v2_preload_ids)
            else:
                _cnt_range = cfg.neutral_loaded_count if neutral_loaded else cfg.preload_count_range
                count = int(self.rng.integers(_cnt_range[0], _cnt_range[1] + 1))
                # fills the magazine only; balls_collected stays 0, so preloaded
                # FUEL never earns collection reward - only its conversion to score
                controller.preload(slot.fuel, count=count)
        if (
            cfg.stage_d
            and cfg.stage_d_preload
            and cfg.spawn_under_trench
            and not preloaded
        ):
            # Official 8-ball robot preload (rules.MAX_FUEL_PRELOAD_PER_ROBOT)
            # at the accepted trench start.  balls_collected stays 0 and custody
            # seeds prev_magazine below, so preloads never earn collect reward;
            # cycle_v2 FIRST_CYCLE protects them at full score value (they are
            # seeded into protected_first_ids via the initial magazine).
            from frc_rebuilt.rules import MAX_FUEL_PRELOAD_PER_ROBOT

            controller.preload(
                slot.fuel, count=int(MAX_FUEL_PRELOAD_PER_ROBOT)
            )

        router = slot.router
        router.pending.clear()
        router.blocked_until_clear = set()
        router.released_watch.clear()
        router.exit_free_at.clear()
        router.funnel_entry.clear()
        router.scored = {"red": 0, "blue": 0}
        router.score_events.clear()   # custody telemetry — must clear with scored (Turn 32)
        router.detected = 0
        router.released = 0
        if cfg.stage_d:
            # Fresh official-match context: the SHIFT 1 decision is re-made at
            # t=23 s each episode (step loop); pre-decision both hubs are active.
            slot.stage_d_first_inactive = None
            router.match_first_inactive = None
            slot.stage_d_episode_seed = int(self.rng.integers(0, 2**31 - 1))
            slot.stage_d_masked_fires = 0

        slot.clock_s = 0.0
        slot.stage_d_lane_end_s = None
        if cfg.stagec_v2 and v2_mode == "bank":
            # stage_d_v1 wave-3: preset the match clock inside the slice and
            # pre-make the parity decision (the t=23 write-once site checks
            # ``first_inactive is None`` and skips itself).  Lane terminates
            # span seconds later; episode_len_s stays the real 160 so idx7/29
            # and return_time_guard keep FULL-match semantics.
            t0 = float(getattr(slot, "stage_d_bank_t0", cfg.stage_d_bank_clock_a))
            slot.clock_s = t0
            # End when the BLACKOUT ends (30->55, 80->105), not span seconds
            # later: running past the edge turns the drill into a scoring window.
            slot.stage_d_lane_t0 = float(slot.clock_s)
            slot.stage_d_lane_end_s = min(
                float(cfg.episode_len_s),
                self._phase_end_after(t0, t0 + float(cfg.stage_d_bank_span_s)),
            )
            slot.stage_d_first_inactive = str(cfg.stage_d_first_inactive)
            router.match_first_inactive = str(cfg.stage_d_first_inactive)
        if cfg.stagec_v2 and v2_mode == "live":
            # Three live windows: SHIFT 2 (55-80), SHIFT 4 (105-130) and the
            # 30 s ENDGAME (130-160).  Sampled uniformly so all three are drilled.
            # Two live blocks for blue: SHIFT 2 (55-80, 25 s) and the
            # contiguous SHIFT 4 + ENDGAME (105-160, 55 s).
            _starts = (
                float(cfg.stage_d_live_clock_a),
                float(cfg.stage_d_live_clock_b),
            )
            slot.clock_s = float(_starts[int(self.rng.integers(0, len(_starts)))])
            slot.stage_d_lane_t0 = float(slot.clock_s)
            slot.stage_d_lane_end_s = min(
                float(cfg.episode_len_s),
                self._live_block_end(
                    slot.clock_s,
                    str(cfg.stage_d_first_inactive),
                    float(cfg.episode_len_s),
                ),
            )
            if cfg.stage_d:
                slot.stage_d_first_inactive = str(cfg.stage_d_first_inactive)
                router.match_first_inactive = str(cfg.stage_d_first_inactive)
        if cfg.stage_d and cfg.stagec_v2 and v2_mode == "postdump":
            # Start at a hub reactivation with the lane's own REAL post-dump
            # state (home, chamber empty, LEAVE) and run to the end of that
            # contiguous live block -- the exact repeat-cycle the top episodes
            # execute and the typical ones skip.
            _starts = (
                float(cfg.stage_d_postdump_clock_a),
                float(cfg.stage_d_postdump_clock_b),
            )
            slot.clock_s = float(_starts[int(self.rng.integers(0, len(_starts)))])
            slot.stage_d_lane_t0 = float(slot.clock_s)
            slot.stage_d_lane_end_s = min(
                float(cfg.episode_len_s),
                self._live_block_end(
                    slot.clock_s,
                    str(cfg.stage_d_first_inactive),
                    float(cfg.episode_len_s),
                ),
            )
            slot.stage_d_first_inactive = str(cfg.stage_d_first_inactive)
            router.match_first_inactive = str(cfg.stage_d_first_inactive)
        if cfg.stagec_v2 and v2_mode == "opener":
            # episode_len_s stays the real 160 so proprio idx7/29 and
            # return_time_guard keep FULL-match semantics; only the lane end
            # moves.  The parity decision is pre-made when stage_d is on.
            slot.clock_s = 0.0
            slot.stage_d_lane_t0 = float(slot.clock_s)
            slot.stage_d_lane_end_s = min(
                float(cfg.episode_len_s), float(cfg.stage_d_opener_span_s)
            )
            if cfg.stage_d:
                slot.stage_d_first_inactive = str(cfg.stage_d_first_inactive)
                router.match_first_inactive = str(cfg.stage_d_first_inactive)
        slot.prev_action[:] = 0.0
        slot.score_seen = 0
        slot.collected_seen = 0
        # custody ledger: seed prev_magazine with any PRELOAD so preloaded balls are not
        # credited as fresh collections (matches balls_collected staying 0 for preloads).
        if slot.custody is None:
            from frc_rebuilt.rl.custody import CustodyState
            slot.custody = CustodyState()
        slot.custody.reset(controller.magazine)
        if cfg.stagec_v2:
            from frc_rebuilt.rl.cycle_v2 import CyclePhase, CycleV2Config, CycleV2State

            phase = {
                "full": CyclePhase.FIRST_CYCLE,
                "postdump": CyclePhase.LEAVE,
                "collect": CyclePhase.COLLECT,
                "return": CyclePhase.RETURN,
                "bank": CyclePhase.COLLECT,
                "opener": CyclePhase.FIRST_CYCLE,
                "live": CyclePhase.SCORE,
            }[v2_mode]
            cycle_cfg = CycleV2Config(
                target_load=int(cfg.cycle_v2_target_load),
                preferred_load=(
                    int(cfg.cycle_v2_preferred_repeat_load)
                    if bool(cfg.cycle_v2_collect_until_preferred)
                    else 0
                ),
                collect_stall_steps=int(cfg.cycle_v2_collect_stall_steps),
                return_time_guard=float(cfg.cycle_v2_return_time_guard),
                live_return_load=int(cfg.stage_d_live_return_load),
                chamber_capacity=int(cfg.cycle_v2_chamber_capacity),
                full_score_reward=float(cfg.score_reward_weight),
                # Repeat-cycle score is valuable only after a neutral-zone
                # collection.  The state-machine default is deliberately zero;
                # overriding it to +2 reintroduced a small recycle incentive.
                unqualified_score_reward=0.0,
                qualified_collect_reward=1.5,
                base_collect_reward=float(cfg.collect_reward_weight),
                cycle_score_fraction=float(cfg.cycle_v2_score_fraction),
                cycle_score_floor=int(cfg.cycle_v2_score_floor),
            )
            slot.cycle_v2 = CycleV2State(cycle_cfg)
            position, _ = controller.chassis_pose()
            slot.cycle_v2.reset(
                controller.magazine,
                position,
                score=0,
                phase=phase,
                qualified_ids=(v2_preload_ids if v2_mode == "return" else ()),
            )
            if cfg.stagec_v2 and v2_mode == "bank":
                # Seeded stockpile balls carry their +10 entitlement so the
                # own-court loop can convert them at reactivation.  They sit on
                # the FLOOR, not in the chamber, so they cannot go through
                # ``reset(qualified_ids=...)`` (its invariant requires chamber
                # membership -- the return-lane preload case).  Mutating after
                # reset mirrors how floor entitlements arise mid-episode: a
                # ferry launch removes the ball from the magazine while its id
                # stays in ``qualified_ids``.
                slot.cycle_v2.qualified_ids.update(
                    int(i) for i in getattr(slot, "stage_d_bank_seed_ids", ())
                )
            slot.cycle_v2_progress_phase = phase
            slot.cycle_v2_best_distance = None
            slot.cycle_v2_leave_corridor_entered = False
            slot.cycle_v2_stats = {
                "full_score_events": 0,
                "unqualified_score_events": 0,
                "qualified_collections": 0,
                "milestones": {},
                "reserve_releases": 0,
                "postdump_depleted_balls": int(
                    slot.cycle_v2_postdump_depleted_count
                ),
                "v2_score_reward": 0.0,
                "v2_collect_reward": 0.0,
                "v2_progress_reward": 0.0,
                "v2_milestone_reward": 0.0,
                "v2_behavior_penalty": 0.0,
                "v2_border_penalty": 0.0,
                "v2_outer_rail_penalty": 0.0,
                "v2_ramp_out_reward": 0.0,
                "v2_load_efficiency_reward": 0.0,
                "ferried_balls": 0,
                "v2_ferry_reward": 0.0,
                "own_court_stockpile": 0,
                "owncourt_score_entries": 0,  # STAGE-D1C: LEAVE/COLLECT->SCORE re-entries
                "owncourt_shots": 0,          # STAGE-D1C: score-dumps in an own-court window
                "owncourt_scored": 0,         # STAGE-D1C: +10 events during those windows
                "owncourt_ledger_scored": 0,  # STAGE-D1D: volley-ledger conversion (incl. late landings)
                "timeline": [],               # STAGE-D1D: per-event (clock_s) shot/ferry/oc log
                "active_ferry_penalty": 0.0,  # STAGE-D1E: penalty for hub-live ferry presses
                "active_ferry_presses": 0,    # STAGE-D1E: count of penalized ferry presses
                "repeat_return_load_count": 0,
                "repeat_return_load_sum": 0,
                "repeat_return_load_max": 0,
                "repeat_scored_load_count": 0,
                "repeat_scored_load_sum": 0,
                "repeat_scored_load_max": 0,
                "outer_rail_active_steps": 0,
                "outer_rail_steps": 0,
                "outer_rail_positive_steps": 0,
                "outer_rail_negative_steps": 0,
                "outer_rail_max_streak": 0,
                "outer_rail_fraction": 0.0,
                "cycle2_outer_rail_active_steps": 0,
                "cycle2_outer_rail_steps": 0,
                "cycle2_outer_rail_fraction": 0.0,
                "cycle3plus_outer_rail_active_steps": 0,
                "cycle3plus_outer_rail_steps": 0,
                "cycle3plus_outer_rail_fraction": 0.0,
                "ramp_side_refreshes": 0,
                "ramp_side_positive_refreshes": 0,
                "ramp_side_negative_refreshes": 0,
                "ramp_out_attempts": 0,
                "ramp_out_successes": 0,
                "off_ramp_outs": 0,
                "cycle2_ramp_out_attempts": 0,
                "cycle2_ramp_out_successes": 0,
                "cycle2_off_ramp_outs": 0,
                "cycle3plus_ramp_out_attempts": 0,
                "cycle3plus_ramp_out_successes": 0,
                "cycle3plus_off_ramp_outs": 0,
                "ramp_out_abs_x_sum": 0.0,
                "ramp_return_attempts": 0,
                "ramp_return_successes": 0,
                "off_ramp_returns": 0,
                "ramp_return_abs_x_sum": 0.0,
                "dump_attempts": 0,
                "dump_empty_completions": 0,
                "partial_dumps": 0,
                "partial_dump_balls_left": 0,
                "collect_exit_target": 0,
                "collect_exit_preferred": 0,
                "collect_exit_stall": 0,
                "collect_exit_clock": 0,
            }
            slot.cycle_v2_penalty_spent = {
                "leave": 0.0,
                "return": 0.0,
                "score": 0.0,
            }
            slot.cycle_v2_border_steps = 0
            slot.cycle_v2_border_spent = 0.0
            slot.cycle_v2_outer_rail_active = False
            slot.cycle_v2_outer_rail_streak = 0
            slot.cycle_v2_outer_rail_spent = 0.0
            slot.stage_d_deep_red_spent = 0.0
            slot.stage_d_deep_red_streak = 0
            slot.stage_d_idle_spent = 0.0
            slot.stage_d_idle_streak = 0
            slot.stage_d_live_dump_paid = False
            slot.stage_d_rescued = False
            slot.stage_d_arrival_paid = False
            slot.cycle_v2_return_loads = {}
            slot.cycle_v2_phase_enter_step = 0
            slot.cycle_v2_terminal_reason = ""
        else:
            slot.cycle_v2 = None
            slot.cycle_v2_stats = {}
            slot.cycle_v2_terminal_reason = ""
            slot.cycle_v2_penalty_spent = {}
            slot.cycle_v2_phase_enter_step = 0
        slot.dumping = False          # not mid-chamber-dump at episode start
        slot.dump_mode = None         # clear ALL per-episode fire/dump telemetry so a
        slot.dump_ticks = 0           # forced mid-run reset cannot leak dump/ferry state
        slot.dump_started_this_step = False
        slot.dump_completed_this_step = False
        slot.dump_aborted_this_step = False
        slot.dump_start_ids = ()
        slot.dump_start_mode = None
        slot.dump_remaining_count = 0
        slot.dump_lost_aim_ticks = 0
        slot.ferry_fires = 0          # across the boundary (Turn 18)
        slot.ferried_ids = set()               # STAGE-D1B: distinct balls
        if cfg.stagec_v2 and v2_mode in ("bank", "live"):
            # Seeds are already "ferried": custody-once means they can convert
            # (+10 via the oc loop) but never re-mint the ferry reward.
            slot.ferried_ids |= set(getattr(slot, "stage_d_bank_seed_ids", ()))
        slot.ferry_prev_mag_ids = frozenset()  # repatriated; credited once each
        slot.owncourt_collectable = 0          # STAGE-D1C: ground balls in own court
        slot.owncourt_score_active = False     # STAGE-D1C: inside an own-court SCORE window
        slot.owncourt_dump_ledgers = []        # STAGE-D1D: own-court volley ledgers
        slot.active_ferry_this_step = False    # STAGE-D1E: hub-live ferry-press flag
        slot.owncourt_prev_ent_in_mag = 0      # STAGE-D1F: re-intake pull tracker
        slot.stage_d_masked_ferries = 0        # STAGE-D1G: live-window ferry no-ops
        slot.stage_d_small_ferries = 0         # stage_d_v1 wave-2: under-min-load ferry no-ops
        slot.forced_reset_settle_s = 0.0  # hidden shared-scene physics time since this
        #                                   env last (re)started; accrues on the settle
        #                                   of OTHER envs' forced resets (Turn 20)

    def reset_all(self) -> np.ndarray:
        for slot in self.slots:
            self._reset_slot(slot)
        # settle one physics step so poses/velocities take effect
        self.sim.step(render=False)
        return np.arange(self.cfg.num_envs)

    def _cycle_v2_target_distance(self, slot: EnvSlot, phase, position) -> float | None:
        """Small ramp-aligned waypoint curriculum used only for positive progress."""

        from frc_rebuilt.rl.cycle_v2 import CyclePhase

        x, y = float(position[0]), float(position[1])
        side = float(slot.cycle_v2_ramp_side)
        if phase is CyclePhase.LEAVE and bool(self.cfg.cycle_v2_require_ramp_out):
            # The nearest outbound ramp is observable from the current pose.
            # Avoid making LEAVE depend on a hidden retained side while the
            # robot is near the hub centre.
            if abs(x) > float(self.cfg.cycle_v2_ramp_side_deadband_x):
                side = -1.0 if x < 0.0 else 1.0
        if phase is CyclePhase.FIRST_CYCLE:
            return None
        targets = {
            CyclePhase.LEAVE: (side * 1.55, -2.35),
            CyclePhase.COLLECT: (
                side * float(self.cfg.cycle_v2_cluster_x),
                float(self.cfg.cycle_v2_cluster_y),
            ),
            CyclePhase.RETURN: (side * 1.55, -3.20),
            # A legal, unobstructed scoring pose instead of the hub centre.
            CyclePhase.SCORE: (side * 1.55, -5.20),
        }
        target = targets.get(phase)
        if target is None:
            return None
        return math.hypot(x - target[0], y - target[1])

    def _cycle_v2_progress_reward(self, slot: EnvSlot, phase, position) -> float:
        from frc_rebuilt.rl.cycle_v2 import CyclePhase

        if bool(self.cfg.cycle_v2_require_ramp_out) and phase is CyclePhase.LEAVE:
            x, y = float(position[0]), float(position[1])
            side = float(slot.cycle_v2_ramp_side)
            if abs(x) > float(self.cfg.cycle_v2_ramp_side_deadband_x):
                side = -1.0 if x < 0.0 else 1.0
            lateral_error = abs(x - side * 1.55)
            half_width = float(self.cfg.cycle_v2_ramp_out_half_width)
            in_corridor = lateral_error <= half_width

            if (
                slot.cycle_v2_progress_phase is not phase
                or slot.cycle_v2_best_distance is None
            ):
                slot.cycle_v2_progress_phase = phase
                slot.cycle_v2_leave_corridor_entered = bool(in_corridor)
                slot.cycle_v2_best_distance = (
                    math.hypot(lateral_error, max(0.0, -2.35 - y))
                    if in_corridor
                    else float(lateral_error)
                )
                return 0.0

            if not slot.cycle_v2_leave_corridor_entered:
                # Until the chassis reaches a ramp lane, only inward motion is
                # progress.  Driving north beside the wall earns exactly zero.
                improvement = max(
                    0.0,
                    float(slot.cycle_v2_best_distance) - float(lateral_error),
                )
                slot.cycle_v2_best_distance = min(
                    float(slot.cycle_v2_best_distance), float(lateral_error)
                )
                if in_corridor:
                    slot.cycle_v2_leave_corridor_entered = True
                    slot.cycle_v2_best_distance = math.hypot(
                        lateral_error, max(0.0, -2.35 - y)
                    )
                return min(
                    float(self.cfg.cycle_v2_progress_step_cap),
                    float(self.cfg.cycle_v2_progress_per_m) * improvement,
                )

            if not in_corridor:
                return 0.0
            distance = math.hypot(lateral_error, max(0.0, -2.35 - y))
            improvement = max(
                0.0, float(slot.cycle_v2_best_distance) - float(distance)
            )
            slot.cycle_v2_best_distance = min(
                float(slot.cycle_v2_best_distance), float(distance)
            )
            return min(
                float(self.cfg.cycle_v2_progress_step_cap),
                float(self.cfg.cycle_v2_progress_per_m) * improvement,
            )

        distance = self._cycle_v2_target_distance(slot, phase, position)
        if distance is None:
            slot.cycle_v2_progress_phase = phase
            slot.cycle_v2_best_distance = None
            return 0.0
        if slot.cycle_v2_progress_phase is not phase or slot.cycle_v2_best_distance is None:
            slot.cycle_v2_progress_phase = phase
            slot.cycle_v2_best_distance = float(distance)
            return 0.0
        improvement = max(0.0, float(slot.cycle_v2_best_distance) - float(distance))
        slot.cycle_v2_best_distance = min(float(slot.cycle_v2_best_distance), float(distance))
        return min(
            float(self.cfg.cycle_v2_progress_step_cap),
            float(self.cfg.cycle_v2_progress_per_m) * improvement,
        )

    def _cycle_v2_is_ramp_crossing(self, x: float) -> bool:
        return abs(abs(float(x)) - 1.55) <= float(
            self.cfg.cycle_v2_ramp_out_half_width
        )

    def _cycle_v2_record_route_crossing(self, slot: EnvSlot, event, x: float) -> bool:
        from frc_rebuilt.rl.cycle_v2 import Milestone

        is_ramp = self._cycle_v2_is_ramp_crossing(x)
        stats = slot.cycle_v2_stats
        if not stats:
            return is_ramp
        if event.name is Milestone.LEFT_HOME:
            cycle_key = "cycle2" if int(event.cycle_index) == 2 else "cycle3plus"
            stats["ramp_out_attempts"] += 1
            stats[f"{cycle_key}_ramp_out_attempts"] += 1
            stats["ramp_out_abs_x_sum"] += abs(float(x))
            if is_ramp:
                stats["ramp_out_successes"] += 1
                stats[f"{cycle_key}_ramp_out_successes"] += 1
                # The actual crossing becomes the observable route context for
                # COLLECT.  This avoids pulling the robot back across the field
                # toward a stale side selected at the preceding score pose.
                slot.cycle_v2_ramp_side = -1.0 if float(x) < 0.0 else 1.0
            else:
                stats["off_ramp_outs"] += 1
                stats[f"{cycle_key}_off_ramp_outs"] += 1
        elif event.name is Milestone.RETURNED_HOME:
            stats["ramp_return_attempts"] += 1
            stats["ramp_return_abs_x_sum"] += abs(float(x))
            if is_ramp:
                stats["ramp_return_successes"] += 1
            else:
                stats["off_ramp_returns"] += 1
        return is_ramp

    def _cycle_v2_milestone_reward(
        self,
        slot: EnvSlot,
        milestones,
        x: float,
        *,
        qualified_load: int | None = None,
    ) -> float:
        from frc_rebuilt.rl.cycle_v2 import (
            Milestone,
            soft_repeat_load_bonus,
            time_decayed_success_bonus,
        )

        schedule = {
            Milestone.LEFT_HOME: (12.0, 500.0),
            Milestone.TARGET_LOAD: (16.0, 500.0),
            Milestone.RETURNED_HOME: (18.0, 450.0),
            Milestone.CYCLE_SCORED: (25.0, 350.0),
        }
        reward = 0.0
        load_reward = 0.0
        for event in milestones:
            base_decay = schedule.get(event.name)
            if base_decay is not None:
                reward += time_decayed_success_bonus(
                    base_decay[0], event.elapsed_steps, base_decay[1]
                )
            if event.name in (Milestone.LEFT_HOME, Milestone.RETURNED_HOME):
                is_ramp = self._cycle_v2_record_route_crossing(slot, event, x)
                if (
                    event.name is Milestone.LEFT_HOME
                    and bool(self.cfg.cycle_v2_require_ramp_out)
                ):
                    route_reward = (
                        float(self.cfg.cycle_v2_ramp_out_bonus)
                        if is_ramp
                        else -float(self.cfg.cycle_v2_off_ramp_exit_penalty)
                    )
                    reward += route_reward
                    if slot.cycle_v2_stats:
                        slot.cycle_v2_stats["v2_ramp_out_reward"] += route_reward
                elif is_ramp:
                    reward += float(self.cfg.cycle_v2_ramp_bonus)
            if (
                event.name is Milestone.RETURNED_HOME
                and qualified_load is not None
            ):
                load = max(0, int(qualified_load))
                slot.cycle_v2_return_loads[int(event.cycle_index)] = load
                if slot.cycle_v2_stats:
                    stats = slot.cycle_v2_stats
                    stats["repeat_return_load_count"] = int(
                        stats.get("repeat_return_load_count", 0)
                    ) + 1
                    stats["repeat_return_load_sum"] = int(
                        stats.get("repeat_return_load_sum", 0)
                    ) + load
                    stats["repeat_return_load_max"] = max(
                        int(stats.get("repeat_return_load_max", 0)), load
                    )
                if int(self.cfg.cycle_v2_preferred_repeat_load) > 0:
                    load_reward += soft_repeat_load_bonus(
                        qualified_load=load,
                        minimum_load=int(self.cfg.cycle_v2_target_load),
                        preferred_load=int(
                            self.cfg.cycle_v2_preferred_repeat_load
                        ),
                        max_bonus=float(
                            self.cfg.cycle_v2_repeat_load_return_bonus
                        ),
                    )
            elif event.name is Milestone.CYCLE_SCORED:
                load = int(
                    slot.cycle_v2_return_loads.pop(
                        int(event.cycle_index), 0
                    )
                )
                if slot.cycle_v2_stats:
                    stats = slot.cycle_v2_stats
                    stats["repeat_scored_load_count"] = int(
                        stats.get("repeat_scored_load_count", 0)
                    ) + 1
                    stats["repeat_scored_load_sum"] = int(
                        stats.get("repeat_scored_load_sum", 0)
                    ) + load
                    stats["repeat_scored_load_max"] = max(
                        int(stats.get("repeat_scored_load_max", 0)), load
                    )
                if int(self.cfg.cycle_v2_preferred_repeat_load) > 0:
                    load_reward += soft_repeat_load_bonus(
                        qualified_load=load,
                        minimum_load=int(self.cfg.cycle_v2_target_load),
                        preferred_load=int(
                            self.cfg.cycle_v2_preferred_repeat_load
                        ),
                        max_bonus=float(
                            self.cfg.cycle_v2_repeat_load_score_bonus
                        ),
                    )
        reward += load_reward
        if slot.cycle_v2_stats:
            slot.cycle_v2_stats["v2_load_efficiency_reward"] = float(
                slot.cycle_v2_stats.get("v2_load_efficiency_reward", 0.0)
            ) + load_reward
        return reward

    def _cycle_v2_owncourt_ready(self, slot: EnvSlot) -> bool:
        """Stage-D1C: own-court short-loop readiness for one env.

        Returns True only in the latched suffix while the blue hub is LIVE and
        >= ``stage_d_owncourt_min_balls`` ENTITLED loose balls sit in own court,
        so a home load may re-enter SCORE without a cross-field trip.  As a
        side effect it caches ``slot.owncourt_collectable`` -- the count of
        entitled loose balls (ids in ``cycle_v2.qualified_ids`` holding an
        unspent +10, not in the magazine) sitting in the robot's own court -- which the LEAVE
        delay-penalty suppression reads.  Returns False and touches nothing else
        unless the flag is on, so flag-off stays byte-identical and zero-cost.
        """

        cfg = self.cfg
        if not (cfg.stage_d and cfg.stage_d_owncourt_loop and cfg.stage_d_ferry):
            return False
        slot.owncourt_collectable = 0
        if not bool(getattr(slot.cycle_v2, "latched", False)):
            return False
        if not (
            cfg.stage_d_owncourt_blackout_intake
            or _stage_d.blue_hub_eligible(slot.clock_s, slot.stage_d_first_inactive)
        ):
            # Blackout without the blackout-intake option: the payoff is dead,
            # so no own-court loop -- the robot should be ferrying.  With
            # stage_d_owncourt_blackout_intake the loop stays armed during a
            # blackout so a home robot can PRE-LOAD the ferried stockpile and
            # hold at the hub: the ineligible-fire mask still makes a dark dump
            # physically impossible and the SCORE delay penalty is paused while
            # ineligible, so the load simply rides to the reactivation edge.
            return False
        # STAGE-D1D fix: count only ENTITLED loose balls -- ids still carrying
        # an unspent +10 (collected AWAY, then ferried/dropped home).  The
        # 200-ball template parks 13 never-qualified balls at the own-court
        # back wall on every reset, so counting raw poses made this gate true
        # from t=0: the loop dumped worthless balls for zero and the LEAVE
        # suppression parked the robot at home (D1c: mean 87->65, 2-cyc 11->6).
        entitled = getattr(slot.cycle_v2, "qualified_ids", None)
        if not entitled:
            return False
        magazine = {int(b) for b in slot.controller.magazine}
        balls, _ = slot.fuel.get_world_poses()
        board_y = float(cfg.stage_d_owncourt_board_y)
        n = len(balls)
        count = 0
        for raw in entitled:
            i = int(raw)
            if i in magazine or not (0 <= i < n):
                continue
            if (
                abs(float(balls[i, 0])) <= 8.2
                and abs(float(balls[i, 1])) <= 8.2
                and float(balls[i, 1]) <= board_y
            ):
                count += 1
        slot.owncourt_collectable = count
        return count >= int(cfg.stage_d_owncourt_min_balls)

    def _cycle_v2_delay_penalty(self, slot: EnvSlot, cycle_step) -> float:
        """Bounded cost for delaying only the active ordered-cycle milestone.

        The old experiment penalised time even after the robot started doing the
        right thing.  Here each cost has an explicit off-switch: crossing away,
        crossing home, or beginning a legal scoring dump.
        """

        from frc_rebuilt.rl.cycle_v2 import (
            CyclePhase,
            FieldRegion,
            capped_phase_delay_penalty,
        )

        phase = cycle_step.phase
        magazine_count = len(slot.controller.magazine)
        active = False
        if phase is CyclePhase.LEAVE:
            # stage_d_v1 ferry-first: the penalty no longer switches off when
            # the magazine holds a stray home pickup.  Under the old
            # ``magazine_count == 0`` condition a single own-court ball made
            # LEAVE camping free for the rest of the match, and 30-56% of
            # deterministic Stage-D episodes ended still parked in LEAVE.
            # Crossing away remains the only off-switch (region flips AWAY ->
            # phase becomes COLLECT), plus the own-court-stockpile suppression
            # below.
            active = cycle_step.region is FieldRegion.HOME
            if (
                active
                and self.cfg.stage_d
                and self.cfg.stage_d_owncourt_loop
                and self.cfg.stage_d_ferry
                and int(getattr(slot, "owncourt_collectable", 0))
                >= int(self.cfg.stage_d_owncourt_min_balls)
                and _stage_d.blue_hub_eligible(
                    slot.clock_s, slot.stage_d_first_inactive
                )
            ):
                # Stage-D1C own-court loop: staying home while ferried fuel
                # still sits in own court AND the hub is LIVE is correct play
                # -- intake those balls and shoot the short loop, not loaf.
                # wave-2 REVERT: during a blackout the penalty stays ACTIVE
                # even with a stockpile (blue2 showed 43% leave-camping with
                # penalty-free stockpile camps among them).  A blackout robot
                # at home must either work the stockpile -- entering SCORE via
                # the still-armed owncourt loop, where the delay penalty is
                # paused -- or cross out to ferry more; idling in LEAVE bleeds.
                active = False
            grace = int(self.cfg.cycle_v2_leave_grace_steps)
            per_step = float(self.cfg.cycle_v2_leave_penalty_per_step)
            cap = float(self.cfg.cycle_v2_leave_penalty_cap)
            key = "leave"
        elif phase is CyclePhase.RETURN:
            active = (
                cycle_step.region is FieldRegion.AWAY
                and int(cycle_step.qualified_load) > 0
            )
            grace = int(self.cfg.cycle_v2_return_grace_steps)
            per_step = float(self.cfg.cycle_v2_return_penalty_per_step)
            cap = float(self.cfg.cycle_v2_return_penalty_cap)
            key = "return"
        elif phase is CyclePhase.SCORE:
            active = (
                cycle_step.region is FieldRegion.HOME
                and magazine_count > 0
                and not bool(slot.dumping)
                and not bool(slot.dump_started_this_step)
            )
            if (
                active
                and self.cfg.stage_d
                and self.cfg.stage_d_pause_shoot_penalty_when_ineligible
                and not _stage_d.blue_hub_eligible(
                    slot.clock_s, slot.stage_d_first_inactive
                )
            ):
                # Stage D: holding a load at the hub while it is inactive is
                # correct play (wait for reactivation), not loafing.
                active = False
            grace = int(self.cfg.cycle_v2_shoot_grace_steps)
            per_step = float(self.cfg.cycle_v2_shoot_penalty_per_step)
            cap = float(self.cfg.cycle_v2_shoot_penalty_cap)
            key = "score"
        else:
            return 0.0
        if not active:
            return 0.0
        spent = float(slot.cycle_v2_penalty_spent.get(key, 0.0))
        penalty = capped_phase_delay_penalty(
            elapsed_steps=int(cycle_step.phase_elapsed_steps),
            grace_steps=grace,
            penalty_per_step=per_step,
            spent=spent,
            cap=cap,
        )
        slot.cycle_v2_penalty_spent[key] = spent + abs(float(penalty))
        return float(penalty)

    def _cycle_v2_border_camp_penalty(self, slot: EnvSlot, y: float) -> float:
        """Charge consecutive dwelling near the home/away borderline (anti-camp).

        A normal crossing passes through the band within the grace window and pays
        nothing.  Only *staying* at the edge beyond the grace is charged, per step,
        up to a per-cycle cap.  ``per_step == 0`` disables the term entirely.
        """

        per_step = float(self.cfg.cycle_v2_border_penalty_per_step)
        if per_step <= 0.0:
            slot.cycle_v2_border_steps = 0
            return 0.0
        if abs(float(y) - float(self.cfg.cycle_v2_border_y)) <= float(
            self.cfg.cycle_v2_border_band
        ):
            slot.cycle_v2_border_steps += 1
        else:
            slot.cycle_v2_border_steps = 0
            return 0.0
        if slot.cycle_v2_border_steps <= int(self.cfg.cycle_v2_border_grace_steps):
            return 0.0
        remaining = float(self.cfg.cycle_v2_border_penalty_cap) - float(
            slot.cycle_v2_border_spent
        )
        if remaining <= 0.0:
            return 0.0
        penalty = min(per_step, remaining)
        slot.cycle_v2_border_spent = float(slot.cycle_v2_border_spent) + penalty
        return -penalty

    def _cycle_v2_select_ramp_side(self, slot: EnvSlot, x: float) -> float:
        """Freeze the nearest ramp lane, retaining the old side near centre."""

        previous = -1.0 if float(slot.cycle_v2_ramp_side) < 0.0 else 1.0
        if abs(float(x)) <= float(self.cfg.cycle_v2_ramp_side_deadband_x):
            selected = previous
        else:
            selected = -1.0 if float(x) < 0.0 else 1.0
        slot.cycle_v2_ramp_side = selected
        stats = slot.cycle_v2_stats
        if stats:
            stats["ramp_side_refreshes"] += 1
            key = (
                "ramp_side_negative_refreshes"
                if selected < 0.0
                else "ramp_side_positive_refreshes"
            )
            stats[key] += 1
        return selected

    @staticmethod
    def _cycle_v2_refresh_outer_rail_fractions(stats: dict[str, Any], cycle_key: str) -> None:
        active_steps = max(1, int(stats["outer_rail_active_steps"]))
        cycle_active = max(1, int(stats[f"{cycle_key}_outer_rail_active_steps"]))
        stats["outer_rail_fraction"] = (
            float(stats["outer_rail_steps"]) / float(active_steps)
        )
        stats[f"{cycle_key}_outer_rail_fraction"] = (
            float(stats[f"{cycle_key}_outer_rail_steps"]) / float(cycle_active)
        )

    def _cycle_v2_outer_rail_penalty(
        self, slot: EnvSlot, cycle_step, x: float, collected_gain: int = 0
    ) -> float:
        """Bound sustained outer-wall dwelling during repeat-cycle navigation.

        The term is symmetric so moving from the right wall to the equally slow
        left wall cannot evade it.  Hysteresis avoids boundary chatter, and the
        depth scale leaves a normal ramp route untouched while making long rail
        loops measurably worse.
        """

        from frc_rebuilt.rl.cycle_v2 import CyclePhase

        active_phase = cycle_step.phase in (
            CyclePhase.LEAVE,
            CyclePhase.COLLECT,
            CyclePhase.RETURN,
        )
        if int(cycle_step.cycle_index) < 2 or not active_phase:
            slot.cycle_v2_outer_rail_active = False
            slot.cycle_v2_outer_rail_streak = 0
            return 0.0

        stats = slot.cycle_v2_stats
        cycle_key = "cycle2" if int(cycle_step.cycle_index) == 2 else "cycle3plus"
        if stats:
            stats["outer_rail_active_steps"] += 1
            stats[f"{cycle_key}_outer_rail_active_steps"] += 1
            self._cycle_v2_refresh_outer_rail_fractions(stats, cycle_key)

        abs_x = abs(float(x))
        if slot.cycle_v2_outer_rail_active:
            if abs_x <= float(self.cfg.cycle_v2_outer_rail_exit_x):
                slot.cycle_v2_outer_rail_active = False
                slot.cycle_v2_outer_rail_streak = 0
                return 0.0
        elif abs_x >= float(self.cfg.cycle_v2_outer_rail_enter_x):
            slot.cycle_v2_outer_rail_active = True
        else:
            slot.cycle_v2_outer_rail_streak = 0
            return 0.0

        slot.cycle_v2_outer_rail_streak += 1
        if stats:
            stats["outer_rail_steps"] += 1
            stats[f"{cycle_key}_outer_rail_steps"] += 1
            side_key = (
                "outer_rail_negative_steps" if float(x) < 0.0
                else "outer_rail_positive_steps"
            )
            stats[side_key] += 1
            stats["outer_rail_max_streak"] = max(
                int(stats["outer_rail_max_streak"]),
                int(slot.cycle_v2_outer_rail_streak),
            )
            self._cycle_v2_refresh_outer_rail_fractions(stats, cycle_key)

        if int(collected_gain) > 0:
            # Productive presence: the intake gained a ball this step. Charging
            # pickups pits the term against the collect/score gradient; retain
            # the proven v3 exemption while route selection is handled by the
            # explicit ramp-crossing outcome above.
            return 0.0
        per_step = float(self.cfg.cycle_v2_outer_rail_penalty_per_step)
        if (
            per_step <= 0.0
            or slot.cycle_v2_outer_rail_streak
            <= int(self.cfg.cycle_v2_outer_rail_grace_steps)
        ):
            return 0.0
        remaining = float(self.cfg.cycle_v2_outer_rail_penalty_cap) - float(
            slot.cycle_v2_outer_rail_spent
        )
        if remaining <= 0.0:
            return 0.0
        depth = float(np.clip(
            (
                abs_x - float(self.cfg.cycle_v2_outer_rail_enter_x)
            )
            / (
                float(self.cfg.cycle_v2_outer_rail_max_x)
                - float(self.cfg.cycle_v2_outer_rail_enter_x)
            ),
            0.0,
            1.0,
        ))
        effective_depth = max(
            depth, float(self.cfg.cycle_v2_outer_rail_min_scale)
        )
        escalation_steps = int(self.cfg.cycle_v2_outer_rail_escalation_steps)
        overdue_steps = (
            int(slot.cycle_v2_outer_rail_streak)
            - int(self.cfg.cycle_v2_outer_rail_grace_steps)
        )
        multiplier = 1.0
        if escalation_steps > 0:
            multiplier += max(0, overdue_steps - 1) / float(escalation_steps)
        multiplier = min(
            multiplier,
            float(self.cfg.cycle_v2_outer_rail_max_multiplier),
        )
        penalty = min(per_step * effective_depth * multiplier, remaining)
        slot.cycle_v2_outer_rail_spent += penalty
        return -penalty

    def _cycle_v2_skill_succeeded(self, slot: EnvSlot, milestones, x: float = 0.0) -> bool:
        if not self.cfg.cycle_v2_skill_terminate:
            return False
        # BANK LANE = CONVERSION lane (2026-07-25).  Measured defect: the bank
        # lane taught "send balls home" and stopped there -- across 745 bank
        # episodes only 20.5% scored ANYTHING (mean 1.5) despite 47.9% ferrying.
        # Its terminal objective was the span timer, so ferrying WAS the goal.
        # Now the lane succeeds only when N entitled own-court balls have been
        # CONVERTED to legal score, making conversion the thing being trained.
        # Pure termination/success logic: no scripted actions, no new reward.
        if (
            slot.cycle_v2_mode == "live"
            and int(self.cfg.stage_d_live_success_conversions) > 0
        ):
            return int(
                (slot.cycle_v2_stats or {}).get("owncourt_ledger_scored", 0) or 0
            ) >= int(self.cfg.stage_d_live_success_conversions)
        if (
            slot.cycle_v2_mode == "opener"
            and int(self.cfg.stage_d_opener_success_cycles) > 0
        ):
            return int(
                (slot.cycle_v2_stats or {}).get("cycles_completed", 0) or 0
            ) >= int(self.cfg.stage_d_opener_success_cycles)
        if (
            slot.cycle_v2_mode == "bank"
            and (
                int(self.cfg.stage_d_bank_success_conversions) > 0
                or int(self.cfg.stage_d_bank_success_chamber) > 0
            )
        ):
            _st = slot.cycle_v2_stats or {}
            # STAGE-D2 (user 2026-07-26): the blackout drill succeeds when the
            # robot comes HOME WITH A FULL CHAMBER -- it is not asked to shoot
            # during a blackout.  Chamber gate and conversion gate are ANDed only
            # over the ones actually enabled, so either can be used alone.
            _ok = True
            _needc = int(self.cfg.stage_d_bank_success_conversions)
            if _needc > 0:
                _ok = _ok and int(_st.get("owncourt_ledger_scored", 0) or 0) >= _needc
            _needm = int(self.cfg.stage_d_bank_success_chamber)
            if _needm > 0:
                _load = int(len(slot.controller.magazine))
                _home = float(slot.controller.chassis_pose()[0][1]) <= -3.05
                _ok = _ok and _load >= _needm and _home
            return _ok
        from frc_rebuilt.rl.cycle_v2 import Milestone

        wanted = {
            "postdump": (
                Milestone.CYCLE_SCORED
                if bool(self.cfg.cycle_v2_postdump_complete_cycle)
                else (
                    Milestone.TARGET_LOAD
                    if bool(self.cfg.cycle_v2_postdump_require_target_load)
                    else Milestone.LEFT_HOME
                )
            ),
            "collect": Milestone.TARGET_LOAD,
            "return": Milestone.CYCLE_SCORED,
        }.get(slot.cycle_v2_mode)
        succeeded = wanted is not None and any(
            event.name is wanted for event in milestones
        )
        if (
            succeeded
            and slot.cycle_v2_mode == "postdump"
            and bool(self.cfg.cycle_v2_require_ramp_out)
        ):
            if bool(
                self.cfg.cycle_v2_postdump_require_target_load
                or self.cfg.cycle_v2_postdump_complete_cycle
            ):
                return int(slot.cycle_v2_stats.get("ramp_out_successes", 0)) > 0
            return self._cycle_v2_is_ramp_crossing(x)
        return succeeded

    def _cycle_v2_skill_failed(self, slot: EnvSlot, milestones, x: float) -> bool:
        if (
            not self.cfg.cycle_v2_skill_terminate
            or slot.cycle_v2_mode != "postdump"
            or not bool(self.cfg.cycle_v2_require_ramp_out)
        ):
            return False
        from frc_rebuilt.rl.cycle_v2 import Milestone

        left_home = any(event.name is Milestone.LEFT_HOME for event in milestones)
        return left_home and not self._cycle_v2_is_ramp_crossing(x)

    def reset_slots(self, indices) -> dict[str, np.ndarray]:
        """Force-reset a SUBSET of envs mid-run (prefix-takeover curriculum no-progress
        reset) and RETURN a fresh full observation batch that the caller MUST adopt as
        its next obs -- otherwise a reset env's next action is computed from stale
        pre-reset state (Turn 18). The CALLER must first mark the final candidate
        transition done/truncated so a 3-step n-step never bootstraps across the
        discontinuity into the next champion-generated prefix (Turns 16-18). Indices are
        the settle renders (like the normal auto-reset) so cameras refresh. Invalid
        indices FAIL FAST (IndexError) rather than being silently dropped -- a silently
        ignored curriculum index could leave a suffix running while the caller believes
        it reset to PREFIX (Turn 20); valid duplicates are de-duplicated. Batch all
        simultaneous resets in ONE call so the shared-scene settle advances untouched
        envs only once."""
        req = [int(i) for i in indices]
        bad = sorted({i for i in req if not (0 <= i < self.cfg.num_envs)})
        if bad:
            raise IndexError(
                f"reset_slots: env indices out of range [0,{self.cfg.num_envs}): {bad}"
            )
        idx = sorted(set(req))
        for i in idx:
            self._reset_slot(self.slots[i])
        if idx:
            self.sim.step(render=bool(self.cameras))  # render so camera products refresh
            # the shared PhysX settle advances EVERY untouched env one physics tick
            # without advancing its logical clock_s -> hidden time. Track it so the
            # caller can hard-stop an episode that accrues > ~0.5 s of it (Turn 20).
            settle_s = 1.0 / 60.0
            # whole-run diagnostic total (Turn 22): NOT a stop trigger -- the 0.5 s
            # hard-stop is per untouched EPISODE via slot.forced_reset_settle_s below.
            self._forced_reset_settle_total_s = getattr(self, "_forced_reset_settle_total_s", 0.0) + settle_s
            reset = set(idx)
            for j, slot in enumerate(self.slots):
                if j not in reset:
                    slot.forced_reset_settle_s = getattr(slot, "forced_reset_settle_s", 0.0) + settle_s
        return self._observe(None)

    # -- stepping ----------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        """Advance one policy step (six 60 Hz physics steps) for every env."""
        cfg = self.cfg
        decoded = decode_policy_actions(actions, self.spec)
        dones = np.zeros(cfg.num_envs, dtype=bool)
        rewards = np.zeros(cfg.num_envs, dtype=np.float32)

        if cfg.stagec_v2:
            for slot in self.slots:
                self._pin_cycle_v2_reserve(slot)
                # Edge-triggered dump telemetry belongs to exactly one policy
                # transition.  Persistent dump state remains in slot.dumping.
                slot.dump_started_this_step = False
                slot.dump_completed_this_step = False
                slot.dump_aborted_this_step = False
                slot.dump_start_ids = ()
                slot.dump_start_mode = None
                slot.dump_remaining_count = 0

        last_k = self.spec.physics_steps_per_action - 1
        for k in range(self.spec.physics_steps_per_action):
            # the final physics step of the window is a full Kit update so the
            # camera render products are fed exactly once per policy step
            self.sim.step(render=bool(self.cameras) and k == last_k)
            # 30 Hz control at k=0,2,4 so the LAST tick precedes the k=last
            # camera render -> the stored frame reflects THIS step's fuel
            # teleports (intake/hub captures, shot spawns), aligned with the
            # proprio/reward the same step already counts (audit finding).
            if k % 2 == 0:
                for slot in self.slots:
                    i = slot.index
                    controller = slot.controller
                    slot.clock_s += 2.0 / 60.0
                    controller.intake_on = bool(decoded.intake_on[i])
                    # wave-5 COMMAND: auto own-court intake -- while the loop
                    # is ready (stockpile count cached by owncourt_ready last
                    # step) and the robot is HOME in the latched suffix, the
                    # intake stays ON regardless of the policy bit, so banked
                    # balls actually enter the chamber.
                    if (
                        cfg.stage_d_auto_oc_intake
                        and bool(getattr(slot.cycle_v2, "latched", False))
                        and int(getattr(slot, "owncourt_collectable", 0))
                        >= int(cfg.stage_d_owncourt_min_balls)
                        and getattr(
                            getattr(slot.cycle_v2, "region", None), "name", ""
                        )
                        == "HOME"
                    ):
                        controller.intake_on = True
                    if not cfg.lock_storage_extended:
                        want_extend = bool(decoded.storage_extended[i])
                        # TRENCH INTERLOCK: the 0.565 m roof physically blocks the
                        # 0.74 m extended envelope, and the storage bit is untrained
                        # in A/B so its Stage-C output is arbitrary. Refuse a deploy
                        # while still under the trench; the policy may extend only
                        # after driving clear. Mirrors the real robot (audit).
                        if want_extend and _under_trench_roof(controller):
                            want_extend = False
                        controller.set_storage_extended(want_extend)
                    controller.step_mechanisms(dt_s=2.0 / 60.0)
                    intake_substeps = (
                        int(cfg.cycle_v2_intake_substeps)
                        if cfg.stagec_v2
                        else 1
                    )
                    for _ in range(intake_substeps):
                        controller.step_intake(
                            slot.fuel,
                            set(slot.router.pending)
                            | set(slot.cycle_v2_reserved_ids),
                            dt_s=2.0 / 60.0,
                        )
                    if (
                        cfg.stage_d
                        and slot.stage_d_first_inactive is None
                        and slot.clock_s >= _stage_d.AUTO_RESULT_DECISION_S
                    ):
                        # Official AUTO result assessed at 23 s (20 s AUTO +
                        # 3 s scoring grace), mirroring the GUI match loop.
                        # The decision is needed at t=30 s, so it always lands
                        # in time; router eligibility then follows the official
                        # shift schedule at sensor time.
                        slot.stage_d_first_inactive = _stage_d.decide_first_inactive(
                            cfg.stage_d_first_inactive,
                            blue_auto_fuel=int(slot.router.scored["blue"]),
                            episode_seed=slot.stage_d_episode_seed,
                            synthetic_red_auto=cfg.stage_d_synthetic_red_auto,
                        )
                        slot.router.match_first_inactive = (
                            slot.stage_d_first_inactive
                        )
                    slot.router.step(slot.clock_s)
                    sm = controller.state_machine
                    sm.set_continuous(False)
                    # Illegal fire is ALWAYS a no-op that keeps the chassis DRIVING.
                    # A press only "counts" (auto-align + the auto-stop below) when a
                    # legal shot exists THIS tick: a HUB shot needs a valid auto-aim
                    # (own court, past the |y|=2.775 board, in range) and a FERRY a
                    # valid ferry solution (refused inside own court). Pressing fire
                    # outside the alliance zone must NEVER freeze the robot -- this is
                    # now unconditional (it used to require mask_illegal_fire=True),
                    # which kills the always-fire freeze collapse for good. Ferry earns
                    # no score, so it cannot drive a collapse; we log its use.
                    has_ammo = bool(controller.magazine)
                    # press-independent legality (also gates the dump-hold below so a
                    # dump can never outlast the ability to actually score)
                    aim_ok = has_ammo and bool(
                        controller.solve_auto_aim("blue").get("valid", False)
                    )
                    ferry_valid = has_ammo and bool(
                        controller.solve_ferry("blue").get("valid", False)
                    )
                    # FIRST is the exact frozen champion, including its ferry
                    # head.  Ferry is disabled only after the verified first
                    # unload, when the ordered suffix policy takes control.  The
                    # shared action wrapper applies the same contract before the
                    # transition is written to replay.
                    v2_phase = (
                        getattr(getattr(slot, "cycle_v2", None), "phase", None)
                        if cfg.stagec_v2
                        else None
                    )
                    v2_phase_name = getattr(v2_phase, "value", None)
                    score_phase = v2_phase_name in (None, "first_cycle", "score")
                    ferry_phase = (
                        not cfg.stagec_v2 or v2_phase_name == "first_cycle"
                    )
                    if cfg.stage_d and cfg.stage_d_ferry and cfg.stagec_v2:
                        # GATE A LIFT (STAGE-D1B): let the trainable suffix ferry
                        # in LEAVE/COLLECT/RETURN -- every phase but SCORE, where a
                        # live hub means we want to shoot, not ferry.
                        ferry_phase = ferry_phase or (v2_phase_name != "score")
                    stage_d_fire_ok = True
                    if (
                        cfg.stage_d
                        and cfg.stage_d_mask_ineligible_fire
                        and cfg.stagec_v2
                        and bool(getattr(slot.cycle_v2, "latched", False))
                    ):
                        # SUFFIX-ONLY Stage D mask: pressing fire at an
                        # ineligible hub is a no-op that keeps the chassis
                        # driving (Stage-B illegal-fire-mask precedent), so a
                        # full load can never be wasted into a dead hub.  The
                        # frozen prefix (pre-latch FIRST_CYCLE) is exempt:
                        # masking would deadlock its baked dump timing.
                        stage_d_fire_ok = _stage_d.blue_hub_eligible(
                            slot.clock_s, slot.stage_d_first_inactive
                        )
                        if (
                            not stage_d_fire_ok
                            and bool(decoded.shoot_blue[i])
                            and aim_ok
                        ):
                            slot.stage_d_masked_fires += 1
                    shoot_ok = (
                        bool(decoded.shoot_blue[i])
                        and aim_ok
                        and score_phase
                        and stage_d_fire_ok
                    )
                    if (
                        cfg.stage_d_ferry_blackout_only
                        and bool(getattr(slot.cycle_v2, "latched", False))
                        and ferry_valid
                        and bool(decoded.ferry[i])
                        and not getattr(slot, "dumping", False)
                        and _stage_d.blue_hub_eligible(
                            slot.clock_s, slot.stage_d_first_inactive
                        )
                    ):
                        # STAGE-D1G (SUFFIX-ONLY, like the shoot mask: the
                        # pre-latch FROZEN PREFIX is exempt -- masking its baked
                        # ferry flings diverged its trajectory and it NEVER
                        # dumped, zero-score episodes): hub LIVE -> ferry no-op
                        # (mask_ineligible_fire precedent: chassis keeps
                        # driving).  slot.dumping is exempt so a blackout-
                        # committed ferry dump keeps its aim validity and rides
                        # to completion across the boundary.
                        ferry_valid = False
                        slot.stage_d_masked_ferries = (
                            getattr(slot, "stage_d_masked_ferries", 0) + 1
                        )
                    if (
                        int(cfg.stage_d_ferry_min_load) > 0
                        and bool(getattr(slot.cycle_v2, "latched", False))
                        and ferry_valid
                        and bool(decoded.ferry[i])
                        and not getattr(slot, "dumping", False)
                        and len(controller.magazine)
                        < int(cfg.stage_d_ferry_min_load)
                    ):
                        # stage_d_v1 wave-2 (SUFFIX-ONLY, prefix exempt like the
                        # D1G mask): ferrying an under-filled chamber is a no-op
                        # -- keep collecting until the volley is worth the
                        # commit.  slot.dumping stays exempt so an in-flight
                        # committed volley finishes normally.
                        ferry_valid = False
                        slot.stage_d_small_ferries = (
                            getattr(slot, "stage_d_small_ferries", 0) + 1
                        )
                    ferry_ok = bool(decoded.ferry[i]) and ferry_valid and ferry_phase
                    # wave-5 COMMAND: auto-ferry.  Same legality the masked
                    # press obeys, minus the press requirement.  The hold
                    # window keeps late-blackout loads for the edge dump
                    # (return-lead then rides them home) -- bank early, hold
                    # late, exactly the intended blackout script.
                    if (
                        int(cfg.stage_d_auto_ferry_load) > 0
                        and bool(getattr(slot.cycle_v2, "latched", False))
                        and not getattr(slot, "dumping", False)
                        and ferry_valid
                        and ferry_phase
                        and len(controller.magazine)
                        >= int(cfg.stage_d_auto_ferry_load)
                        and not _stage_d.blue_hub_eligible(
                            slot.clock_s, slot.stage_d_first_inactive
                        )
                        and not _stage_d.blue_hub_eligible(
                            slot.clock_s + float(cfg.stage_d_auto_ferry_hold_s),
                            slot.stage_d_first_inactive,
                        )
                    ):
                        if not ferry_ok:
                            slot.cycle_v2_stats["stage_d_auto_ferries"] = (
                                int(
                                    slot.cycle_v2_stats.get(
                                        "stage_d_auto_ferries", 0
                                    )
                                )
                                + 1
                            )
                        ferry_ok = True
                    # wave-5 COMMAND: auto score press.  In SCORE at an
                    # eligible hub with valid aim and balls chambered, the
                    # dump commits without waiting for the learned press.
                    if (
                        cfg.stage_d_auto_score_press
                        and bool(getattr(slot.cycle_v2, "latched", False))
                        and score_phase
                        and aim_ok
                        and stage_d_fire_ok
                        and len(controller.magazine) > 0
                        and not getattr(slot, "dumping", False)
                    ):
                        if not shoot_ok:
                            slot.cycle_v2_stats["stage_d_auto_presses"] = (
                                int(
                                    slot.cycle_v2_stats.get(
                                        "stage_d_auto_presses", 0
                                    )
                                )
                                + 1
                            )
                        shoot_ok = True
                    fire = shoot_ok or ferry_ok
                    # prefer the scoring shot when both are legal; ferry only when it
                    # is the legal option
                    fire_mode = "ferry" if (ferry_ok and not shoot_ok) else "score"
                    fire_mode = _effective_fire_mode(
                        fire_mode,
                        dumping=bool(slot.dumping),
                        dump_mode=slot.dump_mode,
                    )
                    if ferry_ok and not shoot_ok and not slot.dumping:
                        slot.ferry_fires = getattr(slot, "ferry_fires", 0) + 1
                        if cfg.stage_d:
                            # wave-6 fix: record the ferry AIM TARGET here (a
                            # free dict read from the solver that already ran
                            # for ferry_valid) instead of a per-step GPU ball-
                            # pose readback in the landing loop.  This verifies
                            # the lane_x/target_y override cheaply and off the
                            # physics hot path.
                            _fev = {"t": round(float(slot.clock_s), 1), "ev": "ferry"}
                            _sol = getattr(controller, "last_aim_solution", None)
                            if isinstance(_sol, dict) and _sol.get("target_xy"):
                                try:
                                    _tx, _ty = _sol["target_xy"]
                                    _fev["tx"] = round(float(_tx), 2)
                                    _fev["ty"] = round(float(_ty), 2)
                                except Exception:
                                    pass
                            slot.cycle_v2_stats.setdefault("timeline", []).append(_fev)
                            # STAGE-D1E: flag a ferry press fired while the hub
                            # is LIVE, so the reward loop can penalize the leak.
                            if _stage_d.blue_hub_eligible(
                                slot.clock_s, slot.stage_d_first_inactive
                            ):
                                slot.active_ferry_this_step = True
                    if cfg.dump_on_press and (
                        fire_mode == "score"
                        or (
                            fire_mode == "ferry"
                            and cfg.stage_d_ferry_dump_on_press
                            and (
                                getattr(slot, "dumping", False)
                                or not _stage_d.blue_hub_eligible(
                                    slot.clock_s, slot.stage_d_first_inactive
                                )
                            )
                        )
                    ):
                        # DUMP-ON-PRESS: one legal fire press COMMITS the robot to stand
                        # completely still, auto-aim, and empty the ENTIRE magazine before
                        # it resumes policy control (get in position -> press once); the FSM
                        # continuous-fires until the chamber is empty. No movement for the
                        # whole dump. Pairs with empty_own_court_penalty.
                        if not getattr(slot, "dumping", False) and fire:
                            slot.dumping = True
                            slot.dump_mode = fire_mode       # lock the aim mode at trigger
                            slot.dump_ticks = 0
                            slot.dump_lost_aim_ticks = 0
                            slot.dump_started_this_step = True
                            slot.dump_start_ids = tuple(int(x) for x in controller.magazine)
                            slot.dump_start_mode = str(fire_mode)
                        if slot.dumping:
                            slot.dump_ticks = getattr(slot, "dump_ticks", 0) + 1
                            # Legality is MODE-SPECIFIC.  A valid ferry must never
                            # keep a lost HUB-shot dump locked.  Give auto-align a
                            # short recovery window for transient chassis wobble,
                            # then release safely; the six-second global cap remains.
                            dump_aim_valid = aim_ok if slot.dump_mode == "score" else ferry_valid
                            if dump_aim_valid:
                                slot.dump_lost_aim_ticks = 0
                            else:
                                slot.dump_lost_aim_ticks = (
                                    getattr(slot, "dump_lost_aim_ticks", 0) + 1
                                )
                            dump_empty = not controller.magazine
                            dump_timed_out = slot.dump_ticks > cfg.max_dump_ticks
                            lost_aim_grace = (
                                int(cfg.cycle_v2_dump_lost_aim_grace_ticks)
                                if cfg.stagec_v2
                                else 0
                            )
                            dump_lost = (
                                slot.dump_lost_aim_ticks
                                > lost_aim_grace
                            )
                            if dump_empty or dump_timed_out or dump_lost:
                                slot.dumping = False
                                slot.dump_remaining_count = len(controller.magazine)
                                if dump_empty:
                                    slot.dump_completed_this_step = True
                                else:
                                    slot.dump_aborted_this_step = True
                        firing = bool(slot.dumping)
                        if firing:
                            fire_mode = slot.dump_mode        # keep aiming the same way
                        sm.set_continuous(firing)             # feed until the magazine empties
                        sm.set_emergency_stop(False)
                        sm.auto_align = firing
                        driver = decoded.driver[i]
                        # ZERO movement while dumping the chamber (aim + fire only)
                        moving = (not firing) and bool(np.any(np.abs(driver) > 0.03))
                    else:
                        # ONE-CLICK SINGLE (default): one legal press latches ONE ball;
                        # holding re-queues -> cooldown-rate rapid fire. Firing auto-stops
                        # the drive so the shooter can auto-align ("one button" scoring).
                        if fire:
                            sm.request_single()
                        sm.set_emergency_stop(False)
                        sm.auto_align = fire
                        driver = decoded.driver[i]
                        moving = bool(np.any(np.abs(driver) > 0.03)) and not fire
                    if moving:
                        # PHASE-GATED SPEED: the frozen prefix owns the FIRST
                        # cycle and can never be retrained, so it keeps the
                        # damped mapping it learned.  Only the trainable
                        # suffix gets the robot's real 3.21 m/s.
                        # PHASE-GATED SPEED CURRICULUM: the frozen prefix
                        # owns the FIRST cycle and can never be retrained,
                        # so it always keeps the damped mapping; only the
                        # trainable suffix follows the ramp.
                        _sc = _policy_speed_scale()
                        _in_first = False
                        if cfg.stagec_v2:
                            _ph = getattr(
                                getattr(slot, 'cycle_v2', None), 'phase', None
                            )
                            _in_first = getattr(_ph, 'value', '') == 'first_cycle'
                        if _in_first:
                            _sc = None
                        _fs = _POLICY_FULL_SPEED and _sc is None and not _in_first
                        controller.drive(
                            float(driver[0]),
                            float(driver[2]),
                            strafe=float(driver[1]),
                            keyboard_scale=not _fs,
                            speed_scale=_sc,
                        )
                    controller.update(
                        slot.fuel,
                        now_s=slot.clock_s,
                        alliance="blue",
                        hub_active=True,
                        allow_drive=not moving,
                        fire_mode=fire_mode,
                    )

        from frc_rebuilt.rl.custody import collect_custody, score_custody
        info: dict[str, Any] = {"reward_components": []}
        for slot in self.slots:
            i = slot.index
            cust = slot.custody
            # score custody: credit each newly-scored blue ball by its index (first=full,
            # repeat=rho). Consumes the router's per-ball score_events since the last step;
            # the event list is appended in lockstep with scored++ so counts stay consistent.
            events = slot.router.score_events
            new_blue = [idx for (al, idx) in events[cust.score_events_seen:] if al == "blue"]
            cust.score_events_seen = len(events)
            custody_score = score_custody(
                new_blue, cust, self.cfg.score_reward_weight, self.cfg.rho_score
            )
            # collect custody: balls newly appearing in the magazine (a recycled/previously
            # scored ball -> rho; a fresh field ball -> full).
            custody_collect = collect_custody(
                slot.controller.magazine, cust, self.collect_weight, self.cfg.rho_collect
            )
            r_action = -self.cfg.action_penalty * float(np.square(actions[i, :3]).sum())
            pos, _ = slot.controller.chassis_pose()
            if cfg.stagec_v2:
                from frc_rebuilt.rl.cycle_v2 import Milestone

                phase_before = slot.cycle_v2.phase
                r_progress = self._cycle_v2_progress_reward(slot, phase_before, pos)
                time_remaining = 1.0 - min(
                    1.0, float(slot.clock_s) / max(1.0, float(cfg.episode_len_s))
                )
                owncourt_ready = self._cycle_v2_owncourt_ready(slot)
                # STAGE-D1E: tell the cycle machine the hub is live so it returns
                # to score at target_load instead of dwelling in COLLECT.
                stage_d_hub_live = bool(
                    cfg.stage_d
                    and cfg.stage_d_return_when_live
                    and (
                        _stage_d.blue_hub_eligible(
                            slot.clock_s, slot.stage_d_first_inactive
                        )
                        or (
                            # stage_d_v1 wave-4 return lead: reactivation is
                            # close -- start the loaded return NOW so the dump
                            # fires at the edge, not after a post-edge commute.
                            float(cfg.stage_d_return_lead_s) > 0.0
                            and _stage_d.blue_hub_eligible(
                                slot.clock_s + float(cfg.stage_d_return_lead_s),
                                slot.stage_d_first_inactive,
                            )
                        )
                    )
                )
                cycle_step = slot.cycle_v2.update(
                    slot.controller.magazine,
                    new_blue,
                    pos,
                    int(slot.router.scored["blue"]),
                    time_remaining=time_remaining,
                    score_dump_started=bool(
                        slot.dump_started_this_step
                        and slot.dump_start_mode == "score"
                    ),
                    score_dump_completed=bool(
                        slot.dump_completed_this_step
                        and slot.dump_mode == "score"
                    ),
                    owncourt_score_ready=owncourt_ready,
                    hub_live=stage_d_hub_live,
                )
                r_score = float(cycle_step.score_reward)
                r_collect = float(cycle_step.collect_reward)
                r_milestone = self._cycle_v2_milestone_reward(
                    slot,
                    cycle_step.milestones,
                    float(pos[0]),
                    qualified_load=int(cycle_step.qualified_load),
                )
                if slot.dump_completed_this_step and slot.dump_mode == "score":
                    # A new leave/collect/return/score sequence starts now.
                    slot.cycle_v2_penalty_spent = {
                        "leave": 0.0,
                        "return": 0.0,
                        "score": 0.0,
                    }
                    slot.cycle_v2_border_spent = 0.0
                    slot.cycle_v2_outer_rail_spent = 0.0
                    slot.stage_d_deep_red_spent = 0.0
                    slot.stage_d_deep_red_streak = 0
                    slot.stage_d_idle_spent = 0.0
                    slot.stage_d_idle_streak = 0
                    slot.stage_d_live_dump_paid = False
                    slot.stage_d_rescued = False
                    slot.stage_d_arrival_paid = False
                    slot.cycle_v2_outer_rail_active = False
                    slot.cycle_v2_outer_rail_streak = 0
                r_behavior = self._cycle_v2_delay_penalty(slot, cycle_step)
                if slot.dump_aborted_this_step and slot.dump_mode == "score":
                    r_behavior -= min(
                        float(cfg.cycle_v2_partial_dump_penalty_cap),
                        float(cfg.cycle_v2_partial_dump_penalty_per_ball)
                        * float(slot.dump_remaining_count),
                    )
                r_border = self._cycle_v2_border_camp_penalty(slot, float(pos[1]))
                r_outer_rail = self._cycle_v2_outer_rail_penalty(
                    slot,
                    cycle_step,
                    float(pos[0]),
                    collected_gain=len(cycle_step.collected_ids),
                )
                rewards[i] = r_score + r_collect + r_progress + r_milestone + r_action
                rewards[i] += r_behavior + r_border + r_outer_rail

                stats = slot.cycle_v2_stats
                # OWN-COURT short-loop telemetry (STAGE-D1C).  An own-court
                # SCORE re-entry is a SCORE phase reached from LEAVE/COLLECT --
                # the ordinary path is RETURN->SCORE, so this isolates short
                # loops.  Count the entries, the score-dumps completed inside the
                # window, and the +10 events landed, to measure short-loop usage.
                if cfg.stage_d and cfg.stage_d_owncourt_loop and cfg.stage_d_ferry:
                    if (
                        phase_before.value in ("leave", "collect")
                        and cycle_step.phase.value == "score"
                    ):
                        slot.owncourt_score_active = True
                        stats["owncourt_score_entries"] = (
                            int(stats.get("owncourt_score_entries", 0)) + 1
                        )
                        stats.setdefault("timeline", []).append(
                            {"t": round(float(slot.clock_s), 1), "ev": "oc_entry"}
                        )
                    if getattr(slot, "owncourt_score_active", False):
                        stats["owncourt_scored"] = int(
                            stats.get("owncourt_scored", 0)
                        ) + int(cycle_step.full_score_events)
                        if (
                            slot.dump_completed_this_step
                            and slot.dump_mode == "score"
                        ):
                            stats["owncourt_shots"] = (
                                int(stats.get("owncourt_shots", 0)) + 1
                            )
                            slot.owncourt_score_active = False
                            led = getattr(
                                slot.cycle_v2, "_score_dump_ledger", None
                            )
                            if led is not None:
                                if not hasattr(slot, "owncourt_dump_ledgers"):
                                    slot.owncourt_dump_ledgers = []
                                slot.owncourt_dump_ledgers.append(led)
                            # STAGE-D1F change-4: re-arm the short loop while the
                            # stockpile still has >= min_balls entitled loose in
                            # own court and the hub is live -- drop back to
                            # COLLECT so the next re-intake re-triggers SCORE and
                            # the whole stockpile converts, not just ~1.4 balls.
                            if (
                                cfg.stage_d_owncourt_rearm
                                and self._cycle_v2_owncourt_ready(slot)
                            ):
                                slot.cycle_v2.rearm_owncourt_collect()
                    # A lobbed volley keeps landing AFTER the chamber-empty
                    # edge closes the window above; each dump ledger keeps
                    # crediting those late +10s, so recompute every step.
                    stats["owncourt_ledger_scored"] = sum(
                        len(l.scored_ids)
                        for l in getattr(slot, "owncourt_dump_ledgers", ())
                    )
                    # STAGE-D1F change-5: owncourt_scored counted full_score_events
                    # only while the window flag was up, but qualified +10s land
                    # after the chamber-empty edge closes it -> it read 0. Alias
                    # the trustworthy ledger count so the metric is measurable.
                    stats["owncourt_scored"] = int(stats["owncourt_ledger_scored"])
                    # STAGE-D1F change-2: re-intake PULL. Reward pulling loose
                    # entitled balls back into the magazine while the loop is
                    # ready (qualified_load>0 is what triggers SCORE re-entry).
                    # owncourt_ready already implies latched + hub live.
                    _ir = float(cfg.stage_d_owncourt_intake_reward)
                    _qim = int(cycle_step.qualified_load)
                    if _ir > 0.0 and owncourt_ready:
                        _prev_qim = int(getattr(slot, "owncourt_prev_ent_in_mag", 0))
                        _gain = _qim - _prev_qim
                        if _gain > 0:
                            rewards[i] += _ir * _gain
                            stats["owncourt_intake_reward"] = (
                                float(stats.get("owncourt_intake_reward", 0.0))
                                + _ir * _gain
                            )
                    slot.owncourt_prev_ent_in_mag = _qim
                    stats["owncourt_qual_in_mag"] = _qim
                if cfg.stage_d:
                    n_full = int(cycle_step.full_score_events)
                    n_unq = int(cycle_step.unqualified_score_events)
                    if n_full + n_unq > 0:
                        stats.setdefault("timeline", []).append({
                            "t": round(float(slot.clock_s), 1),
                            "ev": "score",
                            "q": n_full,
                            "u": n_unq,
                            "oc": bool(getattr(slot, "owncourt_score_active", False)),
                            "elig": bool(_stage_d.blue_hub_eligible(
                                slot.clock_s, slot.stage_d_first_inactive)),
                        })
                    # STALL-RESCUE (2026-07-30): the frozen prefix owns the
                    # protected FIRST cycle; when it wedges (no dump by the
                    # rescue clock) the episode is otherwise a guaranteed ~0.
                    # Flipping the FSM to LEAVE hands the suffix control via
                    # the phase one-hot; masks follow automatically.
                    _prs = float(cfg.stage_d_prefix_rescue_s)
                    if (
                        _prs > 0.0
                        and float(slot.clock_s) >= _prs
                        and getattr(slot.cycle_v2, "phase", None) is not None
                        and getattr(slot.cycle_v2.phase, "value", "") == "first_cycle"
                        and not getattr(slot, "stage_d_rescued", False)
                    ):
                        from frc_rebuilt.rl.cycle_v2 import CyclePhase as _CP
                        _cv = slot.cycle_v2
                        # Mimic the natural first-dump latch transition
                        # (cycle_v2.update lines ~645-657) minus the LATCHED
                        # milestone (no unearned reward): without latched=True
                        # every later phase advance is gated off and the FSM
                        # wedges in LEAVE (proved by smoke A, 2026-07-30).
                        _cv._score_dump_active = False
                        _cv._score_dump_ledger = None
                        _cv.latched = True
                        _cv._set_phase(_CP.LEAVE)
                        _cv.cycle_index = 2
                        _cv.cycle_started_step = _cv.step_count
                        _cv._milestones_this_cycle = set()
                        slot.stage_d_rescued = True
                        stats["prefix_rescues"] = int(stats.get("prefix_rescues", 0)) + 1
                    # STAGE-D2: chamber load every step; the last value on the
                    # episode is the end-of-window load the chamber gate reads.
                    stats["chamber_load"] = int(len(slot.controller.magazine))
                    # Which match window this episode drilled -- without this the
                    # 105 s / 130 s live starts are invisible whenever they score
                    # nothing, so the window split cannot be audited.
                    # FULL mode leaves these as None (no lane window), and
                    # getattr returns that None rather than the default -- so
                    # float(None) crashed every full-match collector.  Coerce.
                    _lt0 = getattr(slot, "stage_d_lane_t0", None)
                    _len = getattr(slot, "stage_d_lane_end_s", None)
                    stats["lane_t0"] = round(float(_lt0), 1) if _lt0 is not None else -1.0
                    stats["lane_end"] = round(float(_len), 1) if _len is not None else -1.0
                    # SECTION 3: pay for STARTING a score dump with a real
                    # load while the hub is live.  Once per episode.
                    _ldr = float(cfg.stage_d_live_dump_reward)
                    if (
                        _ldr > 0.0
                        # PHASE-GATED (was: only in the "live" lane, so it never
                        # fired in a full match).  Hub-live + real load are the
                        # real conditions; the lane was never the point.
                        and slot.dump_started_this_step
                        and slot.dump_start_mode == "score"
                        and not getattr(slot, "stage_d_live_dump_paid", False)
                        and len(slot.controller.magazine)
                        >= int(cfg.stage_d_live_dump_min_load)
                        and _stage_d.blue_hub_eligible(
                            slot.clock_s, slot.stage_d_first_inactive
                        )
                    ):
                        slot.stage_d_live_dump_paid = True
                        rewards[i] += _ldr
                        stats["live_dump_reward"] = (
                            float(stats.get("live_dump_reward", 0.0)) + _ldr
                        )
                    # SECTION 2 objective, in-phase: arriving HOME loaded while
                    # the hub is dark is exactly what the blackout drill wanted.
                    # Ferrying already pays per ball; nothing paid for the
                    # ARRIVAL STATE, which is why chamber-home sat at 4.7 for
                    # 1041 episodes.  Scaled by load, once per blackout.
                    _har = float(cfg.stage_d_home_arrival_reward)
                    if _har > 0.0 and not _stage_d.blue_hub_eligible(
                        slot.clock_s, slot.stage_d_first_inactive
                    ):
                        _home = (
                            getattr(getattr(cycle_step, "region", None), "value", "")
                            == "home"
                        )
                        _load = int(len(slot.controller.magazine))
                        if (
                            _home
                            and _load >= int(cfg.stage_d_home_arrival_min_load)
                            and not getattr(slot, "stage_d_arrival_paid", False)
                        ):
                            slot.stage_d_arrival_paid = True
                            _pay = _har * float(_load)
                            rewards[i] += _pay
                            stats["home_arrival_reward"] = (
                                float(stats.get("home_arrival_reward", 0.0)) + _pay
                            )
                            stats["home_arrival_load"] = _load
                    elif getattr(slot, "stage_d_arrival_paid", False):
                        # hub came back live -> re-arm for the next blackout
                        slot.stage_d_arrival_paid = False
                    # SECTION 1: time pressure -- every step of the opener window
                    # costs, so a faster second cycle is strictly better.
                    _otp = float(cfg.stage_d_opener_time_penalty)
                    if _otp > 0.0 and float(slot.clock_s) < float(
                        cfg.stage_d_opener_span_s
                    ):
                        rewards[i] -= _otp
                        stats["opener_time_penalty"] = (
                            float(stats.get("opener_time_penalty", 0.0)) - _otp
                        )
                    # STAGE-D2 penalties.  Both are charged ONLY in the
                    # trainable latched suffix: the frozen prefix owns FIRST and
                    # its executed actions must stay exactly as trained.
                    if bool(getattr(slot.cycle_v2, "latched", False)):
                        _pos, _ = slot.controller.chassis_pose()
                        _lin, _ = slot.controller.chassis_velocity()
                        _y = float(_pos[1])
                        _speed = math.hypot(float(_lin[0]), float(_lin[1]))
                        _dr = float(cfg.stage_d_deep_red_penalty)
                        if _dr > 0.0 and _y >= float(cfg.stage_d_deep_red_y):
                            slot.stage_d_deep_red_streak += 1
                            stats["deep_red_steps"] = (
                                int(stats.get("deep_red_steps", 0)) + 1
                            )
                            if slot.stage_d_deep_red_streak > int(
                                cfg.stage_d_deep_red_grace_steps
                            ):
                                _room = float(
                                    cfg.stage_d_deep_red_penalty_cap
                                ) - float(slot.stage_d_deep_red_spent)
                                if _room > 0.0:
                                    _p = min(_dr, _room)
                                    rewards[i] -= _p
                                    slot.stage_d_deep_red_spent += _p
                                    stats["deep_red_penalty"] = (
                                        float(stats.get("deep_red_penalty", 0.0)) - _p
                                    )
                        else:
                            slot.stage_d_deep_red_streak = 0
                        _ip = float(cfg.stage_d_idle_penalty)
                        _in_score = (
                            getattr(cycle_step.phase, "value", "") == "score"
                        )
                        if (
                            _ip > 0.0
                            and not _in_score
                            and _speed < float(cfg.stage_d_idle_speed_mps)
                        ):
                            slot.stage_d_idle_streak += 1
                            stats["idle_steps"] = int(stats.get("idle_steps", 0)) + 1
                            if slot.stage_d_idle_streak > int(
                                cfg.stage_d_idle_grace_steps
                            ):
                                _room = float(cfg.stage_d_idle_penalty_cap) - float(
                                    slot.stage_d_idle_spent
                                )
                                if _room > 0.0:
                                    _p = min(_ip, _room)
                                    rewards[i] -= _p
                                    slot.stage_d_idle_spent += _p
                                    stats["idle_penalty"] = (
                                        float(stats.get("idle_penalty", 0.0)) - _p
                                    )
                        else:
                            slot.stage_d_idle_streak = 0
                            slot.stage_d_live_dump_paid = False
                            slot.stage_d_rescued = False
                            slot.stage_d_arrival_paid = False
                # FERRY repatriation shaping (STAGE-D1B): blackout-gated,
                # custody-based, credited ONCE per ball.  A qualified ball
                # ferried home keeps its +10 downstream score entitlement; this
                # small bootstrap term only pays while the blue hub is
                # INELIGIBLE (scoring is strictly better when it is live) and
                # only in the trainable latched suffix.
                if cfg.stage_d and cfg.stage_d_ferry:
                    mag_now = {int(b) for b in slot.controller.magazine}
                    prev_mag = getattr(slot, "ferry_prev_mag_ids", frozenset())
                    r_ferry = 0.0
                    latched = bool(getattr(slot.cycle_v2, "latched", False))
                    blue_elig = _stage_d.blue_hub_eligible(
                        slot.clock_s, slot.stage_d_first_inactive
                    )
                    if latched and not blue_elig:
                        # In the latched suffix during a blackout, scoring is
                        # masked and dumps are score-only, so the ONLY way a
                        # ball leaves the magazine is a ferry launch (the solver
                        # refuses to ferry from own court -> one-way by physics).
                        # Credit each distinct ball ONCE: a re-collected+
                        # re-ferried ball is never paid twice, killing the
                        # ping-pong exploit at the custody layer.
                        if cfg.stage_d_ferry_entitled_only:  # STAGE-D1F
                            ent = {int(x) for x in slot.cycle_v2.qualified_ids}
                            newly = ((prev_mag - mag_now) & ent) - slot.ferried_ids
                        else:
                            newly = (prev_mag - mag_now) - slot.ferried_ids
                        if newly:
                            slot.ferried_ids |= newly
                            n_new = len(newly)
                            r_ferry = float(cfg.stage_d_ferry_reward) * n_new
                            rewards[i] += r_ferry
                            stats["ferried_balls"] = (
                                int(stats.get("ferried_balls", 0)) + n_new
                            )
                            # stage_d_v1 wave-2: timestamped balls-returned event
                            # so the match-time profile can chart actual balls
                            # banked (not just presses).  n = balls that reached
                            # own court on this step of the ferry volley.  wave-6
                            # fix: NO GPU readback here (moved to the aim-target
                            # stamp at ferry-press) -- this runs in the physics
                            # hot loop and a per-step get_world_poses across 12
                            # collectors was a load risk.
                            stats.setdefault("timeline", []).append({
                                "t": round(float(slot.clock_s), 1),
                                "ev": "ferry_land",
                                "n": int(n_new),
                            })
                    stats["v2_ferry_reward"] = (
                        float(stats.get("v2_ferry_reward", 0.0)) + r_ferry
                    )
                    stats["own_court_stockpile"] = len(slot.ferried_ids)
                    stats["stage_d_masked_ferries"] = int(
                        getattr(slot, "stage_d_masked_ferries", 0)
                    )
                    stats["stage_d_small_ferries"] = int(
                        getattr(slot, "stage_d_small_ferries", 0)
                    )
                    # STAGE-D1E: penalize a ferry press fired while the hub was
                    # LIVE (leaked fire that should have been a shot or a
                    # reposition); only in the trainable latched suffix.
                    if getattr(slot, "active_ferry_this_step", False):
                        slot.active_ferry_this_step = False
                        pen = float(cfg.stage_d_active_ferry_penalty)
                        if latched and pen > 0.0:
                            rewards[i] -= pen
                            stats["active_ferry_penalty"] = (
                                float(stats.get("active_ferry_penalty", 0.0)) - pen
                            )
                            stats["active_ferry_presses"] = (
                                int(stats.get("active_ferry_presses", 0)) + 1
                            )
                    slot.ferry_prev_mag_ids = frozenset(mag_now)
                if slot.dump_started_this_step and slot.dump_start_mode == "score":
                    stats["dump_attempts"] += 1
                if slot.dump_completed_this_step and slot.dump_mode == "score":
                    stats["dump_empty_completions"] += 1
                if slot.dump_aborted_this_step and slot.dump_mode == "score":
                    stats["partial_dumps"] += 1
                    stats["partial_dump_balls_left"] += int(slot.dump_remaining_count)
                stats["full_score_events"] += int(cycle_step.full_score_events)
                stats["unqualified_score_events"] += int(
                    cycle_step.unqualified_score_events
                )
                stats["qualified_collections"] += sum(
                    int(ball_id in cycle_step.qualified_ids)
                    for ball_id in cycle_step.collected_ids
                )
                stats["v2_score_reward"] += r_score
                stats["v2_collect_reward"] += r_collect
                stats["v2_progress_reward"] += r_progress
                stats["v2_milestone_reward"] += r_milestone
                stats["v2_behavior_penalty"] += r_behavior
                stats["v2_border_penalty"] += r_border
                stats["v2_outer_rail_penalty"] += r_outer_rail
                if cycle_step.collect_exit_reason is not None:
                    key = f"collect_exit_{cycle_step.collect_exit_reason}"
                    stats[key] = int(stats.get(key, 0)) + 1
                for event in cycle_step.milestones:
                    key = event.name.value
                    stats["milestones"][key] = int(stats["milestones"].get(key, 0)) + 1
                if (
                    slot.cycle_v2_mode == "full"
                    and any(event.name is Milestone.LATCHED for event in cycle_step.milestones)
                ):
                    # The champion has finished; select the nearer ramp without
                    # retroactively perturbing its reset RNG stream.
                    self._cycle_v2_select_ramp_side(slot, float(pos[0]))
                if (
                    bool(cfg.cycle_v2_refresh_ramp_side_on_dump)
                    and any(
                        event.name is Milestone.CYCLE_DUMPED
                        for event in cycle_step.milestones
                    )
                ):
                    # Cycle 3+ must inherit the ramp nearest the newly completed
                    # scoring pose, not the stale side selected after cycle 1.
                    self._cycle_v2_select_ramp_side(slot, float(pos[0]))
                if any(
                    event.name in (
                        Milestone.LATCHED,
                        Milestone.CYCLE_DUMPED,
                    )
                    for event in cycle_step.milestones
                ):
                    released = self.release_cycle_v2_reserve(slot.index)
                    stats["reserve_releases"] += int(bool(released))
                if self._cycle_v2_skill_succeeded(
                    slot, cycle_step.milestones, float(pos[0])
                ):
                    dones[i] = True
                    slot.cycle_v2_terminal_reason = "skill_success"
                elif self._cycle_v2_skill_failed(
                    slot, cycle_step.milestones, float(pos[0])
                ):
                    dones[i] = True
                    slot.cycle_v2_terminal_reason = "off_ramp_exit"

                info["reward_components"].append(
                    {
                        "score": r_score,
                        "collect": r_collect,
                        "progress": r_progress,
                        "milestone": r_milestone,
                        "action": r_action,
                        "behavior": r_behavior,
                        "border": r_border,
                        "outer_rail": r_outer_rail,
                        "legacy_custody_score": float(custody_score),
                        "legacy_custody_collect": float(custody_collect),
                    }
                )
            else:
                r_score = custody_score
                r_collect = custody_collect
                empty_home = len(slot.controller.magazine) == 0 and float(pos[1]) < -2.775
                r_behavior = -self.cfg.empty_own_court_penalty * float(empty_home)
                rewards[i] = r_score + r_collect + r_action + r_behavior
                info["reward_components"].append(
                    {"score": r_score, "collect": r_collect, "action": r_action, "behavior": r_behavior}
                )
            slot.score_seen = int(slot.router.scored["blue"])
            slot.collected_seen = int(slot.controller.balls_collected)
            slot.prev_action[:] = np.clip(actions[i], -1.0, 1.0)
            _lane_end = getattr(slot, "stage_d_lane_end_s", None)
            if slot.clock_s >= cfg.episode_len_s or (
                _lane_end is not None and slot.clock_s >= float(_lane_end)
            ):
                dones[i] = True
                if cfg.stagec_v2 and not slot.cycle_v2_terminal_reason:
                    slot.cycle_v2_terminal_reason = "horizon"
        info["ferry_fires"] = int(sum(getattr(s, "ferry_fires", 0) for s in self.slots))

        # physics hygiene: force-reset (and end the episode for) any env whose
        # robot/fuel state has gone non-finite or exploded, BEFORE it is observed
        # or poisons neighbours through the shared GPU PhysX solver.
        unhealthy = self._detect_unhealthy()
        if bool(unhealthy.any()):
            self._unhealthy_trips = getattr(self, "_unhealthy_trips", 0) + int(unhealthy.sum())
            for slot in self.slots:
                if unhealthy[slot.index]:
                    rewards[slot.index] = 0.0
                    dones[slot.index] = True
                    if cfg.stagec_v2:
                        slot.cycle_v2_terminal_reason = "unhealthy"
            # A single all-bad step is usually a correlated settle-step blip that the
            # per-env force-reset (above) + jittered respawn recover from. Only a
            # SUSTAINED all-bad run means the shared solver is truly, globally corrupt
            # (nothing healthy left to protect): bail so the supervisor rebuilds.
            self._consec_all_bad = (
                getattr(self, "_consec_all_bad", 0) + 1 if bool(unhealthy.all()) else 0
            )
            if self._consec_all_bad >= 3 or self._unhealthy_trips > cfg.max_unhealthy_trips:
                raise SimulationUnstable(
                    f"physics unstable: {int(unhealthy.sum())}/{cfg.num_envs} envs this "
                    f"step, {self._unhealthy_trips} force-resets total, "
                    f"{self._consec_all_bad} consecutive all-bad steps"
                )
        else:
            self._consec_all_bad = 0
        info["unhealthy"] = [int(x) for x in np.flatnonzero(unhealthy)]

        # terminal episode stats, captured BEFORE auto-reset wipes the counters
        info["episode_stats"] = {
            int(slot.index): {
                "scored": int(slot.router.scored["blue"]),
                "collected": int(slot.controller.balls_collected),
                "shots_fired": int(slot.controller.shots_fired),
                # custody-bite telemetry (Turn 31 §6): fresh vs recycled credits
                "fresh_score": int(slot.custody.fresh_score),
                "recycled_score": int(slot.custody.recycled_score),
                "fresh_collect": int(slot.custody.fresh_collect),
                "recycled_collect": int(slot.custody.recycled_collect),
                "neutral_loaded": bool(slot.episode_neutral_loaded),
                **(
                    {
                        "stagec_v2": True,
                        "reset_mode": str(slot.cycle_v2_mode),
                        "terminal_phase": slot.cycle_v2.phase.value,
                        "latched": bool(slot.cycle_v2.latched),
                        "cycle_index": int(slot.cycle_v2.cycle_index),
                        "cycles_attempted": int(
                            getattr(slot.cycle_v2, "cycles_attempted", 0)
                        ),
                        "cycles_completed": int(slot.cycle_v2.cycles_completed),
                        "qualified_load": int(
                            len(
                                slot.cycle_v2.qualified_ids.intersection(
                                    slot.controller.magazine
                                )
                            )
                        ),
                        "terminal_reason": str(slot.cycle_v2_terminal_reason or "done"),
                        "dump_on_press": bool(cfg.dump_on_press),
                        "reserve_ids_remaining": int(len(slot.cycle_v2_reserved_ids)),
                        "reserve_batches_remaining": int(
                            len(slot.cycle_v2_reserved_batches)
                        ),
                        "return_skill_preload": int(
                            slot.cycle_v2_return_preload_count
                        ),
                        **(
                            {
                                "stage_d": True,
                                "stage_d_first_inactive": str(
                                    slot.stage_d_first_inactive
                                ),
                                "stage_d_masked_fires": int(
                                    slot.stage_d_masked_fires
                                ),
                            }
                            if cfg.stage_d
                            else {}
                        ),
                        **slot.cycle_v2_stats,
                    }
                    if cfg.stagec_v2
                    else {}
                ),
            }
            for slot in self.slots
            if dones[slot.index]
        }
        # Auto-reset BEFORE observing, so the returned observation for a done
        # env is the FIRST frame of its new episode (never a stale terminal
        # frame).  Terminal next-observations are unused by the learner: the
        # stored done flag zeroes the bootstrap on episode-ending transitions.
        if bool(dones.any()):
            for slot in self.slots:
                if dones[slot.index]:
                    self._reset_slot(slot)
            # settle step; rendered so post-reset camera frames are fresh
            self.sim.step(render=bool(self.cameras))
        observations = self._observe(decoded)
        return observations, rewards, dones, info

    # -- observations -------------------------------------------------------
    # Official phase edges.  A section lane must stop when its phase stops --
    # the endgame is 30 s while every shift is 25 s, so a single span cannot
    # express all of them.
    _PHASE_EDGES = (30.0, 55.0, 80.0, 105.0, 130.0, 160.0)

    @staticmethod
    def _live_block_end(t0: float, first_inactive, horizon: float) -> float:
        """End of the CONTIGUOUS window in which our hub stays live.

        For blue the hub comes up at 105 and stays up through 160 -- the ENDGAME
        does not deactivate it, it only brings red's hub up as well.  So 105-160
        is ONE 55 s live block, not a 25 s shift plus a 30 s endgame.  Walking
        eligibility forward keeps this correct for either parity.
        """
        t = float(t0)
        step = 0.5
        while t + step < float(horizon):
            if not _stage_d.blue_hub_eligible(t + step, first_inactive):
                return t + step
            t += step
        return float(horizon)

    @classmethod
    def _phase_end_after(cls, t0: float, fallback: float) -> float:
        for _e in cls._PHASE_EDGES:
            if _e > float(t0) + 1e-6:
                return float(_e)
        return float(fallback)

    def _observe(self, decoded) -> dict[str, np.ndarray]:
        n = self.cfg.num_envs
        obs: dict[str, np.ndarray] = {}
        if self.cameras:
            w, h = self.camera_resolution
            frames = np.zeros((n, len(self.camera_names), h, w, 3), np.uint8)
            for (i, name), camera in self.cameras.items():
                rgba = np.asarray(camera.get_rgba())
                if rgba.size and rgba.ndim == 3:
                    frames[i, self.camera_names.index(name)] = rgba[..., :3]
            obs["rgb"] = frames

        proprio_dim = 30 if self.cfg.stagec_v2 else 22
        proprio = np.zeros((n, proprio_dim), np.float32)
        privileged = np.zeros((n, 26), np.float32)
        for slot in self.slots:
            i = slot.index
            controller = slot.controller
            position, quat = controller.chassis_pose()
            yaw = controller.chassis_yaw()
            linear, yaw_rate = controller.chassis_velocity()
            noise = self.rng.normal(0.0, self.cfg.proprio_noise_std, 4).astype(np.float32)
            mag = len(controller.magazine)
            state = controller.state_machine.state.value
            legacy_proprio = np.concatenate(
                [
                    np.asarray(
                        [
                            (position[0] + noise[0]) / 8.0,
                            (position[1] + noise[1]) / 8.0,
                            math.sin(yaw) + noise[2] * 0.1,
                            math.cos(yaw) + noise[3] * 0.1,
                            linear[0] / 4.0,
                            linear[1] / 4.0,
                            yaw_rate / 6.0,
                            slot.clock_s / max(1.0, self.cfg.episode_len_s),
                            mag / 8.0,
                            1.0 if controller.intake_on else 0.0,
                            controller.storage_position,
                            1.0 if state in ("READY", "FEEDING") else 0.0,
                            (
                                # Stage D: real blue-hub eligibility (1.0
                                # active/grace, 0.5 deactivation warning,
                                # 0.0 inactive).  Pre-Stage-D: the constant
                                # 1.0 the frozen prefix trained on.
                                _stage_d.blue_hub_obs(
                                    slot.clock_s, slot.stage_d_first_inactive
                                )
                                if self.cfg.stage_d
                                else 1.0
                            ),
                            float(controller.shots_fired) / 20.0,
                            float(slot.router.scored["blue"]) / 20.0,
                        ],
                        np.float32,
                    ),
                    slot.prev_action.astype(np.float32),
                ]
            )
            if self.cfg.stagec_v2:
                remaining = 1.0 - min(
                    1.0, float(slot.clock_s) / max(1.0, float(self.cfg.episode_len_s))
                )
                phase_features = np.asarray(
                    slot.cycle_v2.feature_vector(
                        controller.magazine, time_remaining=remaining
                    ),
                    np.float32,
                )
                if phase_features.shape != (8,):
                    raise RuntimeError(
                        f"Stage C v2 phase feature contract changed: {phase_features.shape}"
                    )
                proprio[i] = np.concatenate([legacy_proprio, phase_features])
            else:
                proprio[i] = legacy_proprio
            # privileged (critic-only): true pose + nearest COLLECTIBLE-fuel
            # geometry. Exclude balls carried in the robot (captured_indices) and
            # hub-parked / out-of-field balls, so the critic sees the real nearest
            # collectible target during carrying/scoring, not a ball at distance 0.
            balls, _ = slot.fuel.get_world_poses()
            deltas = balls[:, :2] - position[None, :2]
            distances = np.linalg.norm(deltas, axis=1)
            excl = np.zeros(len(balls), dtype=bool)
            captured = getattr(controller, "captured_indices", None)
            if captured:
                ci = np.fromiter(
                    (j for j in captured if 0 <= j < len(balls)), dtype=np.int64
                )
                if ci.size:
                    excl[ci] = True
            excl |= (np.abs(balls[:, 0]) > 8.2) | (np.abs(balls[:, 1]) > 8.2)
            eff_dist = np.where(excl, np.inf, distances)
            nearest = np.argsort(eff_dist)[:4]
            slip = np.isinf(eff_dist[nearest])  # <4 collectible left -> far sentinel
            near_delta = np.where(slip[:, None], 8.0, deltas[nearest]).astype(np.float32)
            near_dist = np.where(slip, 8.0, distances[nearest]).astype(np.float32)
            privileged[i] = np.concatenate(
                [
                    np.asarray(
                        [
                            position[0] / 8.0,
                            position[1] / 8.0,
                            math.sin(yaw),
                            math.cos(yaw),
                            linear[0] / 4.0,
                            linear[1] / 4.0,
                            yaw_rate / 6.0,
                            mag / 8.0,
                            float(slot.router.scored["blue"]) / 20.0,
                            float((eff_dist < 1.5).sum()) / 16.0,
                        ],
                        np.float32,
                    ),
                    (near_delta / 8.0).reshape(-1),
                    (near_dist[:, None] / 8.0).reshape(-1),
                    np.asarray(
                        [
                            1.0 if controller.intake_on else 0.0,
                            controller.storage_position,
                            float(controller.shots_fired) / 20.0,
                            slot.clock_s / max(1.0, self.cfg.episode_len_s),
                        ],
                        np.float32,
                    ),
                ]
            )
        obs["proprio"] = proprio
        obs["privileged"] = privileged
        return obs

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass
