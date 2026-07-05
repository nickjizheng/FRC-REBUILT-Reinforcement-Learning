"""Scripted privileged "ceiling" policy for the xRC REBUILT match.

A hand-coded greedy match-player used ONLY to measure the achievable legal-score
ceiling and to validate (or lower) the >=200 target *before* any RL training
(Converged decision #1 in ``RL_BRAINSTORM.md``).  It is **not** a teacher and
produces **no** training data.  It consumes privileged state (true FUEL
positions) and emits the SAME 7-D action vector the RL policy will
(see :data:`xrc_rebuilt.rl.spec.ACTION_NAMES`), so it also exercises the
action-decode path end to end.  It deliberately has no Isaac Sim dependency so
the strategy is unit-testable without booting a heavyweight simulator.

Strategy - a tight collect->score loop:

The HUB recycles scored FUEL back onto the field near its exit chutes, so
camping near the blue HUB and re-collecting ejected FUEL is the dominant scoring
engine.  Nearest-ball collection produces that behaviour naturally.  Ferrying is
a *repositioning* tool, not a scorer, so this ceiling policy leaves it off and
measures the pure collect->score ceiling - a defensible lower bound on what a
learned policy could reach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xrc_rebuilt.rl.spec import ACTION_NAMES, CompetitionRLSpec

# Blue HUB ground target (mirrors competition_robot.HUB_TARGETS["blue"][:2];
# hardcoded to keep this module Isaac-free and dependency-light like spec.py).
BLUE_HUB_XY = np.array([-0.0199, -3.6874], dtype=np.float32)
NEUTRAL_ZONE_HALF_Y_M = 2.775  # |y| beyond this on the blue side is our own zone

COLLECT_TARGET = 7          # collect until the magazine holds this many, then score
SCORE_RANGE_M = (1.7, 5.2)  # hold-and-fire band inside the calibrated 1.4-5.7 m
SHOOT_STANDOFF_M = 3.2      # ideal range to sit at while dumping a load
TURN_GAIN = 1.6            # heading error (rad) -> normalized turn request
STOP_RADIUS_M = 0.25       # within this of the aim pose, stop and let auto-aim fire

_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class CeilingObs:
    """Privileged observation for the scripted ceiling policy."""

    robot_xy: np.ndarray        # (2,) field position
    robot_yaw: float            # radians; robot +X (intake) points along this
    ball_xy: np.ndarray         # (M, 2) collectible in-field FUEL; may be empty
    magazine: int               # FUEL currently held
    elapsed_s: float
    match_remaining_s: float


def ceiling_action(obs: CeilingObs, spec: CompetitionRLSpec | None = None) -> np.ndarray:
    """Return a 7-D action in ``[-1, 1]`` following :data:`ACTION_NAMES` order.

    COLLECT: drive to the nearest FUEL and rotate the +X intake toward it.
    SCORE:   reach the calibrated HUB range, then emit ~zero drive so the
             harness frees the controller's auto-aim/feed FSM to fire.
    """
    _ = spec or CompetitionRLSpec()
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action[_IDX["storage"]] = 1.0   # always deployed (needed to collect and feed)
    action[_IDX["ferry"]] = -1.0    # off: this ceiling measures pure collect->score

    robot = np.asarray(obs.robot_xy, dtype=np.float32).reshape(2)
    balls = np.asarray(obs.ball_xy, dtype=np.float32).reshape(-1, 2)

    want_collect = (
        obs.magazine < COLLECT_TARGET
        and len(balls) > 0
        and obs.match_remaining_s > 3.0
    )

    if want_collect:
        to = balls[int(np.argmin(((balls - robot[None, :]) ** 2).sum(1)))] - robot
        step = _unit(to)
        action[_IDX["forward"]] = float(step[0])   # field +X velocity request
        action[_IDX["strafe"]] = float(step[1])    # field +Y velocity request
        action[_IDX["intake"]] = 1.0
        bearing = math.atan2(float(to[1]), float(to[0]))
        action[_IDX["turn"]] = float(
            np.clip(TURN_GAIN * _wrap(bearing - obs.robot_yaw), -1.0, 1.0)
        )
        return action

    # SCORE ---------------------------------------------------------------
    dist = float(np.linalg.norm(BLUE_HUB_XY - robot))
    if SCORE_RANGE_M[0] <= dist <= SCORE_RANGE_M[1]:
        # in the band: stop moving and hold shoot; the controller aims + feeds
        action[_IDX["shoot_blue"]] = 1.0
        return action

    # approach a standoff point at ideal range, on the side we're already on
    aim_pose = BLUE_HUB_XY + _unit(robot - BLUE_HUB_XY) * SHOOT_STANDOFF_M
    to_pose = aim_pose - robot
    if float(np.linalg.norm(to_pose)) > STOP_RADIUS_M:
        step = _unit(to_pose)
        action[_IDX["forward"]] = float(step[0])
        action[_IDX["strafe"]] = float(step[1])
    # request the shot whenever we are already within max range (auto-aim will
    # refuse if the geometry is actually invalid, so this is safe)
    action[_IDX["shoot_blue"]] = 1.0 if dist <= SCORE_RANGE_M[1] else 0.0
    return action
