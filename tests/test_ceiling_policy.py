"""Unit tests for the scripted ceiling policy (no Isaac dependency)."""
from __future__ import annotations

import math

import numpy as np

from xrc_rebuilt.rl.ceiling_policy import (
    BLUE_HUB_XY,
    COLLECT_TARGET,
    SCORE_RANGE_M,
    CeilingObs,
    ceiling_action,
)
from xrc_rebuilt.rl.spec import ACTION_NAMES

IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def _obs(**kw) -> CeilingObs:
    base = dict(
        robot_xy=np.array([0.0, 0.0], np.float32),
        robot_yaw=0.0,
        ball_xy=np.zeros((0, 2), np.float32),
        magazine=0,
        elapsed_s=10.0,
        match_remaining_s=150.0,
    )
    base.update(kw)
    return CeilingObs(**base)


def test_action_is_bounded_and_shaped():
    a = ceiling_action(_obs(ball_xy=np.array([[2.0, 0.0]], np.float32)))
    assert a.shape == (len(ACTION_NAMES),)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_storage_always_deployed_and_ferry_off():
    for mag in (0, COLLECT_TARGET, 20):
        a = ceiling_action(_obs(magazine=mag, ball_xy=np.array([[1.0, 1.0]], np.float32)))
        assert a[IDX["storage"]] > 0.25   # extended
        assert a[IDX["ferry"]] <= -0.25   # off


def test_collect_drives_and_turns_toward_ball_ahead():
    # ball straight along field +X, robot facing +X -> forward drive, intake on,
    # little turn needed
    a = ceiling_action(_obs(ball_xy=np.array([[3.0, 0.0]], np.float32)))
    assert a[IDX["forward"]] > 0.5
    assert abs(a[IDX["strafe"]]) < 1e-3
    assert a[IDX["intake"]] > 0.25
    assert abs(a[IDX["turn"]]) < 1e-3
    assert a[IDX["shoot_blue"]] <= 0.25


def test_collect_turns_intake_toward_a_side_ball():
    # ball at field +Y while robot faces +X: must strafe +Y and turn to point the
    # +X intake toward +Y (positive yaw)
    a = ceiling_action(_obs(ball_xy=np.array([[0.0, 3.0]], np.float32)))
    assert a[IDX["strafe"]] > 0.5
    assert a[IDX["turn"]] > 0.25


def test_picks_nearest_ball():
    balls = np.array([[5.0, 0.0], [1.0, 0.0], [8.0, 0.0]], np.float32)
    a = ceiling_action(_obs(ball_xy=balls))
    # nearest is +X at 1.0 -> forward positive, no strafe
    assert a[IDX["forward"]] > 0.5 and abs(a[IDX["strafe"]]) < 1e-3


def test_full_magazine_switches_to_score_and_holds_fire_in_range():
    # sit exactly at 3.2 m on the +Y (neutral) side of the blue hub, full load
    pose = BLUE_HUB_XY + np.array([0.0, 3.2], np.float32)
    a = ceiling_action(_obs(robot_xy=pose, magazine=COLLECT_TARGET,
                            ball_xy=np.array([[0.0, 0.0]], np.float32)))
    dist = float(np.linalg.norm(BLUE_HUB_XY - pose))
    assert SCORE_RANGE_M[0] <= dist <= SCORE_RANGE_M[1]
    assert a[IDX["shoot_blue"]] > 0.25
    # in-band: stop moving so the controller can auto-aim
    assert abs(a[IDX["forward"]]) < 1e-3 and abs(a[IDX["strafe"]]) < 1e-3


def test_full_magazine_far_from_hub_drives_toward_it():
    far = np.array([6.0, 6.0], np.float32)  # well outside range
    a = ceiling_action(_obs(robot_xy=far, magazine=COLLECT_TARGET))
    dist = float(np.linalg.norm(BLUE_HUB_XY - far))
    assert dist > SCORE_RANGE_M[1]
    # should move (nonzero drive) generally toward the hub (negative-ish y/x)
    drive = np.array([a[IDX["forward"]], a[IDX["strafe"]]])
    assert float(np.linalg.norm(drive)) > 0.5
    assert a[IDX["intake"]] <= 0.25  # not collecting while ferrying a full load home


def test_no_balls_forces_score_mode_even_if_not_full():
    a = ceiling_action(_obs(robot_xy=np.array([0.0, 0.0], np.float32), magazine=2,
                            ball_xy=np.zeros((0, 2), np.float32)))
    # score mode: drives toward hub standoff, not collecting
    assert a[IDX["intake"]] <= 0.25


def test_end_of_match_stops_collecting():
    a = ceiling_action(_obs(magazine=0, match_remaining_s=1.0,
                            ball_xy=np.array([[1.0, 0.0]], np.float32)))
    assert a[IDX["intake"]] <= 0.25  # no time left -> don't start a new collect
