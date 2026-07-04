"""Phase B acceptance: continuous-fire controls + shooter/indexer FSM.

These are pure-Python tests of the exact decision logic that drives the GUI
and the future RL policy (``RobotController.update`` / ``ShooterStateMachine``).
The full-geometry PhysX acceptance (160/160 scoring) runs in Isaac via
``tools/validate_robot.py`` scenario G; here we prove the state machine and
magazine integrity without launching Isaac.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from xrc_rebuilt.robot_model import (
    AUTO_ALIGN_TOLERANCE_DEG,
    FIRE_MAX_SPEED_MPS,
    FIRE_MAX_YAW_RATE_DPS,
    RobotController,
    ShooterInputs,
    ShooterState,
    ShooterStateMachine,
)

COOLDOWN = 0.07


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _inputs(mag: int = 8, **kw) -> ShooterInputs:
    base = dict(
        magazine_count=mag,
        hub_active=True,
        shot_valid=True,
        yaw_error_deg=0.0,
        chassis_speed_mps=0.0,
        yaw_rate_dps=0.0,
        muzzle_clear=True,
        blocked_reason="",
    )
    base.update(kw)
    return ShooterInputs(**base)


def _simulate(sm: ShooterStateMachine, ticks: int = 120, dt: float = 0.02,
              mag: int = 8, start: float = 0.0, **kw) -> list[float]:
    """Tick the FSM, decrementing a simulated magazine on each feed."""
    feeds: list[float] = []
    t = start
    for _ in range(ticks):
        status = sm.tick(_inputs(mag=mag, **kw), t)
        if status["should_feed"]:
            feeds.append(t)
            mag = max(0, mag - 1)
        t += dt
    return feeds


# --------------------------------------------------------------------------- #
# control-mode semantics (B1)
# --------------------------------------------------------------------------- #
def test_single_click_produces_exactly_one_shot():
    sm = ShooterStateMachine()
    sm.request_single()
    feeds = _simulate(sm, ticks=200)
    assert len(feeds) == 1


def test_continuous_toggle_fires_all_eight_without_extra_clicks():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    feeds = _simulate(sm, ticks=200, mag=8)
    assert len(feeds) == 8


def test_hold_fires_all_eight_then_blocks_empty():
    sm = ShooterStateMachine()
    sm.press_hold()
    feeds = _simulate(sm, ticks=200, mag=8)
    assert len(feeds) == 8
    # once empty the FSM must report BLOCKED, not keep FEEDING
    assert sm.tick(_inputs(mag=0), 99.0)["state"] == ShooterState.BLOCKED.value


def test_release_stops_before_the_next_feed_cycle():
    sm = ShooterStateMachine()
    sm.press_hold()
    t, dt, feeds, mag = 0.0, 0.02, 0, 8
    # fire exactly one, then release
    while feeds == 0:
        if sm.tick(_inputs(mag=mag), t)["should_feed"]:
            feeds += 1
            mag -= 1
        t += dt
    sm.release_hold()
    after = _simulate(sm, ticks=100, start=t, mag=mag)
    assert after == []  # no further feeds after release
    assert feeds == 1


def test_continuous_enables_target_lock_automatically():
    sm = ShooterStateMachine()
    sm.auto_align = False
    sm.set_continuous(True)
    assert sm.auto_align is True


# --------------------------------------------------------------------------- #
# fire gates (B2) — no shot while turning / translating / blocked / empty
# --------------------------------------------------------------------------- #
def test_no_fire_while_turning_too_fast():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    feeds = _simulate(sm, yaw_error_deg=5.0)
    assert feeds == []
    assert sm.state == ShooterState.TURNING


def test_no_fire_while_translating():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    feeds = _simulate(sm, chassis_speed_mps=0.5)
    assert feeds == []
    assert sm.state == ShooterState.BRAKING


def test_no_fire_while_yaw_rate_high():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    feeds = _simulate(sm, yaw_rate_dps=25.0)
    assert feeds == []
    assert sm.state == ShooterState.BRAKING


def test_no_fire_when_shot_invalid_and_reason_propagates():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    status = sm.tick(_inputs(shot_valid=False, blocked_reason="out of range"), 1.0)
    assert status["should_feed"] is False
    assert status["state"] == ShooterState.BLOCKED.value
    assert status["blocked_reason"] == "out of range"


def test_no_fire_when_magazine_empty():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    status = sm.tick(_inputs(mag=0), 1.0)
    assert status["state"] == ShooterState.BLOCKED.value
    assert status["blocked_reason"] == "magazine empty"


def test_no_fire_when_hub_inactive():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    status = sm.tick(_inputs(hub_active=False), 1.0)
    assert status["state"] == ShooterState.BLOCKED.value
    assert status["blocked_reason"] == "target HUB inactive"


def test_no_fire_until_muzzle_clears():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    # lane never clears -> never feeds
    assert _simulate(sm, muzzle_clear=False) == []
    assert sm.state == ShooterState.COOLDOWN
    # once clear it feeds
    assert len(_simulate(sm, muzzle_clear=True)) >= 1


# --------------------------------------------------------------------------- #
# cooldown + emergency stop (B2/B3)
# --------------------------------------------------------------------------- #
def test_shot_intervals_never_violate_cooldown():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    feeds = _simulate(sm, ticks=400, dt=0.02, mag=8)
    assert len(feeds) == 8
    intervals = np.diff(feeds)
    assert float(intervals.min()) >= COOLDOWN - 1e-9


def test_finer_dt_still_respects_cooldown():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    feeds = _simulate(sm, ticks=2000, dt=0.004, mag=8)
    assert len(feeds) == 8
    assert float(np.diff(feeds).min()) >= COOLDOWN - 1e-9


def test_emergency_stop_overrides_continuous_and_resumes():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    sm.set_emergency_stop(True)
    assert _simulate(sm, ticks=50) == []
    assert sm.state == ShooterState.IDLE
    assert sm.request_mode() == "ESTOP"
    # clearing the stop resumes the still-latched continuous request
    sm.set_emergency_stop(False)
    assert len(_simulate(sm, ticks=50)) >= 1


# --------------------------------------------------------------------------- #
# status reporting (B3) + state reachability
# --------------------------------------------------------------------------- #
def test_status_reports_all_required_fields():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    status = sm.tick(_inputs(), 0.0)
    for key in ("state", "should_feed", "request_mode", "blocked_reason",
                "magazine_count", "feeds", "fire_rate_hz"):
        assert key in status


def test_actual_rate_is_positive_during_continuous_fire():
    sm = ShooterStateMachine()
    sm.set_continuous(True)
    last = 0.0
    for i in range(400):
        last = i * 0.02
        sm.tick(_inputs(mag=8), last)
    assert sm.actual_rate_hz(last) > 0.0


def test_full_ready_progression_visits_expected_states():
    sm = ShooterStateMachine()
    sm.auto_align = True  # aim engaged, no fire request yet
    seen = set()
    # start misaligned + moving, then converge
    for i in range(6):
        seen.add(sm.tick(_inputs(yaw_error_deg=5.0), i * 0.02)["state"])
    for i in range(6):
        seen.add(sm.tick(_inputs(chassis_speed_mps=0.5), 1.0 + i * 0.02)["state"])
    for i in range(6):
        seen.add(sm.tick(_inputs(), 2.0 + i * 0.02)["state"])
    assert {ShooterState.ACQUIRE_TARGET.value, ShooterState.TURNING.value,
            ShooterState.BRAKING.value, ShooterState.READY.value} <= seen


def test_gate_constants_match_plan():
    assert FIRE_MAX_SPEED_MPS == 0.08
    assert FIRE_MAX_YAW_RATE_DPS == 3.0
    assert AUTO_ALIGN_TOLERANCE_DEG == 0.25


# --------------------------------------------------------------------------- #
# RobotController integration with lightweight fakes (no Isaac)
# --------------------------------------------------------------------------- #
class FakeFuelView:
    def __init__(self, count: int = 24):
        self.count = count
        self.positions = np.tile(np.array([0.0, 0.0, -5.0], np.float32), (count, 1))
        self.lin = np.zeros((count, 3), np.float32)
        self.ang = np.zeros((count, 3), np.float32)

    def get_world_poses(self):
        return self.positions.copy(), None

    def get_linear_velocities(self):
        return self.lin.copy()

    def set_world_poses(self, positions=None, indices=None):
        if positions is not None and indices is not None:
            self.positions[np.asarray(indices)] = np.asarray(positions, np.float32)

    def set_linear_velocities(self, v, indices=None):
        self.lin[np.asarray(indices)] = np.asarray(v, np.float32)

    def set_angular_velocities(self, v, indices=None):
        self.ang[np.asarray(indices)] = np.asarray(v, np.float32)


class FakeArticulation:
    def __init__(self, pos, yaw_rad=0.0):
        self._pos = np.asarray(pos, np.float32)
        self.set_yaw(yaw_rad)

    def set_yaw(self, yaw_rad):
        self._quat = np.array(
            [math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)], np.float32
        )

    def get_world_pose(self):
        return self._pos.copy(), self._quat.copy()

    def get_linear_velocity(self):
        return np.zeros(3, np.float32)

    def get_angular_velocity(self):
        return np.zeros(3, np.float32)


def test_fire_pops_eight_distinct_balls_in_order_no_double_fire():
    c = RobotController()
    c.articulation = FakeArticulation(pos=[1.52, -5.55, 0.2])
    fv = FakeFuelView()
    c.preload(fv, count=8)
    fired, t = [], 0.0
    for _ in range(300):
        idx = c.fire(fv, aim=0.118, now_s=t)
        if idx is not None:
            fired.append(idx)
        t += 0.02
    assert fired == list(range(8))          # exactly 8, in magazine order
    assert len(set(fired)) == 8             # no double-fire of any index
    assert c.magazine == []
    assert c.shots_fired == 8


def test_fire_respects_cooldown_timestamps():
    c = RobotController()
    c.articulation = FakeArticulation(pos=[1.52, -5.55, 0.2])
    fv = FakeFuelView()
    c.preload(fv, count=8)
    times, t = [], 0.0
    for _ in range(300):
        if c.fire(fv, aim=0.118, now_s=t) is not None:
            times.append(t)
        t += 0.02
    assert float(np.diff(times).min()) >= COOLDOWN - 1e-9


def test_update_continuous_scores_eight_from_accepted_blue_pose():
    """Full FSM+solver+feed pipeline (drive actuation is Isaac-only)."""
    c = RobotController()
    c.articulation = FakeArticulation(pos=[1.52, -5.55, 0.2])
    fv = FakeFuelView()
    c.preload(fv, count=8)
    solution = c.solve_auto_aim("blue")
    assert solution["valid"], solution.get("blocked_reason")
    # align the chassis onto the solved bearing
    c.articulation.set_yaw(float(solution["desired_yaw_rad"]))
    c.state_machine.set_continuous(True)
    feeds, t = 0, 0.0
    for _ in range(400):
        status = c.update(fv, now_s=t, alliance="blue", hub_active=True, allow_drive=False)
        if status["fired_index"] is not None:
            feeds += 1
            # simulate the ball leaving the muzzle keep-clear volume in flight
            fv.positions[status["fired_index"]] = np.array([5.0, 5.0, 1.0], np.float32)
        t += 0.02
    assert feeds == 8
    assert c.magazine == []


def test_fire_volley_releases_horizontal_row():
    c = RobotController()
    c.articulation = FakeArticulation(pos=[1.52, -5.55, 0.2])
    c.barrels = 3
    fv = FakeFuelView()
    c.preload(fv, count=8)
    fired = c.fire_volley(fv, aim=0.118, now_s=0.0)
    assert fired == [0, 1, 2]              # three balls, distinct, in order
    assert len(c.magazine) == 5
    # the three land in a horizontal row (spread on the world y axis here)
    ys = sorted(float(fv.positions[i][1]) for i in fired)
    assert (ys[-1] - ys[0]) > 0.2         # spread ~2 * spacing


def test_fire_volley_respects_cooldown_and_empties_in_volleys():
    c = RobotController()
    c.articulation = FakeArticulation(pos=[1.52, -5.55, 0.2])
    c.barrels = 3
    fv = FakeFuelView()
    c.preload(fv, count=8)
    total, t = 0, 0.0
    for _ in range(300):
        total += len(c.fire_volley(fv, aim=0.118, now_s=t))
        t += 0.02
    assert total == 8                     # 3 + 3 + 2, no ball lost or duplicated
    assert c.magazine == []


def test_update_reports_blocked_when_out_of_range():
    c = RobotController()
    # far downfield: no valid direct shot -> BLOCKED, never feeds
    c.articulation = FakeArticulation(pos=[5.0, 0.0, 0.2])
    fv = FakeFuelView()
    c.preload(fv, count=8)
    c.state_machine.set_continuous(True)
    status = c.update(fv, now_s=1.0, alliance="blue", hub_active=True, allow_drive=False)
    assert status["fired_index"] is None
    assert status["state"] == ShooterState.BLOCKED.value
    assert c.magazine == [0, 1, 2, 3, 4, 5, 6, 7]
