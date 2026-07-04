"""Swerve kinematics (robot config) — rebuild step 1: driving/steering."""
from __future__ import annotations

import math

import pytest

from xrc_rebuilt import swerve
from xrc_rebuilt.swerve import (
    MAX_WHEEL_SPEED_MPS,
    MODULE_ORDER,
    ModuleState,
    field_to_robot,
    forward_kinematics,
    inverse_kinematics,
    optimize_module,
)


def test_config_matches_robot():
    assert swerve.WHEEL_RADIUS_M == pytest.approx(2.008 * 0.0254)
    assert swerve.MAX_WHEEL_SPEED_MPS == pytest.approx(4.59)
    assert swerve.DRIVE_GEAR_RATIO == pytest.approx(6.746031746031747)
    assert swerve.STEER_GEAR_RATIO == pytest.approx(21.428571428571427)
    assert set(MODULE_ORDER) == set(swerve.MODULE_POSITIONS)
    # rectangular base, symmetric about center
    xs = {round(p[0], 4) for p in swerve.MODULE_POSITIONS.values()}
    ys = {round(p[1], 4) for p in swerve.MODULE_POSITIONS.values()}
    assert xs == {0.2762, -0.2762} and ys == {0.2762, -0.2762}


def test_pure_forward():
    states = inverse_kinematics(1.0, 0.0, 0.0)
    for s in states.values():
        assert s.speed_mps == pytest.approx(1.0)
        assert math.cos(s.angle_rad) == pytest.approx(1.0, abs=1e-9)  # angle ~0


def test_pure_strafe():
    states = inverse_kinematics(0.0, 1.0, 0.0)
    for s in states.values():
        assert s.speed_mps == pytest.approx(1.0)
        assert s.angle_rad == pytest.approx(math.pi / 2)


def test_pure_rotation_is_tangential():
    states = inverse_kinematics(0.0, 0.0, 1.0)
    for name, (mx, my) in swerve.MODULE_POSITIONS.items():
        s = states[name]
        assert s.speed_mps == pytest.approx(math.hypot(mx, my))  # omega * radius
        # module direction is perpendicular to its position vector
        dot = math.cos(s.angle_rad) * mx + math.sin(s.angle_rad) * my
        assert dot == pytest.approx(0.0, abs=1e-9)


def test_ik_fk_roundtrip():
    vx, vy, omega = 1.0, 0.5, 0.8
    got = forward_kinematics(inverse_kinematics(vx, vy, omega))
    assert got[0] == pytest.approx(vx, abs=1e-6)
    assert got[1] == pytest.approx(vy, abs=1e-6)
    assert got[2] == pytest.approx(omega, abs=1e-6)


def test_desaturation_caps_speed():
    states = inverse_kinematics(10.0, 0.0, 5.0)  # way over max
    assert max(s.speed_mps for s in states.values()) == pytest.approx(MAX_WHEEL_SPEED_MPS)


def test_field_to_robot_rotation():
    # heading +90 deg: field +x maps to robot -y
    rx, ry = field_to_robot(1.0, 0.0, math.pi / 2)
    assert rx == pytest.approx(0.0, abs=1e-9)
    assert ry == pytest.approx(-1.0, abs=1e-9)


def test_optimize_module_flips_when_shorter():
    # target 170 deg from current 0 -> flip to ~-10 deg with reversed speed
    opt = optimize_module(ModuleState(1.0, math.radians(170)), 0.0)
    assert opt.speed_mps == pytest.approx(-1.0)
    assert abs(math.degrees(opt.angle_rad)) == pytest.approx(10.0, abs=1e-6)


def test_optimize_module_keeps_when_short():
    opt = optimize_module(ModuleState(1.0, math.radians(20)), 0.0)
    assert opt.speed_mps == pytest.approx(1.0)
    assert math.degrees(opt.angle_rad) == pytest.approx(20.0, abs=1e-6)
