"""Swerve drivetrain kinematics following the competition robot's 2026 configuration.

Transcribed from CompetitionRobot_2026-PID ``generated/TunerConstants.h`` so the
imported robot CAD drives/steers exactly like the team code:

- 4 modules at (+/-0.276225, +/-0.276225) m in robot coordinates,
- wheel radius 0.0510032 m (2.008 in), drive gear 6.7460, steer gear 21.4286,
- max wheel speed 4.59 m/s (``kSpeedAt12Volts``).

Holonomic: command (vx, vy, omega); this maps to each module's (speed, angle).
Pure Python + numpy (kinematics only); the Isaac articulation that drives the
physical steer/drive joints consumes these module states.  Replaces the
6-wheel differential -- there IS lateral (strafe) motion now, by design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WHEEL_RADIUS_M = 2.008 * 0.0254
MAX_WHEEL_SPEED_MPS = 4.59
DRIVE_GEAR_RATIO = 6.746031746031747
STEER_GEAR_RATIO = 21.428571428571427

# Module positions (x forward, y left), metres — robot's rectangular base.
MODULE_POSITIONS: dict[str, tuple[float, float]] = {
    "front_left": (0.276225, 0.276225),
    "front_right": (0.276225, -0.276225),
    "back_left": (-0.276225, 0.276225),
    "back_right": (-0.276225, -0.276225),
}
MODULE_ORDER = ("front_left", "front_right", "back_left", "back_right")


@dataclass
class ModuleState:
    speed_mps: float
    angle_rad: float


def inverse_kinematics(
    vx: float, vy: float, omega: float,
    positions: dict[str, tuple[float, float]] = MODULE_POSITIONS,
    max_speed: float = MAX_WHEEL_SPEED_MPS,
) -> dict[str, ModuleState]:
    """Chassis twist (m/s, m/s, rad/s, robot frame) -> per-module (speed, angle).

    Standard swerve IK: module velocity = chassis linear + omega x r.  Speeds
    are desaturated (scaled down together) so none exceeds ``max_speed``.
    """
    states: dict[str, ModuleState] = {}
    for name, (mx, my) in positions.items():
        module_vx = vx - omega * my
        module_vy = vy + omega * mx
        states[name] = ModuleState(math.hypot(module_vx, module_vy),
                                   math.atan2(module_vy, module_vx))
    top = max((s.speed_mps for s in states.values()), default=0.0)
    if top > max_speed:
        scale = max_speed / top
        for s in states.values():
            s.speed_mps *= scale
    return states


def forward_kinematics(
    states: dict[str, ModuleState],
    positions: dict[str, tuple[float, float]] = MODULE_POSITIONS,
) -> tuple[float, float, float]:
    """Per-module states -> chassis twist (vx, vy, omega) by least squares."""
    rows: list[list[float]] = []
    rhs: list[float] = []
    for name, (mx, my) in positions.items():
        s = states[name]
        rows.append([1.0, 0.0, -my]); rhs.append(s.speed_mps * math.cos(s.angle_rad))
        rows.append([0.0, 1.0, mx]); rhs.append(s.speed_mps * math.sin(s.angle_rad))
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return float(solution[0]), float(solution[1]), float(solution[2])


def field_to_robot(vx_field: float, vy_field: float, heading_rad: float) -> tuple[float, float]:
    """Rotate a field-relative velocity into the robot frame for field-centric drive."""
    c, s = math.cos(-heading_rad), math.sin(-heading_rad)
    return vx_field * c - vy_field * s, vx_field * s + vy_field * c


def optimize_module(target: ModuleState, current_angle_rad: float) -> ModuleState:
    """Flip a module (reverse speed, +180 deg) when that is the shorter turn --
    the standard swerve trick so a module never rotates more than 90 deg."""
    delta = math.atan2(math.sin(target.angle_rad - current_angle_rad),
                       math.cos(target.angle_rad - current_angle_rad))
    if abs(delta) > math.pi / 2:
        return ModuleState(-target.speed_mps,
                           math.atan2(math.sin(target.angle_rad + math.pi),
                                      math.cos(target.angle_rad + math.pi)))
    return target
