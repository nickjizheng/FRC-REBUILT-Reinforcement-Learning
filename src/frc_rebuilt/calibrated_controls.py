"""Competition robot shooter, swerve, and moving-shot control math for Isaac.

Values are generated from the local FRC sources by
``tools/import_robot_controls.py``.  This module works without WPILib so the
same calculations can be tested and called by the live simulator.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT / "assets" / "robot_runtime" / "control_spec.json"


def load_control_spec(path: Path = SPEC_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ShooterSetpoint:
    distance_m: float
    exit_speed_mps: float
    pitch_deg: float
    motor_target_rps: float
    chassis_yaw_rad: float
    shot_yaw_rad: float
    flight_time_s: float
    clamped: bool


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class CalibratedRobotControl:
    def __init__(self, spec: dict | None = None):
        self.spec = spec or load_control_spec()
        shooter = self.spec["shooter"]
        distance = np.asarray(shooter["distance_table"], dtype=np.float64)
        motor = np.asarray(shooter["motor_table"], dtype=np.float64)
        self.distance_m = distance[:, 0]
        self.exit_speed_mps = distance[:, 1]
        self.pitch_deg = distance[:, 2]
        self.motor_speed_axis = motor[:, 0]
        self.motor_target_rps = motor[:, 1]
        self.mount_yaw_rad = math.radians(float(shooter["mount_yaw_deg"]))

    def stationary_calibration(self, distance_m: float) -> tuple[float, float, float, bool]:
        value = float(distance_m)
        bounded = float(np.clip(value, self.distance_m[0], self.distance_m[-1]))
        speed = float(np.interp(bounded, self.distance_m, self.exit_speed_mps))
        pitch = float(np.interp(bounded, self.distance_m, self.pitch_deg))
        motor = float(np.interp(speed, self.motor_speed_axis, self.motor_target_rps))
        return speed, pitch, motor, bounded != value

    def solve_shot(
        self,
        robot_xy: np.ndarray,
        robot_velocity_xy: np.ndarray,
        hub_xy: np.ndarray,
        compensate_motion: bool = True,
    ) -> ShooterSetpoint:
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        hub_xy = np.asarray(hub_xy, dtype=np.float64)
        velocity = np.asarray(robot_velocity_xy, dtype=np.float64)
        delta = hub_xy - robot_xy
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction = delta / distance
        base_speed, base_pitch_deg, _, clamped = self.stationary_calibration(distance)
        base_pitch = math.radians(base_pitch_deg)
        desired_horizontal = direction * (base_speed * math.cos(base_pitch))
        relative_horizontal = desired_horizontal - velocity if compensate_motion else desired_horizontal
        vertical = base_speed * math.sin(base_pitch)
        corrected_speed = float(math.sqrt(float(relative_horizontal @ relative_horizontal) + vertical * vertical))
        corrected_pitch = math.degrees(math.atan2(vertical, float(np.linalg.norm(relative_horizontal))))
        shot_yaw = math.atan2(float(relative_horizontal[1]), float(relative_horizontal[0]))
        chassis_yaw = wrap_angle(shot_yaw - self.mount_yaw_rad)
        motor_rps = float(
            np.interp(
                corrected_speed,
                self.motor_speed_axis,
                self.motor_target_rps,
                left=self.motor_target_rps[0],
                right=self.motor_target_rps[-1],
            )
        )
        horizontal_speed = max(1e-6, float(np.linalg.norm(relative_horizontal + velocity)))
        return ShooterSetpoint(
            distance_m=distance,
            exit_speed_mps=corrected_speed,
            pitch_deg=corrected_pitch,
            motor_target_rps=motor_rps,
            chassis_yaw_rad=chassis_yaw,
            shot_yaw_rad=shot_yaw,
            flight_time_s=distance / horizontal_speed,
            clamped=clamped,
        )

    def launch_velocity_world(self, setpoint: ShooterSetpoint, chassis_velocity_xy: np.ndarray) -> np.ndarray:
        pitch = math.radians(setpoint.pitch_deg)
        relative = np.array(
            [
                setpoint.exit_speed_mps * math.cos(pitch) * math.cos(setpoint.shot_yaw_rad),
                setpoint.exit_speed_mps * math.cos(pitch) * math.sin(setpoint.shot_yaw_rad),
                setpoint.exit_speed_mps * math.sin(pitch),
            ],
            dtype=np.float32,
        )
        relative[:2] += np.asarray(chassis_velocity_xy, dtype=np.float32)
        return relative
