#!/usr/bin/env python3
"""Extract authoritative competition-robot control constants into a sim artifact.

The shooter table comes from a generated calibration with explicit unit tests
and a 431-point 1.4..5.7 m lookup. Swerve and aim bindings are cross-checked
against the PID working tree. Source hashes make the selection auditable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("ROBOT_SOURCE_ROOT", Path.home() / "Desktop" / "FRC"))
OUTPUT = PROJECT / "assets" / "robot_runtime" / "control_spec.json"


def discover(filename: str, preferred_tree: str) -> Path:
    """Find a source file without embedding workstation- or team-specific paths."""

    candidates = sorted(SOURCE_ROOT.rglob(filename))
    preferred = [
        path for path in candidates if preferred_tree.casefold() in str(path).casefold()
    ]
    matches = preferred or candidates
    if not matches:
        return SOURCE_ROOT / preferred_tree / filename
    return matches[0]

SOURCES = {
    "shooter_table": discover("GeneratedShooterCalibration4.h", "beta"),
    "shooter_api": discover("ShooterCalibration.cpp", "beta"),
    "shooter_tests": discover("ShooterCalibrationTest.cpp", "beta"),
    "swerve": discover("TunerConstants.h", "PID"),
    "moving_aim": discover("RealTimeAimDrive.cpp", "PID"),
    "auto_moving_aim": discover("AutoRealTimeAimDrive.cpp", "PID"),
    "bindings": discover("RobotContainer.cpp", "PID"),
    "intake": discover("GroundIntakeSubsystem.cpp", "PID"),
    "vision": discover("VisionSubsystem.cpp", "PID"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_points(text: str, array_name: str, next_marker: str) -> list[list[float]]:
    start = text.index(array_name)
    end = text.index(next_marker, start)
    block = text[start:end]
    return [
        [float(a), float(b), float(c)]
        for a, b, c in re.findall(
            r"DistancePoint\{\s*([\d.+-]+),\s*([\d.+-]+),\s*([\d.+-]+)\}", block
        )
    ]


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing competition robot sources:\n" + "\n".join(missing))

    generated = SOURCES["shooter_table"].read_text(encoding="utf-8")
    distance_table = parse_points(
        generated, "kDistanceLookupFull", "kDistanceLookupCompact"
    )
    motor_block = generated[
        generated.index("kLookup") : generated.index("LookupLinear")
    ]
    motor_table = [
        [float(velocity), float(rps)]
        for velocity, rps in re.findall(
            r"CalibrationPoint\{\s*([\d.+-]+),\s*([\d.+-]+)\}", motor_block
        )
    ]
    if len(distance_table) != 431 or len(motor_table) != 9:
        raise RuntimeError(
            f"Unexpected robot calibration sizes: {len(distance_table)}, {len(motor_table)}"
        )

    source_records = {
        name: {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in SOURCES.items()
    }
    spec = {
        "robot": "FRC competition robot CAD/control strategy",
        "selection": {
            "shooter": "beta_dev GeneratedShooterCalibration4 (cali4, unit-tested)",
            "swerve_and_bindings": "PID working tree",
            "reason": "cali4 is the densest tested shooter calibration; PID has the latest driver bindings and motion compensation",
        },
        "sources": source_records,
        "swerve": {
            "max_speed_mps": 4.59,
            "wheel_radius_m": 2.008 * 0.0254,
            "drive_gear_ratio": 6.746031746031747,
            "steer_gear_ratio": 21.428571428571427,
            "heading_pid": [8.0, 0.0, 0.1],
            "max_aim_rotation_rad_s": 3.141592653589793 * 0.8,
            "module_positions_m": {
                "front_left": [10.875 * 0.0254, 10.875 * 0.0254],
                "front_right": [10.875 * 0.0254, -10.875 * 0.0254],
                "back_left": [-10.875 * 0.0254, 10.875 * 0.0254],
                "back_right": [-10.875 * 0.0254, -10.875 * 0.0254],
            },
        },
        "shooter": {
            "distance_min_m": 1.4,
            "distance_max_m": 5.7,
            "height_m": 0.46932,
            "mount_yaw_deg": 180.0,
            "distance_table_columns": ["distance_m", "theory_exit_speed_mps", "pitch_deg"],
            "distance_table": distance_table,
            "motor_table_columns": ["theory_exit_speed_mps", "motor_target_rps"],
            "motor_table": motor_table,
            "ready_tolerance_rps": 0.7,
            "teleop_motion_compensation": {
                "source_coefficient": 0.45,
                "implementation_note": "sim applies the same vector subtraction directly in physical m/s, then maps corrected exit speed back to motor RPS",
            },
            "auto_latency_seconds": 0.45,
        },
        "vision": {
            "strategy": "Limelight MegaTag2 pose fused with external Pigeon yaw; close reliable MT1 yaw may be mixed",
            "max_tag_distance_m": 4.5,
            "min_tag_distance_m": 0.26,
            "max_ambiguity": 0.5,
        },
        "intake": {
            "roller_supply_current_limit_a": 40.0,
            "pitch_gear_ratio": 18.67,
            "pitch_pid": [2.0, 0.0, 0.0],
            "pitch_homing_current_a": -50.0,
            "pitch_homing_threshold_a": -46.0,
            "pitch_homing_cycles": 3,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(
        f"Wrote {OUTPUT}: distance_points={len(distance_table)} motor_points={len(motor_table)}"
    )


if __name__ == "__main__":
    main()
