#!/usr/bin/env python3
"""Empirical drive calibration for the articulated legacy.

Sweeps wheel friction x turn scale x drive torque and measures full-stick
forward speed and turn-in-place yaw rate, targeting the xRC-measured
~3.5 m/s and ~100 deg/s.  Prints CALIB lines and writes
runs/drive_calibration.json.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main() -> None:
    sys.argv = [sys.argv[0]]
    import os

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation

        from xrc_rebuilt.isaac_scene import SceneBuilder
        from xrc_rebuilt import robot_model
        from xrc_rebuilt.robot_model import RobotController

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        builder = SceneBuilder(stage, max_fuel=2, articulated_robot=True)
        builder.build()
        sim = SimulationContext(physics_dt=0.004, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        robot = SingleArticulation("/World/Robot/LegacyRobot")
        robot.initialize()
        controller = RobotController()
        controller.initialize(robot)
        material_prim = stage.GetPrimAtPath("/World/PhysicsMaterials/WheelRubber")

        def set_friction(mu: float) -> None:
            material_prim.GetAttribute("physics:staticFriction").Set(mu * 1.1)
            material_prim.GetAttribute("physics:dynamicFriction").Set(mu)

        def set_torque(torque: float) -> None:
            art_controller = robot.get_articulation_controller()
            count = len(robot.dof_names)
            try:
                art_controller.set_max_efforts(np.full(count, torque, dtype=np.float32))
            except AttributeError:
                robot.set_max_efforts(np.full(count, torque, dtype=np.float32))

        def teleport(x: float, y: float, yaw_deg: float) -> None:
            half = math.radians(yaw_deg) * 0.5
            robot.set_world_pose(
                position=np.array([x, y, 0.02], dtype=np.float32),
                orientation=np.array([math.cos(half), 0, 0, math.sin(half)], dtype=np.float32),
            )
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            controller.drive(0.0, 0.0)
            for _ in range(40):
                sim.step(render=False)

        def yaw_of(orientation) -> float:
            qw, qx, qy, qz = (float(v) for v in orientation)
            return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        set_friction(1.0)
        set_torque(4.0)
        results = []

        # forward speed vs commanded wheel-speed ceiling (clear +x side corridor)
        for max_speed in (3.5, 3.68, 3.95, 4.2):
            robot_model.MAX_DRIVE_SPEED_MPS = max_speed
            teleport(3.1, -6.8, 90.0)
            speeds = []
            for step in range(750):
                controller.drive(1.0, 0.0)
                sim.step(render=False)
                if step > 400 and step % 25 == 0:
                    velocity = np.asarray(robot.get_linear_velocity())
                    speeds.append(float(np.linalg.norm(velocity[:2])))
            row = {"cmd_max_mps": max_speed, "forward_mps": round(float(np.mean(speeds)), 3)}
            results.append(row)
            print("CALIB_FWD", json.dumps(row), flush=True)

        # turn-in-place yaw rate vs turn scale (open neutral zone)
        robot_model.MAX_DRIVE_SPEED_MPS = 3.68
        for turn_scale in (0.10, 0.14, 0.18, 0.24, 0.28):
            robot_model.TURN_SPEED_SCALE = turn_scale
            teleport(0.0, -0.8, 0.0)
            yaws, times = [], []
            for step in range(750):
                controller.drive(0.0, 1.0)
                sim.step(render=False)
                if step > 250 and step % 25 == 0:
                    _, orientation = robot.get_world_pose()
                    yaws.append(yaw_of(np.asarray(orientation)))
                    times.append(step * 0.004)
            yaw_dps = float(math.degrees(np.polyfit(times, np.unwrap(yaws), 1)[0]))
            row = {"turn_scale": turn_scale, "yaw_dps": round(yaw_dps, 1)}
            results.append(row)
            print("CALIB_TURN", json.dumps(row), flush=True)

        (PROJECT / "runs" / "drive_calibration.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print("CALIB_DONE", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
