#!/usr/bin/env python3
"""PhysX acceptance for robot field-centric teleop and xRC-style intake."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main() -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import SceneBuilder
        from xrc_rebuilt.competition_robot import (
            HOPPER_MAX_LOCAL,
            HOPPER_MIN_LOCAL,
            INTAKE_CENTER_LOCAL,
            ROBOT_ROOT_PATH,
            XRC_PRELOAD_COUNT,
            CompetitionRobotController,
        )
        from xrc_rebuilt.robot_model import quat_wxyz_to_matrix

        context = omni.usd.get_context()
        context.new_stage()
        SceneBuilder(context.get_stage(), max_fuel=32, articulated_robot=True).build()
        fuel = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        sim = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel.initialize()
        robot = SingleArticulation(ROBOT_ROOT_PATH)
        robot.initialize()
        controller = CompetitionRobotController()
        controller.initialize(robot)
        controller.preload(fuel, XRC_PRELOAD_COUNT)

        def reset_robot(yaw: float, x: float = 0.0, y: float = 0.0) -> None:
            robot.set_world_pose(
                position=np.array([x, y, 0.03], np.float32),
                orientation=np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], np.float32),
            )
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), np.float32))
            robot.set_linear_velocity(np.zeros(3, np.float32))
            robot.set_angular_velocity(np.zeros(3, np.float32))
            controller._driver_field_velocity[:] = 0.0
            controller._driver_omega = 0.0
            for _ in range(30):
                controller.drive_swerve(0, 0, 0)
                sim.step(render=False)
                controller.sync_magazine(fuel)

        # At a 90 degree chassis heading, field-forward W must still move +X.
        reset_robot(math.pi / 2)
        start_w, _ = controller.chassis_pose()
        for _ in range(180):
            controller.drive(1.0, 0.0, 0.0)
            sim.step(render=False)
            controller.sync_magazine(fuel)
        end_w, _ = controller.chassis_pose()

        # A is field-left (+Y), also independent of chassis heading.
        reset_robot(-math.pi / 3)
        start_a, _ = controller.chassis_pose()
        for _ in range(180):
            controller.drive(0.0, 0.0, 1.0)
            sim.step(render=False)
            controller.sync_magazine(fuel)
        end_a, _ = controller.chassis_pose()

        # Put a loose ball at the actual +X intake mouth, then observe conveyor.
        reset_robot(0.4)
        robot_pos, robot_q = controller.chassis_pose()
        intake_world = robot_pos + quat_wxyz_to_matrix(robot_q) @ INTAKE_CENTER_LOCAL
        ball_index = XRC_PRELOAD_COUNT
        idx = np.asarray([ball_index], np.int32)
        fuel.set_world_poses(positions=intake_world[None, :], indices=idx)
        fuel.set_linear_velocities(np.zeros((1, 3), np.float32), indices=idx)
        controller.intake_on = True
        observed: list[list[float]] = []
        velocity_samples: list[dict[str, list[float]]] = []
        # The collision-valid path now passes below the front board and then
        # rises through the roller throat before indexing into the dynamic pile.
        # Allow the same bounded re-routing window used by the full-capacity
        # acceptance test.
        for step in range(180):
            controller.step_intake(fuel, set(), dt_s=1 / 30)
            commanded = np.asarray(fuel.get_linear_velocities())[ball_index].tolist()
            for _ in range(4):
                sim.step(render=False)
            after = np.asarray(fuel.get_linear_velocities())[ball_index].tolist()
            poses, _ = fuel.get_world_poses()
            observed.append(np.asarray(poses)[ball_index].tolist())
            velocity_samples.append({"commanded": commanded, "after": after})
            if ball_index in controller.magazine:
                break
        intake_success = ball_index in controller.magazine
        poses, _ = fuel.get_world_poses()
        intake_robot_p, intake_robot_q = controller.chassis_pose()
        intake_local = (
            np.asarray(poses)[ball_index] - intake_robot_p
        ) @ quat_wxyz_to_matrix(intake_robot_q)
        intake_debug = {
            "local_xyz": [float(value) for value in intake_local],
            "magazine": list(controller.magazine),
            "captured": sorted(controller.captured_indices),
            "transit": {
                str(index): waypoint
                for index, waypoint in controller.intake_transit.items()
            },
            "release_target": {
                str(index): [float(value) for value in target]
                for index, target in controller.intake_release_target.items()
            },
        }

        # Cross the +X BUMP lane at full binary-key input, then brake.  The
        # keyboard slew limiter and closed Weidai gate must prevent launch and
        # retain all eight physical preloads.
        reset_robot(0.0, x=1.52, y=-5.55)
        controller.preload(fuel, XRC_PRELOAD_COUNT)
        max_robot_z = -1e9
        max_tilt_rad = 0.0
        for _ in range(360):
            controller.drive(0.0, 0.0, 1.0)
            sim.step(render=False)
            controller.sync_magazine(fuel)
            p, q = controller.chassis_pose()
            max_robot_z = max(max_robot_z, float(p[2]))
            up_z = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
            max_tilt_rad = max(max_tilt_rad, math.acos(float(np.clip(up_z, -1.0, 1.0))))
        for _ in range(120):
            controller.drive(0.0, 0.0, 0.0)
            sim.step(render=False)
            controller.sync_magazine(fuel)
        ball_poses, _ = fuel.get_world_poses()
        robot_pos, robot_q = controller.chassis_pose()
        local_balls = (np.asarray(ball_poses) - robot_pos) @ quat_wxyz_to_matrix(robot_q)
        inside = np.all(
            (local_balls >= HOPPER_MIN_LOCAL - 0.03)
            & (local_balls <= HOPPER_MAX_LOCAL + 0.03),
            axis=1,
        )
        inside_feeder = (
            (local_balls[:, 0] >= -0.45)
            & (local_balls[:, 0] <= -0.13)
            & (np.abs(local_balls[:, 1]) <= 0.38)
            & (local_balls[:, 2] >= 0.10)
            & (local_balls[:, 2] <= 0.50)
        )
        inside |= inside_feeder
        retained = sum(bool(inside[i]) for i in range(XRC_PRELOAD_COUNT))
        stable_pass = max_robot_z < 0.65 and max_tilt_rad < math.radians(45)
        retention_pass = retained == XRC_PRELOAD_COUNT

        result = {
            "w_field_displacement": (end_w - start_w).tolist(),
            "a_field_displacement": (end_a - start_a).tolist(),
            "w_pass": float(end_w[0] - start_w[0]) > 0.35,
            "a_pass": float(end_a[1] - start_a[1]) > 0.35,
            "intake_pass": intake_success,
            "intake_debug": intake_debug,
            "intake_samples": observed,
            "intake_velocity_samples": velocity_samples,
            "hopper": len(controller.magazine),
            "transit": len(controller.intake_transit),
            "bump_max_robot_z_m": max_robot_z,
            "bump_max_tilt_deg": math.degrees(max_tilt_rad),
            "stability_pass": stable_pass,
            "retained_preloads": retained,
            "retention_pass": retention_pass,
            "retention_corrections": controller.retention_corrections,
        }
        result["passed"] = bool(
            result["w_pass"]
            and result["a_pass"]
            and result["intake_pass"]
            and stable_pass
            and retention_pass
        )
        output = PROJECT / "runs" / "robot_drive_intake_validation.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("CompetitionRobot_DRIVE_INTAKE " + json.dumps(result), flush=True)
        sim.stop()
        if not result["passed"]:
            raise RuntimeError("robot drive/intake acceptance failed")
    finally:
        app.close()


if __name__ == "__main__":
    main()
