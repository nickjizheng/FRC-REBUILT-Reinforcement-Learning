#!/usr/bin/env python3
"""Fill the robot through its intake, then verify physical retention."""
from __future__ import annotations

import json
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

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import SceneBuilder
        from xrc_rebuilt.competition_robot import (
            HOPPER_PRESSURE_CAPACITY,
            INTAKE_CENTER_LOCAL,
            ROBOT_ROOT_PATH,
            XRC_PRELOAD_COUNT,
            CompetitionRobotController,
        )
        from xrc_rebuilt.robot_model import quat_wxyz_to_matrix

        omni.usd.get_context().new_stage()
        SceneBuilder(
            omni.usd.get_context().get_stage(),
            max_fuel=HOPPER_PRESSURE_CAPACITY + 8,
            articulated_robot=True,
        ).build()
        sim = SimulationContext(
            physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1.0
        )
        sim.reset()
        fuel = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        fuel.initialize()
        robot = SingleArticulation(ROBOT_ROOT_PATH)
        robot.initialize()
        controller = CompetitionRobotController()
        controller.initialize(robot)
        controller.preload(fuel, XRC_PRELOAD_COUNT)
        controller.intake_on = True
        first_escape: dict[str, dict[str, object]] = {}

        def trace_escaped(phase: str, step: int) -> None:
            poses, _ = fuel.get_world_poses()
            robot_p, robot_q = controller.chassis_pose()
            local = (
                np.asarray(poses) - robot_p
            ) @ quat_wxyz_to_matrix(robot_q)
            for captured in controller.captured_indices:
                point = local[captured]
                inside = bool(
                    -0.50 <= float(point[0]) <= 0.52
                    and abs(float(point[1])) <= 0.38
                    and 0.045 <= float(point[2]) <= 0.78
                )
                if not inside and str(captured) not in first_escape:
                    first_escape[str(captured)] = {
                        "phase": phase,
                        "step": step,
                        "local_xyz": [
                            round(float(value), 4) for value in point
                        ],
                    }

        # Park unused FUEL outside the test and feed one body at a time through
        # the actual horizontal intake trigger/conveyor.
        for index in range(XRC_PRELOAD_COUNT, fuel.count):
            ids = np.asarray([index], np.int32)
            fuel.set_world_poses(
                positions=np.asarray([[-7.0, -7.0, 0.08]], np.float32), indices=ids
            )
            fuel.set_linear_velocities(np.zeros((1, 3), np.float32), indices=ids)

        failed: list[int] = []
        failed_state: dict[str, object] = {}
        for index in range(XRC_PRELOAD_COUNT, HOPPER_PRESSURE_CAPACITY):
            robot_p, robot_q = controller.chassis_pose()
            mouth = robot_p + quat_wxyz_to_matrix(robot_q) @ INTAKE_CENTER_LOCAL
            ids = np.asarray([index], np.int32)
            fuel.set_world_poses(positions=mouth[None, :], indices=ids)
            fuel.set_linear_velocities(np.zeros((1, 3), np.float32), indices=ids)
            # A packed dynamic pile can make the indexer choose another free
            # cell; allow that physical re-routing to finish before declaring
            # the intake jammed.
            for _ in range(180):
                controller.step_intake(fuel, set(), dt_s=1 / 30)
                for _ in range(4):
                    sim.step(render=False)
                if index in controller.magazine:
                    break
            trace_escaped("fill", index)
            if index not in controller.magazine:
                failed.append(index)
                poses, _ = fuel.get_world_poses()
                robot_p, robot_q = controller.chassis_pose()
                point = (
                    np.asarray(poses)[index] - robot_p
                ) @ quat_wxyz_to_matrix(robot_q)
                target = controller.intake_release_target.get(index)
                failed_state = {
                    "index": index,
                    "waypoint": controller.intake_transit.get(index),
                    "local_xyz": [round(float(value), 4) for value in point],
                    "release_target": (
                        [round(float(value), 4) for value in target]
                        if target is not None else None
                    ),
                }
                break

        # Abrupt but bounded motion exposes gaps at the bumper, belly and lower
        # roller bed without driving the robot into field structures.
        for step in range(480):
            direction = 0.35 if (step // 120) % 2 == 0 else -0.35
            controller.drive(0.0, 0.0, direction)
            controller.step_intake(fuel, set(), dt_s=1 / 120)
            sim.step(render=False)
            controller.sync_magazine(fuel)
            trace_escaped("motion", step)

        poses, _ = fuel.get_world_poses()
        poses = np.asarray(poses)
        robot_p, robot_q = controller.chassis_pose()
        local = (poses - robot_p) @ quat_wxyz_to_matrix(robot_q)
        loaded = sorted(controller.captured_indices)
        physical_inside = [
            i for i in loaded
            if (
                -0.50 <= float(local[i, 0]) <= 0.52
                and abs(float(local[i, 1])) <= 0.38
                and 0.045 <= float(local[i, 2]) <= 0.78
            )
        ]
        leaked_below = [i for i in loaded if float(local[i, 2]) < 0.045]
        outside = {
            str(i): [round(float(value), 4) for value in local[i]]
            for i in loaded if i not in physical_inside
        }
        result = {
            "target": HOPPER_PRESSURE_CAPACITY,
            "captured": len(loaded),
            "physical_inside": len(physical_inside),
            "feeder_staged": len(controller.feeder_queue),
            "failed_indices": failed,
            "failed_state": failed_state,
            "leaked_below": leaked_below,
            "outside_local_xyz": outside,
            "first_escape": first_escape,
        }
        result["passed"] = bool(
            not failed
            and len(loaded) == HOPPER_PRESSURE_CAPACITY
            and len(physical_inside) == HOPPER_PRESSURE_CAPACITY
            and not leaked_below
        )
        out = PROJECT / "runs" / "robot_fill_validation.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("CompetitionRobot_FILL " + json.dumps(result), flush=True)
        sim.stop()
        if not result["passed"]:
            raise RuntimeError("robot intake fill/retention acceptance failed")
    finally:
        app.close()


if __name__ == "__main__":
    main()
