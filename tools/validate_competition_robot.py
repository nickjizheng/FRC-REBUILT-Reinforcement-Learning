#!/usr/bin/env python3
"""Detailed PhysX validation for the imported Competition Robot CAD/controller."""
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
    results: dict[str, object] = {}
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import HubRouter, SceneBuilder
        from xrc_rebuilt.competition_robot import ROBOT_ROOT_PATH, CompetitionRobotController

        context = omni.usd.get_context()
        context.new_stage()
        builder = SceneBuilder(context.get_stage(), max_fuel=64, articulated_robot=True)
        stats = builder.build()
        fuel = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        sim = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel.initialize()
        robot = SingleArticulation(ROBOT_ROOT_PATH)
        robot.initialize()
        controller = CompetitionRobotController()
        controller.initialize(robot)
        router = HubRouter(fuel, 64)

        def settle(steps: int = 30) -> None:
            controller.drive_swerve(0.0, 0.0, 0.0)
            for _ in range(steps):
                sim.step(render=False)
                controller.sync_magazine(fuel)

        def teleport(x: float, y: float, yaw_rad: float) -> None:
            robot.set_world_pose(
                position=np.array([x, y, 0.02], dtype=np.float32),
                orientation=np.array(
                    [math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)],
                    dtype=np.float32,
                ),
            )
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            settle()

        def park_all() -> None:
            for i in range(64):
                idx = np.asarray([i], dtype=np.int32)
                p = np.array([[-7.5 + 0.22 * (i % 12), -7.2 - 0.2 * (i // 12), 0.078]], dtype=np.float32)
                fuel.set_world_poses(positions=p, indices=idx)
                fuel.set_linear_velocities(np.zeros((1, 3), dtype=np.float32), indices=idx)

        park_all()
        shots: list[dict[str, object]] = []
        tests = [
            (1.52, -5.55, "blue"),
            (0.0, -5.55, "blue"),
            (-1.0, -5.55, "blue"),
            (-1.52, 5.55, "red"),
        ]
        for attempt, (x, y, alliance) in enumerate(tests):
            teleport(x, y, 0.0)
            initial = controller.solve_auto_aim(alliance)
            teleport(x, y, float(initial["desired_yaw_rad"]))
            volley_indices = list(range(attempt * 3, attempt * 3 + 3))
            controller.preload(fuel, count=3, start_index=attempt * 3)
            settle(20)
            solution = controller.solve_auto_aim(alliance)
            detected_before = router.detected
            fired = controller.fire_setpoint(fuel, solution, now_s=100.0 + attempt * 10.0)
            fired_indices = controller.last_fired_indices[:]
            max_z = {index: -1e9 for index in fired_indices}
            closest = {index: 1e9 for index in fired_indices}
            closest_position: dict[int, list[float] | None] = {
                index: None for index in fired_indices
            }
            trace: list[list[float]] = []
            target = np.array(
                [0.0199, 3.6874, 1.3646]
                if alliance == "red"
                else [-0.0199, -3.6874, 1.3646],
                dtype=np.float32,
            )
            for step in range(360):
                sim.step(render=False)
                router.step(100.0 + attempt * 10.0 + step / 120)
                poses, _ = fuel.get_world_poses()
                poses_np = np.asarray(poses)
                for index in fired_indices:
                    p = poses_np[index]
                    max_z[index] = max(max_z[index], float(p[2]))
                    distance = float(np.linalg.norm(p - target))
                    if distance < closest[index]:
                        closest[index], closest_position[index] = distance, p.tolist()
                if step % 10 == 0:
                    trace.append(poses_np[fired_indices[0]].tolist())
                if router.detected >= detected_before + len(fired_indices):
                    break
            scored_count = min(len(fired_indices), router.detected - detected_before)
            shots.append(
                {
                    "pose": [x, y],
                    "alliance": alliance,
                    "fired": fired is not None,
                    "fired_indices": fired_indices,
                    "scored": scored_count == len(fired_indices),
                    "scored_count": scored_count,
                    "max_z": max_z,
                    "closest_to_markball_m": closest,
                    "closest_position": closest_position,
                    "solution": {k: v for k, v in solution.items() if k != "setpoint"},
                    "trace": trace,
                }
            )
            router.pending.clear()
            router.blocked_until_clear.clear()
            router.released_watch.clear()

        results = {
            "scene": stats,
            "shots": shots,
            "volleys_passed": sum(bool(s["scored"]) for s in shots),
            "balls_scored": sum(int(s["scored_count"]) for s in shots),
        }
        out = PROJECT / "runs" / "competition_robot_validation.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("CompetitionRobot_VALIDATION " + json.dumps(results), flush=True)
        sim.stop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
