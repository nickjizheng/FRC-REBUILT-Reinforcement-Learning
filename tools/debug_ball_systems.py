#!/usr/bin/env python3
"""Dense telemetry for the intake-capture and shooter-scoring scenarios."""
from __future__ import annotations

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
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import HubRouter, SceneBuilder
        from xrc_rebuilt.robot_model import RobotController, quat_wxyz_to_matrix

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        builder = SceneBuilder(stage, max_fuel=24, articulated_robot=True)
        builder.build()
        fuel_view = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        sim = SimulationContext(physics_dt=0.004, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel_view.initialize()
        robot = SingleArticulation("/World/Robot/LegacyRobot")
        robot.initialize()
        controller = RobotController()
        controller.initialize(robot)
        router = HubRouter(fuel_view, 24)

        to_np = lambda v: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v)

        def teleport(x, y, yaw_deg):
            half = math.radians(yaw_deg) * 0.5
            robot.set_world_pose(
                position=np.array([x, y, 0.02], dtype=np.float32),
                orientation=np.array([math.cos(half), 0, 0, math.sin(half)], dtype=np.float32),
            )
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
            controller.drive(0.0, 0.0)
            for _ in range(40):
                sim.step(render=False)

        def park_ball(index, x, y):
            idx = np.asarray([index], dtype=np.int32)
            fuel_view.set_world_poses(np.array([[x, y, 0.077]], dtype=np.float32), indices=idx)
            fuel_view.set_linear_velocities(np.zeros((1, 3), dtype=np.float32), indices=idx)

        for i in range(24):
            park_ball(i, -7.5 + 0.3 * (i % 8), -7.0 - 0.3 * (i // 8))

        # ============ C: intake approach ============
        print("=== SCENARIO C DEBUG ===", flush=True)
        teleport(2.5, -6.8, 90.0)
        park_ball(0, 2.5, -6.15)
        controller.intake_on = True
        for step in range(700):
            controller.drive(0.25, 0.0)
            sim.step(render=False)
            if step % 5 == 0:
                controller.step_intake(fuel_view, set())
            if step % 50 == 0:
                position, orientation = controller.chassis_pose()
                rotation = quat_wxyz_to_matrix(orientation)
                balls = to_np(fuel_view.get_world_poses()[0])
                local0 = (balls[0] - position) @ rotation
                print(
                    f"t={step*0.004:5.2f} robot=({position[0]:6.3f},{position[1]:6.3f},{position[2]:5.3f}) "
                    f"ball0_world=({balls[0][0]:6.3f},{balls[0][1]:6.3f},{balls[0][2]:5.3f}) "
                    f"ball0_local=({local0[0]:6.3f},{local0[1]:6.3f},{local0[2]:6.3f}) "
                    f"mag={len(controller.magazine)}",
                    flush=True,
                )
        print("captured:", controller.balls_collected, flush=True)

        # ============ D: shot trajectory ============
        print("=== SCENARIO D DEBUG ===", flush=True)
        teleport(0.0, 3.269 - 1.8, 90.0)
        controller.magazine = [10]
        controller.pen_reserved = {10}
        controller._park(fuel_view, [10])
        for _ in range(30):
            sim.step(render=False)
        fired = controller.fire(fuel_view, aim=0.25, now_s=100.0)
        print("fired ball index:", fired, flush=True)
        for step in range(200):
            sim.step(render=False)
            router.step(step * 0.004)
            if step % 5 == 0:
                balls = to_np(fuel_view.get_world_poses()[0])
                velocity = to_np(fuel_view.get_linear_velocities())[fired]
                b = balls[fired]
                print(
                    f"t={step*0.004:5.3f} ball=({b[0]:6.3f},{b[1]:6.3f},{b[2]:6.3f}) "
                    f"v=({velocity[0]:5.2f},{velocity[1]:5.2f},{velocity[2]:5.2f}) "
                    f"detected={router.detected}",
                    flush=True,
                )
            if router.detected:
                print("DETECTED at t=", step * 0.004, flush=True)
                break
    finally:
        app.close()


if __name__ == "__main__":
    main()
