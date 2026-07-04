#!/usr/bin/env python3
"""Measure the real PhysX scoring envelope for both HUBs (Phase D2/D3).

For each HUB we sweep open-side poses over range x lateral angle, aim with the
exact xRC solver (no artificial range gate), fire one ball through the full
field/HUB geometry, and record whether it scored.  The measured min/max
scoring range then replaces the conservative 2.45-3.05 m gate so the robot
shoots wherever a shot actually scores.

Run:  OMNI_KIT_ACCEPT_EULA=YES C:/il/venv/Scripts/python.exe tools/sweep_envelope.py
Writes runs/shot_envelope.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

RANGES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0]
ANGLES_DEG = [-40.0, -20.0, 0.0, 20.0, 40.0]


def main() -> None:
    sys.argv = [sys.argv[0]]
    import os

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    out: dict = {"ranges": RANGES, "angles_deg": ANGLES_DEG, "by_hub": {}}
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import HubRouter, SceneBuilder
        from xrc_rebuilt.robot_model import (
            HUB_MARKBALL_TARGETS,
            RobotController,
            solve_shot_geometry,
        )

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        SceneBuilder(stage, max_fuel=24, articulated_robot=True).build()
        fuel_view = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        sim = SimulationContext(physics_dt=0.004, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel_view.initialize()
        robot = SingleArticulation("/World/Robot/LegacyRobot")
        robot.initialize()
        controller = RobotController()
        controller.initialize(robot)
        router = HubRouter(fuel_view, 24)

        def settle(steps=30):
            controller.drive(0.0, 0.0)
            for _ in range(steps):
                sim.step(render=False)
                controller.sync_magazine(fuel_view)

        def teleport(x, y, yaw_deg):
            half = math.radians(yaw_deg) * 0.5
            robot.set_world_pose(
                position=np.array([x, y, 0.02], np.float32),
                orientation=np.array([math.cos(half), 0.0, 0.0, math.sin(half)], np.float32),
            )
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), np.float32))
            robot.set_linear_velocity(np.zeros(3, np.float32))
            robot.set_angular_velocity(np.zeros(3, np.float32))
            settle(30)

        def park(idx, x, y):
            i = np.asarray([idx], np.int32)
            fuel_view.set_world_poses(positions=np.array([[x, y, 0.077]], np.float32), indices=i)
            fuel_view.set_linear_velocities(np.zeros((1, 3), np.float32), indices=i)

        for i in range(24):
            park(i, -7.5 + 0.3 * (i % 8), -7.0 - 0.3 * (i // 8))

        clock = 100.0
        for hub, target in HUB_MARKBALL_TARGETS.items():
            open_sign = -1.0 if hub == "blue" else 1.0
            results = []
            for r in RANGES:
                for th in ANGLES_DEG:
                    rad = math.radians(th)
                    x = float(target[0]) + r * math.sin(rad)
                    y = float(target[1]) + open_sign * r * math.cos(rad)
                    if abs(x) > 3.7 or abs(y) > 8.9:
                        continue
                    pos = np.array([x, y, 0.20], np.float32)
                    geo = solve_shot_geometry(pos, np.asarray(target, np.float32))
                    trajectory_ok = abs(float(geo["vertical_error_m"])) <= 0.03
                    scored = False
                    if trajectory_ok:
                        yaw_deg = math.degrees(float(geo["desired_yaw_rad"]))
                        try:
                            teleport(x, y, yaw_deg)
                            # re-solve from the settled chassis for the final aim
                            pose_now, _ = controller.chassis_pose()
                            geo2 = solve_shot_geometry(pose_now, np.asarray(target, np.float32))
                            controller.magazine = [0]
                            controller.pen_reserved = {0}
                            controller.sync_magazine(fuel_view)
                            settle(15)
                            before = router.detected
                            clock += 5.0
                            controller.last_shot_time = -1.0
                            controller.fire(fuel_view, aim=float(geo2["aim"]), now_s=clock)
                            for step in range(600):
                                sim.step(render=False)
                                if step % 5 == 0:
                                    router.step(clock + step * 0.004)
                                if router.detected > before:
                                    break
                            scored = router.detected > before
                            park(0, -7.5, -7.0)
                            router.pending.clear()
                            router.blocked_until_clear.clear()
                            router.released_watch.clear()
                        except Exception as exc:  # noqa: BLE001
                            print("SWEEP_POSE_ERROR", hub, round(r, 2), th, repr(exc), flush=True)
                    results.append({
                        "range_m": round(float(geo["range_m"]), 3),
                        "angle_deg": th,
                        "x": round(x, 3), "y": round(y, 3),
                        "vertical_error_m": round(float(geo["vertical_error_m"]), 4),
                        "trajectory_ok": bool(trajectory_ok),
                        "scored": bool(scored),
                    })
                    print("SWEEP", hub, "r=%.2f" % r, "th=%+.0f" % th,
                          "traj=%d" % trajectory_ok, "score=%d" % scored, flush=True)
            scored_ranges = [x["range_m"] for x in results if x["scored"]]
            out["by_hub"][hub] = {
                "results": results,
                "scored_count": len(scored_ranges),
                "min_scored_range_m": round(min(scored_ranges), 3) if scored_ranges else None,
                "max_scored_range_m": round(max(scored_ranges), 3) if scored_ranges else None,
            }
            print("SWEEP_HUB_DONE", hub, out["by_hub"][hub]["min_scored_range_m"],
                  out["by_hub"][hub]["max_scored_range_m"], flush=True)

        (PROJECT / "runs" / "shot_envelope.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("SWEEP_SUMMARY", json.dumps({h: {"min": v["min_scored_range_m"], "max": v["max_scored_range_m"],
              "n": v["scored_count"]} for h, v in out["by_hub"].items()}), flush=True)
    except BaseException as error:  # noqa: BLE001
        import traceback
        print("SWEEP_ERROR", repr(error), flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
