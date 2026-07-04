#!/usr/bin/env python3
"""Phase D4 acceptance: drive-to-shoot from arbitrary starts (closed loop).

From each start pose the planner produces a collision-free path to an open-side
firing pose; the differential pure-pursuit follower drives there in PhysX; then
the shooter FSM fires and we check the ball scores through the real HUB.  This
proves the robot autonomously scores "from anywhere" by navigating, not by
teleporting or blind shooting.

Run:  OMNI_KIT_ACCEPT_EULA=YES C:/il/venv/Scripts/python.exe tools/validate_drive_to_shoot.py
Writes runs/drive_to_shoot.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

# Arbitrary legal starts that require driving (neutral zone / mid-field).
import os as _os

if _os.environ.get("XRC_DRIVE_NEUTRAL"):
    # hard case: neutral-zone starts that must round the central HUB
    STARTS = [(0.0, 0.0), (2.2, -1.0), (-2.2, 1.0), (1.5, 2.5), (-1.5, -2.5), (3.0, 0.5)]
else:
    # realistic case: reposition to a firing pose within the blue alliance zone
    STARTS = [(0.0, -8.4), (3.2, -7.6), (-3.2, -7.6), (3.4, -6.6), (-3.4, -6.6), (1.6, -8.2)]
DIAG = bool(_os.environ.get("XRC_DRIVE_DIAG"))
HUB = "blue"


def main() -> None:
    sys.argv = [sys.argv[0]]
    import os

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    out: dict = {"hub": HUB, "starts": []}
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import HubRouter, SceneBuilder
        from xrc_rebuilt.robot_model import RobotController
        from xrc_rebuilt.shot_planner import FieldState, RobotState, plan_global_score

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
        field = FieldState()
        field.occupancy()

        def settle(steps=30):
            controller.drive(0.0, 0.0)
            for _ in range(steps):
                sim.step(render=False)
                controller.sync_magazine(fuel_view)

        def teleport(x, y, yaw_deg):
            half = math.radians(yaw_deg) * 0.5
            robot.set_world_pose(position=np.array([x, y, 0.02], np.float32),
                                 orientation=np.array([math.cos(half), 0, 0, math.sin(half)], np.float32))
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), np.float32))
            robot.set_linear_velocity(np.zeros(3, np.float32))
            robot.set_angular_velocity(np.zeros(3, np.float32))
            settle(30)

        def park(idx, x, y):
            i = np.asarray([idx], np.int32)
            fuel_view.set_world_poses(positions=np.array([[x, y, 0.077]], np.float32), indices=i)
            fuel_view.set_linear_velocities(np.zeros((1, 3), np.float32), indices=i)

        for i in range(24):
            park(i, -7.5 + 0.3 * (i % 8), -7.5)

        clock = 500.0
        for si, (sx, sy) in enumerate(STARTS):
            teleport(sx, sy, 0.0)
            pos, _ = controller.chassis_pose()
            plan = plan_global_score(RobotState(float(pos[0]), float(pos[1])), HUB, field)
            record = {"start": [sx, sy], "reason": plan.reason, "valid": bool(plan.valid),
                      "arrived": False, "scored": False, "drive_ticks": 0}
            if not plan.valid:
                out["starts"].append(record)
                print("DRIVE", si, "start", (sx, sy), "-> UNREACHABLE", flush=True)
                continue
            # load one ball; it rides in the magazine while driving
            controller.magazine = [si]
            controller.pen_reserved = {si}
            controller._muzzle_watch.clear()
            controller.sync_magazine(fuel_view)
            arrival_yaw = float(plan.arrival_yaw_rad)
            path = plan.path
            arrived = plan.reason == "direct"
            controller.begin_follow()
            ticks = 0
            for step in range(5000):
                if step % 5 == 0 and not arrived:
                    cmd = controller.follow(path, arrival_yaw)
                    controller.sync_magazine(fuel_view)
                    ticks += 1
                    if DIAG and ticks % 40 == 0:
                        p, _ = controller.chassis_pose()
                        print("  DBG t=%d pos=(%.2f,%.2f) dist=%.2f phase=%s"
                              % (ticks, float(p[0]), float(p[1]), cmd["dist_goal"], cmd["phase"]), flush=True)
                    if cmd["phase"] == "arrived":
                        arrived = True
                sim.step(render=False)
                if arrived:
                    break
            final_pos, _ = controller.chassis_pose()
            record["arrived"] = bool(arrived)
            record["drive_ticks"] = ticks
            record["path_len"] = len(path)
            record["firing_pose"] = [round(v, 2) for v in (plan.firing_pose or (0, 0, 0))]
            record["final_pos"] = [round(float(final_pos[0]), 2), round(float(final_pos[1]), 2)]
            record["final_dist_to_goal"] = round(math.dist(
                (float(final_pos[0]), float(final_pos[1])), path[-1]), 3)
            if arrived:
                settle(20)
                controller.state_machine.set_continuous(True)
                controller.state_machine.last_feed_time = -1.0
                before = router.detected
                for step in range(900):
                    now = clock + step * 0.004
                    if step % 5 == 0:
                        controller.update(fuel_view, now_s=now, alliance=HUB, hub_active=True)
                    sim.step(render=False)
                    if step % 5 == 0:
                        router.step(now)
                    if router.detected > before:
                        break
                controller.state_machine.set_continuous(False)
                record["scored"] = bool(router.detected > before)
                clock += 20.0
            out["starts"].append(record)
            print("DRIVE", si, "start", (sx, sy), "reason", plan.reason,
                  "arrived", record["arrived"], "scored", record["scored"],
                  "ticks", ticks, flush=True)
            park(si, -7.5, -7.5)
            router.pending.clear()
            router.blocked_until_clear.clear()
            router.released_watch.clear()

        scored = sum(1 for r in out["starts"] if r["scored"])
        out["summary"] = {"total": len(STARTS), "scored": scored,
                          "success_rate": round(scored / len(STARTS), 3)}
        (PROJECT / "runs" / "drive_to_shoot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("DRIVE_SUMMARY", json.dumps(out["summary"]), flush=True)
    except BaseException as error:  # noqa: BLE001
        import traceback
        print("DRIVE_ERROR", repr(error), flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
