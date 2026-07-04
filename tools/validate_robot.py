#!/usr/bin/env python3
"""Acceptance tests for the articulated Legacy Robot robot model.

Scenarios (single headless Isaac session, sequential):
  A. flat-ground top speed at full stick (xRC target ~3.5 m/s),
  B. turn-in-place yaw rate at full turn stick (xRC target ~100 deg/s),
  C. intake capture: drive over three staged balls with the intake on,
  D. exact-xRC auto-aim scoring through the physical blue HUB,
  E. continuous eight-ball firing with the real indexer cooldown,
  F. visible magazine stability in the extracted preload positions,
  G. FSM-driven continuous fire over N magazine runs via the shared
     RobotController.update path (XRC_CONTINUOUS_RUNS=20 -> full 160/160),
  H. reference-style multi-ball volley (one shot -> 3 balls, all must score).

Scenario C also confirms intake still works with the containment colliders/panels
present (the panels are visual-only over the existing solid colliders).

Writes runs/robot_validation.json and prints ROBOT_VALIDATION lines.

Run:  OMNI_KIT_ACCEPT_EULA=YES C:/il/venv/Scripts/python.exe tools/validate_robot.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def yaw_of(orientation: np.ndarray) -> float:
    qw, qx, qy, qz = (float(v) for v in orientation)
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def main() -> None:
    sys.argv = [sys.argv[0]]
    import os

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    results: dict[str, dict] = {}
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

        def settle(steps: int = 50) -> None:
            controller.drive(0.0, 0.0)
            for _ in range(steps):
                sim.step(render=False)
                controller.sync_magazine(fuel_view)

        def teleport(x: float, y: float, yaw_deg: float) -> None:
            half = math.radians(yaw_deg) * 0.5
            robot.set_world_pose(
                position=np.array([x, y, 0.02], dtype=np.float32),
                orientation=np.array(
                    [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32
                ),
            )
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
            # base velocity must be zeroed too or momentum carries across scenarios
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            settle(30)

        def park_ball(index: int, x: float, y: float) -> None:
            idx = np.asarray([index], dtype=np.int32)
            fuel_view.set_world_poses(
                positions=np.array([[x, y, 0.077]], dtype=np.float32), indices=idx
            )
            fuel_view.set_linear_velocities(np.zeros((1, 3), dtype=np.float32), indices=idx)

        # park all 24 balls far away in a grid so scenarios are clean
        for i in range(24):
            park_ball(i, -7.5 + 0.3 * (i % 8), -7.0 - 0.3 * (i // 8))

        # ---- A: top speed (clear lane; sample the flat stretch before the bump)
        teleport(-2.0, -6.8, 90.0)
        speeds = []
        for step in range(280):
            controller.drive(1.0, 0.0)
            sim.step(render=False)
            if 160 < step and step % 10 == 0:
                velocity = robot.get_linear_velocity()
                speeds.append(float(np.linalg.norm(np.asarray(velocity)[:2])))
        results["A_top_speed"] = {
            "mean_mps": round(float(np.mean(speeds)), 3),
            "target_mps": 3.5,
            "pass": bool(3.2 <= float(np.mean(speeds)) <= 3.8),
        }
        print("ROBOT_VALIDATION A", results["A_top_speed"], flush=True)

        # ---- B: yaw rate (open neutral zone; y=-6.8 wedges the robot against
        # the alliance-wall uprights and reads ~2 deg/s) ------------------------
        teleport(0.0, -0.8, 0.0)
        yaws, times = [], []
        for step in range(750):
            controller.drive(0.0, 1.0)
            sim.step(render=False)
            if step > 250 and step % 25 == 0:
                _, orientation = controller.chassis_pose()
                yaws.append(yaw_of(orientation))
                times.append(step * 0.004)
        unwrapped = np.unwrap(yaws)
        yaw_rate = float(np.polyfit(times, unwrapped, 1)[0])
        results["B_yaw_rate"] = {
            "deg_per_s": round(math.degrees(yaw_rate), 1),
            "xrc_target_deg_per_s": 100.0,
            # +turn must be CCW (+yaw) and near the xRC-calibrated magnitude
            "pass": bool(75.0 <= math.degrees(yaw_rate) <= 130.0),
        }
        print("ROBOT_VALIDATION B", results["B_yaw_rate"], flush=True)

        # ---- C: intake capture ---------------------------------------------
        teleport(2.5, -6.8, 90.0)
        for offset, index in zip((0.65, 0.95, 1.25), (0, 1, 2)):
            park_ball(index, 2.5, -6.8 + offset)
        settle(20)
        controller.intake_on = True
        collected_before = controller.balls_collected
        for step in range(900):
            controller.drive(0.25, 0.0)
            sim.step(render=False)
            if step % 5 == 0:
                controller.step_intake(fuel_view, set())
        captured = controller.balls_collected - collected_before
        results["C_intake"] = {"staged": 3, "captured": int(captured), "pass": bool(captured >= 2)}
        print("ROBOT_VALIDATION C", results["C_intake"], flush=True)

        # ---- D: repeated one-button auto shots from a calibrated pose -------
        # This uses the detailed articulation, full field collision geometry,
        # xRC's coupled 6..11 m/s / 70.355..50.355 degree mechanism, and the
        # actual scorer route.  A miss is a failed acceptance, not a weak
        # "at least one hit" pass.
        attempts = 20
        shot_results = []
        clock = 100.0
        for attempt in range(attempts):
            # Alliance-wall side of the blue HUB: the xRC funnel accepts this
            # arc.  The opposite side is physically screened and is correctly
            # treated as an unsafe shooting pose.
            teleport(1.52, -5.55, 90.0)
            initial = controller.solve_auto_aim("blue")
            teleport(1.52, -5.55, math.degrees(float(initial["desired_yaw_rad"])))
            controller.magazine = [10]
            controller.pen_reserved = {10}
            controller.sync_magazine(fuel_view)
            settle(25)
            detected_before = router.detected
            clock += 10.0
            solution = controller.solve_auto_aim("blue")
            fired = controller.fire_auto(fuel_view, now_s=clock, alliance="blue")
            for step in range(550):  # 2.2 s: flight plus funnel fall
                sim.step(render=False)
                if step % 5 == 0:
                    router.step(clock + step * 0.004)
                if router.detected > detected_before:
                    break
            scored = router.detected > detected_before
            shot_results.append(
                {
                    "attempt": attempt + 1,
                    "fired": fired is not None,
                    "scored": bool(scored),
                    "aim": float(solution["aim"]),
                    "yaw_error_deg": float(solution["yaw_error_deg"]),
                    "vertical_error_m": float(solution["vertical_error_m"]),
                }
            )
            if not scored:
                park_ball(10, -7.5, -7.0)
            router.pending.clear()
            router.blocked_until_clear.clear()
            router.released_watch.clear()
        scoring_shots = [s for s in shot_results if s["scored"]]
        results["D_shooter"] = {
            "attempts": attempts,
            "scoring_count": len(scoring_shots),
            "accuracy": len(scoring_shots) / attempts,
            "shots": shot_results,
            "pass": len(scoring_shots) == attempts,
        }
        print("ROBOT_VALIDATION D", json.dumps(results["D_shooter"]), flush=True)

        # ---- E: continuous eight-ball fire ----------------------------------
        teleport(1.52, -5.55, 90.0)
        initial = controller.solve_auto_aim("blue")
        teleport(1.52, -5.55, math.degrees(float(initial["desired_yaw_rad"])))
        continuous_indices = list(range(12, 20))
        controller.magazine = continuous_indices[:]
        controller.pen_reserved = set(continuous_indices)
        controller.sync_magazine(fuel_view)
        settle(25)
        detected_before = router.detected
        fired_times: list[float] = []
        continuous_clock = clock + 100.0
        for step in range(1200):  # 4.8 s, enough for all flights to reach HUB
            now = continuous_clock + step * 0.004
            if step % 5 == 0 and controller.magazine:
                fired = controller.fire_auto(fuel_view, now_s=now, alliance="blue")
                if fired is not None:
                    fired_times.append(now)
            sim.step(render=False)
            if step % 5 == 0:
                router.step(now)
            if not controller.magazine and router.detected - detected_before >= 8:
                break
        scored_continuous = router.detected - detected_before
        intervals = np.diff(fired_times)
        results["E_continuous_fire"] = {
            "loaded": 8,
            "fired": len(fired_times),
            "scored": int(scored_continuous),
            "minimum_interval_s": round(float(intervals.min()), 3) if len(intervals) else None,
            "pass": bool(
                len(fired_times) == 8
                and scored_continuous == 8
                and (not len(intervals) or float(intervals.min()) >= 0.069)
            ),
        }
        print("ROBOT_VALIDATION E", results["E_continuous_fire"], flush=True)
        router.pending.clear()
        router.blocked_until_clear.clear()
        router.released_watch.clear()

        # ---- F: visible magazine stability -----------------------------------
        teleport(0.0, -0.8, 0.0)
        controller.magazine = [20, 21, 22]
        controller.pen_reserved = {20, 21, 22}
        controller.sync_magazine(fuel_view)
        for step in range(500):
            sim.step(render=False)
            if step % 5 == 0:
                controller.step_intake(fuel_view, set())
        positions, _ = fuel_view.get_world_poses()
        positions_np = np.asarray(
            positions.detach().cpu().numpy() if hasattr(positions, "detach") else positions
        )
        position, orientation = controller.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        local = (positions_np[list(controller.magazine)] - position) @ rotation
        slot_error = np.linalg.norm(local - controller.preload_slots[:3], axis=1)
        results["F_visible_magazine"] = {
            "magazine": len(controller.magazine),
            "world_min_z": round(float(positions_np[list(controller.magazine), 2].min()), 3),
            "max_slot_error_m": round(float(slot_error.max()), 5),
            "pass": bool(slot_error.max() < 0.01 and positions_np[list(controller.magazine), 2].min() > 0.1),
        }
        print("ROBOT_VALIDATION F", results["F_visible_magazine"], flush=True)

        # ---- G: FSM-driven continuous fire over multiple magazine runs -------
        # Exercises the exact GUI/RL code path (RobotController.update -> the
        # shooter/indexer state machine) end to end in PhysX.  Default 3 runs
        # for a quick gate; set XRC_CONTINUOUS_RUNS=20 for the full 160/160.
        runs_target = int(os.environ.get("XRC_CONTINUOUS_RUNS", "3"))
        teleport(1.52, -5.55, 90.0)
        aim0 = controller.solve_auto_aim("blue")
        teleport(1.52, -5.55, math.degrees(float(aim0["desired_yaw_rad"])))
        settle(25)
        mag_indices = list(range(8))
        per_run_fired: list[int] = []
        per_run_scored: list[int] = []
        all_intervals: list[float] = []
        fsm_clock = continuous_clock + 200.0
        now = fsm_clock
        for _run in range(runs_target):
            controller.magazine = mag_indices[:]
            controller.pen_reserved = set(mag_indices)
            controller._muzzle_watch.clear()
            controller.state_machine.last_feed_time = -1.0
            controller.state_machine.set_continuous(True)
            controller.sync_magazine(fuel_view)
            settle(20)
            detected_before = router.detected
            run_fired, fired_ts = 0, []
            for step in range(1500):  # up to 6 s per magazine
                now = fsm_clock + step * 0.004
                if step % 5 == 0:
                    status = controller.update(
                        fuel_view, now_s=now, alliance="blue", hub_active=True
                    )
                    if status["fired_index"] is not None:
                        run_fired += 1
                        fired_ts.append(now)
                sim.step(render=False)
                if step % 5 == 0:
                    router.step(now)
                if not controller.magazine and router.detected - detected_before >= 8:
                    break
            per_run_fired.append(run_fired)
            per_run_scored.append(int(router.detected - detected_before))
            all_intervals.extend(np.diff(fired_ts).tolist())
            fsm_clock = now + 5.0
            for idx in mag_indices:
                park_ball(idx, -7.5 + 0.3 * idx, -7.4)
            router.pending.clear()
            router.blocked_until_clear.clear()
            router.released_watch.clear()
        min_interval = min(all_intervals) if all_intervals else None
        results["G_continuous_fsm"] = {
            "runs": runs_target,
            "per_run_fired": per_run_fired,
            "per_run_scored": per_run_scored,
            "total_fired": int(sum(per_run_fired)),
            "total_scored": int(sum(per_run_scored)),
            "minimum_interval_s": round(float(min_interval), 4) if min_interval is not None else None,
            "pass": bool(
                all(f == 8 for f in per_run_fired)
                and all(s == 8 for s in per_run_scored)
                and (min_interval is None or float(min_interval) >= 0.069)
            ),
        }
        print("ROBOT_VALIDATION G", json.dumps(results["G_continuous_fsm"]), flush=True)
        router.pending.clear()
        router.blocked_until_clear.clear()
        router.released_watch.clear()

        # ---- H: reference-style multi-ball volley ----------------------------
        # One shot releases a horizontal row of 3 balls; all must score.
        teleport(1.52, -5.55, 90.0)
        aimH = controller.solve_auto_aim("blue")
        teleport(1.52, -5.55, math.degrees(float(aimH["desired_yaw_rad"])))
        controller.barrels = 3
        volley_idx = [0, 1, 2]
        controller.magazine = volley_idx[:]
        controller.pen_reserved = set(volley_idx)
        controller._muzzle_watch.clear()
        controller.sync_magazine(fuel_view)
        settle(25)
        before_h = router.detected
        hclock = 2000.0
        fired_h = controller.fire_volley(fuel_view, aim=float(aimH["aim"]), now_s=hclock)
        for step in range(700):
            sim.step(render=False)
            if step % 5 == 0:
                router.step(hclock + step * 0.004)
            if router.detected - before_h >= 3:
                break
        controller.barrels = 1
        results["H_multiball_volley"] = {
            "barrels": 3,
            "fired": len(fired_h),
            "scored": int(router.detected - before_h),
            "pass": bool(len(fired_h) == 3 and (router.detected - before_h) == 3),
        }
        print("ROBOT_VALIDATION H", json.dumps(results["H_multiball_volley"]), flush=True)
        router.pending.clear()
        router.blocked_until_clear.clear()
        router.released_watch.clear()

        results["all_pass"] = all(v.get("pass") for k, v in results.items() if isinstance(v, dict))
        (PROJECT / "runs" / "robot_validation.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print("ROBOT_VALIDATION SUMMARY", json.dumps(results), flush=True)
    except BaseException as error:  # noqa: BLE001
        import traceback

        print("ROBOT_VALIDATION_ERROR", repr(error), flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
