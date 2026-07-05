#!/usr/bin/env python3
"""PhysX acceptance for robot square/rectangle Weidai and trench passage."""
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
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        from xrc_rebuilt.isaac_scene import SceneBuilder
        from xrc_rebuilt.competition_robot import (
            CAMERA_RIG,
            ROBOT_ROOT_PATH,
            STORAGE_LOWERED_POSITION,
            XRC_PRELOAD_COUNT,
            CompetitionRobotController,
        )

        context = omni.usd.get_context()
        context.new_stage()
        stats = SceneBuilder(context.get_stage(), max_fuel=24, articulated_robot=True).build()
        fuel = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        sim = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel.initialize()
        robot = SingleArticulation(ROBOT_ROOT_PATH)
        robot.initialize()
        controller = CompetitionRobotController()
        controller.initialize(robot)

        # Begin at the actual trench approach pose while still deployed.  The
        # old test compacted elsewhere, teleported only the robot, then respawned
        # balls inside the already-closed net, creating an artificial overlap.
        robot.set_world_pose(
            # Nose along +X; commanded +Y motion below is sideways.
            # x=-3.40 keeps the long-axis rear bumper clear of the west wall;
            # x=-3.60 is only valid when the chassis is rotated lengthwise.
            position=np.array([-3.40, -4.30, 0.005], np.float32),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        )
        robot.set_linear_velocity(np.zeros(3, np.float32))
        robot.set_angular_velocity(np.zeros(3, np.float32))
        controller.preload(fuel, XRC_PRELOAD_COUNT)
        for _ in range(180):
            controller.drive(0.0, 0.0, 0.0)
            controller.step_intake(fuel, set(), dt_s=1 / 120)
            sim.step(render=False)
            controller.sync_magazine(fuel)

        # Equivalent to CloseStorageCommand: square footprint, intake folded,
        # upper frame/net below the 22.25 in opening.
        controller.set_storage_extended(False)
        # The source-faithful compact program is staged: intake retracts first,
        # then the container lowers.  Wait for the state machine instead of the
        # old fixed one-second budget, which stopped halfway through lowering.
        for _ in range(720):
            controller.step_mechanisms(1 / 120)
            controller.step_intake(fuel, set(), dt_s=1 / 120)
            controller.drive(0.0, 0.0, 0.0)
            sim.step(render=False)
            controller.sync_magazine(fuel)
            if controller.mechanism_phase == "COMPACT":
                # Continue a short settling window so the physical prismatic
                # joints, not only the command state, reach their targets.
                pass

        compact_position = controller.storage_position
        compact_top = controller.storage_top
        intake_stowed = not controller.intake_deployed
        joint_names = list(robot.dof_names)
        joint_positions = np.asarray(robot.get_joint_positions())
        compact_joints = {
            name: float(joint_positions[joint_names.index(name)])
            for name in (
                "intake_fold",
                "horizontal_retract",
                "vertical_lower",
                "shooter_lower",
            )
        }
        compact_ball_world, _ = fuel.get_world_poses()
        compact_robot_p, compact_robot_q = controller.chassis_pose()
        from xrc_rebuilt.robot_model import quat_wxyz_to_matrix
        compact_ball_local = (
            np.asarray(compact_ball_world) - compact_robot_p
        ) @ quat_wxyz_to_matrix(compact_robot_q)

        # Settle the compact articulation before driving; no teleport or ball
        # respawn occurs after contraction.
        for _ in range(30):
            controller.drive(0.0, 0.0, 0.0)
            controller.step_intake(fuel, set(), dt_s=1 / 120)
            sim.step(render=False)
            controller.sync_magazine(fuel)

        # Cross from the south side of the narrow overhead beam to the north.
        max_z = -1e9
        max_tilt = 0.0
        entered = False
        # Continue until the complete robot has exited the far edge.  Four
        # seconds was only enough to reach the middle at keyboard slew limits.
        for _ in range(1800):
            controller.drive(0.0, 0.0, 1.0)
            controller.step_mechanisms(1 / 120)
            controller.step_intake(fuel, set(), dt_s=1 / 120)
            sim.step(render=False)
            controller.sync_magazine(fuel)
            p, q = controller.chassis_pose()
            entered |= (
                -3.90 < float(p[0]) < -3.20
                and -3.74 < float(p[1]) < -3.55
            )
            max_z = max(max_z, float(p[2]))
            up_z = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
            max_tilt = max(max_tilt, math.acos(float(np.clip(up_z, -1.0, 1.0))))
            if float(p[1]) > -2.95:
                break
        end, _ = controller.chassis_pose()
        # AABB contact audit: identify the exact compact robot proxy nearest to
        # each field obstacle when a passage attempt stalls.  This avoids
        # mistaking a low side/end support for a roof-height problem.
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )

        def collision_bounds(prefix: str, exclude: tuple[str, ...] = ()):
            items = []
            for prim in context.get_stage().Traverse():
                path = str(prim.GetPath())
                if not path.startswith(prefix) or any(path.startswith(p) for p in exclude):
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                items.append(
                    (
                        path,
                        np.asarray(aligned.GetMin(), dtype=np.float64),
                        np.asarray(aligned.GetMax(), dtype=np.float64),
                    )
                )
            return items

        robot_bounds = collision_bounds(ROBOT_ROOT_PATH)
        field_bounds = collision_bounds("/World", (ROBOT_ROOT_PATH, "/World/Fuel"))
        roof_min = np.asarray([-4.2134, -3.7210, 0.5910], dtype=np.float64)
        roof_max = np.asarray([-2.5030, -3.5680, 0.6660], dtype=np.float64)
        roof_overlaps = []
        for robot_path, robot_min, robot_max in robot_bounds:
            overlap = np.minimum(robot_max, roof_max) - np.maximum(robot_min, roof_min)
            if np.all(overlap > 0.0):
                roof_overlaps.append(
                    {
                        "robot": robot_path,
                        "overlap_xyz_m": overlap.round(4).tolist(),
                        "robot_min": robot_min.round(4).tolist(),
                        "robot_max": robot_max.round(4).tolist(),
                    }
                )
        nearest_pairs = []
        for robot_path, robot_min, robot_max in robot_bounds:
            for field_path, field_min, field_max in field_bounds:
                separation = np.maximum(
                    np.maximum(robot_min - field_max, field_min - robot_max), 0.0
                )
                distance = float(np.linalg.norm(separation))
                if distance <= 0.035:
                    nearest_pairs.append(
                        {
                            "distance_m": distance,
                            "robot": robot_path,
                            "field": field_path,
                            "robot_min": robot_min.round(4).tolist(),
                            "robot_max": robot_max.round(4).tolist(),
                            "field_min": field_min.round(4).tolist(),
                            "field_max": field_max.round(4).tolist(),
                        }
                    )
        nearest_pairs.sort(key=lambda item: item["distance_m"])
        # Exact compact CAD envelope (all six mechanism assets): the upper
        # carriage, shooter crown and folded intake each finish at ~0.533 m.
        # The xRC BigGate collider underside is 0.591 m at this lane.
        compact_envelope_top = 0.533
        roof_bottom = 0.591
        envelope_clearance = roof_bottom - (compact_envelope_top + max_z)
        camera_envelope_top = max(
            float(spec["compact_top_m"])
            for spec in CAMERA_RIG.values()
        )
        camera_clearance = roof_bottom - (camera_envelope_top + max_z)
        dynamic_drive_completed = float(end[1]) > -2.95

        result = {
            "source_sequence": "CloseStorage(0.02) -> PassTrench -> StartStorage(0.99)",
            "compact_position": compact_position,
            "compact_top_m": compact_top,
            "trench_clear_height_m": 22.25 * 0.0254,
            "intake_stowed": intake_stowed,
            "compact_joint_positions": compact_joints,
            "compact_preload_local_xyz": compact_ball_local[
                :XRC_PRELOAD_COUNT
            ].round(4).tolist(),
            "entered_trench": entered,
            "end_position": end.tolist(),
            "max_robot_origin_z_m": max_z,
            "max_tilt_deg": math.degrees(max_tilt),
            "retained_preloads": len(set(range(XRC_PRELOAD_COUNT)) & set(controller.magazine)),
            "feeder_staged": len(controller.feeder_queue),
            "scene_compact_height_m": stats.get("robot_storage_height_m", [None])[0],
            "compact_collision_envelope_top_m": compact_envelope_top,
            "camera_collision_envelope_top_m": camera_envelope_top,
            "xrc_roof_bottom_m": roof_bottom,
            "minimum_envelope_clearance_m": envelope_clearance,
            "minimum_camera_clearance_m": camera_clearance,
            "stable_under_roof": max_tilt < math.radians(5) and max_z < 0.02,
            "dynamic_drive_completed": dynamic_drive_completed,
            "nearest_collision_pairs": nearest_pairs[:24],
            "roof_overlaps": roof_overlaps,
        }
        result["passed"] = bool(
            abs(compact_position - STORAGE_LOWERED_POSITION) < 1e-5
            and abs(compact_joints["vertical_lower"] + 0.208) < 0.025
            and abs(compact_joints["horizontal_retract"] + 0.145) < 0.025
            and abs(compact_joints["shooter_lower"] + 0.140) < 0.025
            and compact_top < result["trench_clear_height_m"]
            and intake_stowed
            and entered
            and dynamic_drive_completed
            and envelope_clearance > 0.02
            and camera_envelope_top <= compact_envelope_top
            and camera_clearance > 0.02
            and max_z < 0.25
            and max_tilt < math.radians(35)
            and result["retained_preloads"] == XRC_PRELOAD_COUNT
        )
        output = PROJECT / "runs" / "robot_trench_validation.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("CompetitionRobot_TRENCH " + json.dumps(result), flush=True)
        sim.stop()
        if not result["passed"]:
            raise RuntimeError("robot compact/trench acceptance failed")
    finally:
        app.close()


if __name__ == "__main__":
    main()
