#!/usr/bin/env python3
"""PhysX acceptance for the staged reference intake/container compact program."""
from __future__ import annotations

import json
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preload", type=int, default=8)
    args, _ = parser.parse_known_args()
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
            CAD_HORIZONTAL_RETRACT_M,
            CAD_SHOOTER_LOWER_M,
            CAD_VERTICAL_LOWER_M,
            INTAKE_STOW_ANGLE_DEG,
            ROBOT_ROOT_PATH,
            XRC_PRELOAD_COUNT,
            CompetitionRobotController,
        )

        context = omni.usd.get_context()
        context.new_stage()
        SceneBuilder(context.get_stage(), max_fuel=16, articulated_robot=True).build()
        sim = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        fuel.initialize()
        robot = SingleArticulation(ROBOT_ROOT_PATH)
        robot.initialize()
        controller = CompetitionRobotController()
        controller.initialize(robot)
        controller.preload(fuel, args.preload)

        def run_target(extended: bool) -> dict[str, float]:
            controller.set_storage_extended(extended)
            for _ in range(140):
                controller.step_mechanisms(1 / 30)
                for _ in range(4):
                    sim.step(render=False)
                controller.sync_magazine(fuel)
            positions = np.asarray(robot.get_joint_positions())
            names = list(robot.dof_names)
            return {
                name: float(positions[names.index(name)])
                for name in (
                    "intake_fold", "horizontal_retract", "vertical_lower", "shooter_lower"
                )
            }

        compact = run_target(False)
        applied = robot.get_applied_action()
        retained_compact = len(controller.magazine)
        deployed = run_target(True)
        result = {
            "compact_phase": "COMPACT" if compact else controller.mechanism_phase,
            "compact_joint_positions": compact,
            "deployed_joint_positions": deployed,
            "retained_compact": retained_compact,
            "expected": {
                "intake_fold": math.radians(INTAKE_STOW_ANGLE_DEG),
                "horizontal_retract": -CAD_HORIZONTAL_RETRACT_M,
                "vertical_lower": -CAD_VERTICAL_LOWER_M,
                "shooter_lower": -CAD_SHOOTER_LOWER_M,
            },
            "dof_names": list(robot.dof_names),
            "applied_targets": (
                np.asarray(applied.joint_positions).tolist()
                if applied is not None and applied.joint_positions is not None
                else None
            ),
        }
        result["passed"] = bool(
            abs(compact["intake_fold"] - math.radians(INTAKE_STOW_ANGLE_DEG)) < 0.08
            and abs(compact["horizontal_retract"] + CAD_HORIZONTAL_RETRACT_M) < 0.025
            and abs(compact["vertical_lower"] + CAD_VERTICAL_LOWER_M) < 0.025
            and abs(compact["shooter_lower"] + CAD_SHOOTER_LOWER_M) < 0.025
            and all(abs(value) < 0.025 for value in deployed.values())
            and retained_compact == args.preload
        )
        out = PROJECT / "runs/robot_compact_validation.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("ROBOT_COMPACT " + json.dumps(result), flush=True)
        sim.stop()
        if not result["passed"]:
            raise RuntimeError("reference compact mechanism acceptance failed")
    finally:
        app.close()


if __name__ == "__main__":
    main()
