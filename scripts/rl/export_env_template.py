"""Milestone 1: export the procedurally-built scene as a reusable USD asset.

Isaac Lab clones env templates from USD; our scene is built procedurally by
SceneBuilder.  This bridges the two: build once, export to USD, and verify the
exported layer reloads with the robot articulation, FUEL, colliders, and camera
rig intact.  The exported asset is the input for the vectorized DirectRLEnv
(see docs/VECTORIZATION_PLAN.md).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "assets" / "rl" / "env_template.usd"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-fuel", type=int, default=456)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    report: dict[str, object] = {"out": str(args.out), "max_fuel": args.max_fuel}
    try:
        import omni.usd

        from xrc_rebuilt.isaac_scene import SceneBuilder

        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        stats = SceneBuilder(stage, max_fuel=args.max_fuel, articulated_robot=True).build()
        report["build_stats"] = {
            k: stats[k]
            for k in ("fuel_bodies", "robot_visual_triangles", "robot_chassis_colliders")
            if k in stats
        }

        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Strip the singleton PhysicsScene + lighting so the template is pure
        # per-env content, and set /World as the default prim so it can be
        # referenced/cloned under each env root.  The vectorized benchmark
        # supplies one shared PhysicsScene + dome light for all clones.
        for _singleton in ("/World/PhysicsScene", "/World/Lights"):
            if stage.GetPrimAtPath(_singleton):
                stage.RemovePrim(_singleton)
        _world = stage.GetPrimAtPath("/World")
        if _world:
            stage.SetDefaultPrim(_world)
        stage.GetRootLayer().Export(str(args.out))
        report["export_bytes"] = args.out.stat().st_size
        print(f"EXPORT_OK {args.out} ({report['export_bytes']:,} bytes)", flush=True)

        # ---- verify: open the exported layer on a fresh stage and inventory it
        from pxr import Usd, UsdGeom, UsdPhysics

        vstage = Usd.Stage.Open(str(args.out))
        prims = list(vstage.Traverse())
        articulations = [
            p.GetPath().pathString
            for p in prims
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        cameras = [
            p.GetPath().pathString for p in prims if p.IsA(UsdGeom.Camera)
        ]
        fuel = [
            p for p in prims
            if p.GetName().startswith("Fuel_") and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        report["verify"] = {
            "total_prims": len(prims),
            "articulation_roots": articulations,
            "camera_count": len(cameras),
            "camera_paths": cameras,
            "fuel_rigid_bodies": len(fuel),
        }
        ok = bool(articulations) and len(cameras) >= 2 and len(fuel) >= 1
        report["verify"]["reload_ok"] = ok
        print("VERIFY " + json.dumps(report["verify"], indent=2), flush=True)
        print("EXPORT_TEMPLATE_DONE " + ("OK" if ok else "INCOMPLETE"), flush=True)

        (args.out.parent / "env_template_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
