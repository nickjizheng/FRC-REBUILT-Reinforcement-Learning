"""Milestone 2/3: vectorized physics throughput.

Clones the exported env template (``assets/rl/env_template.usd``) with
``GridCloner`` across N envs on ONE shared GPU PhysicsScene, then measures
aggregate physics-steps/s, env-steps/s, and policy-transitions/s (action-repeat
6) plus VRAM.  This is the go/no-go for the >=8 policy-tx/s throughput gate
(see docs/VECTORIZATION_PLAN.md).  Physics only for now; tiled cameras next.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TEMPLATE = PROJECT_ROOT / "assets" / "rl" / "env_template.usd"
ACTION_REPEAT = 6


def vram_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--spacing", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    args = ap.parse_args()
    template_path = args.template

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    report: dict[str, object] = {
        "num_envs": args.num_envs,
        "spacing": args.spacing,
        "steps": args.steps,
    }
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.prims import Articulation, RigidPrim

        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        # one shared PhysicsScene (GPU dynamics, matching the single-env sim)
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateEnableGPUDynamicsAttr(True)
        physx.CreateBroadphaseTypeAttr("GPU")
        physx.CreateTimeStepsPerSecondAttr(60)
        physx.CreateGpuFoundLostPairsCapacityAttr(2097152)
        physx.CreateGpuTotalAggregatePairsCapacityAttr(2097152)
        physx.CreateGpuMaxRigidContactCountAttr(2097152)
        physx.CreateGpuMaxRigidPatchCountAttr(327680)

        UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(800)

        # reference the template onto env_0, then grid-clone to N envs
        envs_root = "/World/envs"
        UsdGeom.Xform.Define(stage, envs_root)
        env0 = f"{envs_root}/env_0"
        UsdGeom.Xform.Define(stage, env0)
        stage.GetPrimAtPath(env0).GetReferences().AddReference(str(template_path))
        print("REF_ADDED", flush=True)

        cloner = GridCloner(spacing=args.spacing)
        cloner.define_base_env(envs_root)
        target_paths = cloner.generate_paths(f"{envs_root}/env", args.num_envs)
        cloner.clone(
            source_prim_path=env0,
            prim_paths=target_paths,
            replicate_physics=True,
            base_env_path=envs_root,
        )
        print(f"CLONED n={args.num_envs}", flush=True)

        # inventory the cloned structure so we validate paths without a sim
        art_roots = [
            p.GetPath().pathString
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        fuel_prims = [
            p.GetPath().pathString
            for p in stage.Traverse()
            if p.GetName().startswith("Fuel_") and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        print(f"STRUCT art_roots={len(art_roots)} first={art_roots[:2]}", flush=True)
        print(f"STRUCT fuel={len(fuel_prims)} first={fuel_prims[:1]}", flush=True)
        report["stage_articulation_roots"] = art_roots[:4]
        report["stage_fuel_bodies"] = len(fuel_prims)

        import traceback

        robot_expr = (
            art_roots[0].replace("/env_0/", "/env_.*/")
            if art_roots
            else f"{envs_root}/env_.*/Robot/FRC_8011"
        )
        fuel_expr = (
            fuel_prims[0].rsplit("/", 1)[0].replace("/env_0/", "/env_.*/") + "/Fuel_.*"
            if fuel_prims
            else f"{envs_root}/env_.*/Fuel/Fuel_.*"
        )
        print(f"EXPR robot='{robot_expr}' fuel='{fuel_expr}'", flush=True)
        try:
            sim = SimulationContext(
                physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.0
            )
            sim.reset()
            print("SIM_RESET_OK", flush=True)
            robots = Articulation(robot_expr)
            robots.initialize()
            fuel = RigidPrim(fuel_expr)
            fuel.initialize()
            report["robot_count"] = int(robots.count)
            report["fuel_count"] = int(fuel.count)
            print(f"VIEWS robots={robots.count} fuel={fuel.count}", flush=True)
        except Exception:
            print("BENCH_ERROR\n" + traceback.format_exc(), flush=True)
            raise

        for _ in range(30):  # warmup / settle
            sim.step(render=False)
        report["vram_mb_after_warmup"] = vram_mb()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            sim.step(render=False)
        dt = time.perf_counter() - t0
        steps_per_s = args.steps / dt
        env_steps_per_s = steps_per_s * args.num_envs
        policy_tx = env_steps_per_s / ACTION_REPEAT
        report.update(
            {
                "sim_steps_per_s": round(steps_per_s, 2),
                "aggregate_env_steps_per_s": round(env_steps_per_s, 2),
                "aggregate_policy_tx_per_s": round(policy_tx, 2),
                "gate_8_tx_per_s_cleared": bool(policy_tx >= 8.0),
                "vram_mb_peak": vram_mb(),
            }
        )
        print("VEC_THROUGHPUT " + json.dumps(report), flush=True)
        out = PROJECT_ROOT / "runs" / f"vec_throughput_n{args.num_envs}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
