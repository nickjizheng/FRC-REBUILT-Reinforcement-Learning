"""Timing audit probe: verify SimulationContext.step() semantics empirically.

Run: OMNI_KIT_ACCEPT_EULA=YES C:/il/venv/Scripts/python.exe tools/timing_probe.py
Mimics isaac_scene.py: physics scene with TimeStepsPerSecond=250 built BEFORE
SimulationContext(physics_dt=0.004, rendering_dt=1/60), then measures how much
simulated time each step(render=False) and step(render=True) advances.
step(render=True) -> _app.update() behaves identically headless (same loop
runner / physx update path), only the render workload differs.
"""
import json
import sys

sys.argv = [sys.argv[0]]
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
report = {}
try:
    import carb
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Same physics scene the SceneBuilder creates (isaac_scene.py build_physics_scene)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    from pxr import Gf
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateEnableCCDAttr(True)
    physx.CreateEnableEnhancedDeterminismAttr(True)
    physx.CreateTimeStepsPerSecondAttr(250)

    # One rigid body so the scene actually simulates something
    ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
    ball.CreateRadiusAttr(0.076)
    ball.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2.0))
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
    UsdPhysics.MassAPI.Apply(ball.GetPrim()).CreateMassAttr(0.08)

    sim = SimulationContext(physics_dt=0.004, rendering_dt=1 / 60, stage_units_in_meters=1.0)
    sim.reset()

    settings = carb.settings.get_settings()
    report["after_reset"] = {
        "current_time": sim.current_time,
        "step_index": sim.current_time_step_index,
    }
    report["config"] = {
        "get_physics_dt": sim.get_physics_dt(),
        "get_rendering_dt": sim.get_rendering_dt(),
        "TimeStepsPerSecond_attr": physx.GetTimeStepsPerSecondAttr().Get(),
        "minFrameRate_persistent": settings.get("/persistent/simulation/minFrameRate"),
        "minFrameRate_nonpersistent": settings.get("persistent/simulation/minFrameRate"),
        "rateLimitEnabled": settings.get("/app/runLoops/main/rateLimitEnabled"),
        "rateLimitFrequency": settings.get("/app/runLoops/main/rateLimitFrequency"),
    }
    try:
        import omni.kit.loop._loop as omni_loop
        loop = omni_loop.acquire_loop_interface()
        report["config"]["loop_manual_mode"] = loop.get_manual_mode()
        report["config"]["loop_manual_step_size"] = loop.get_manual_step_size()
    except Exception as err:  # pragma: no cover
        report["config"]["loop_error"] = repr(err)

    # Experiment 1: 250 physics-only steps -> expect exactly 1.000 s / 250 steps
    t0, n0 = sim.current_time, sim.current_time_step_index
    for _ in range(250):
        sim.step(render=False)
    report["exp1_250x_render_false"] = {
        "delta_time": sim.current_time - t0,
        "delta_steps": sim.current_time_step_index - n0,
    }

    # Experiment 2: 60 render steps (the GUI path: _app.update())
    t0, n0 = sim.current_time, sim.current_time_step_index
    per_call = []
    for _ in range(60):
        a, b = sim.current_time, sim.current_time_step_index
        sim.step(render=True)
        per_call.append((round(sim.current_time - a, 9), sim.current_time_step_index - b))
    report["exp2_60x_render_true"] = {
        "delta_time": sim.current_time - t0,
        "delta_steps": sim.current_time_step_index - n0,
        "first_10_calls_(dt,steps)": per_call[:10],
        "steps_histogram": {str(k): sum(1 for _, s in per_call if s == k) for k in sorted({s for _, s in per_call})},
    }

    # Experiment 3: the exact GUI-mode loop pattern from isaac_scene.py main():
    # frame % 4 == 0 -> render=True, else render=False, 400 iterations.
    t0, n0 = sim.current_time, sim.current_time_step_index
    for frame in range(400):
        sim.step(render=(frame % 4 == 0))
    report["exp3_gui_pattern_400_iters"] = {
        "assumed_clock_frame_x_0.004": 400 * 0.004,
        "actual_delta_time": sim.current_time - t0,
        "actual_delta_steps": sim.current_time_step_index - n0,
        "skew_factor": (sim.current_time - t0) / (400 * 0.004),
    }

    # Experiment 4: headless pattern drift check: frame*0.004 vs current_time
    sim.stop()
    sim.play()
    t0 = sim.current_time
    for frame in range(1, 2501):
        sim.step(render=False)
    report["exp4_headless_2500_frames"] = {
        "frame_clock": 2500 * 0.004,
        "current_time_minus_t0": sim.current_time - t0,
        "current_time_raw": sim.current_time,
        "abs_drift": abs((sim.current_time - t0) - 2500 * 0.004),
    }

    print("TIMING_PROBE_RESULT " + json.dumps(report, indent=2), flush=True)
except BaseException as error:  # pragma: no cover
    import traceback
    print("TIMING_PROBE_ERROR", repr(error), flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
