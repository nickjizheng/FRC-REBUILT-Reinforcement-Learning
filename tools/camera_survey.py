"""Camera-angle survey for vision-based RL on the xRC REBUILT scene.

Builds the full REBUILT field + Legacy Robot robot via SceneBuilder (no physics
stepping), places the robot in a representative neutral-zone shooting pose
(facing the red hub), creates one USD Camera per candidate placement, and
renders 512x512 RGB stills one camera at a time.

Primary path: omni.replicator.core render products + rgb annotator,
ONE render product at a time (TiledCamera hangs on sm_120; multiple
simultaneous render products are avoided for the same reason).
Fallback path (--mode viewport): set_camera_view + capture_viewport_to_file.

Usage (bash):
  cd C:/Users/nickj/Desktop/xrc-rebuilt-robot-rl && \
  OMNI_KIT_ACCEPT_EULA=YES /c/il/venv/Scripts/python.exe tools/camera_survey.py \
      --headless --max-fuel 120 --mode replicator
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
import struct
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RES = (512, 512)

# Robot survey pose: neutral zone, +X_robot (intake/shooter axis) toward the
# red hub at world y=+3.269.  yaw=+90 deg about Z maps robot +X -> world +Y.
ROBOT_POS = (0.0, -1.2, 0.035)
ROBOT_YAW_DEG = 90.0

# Candidate cameras.  frame: "robot" = coords in robot chassis frame
# (+X intake/front, origin on ground under chassis center); "world" = field.
CANDIDATES = [
    {
        "name": "onboard_intake",
        "frame": "robot",
        "eye": (0.35, 0.0, 0.35),
        "target": (2.35, 0.0, -0.075),   # +X, pitched ~12 deg down
        "focal": 12.0,                    # ~82 deg HFOV, wide intake view
        "note": "sees intake mouth + fuel on floor ahead",
    },
    {
        "name": "onboard_shooter",
        "frame": "robot",
        "eye": (-0.35, 0.0, 0.65),
        "target": (3.65, 0.0, 1.0),       # +X, pitched ~5 deg up
        "focal": 14.0,                    # ~74 deg HFOV
        "note": "sees hub opening when robot is aimed",
    },
    {
        "name": "chase",
        "frame": "robot",
        "eye": (-1.6, 0.0, 1.0),
        "target": (2.264, 0.0, -0.035),   # 15 deg down along +X
        "focal": 16.0,
        "note": "third-person: robot + local context",
    },
    {
        "name": "overhead_field",
        "frame": "world",
        "eye": (0.0, 0.0, 14.0),
        "target": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),            # long axis (y) vertical in image
        "focal": 14.0,                    # covers full 16.5 x 8.1 m field
        "note": "god view, full field",
    },
    {
        "name": "hub_pov",
        "frame": "world",
        "eye": (0.0, 3.05, 1.35),         # just field-side of red hub center
        "target": (0.0, -1.2, 0.30),      # looking at the robot / neutral zone
        "focal": 14.0,
        "note": "what the red hub 'sees' of an aiming robot",
    },
]

# Fuel repositioned in front of the robot so the intake camera has targets.
BALL_SPOTS_WORLD = [
    (0.00, -0.20), (0.30, 0.30), (-0.35, 0.50), (0.15, 1.00), (-0.20, 1.40),
    (0.50, 0.90), (-0.55, 0.10), (0.25, 1.80), (-0.10, 2.30), (0.45, -0.50),
]


def robot_local_to_world(p) -> tuple[float, float, float]:
    yaw = math.radians(ROBOT_YAW_DEG)
    c, s = math.cos(yaw), math.sin(yaw)
    x = c * p[0] - s * p[1] + ROBOT_POS[0]
    y = s * p[0] + c * p[1] + ROBOT_POS[1]
    return (x, y, p[2] + ROBOT_POS[2])


def write_png(path: Path, arr: np.ndarray) -> None:
    """Dependency-free RGB8 PNG writer (PIL fallback kept for robustness)."""
    try:
        from PIL import Image

        Image.fromarray(arr, "RGB").save(str(path))
        return
    except Exception:
        pass
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xRC REBUILT camera survey")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-fuel", type=int, default=120)
    parser.add_argument("--mode", choices=("replicator", "viewport"), default="replicator")
    parser.add_argument("--warmup", type=int, default=16, help="app updates per camera before grab")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "cameras")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.argv = [sys.argv[0]]  # keep Kit from eating our flags
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": RES[0], "height": RES[1]})
    print("CAMERA_SURVEY_APP_READY", flush=True)
    saved: dict[str, str] = {}
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        from xrc_rebuilt.isaac_scene import SceneBuilder

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        builder = SceneBuilder(stage, max_fuel=args.max_fuel)
        stats = builder.build()
        print("CAMERA_SURVEY_SCENE", json.dumps(stats), flush=True)

        # --- pose the robot in a representative aiming position -------------
        # Robust to both robot variants (compound rigid body: translate-only
        # ops; articulated: translate+orient ops on the root): wipe the op
        # order and author a single transform. No physics is stepped, so
        # rewriting the articulation root's ops is safe.
        robot_root = stage.GetPrimAtPath("/World/Robot/LegacyRobot")
        if (not robot_root or not robot_root.IsValid()) and builder.robot_prim is not None:
            robot_root = builder.robot_prim
        have_robot = bool(robot_root and robot_root.IsValid())
        if have_robot:
            xform = UsdGeom.Xformable(robot_root)
            xform.ClearXformOpOrder()
            pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), ROBOT_YAW_DEG))
            pose = pose * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*ROBOT_POS))
            xform.AddTransformOp().Set(pose)
            robot_cam_parent = robot_root.GetPath().pathString
        else:
            print("CAMERA_SURVEY_WARN no robot prim; robot-relative cams computed in world", flush=True)
            robot_cam_parent = None

        # --- scatter some fuel in front of the intake ------------------------
        for i, (bx, by) in enumerate(BALL_SPOTS_WORLD):
            if i >= len(builder.ball_prims):
                break
            try:
                prim = builder.ball_prims[-(i + 1)]
                UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(bx, by, 0.076))
            except Exception as error:
                print(f"CAMERA_SURVEY_WARN fuel reposition {i}: {error!r}", flush=True)

        # --- create camera prims ---------------------------------------------
        def define_camera(path: str, eye, target, up, focal: float) -> str:
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(float(focal))
            cam.CreateHorizontalApertureAttr(20.955)
            cam.CreateVerticalApertureAttr(20.955)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
            view = Gf.Matrix4d()
            view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
            UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
            return cam.GetPath().pathString

        cameras: list[tuple[str, str]] = []   # (name, prim path)
        poses_meta = []
        for cand in CANDIDATES:
            up = cand.get("up", (0.0, 0.0, 1.0))
            if cand["frame"] == "robot" and have_robot:
                # parent under the robot xform: local look-at, robot pose applied by USD
                path = define_camera(
                    f"{robot_cam_parent}/Cameras/{cand['name']}",
                    cand["eye"], cand["target"], up, cand["focal"])
                eye_w = robot_local_to_world(cand["eye"])
                target_w = robot_local_to_world(cand["target"])
            else:
                eye_w = robot_local_to_world(cand["eye"]) if cand["frame"] == "robot" else cand["eye"]
                target_w = robot_local_to_world(cand["target"]) if cand["frame"] == "robot" else cand["target"]
                path = define_camera(f"/World/Cameras/{cand['name']}", eye_w, target_w, up, cand["focal"])
            cameras.append((cand["name"], path))
            poses_meta.append({
                "name": cand["name"], "prim": path, "frame": cand["frame"],
                "eye_local": list(cand["eye"]) if cand["frame"] == "robot" else None,
                "eye_world": list(eye_w), "target_world": list(target_w),
                "focal_mm": cand["focal"], "aperture_mm": 20.955,
                "hfov_deg": round(2 * math.degrees(math.atan(20.955 / 2 / cand["focal"])), 1),
                "note": cand["note"],
            })
        (out_dir / "poses.json").write_text(
            json.dumps({"robot_pos": ROBOT_POS, "robot_yaw_deg": ROBOT_YAW_DEG,
                        "resolution": RES, "cameras": poses_meta}, indent=2),
            encoding="utf-8")

        for _ in range(4):   # settle stage/renderer before first capture
            app.update()

        # --- render ----------------------------------------------------------
        if args.mode == "replicator":
            import time

            import omni.replicator.core as rep

            timings: dict[str, float] = {}
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            for name, path in cameras:
                print(f"CAMERA_SURVEY_RENDER {name} {path}", flush=True)
                try:
                    rp = rep.create.render_product(path, RES)
                    rgb.attach([rp])
                    data = None
                    for attempt in range(3):
                        try:
                            rep.orchestrator.step(delta_time=0.0, rt_subframes=8, pause_timeline=True)
                        except Exception as step_error:
                            print(f"CAMERA_SURVEY_STEP_WARN {name} {step_error!r}", flush=True)
                        for _ in range(args.warmup):
                            app.update()
                        data = rgb.get_data()
                        if data is not None and getattr(data, "size", 0) and np.asarray(data).max() > 0:
                            break
                    arr = np.asarray(data)
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        target = out_dir / f"{name}.png"
                        write_png(target, arr[:, :, :3].astype(np.uint8))
                        saved[name] = str(target)
                        print(f"CAMERA_SURVEY_SAVED {name} {target}", flush=True)
                    else:
                        print(f"CAMERA_SURVEY_FAIL {name} bad data shape {arr.shape}", flush=True)
                    # steady-state per-frame cost with this render product attached
                    t0 = time.perf_counter()
                    for _ in range(20):
                        app.update()
                    timings[name] = (time.perf_counter() - t0) / 20 * 1000.0
                    rgb.detach()
                    rp.destroy()
                except Exception as error:  # keep going: one bad camera != dead survey
                    print(f"CAMERA_SURVEY_FAIL {name} {error!r}", flush=True)
            # baseline app.update() cost with no render product attached
            t0 = time.perf_counter()
            for _ in range(20):
                app.update()
            timings["_baseline_no_render_product"] = (time.perf_counter() - t0) / 20 * 1000.0
            print("CAMERA_SURVEY_TIMINGS_MS", json.dumps({k: round(v, 2) for k, v in timings.items()}), flush=True)
            (out_dir / "timings_ms.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        else:
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

            viewport = get_active_viewport()
            try:
                viewport.resolution = RES
            except Exception:
                pass
            for meta in poses_meta:
                name = meta["name"]
                print(f"CAMERA_SURVEY_RENDER {name} (viewport)", flush=True)
                try:
                    set_camera_view(eye=np.array(meta["eye_world"]),
                                    target=np.array(meta["target_world"]),
                                    camera_prim_path="/OmniverseKit_Persp")
                    for _ in range(args.warmup):
                        app.update()
                    target = out_dir / f"{name}.png"
                    capture_viewport_to_file(viewport, str(target))
                    for _ in range(60):
                        app.update()
                        if target.exists() and target.stat().st_size > 1000:
                            break
                    if target.exists():
                        saved[name] = str(target)
                        print(f"CAMERA_SURVEY_SAVED {name} {target}", flush=True)
                    else:
                        print(f"CAMERA_SURVEY_FAIL {name} capture never landed", flush=True)
                except Exception as error:
                    print(f"CAMERA_SURVEY_FAIL {name} {error!r}", flush=True)

        print("CAMERA_SURVEY_DONE", json.dumps(saved), flush=True)
    except BaseException as error:
        import traceback

        print("CAMERA_SURVEY_ERROR", repr(error), flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
