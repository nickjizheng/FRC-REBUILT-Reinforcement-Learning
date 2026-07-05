"""Render and benchmark the real chassis-mounted onboard camera rig.

The robot builder authors three physical cameras and conservative protected
housing colliders.  This tool validates the irreducible two-camera baseline
(intake + rear shooter) and, with ``--include-navigation``, the optional third
navigation camera under identical conditions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FUEL_DIAMETER_M = 0.152
POLICY_ACTION_REPEAT = 6
CAMERA_READY_MAX_FRAMES = 90
CAMERA_READY_STD_THRESHOLD = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview and benchmark the chassis-mounted RL camera rig"
    )
    parser.add_argument(
        "--two-camera",
        action="store_true",
        help="run the legacy intake+shooter ablation instead of the 3-view baseline",
    )
    parser.add_argument("--max-fuel", type=int, default=456)
    parser.add_argument("--benchmark-transitions", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs" / "camera_preview",
    )
    return parser.parse_args()


def yaw_quaternion(degrees: float) -> np.ndarray:
    half = math.radians(degrees) * 0.5
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], np.float32)


def save_png(path: Path, rgba: np.ndarray) -> np.ndarray:
    from PIL import Image

    image = np.asarray(rgba)
    if image.dtype != np.uint8:
        scale = 255.0 if image.size and float(image.max()) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    rgb = image[..., :3]
    Image.fromarray(rgb).save(path)
    return rgb


def image_metrics(rgb: np.ndarray) -> dict[str, float | int]:
    """Cheap validity and yellow-FUEL visibility metrics."""

    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.asarray([18, 75, 75], np.uint8)
    upper = np.asarray([42, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [
        row for row in stats[1:] if int(row[cv2.CC_STAT_AREA]) >= 4
    ]
    diameters = [
        float(max(row[cv2.CC_STAT_WIDTH], row[cv2.CC_STAT_HEIGHT]))
        for row in components
    ]
    return {
        "rgb_std": round(float(rgb.std()), 3),
        "luminance_mean": round(float(rgb.mean()), 3),
        "unique_rgb_sampled": int(
            len(np.unique(rgb[::4, ::4].reshape(-1, 3), axis=0))
        ),
        "yellow_component_count": len(components),
        "yellow_max_diameter_px": round(max(diameters, default=0.0), 2),
        "yellow_mask_fraction": round(float(np.count_nonzero(mask) / mask.size), 6),
    }


def expected_ball_pixels(width: int, hfov_deg: float) -> dict[str, float]:
    result = {}
    for distance in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0):
        angle = 2.0 * math.atan(FUEL_DIAMETER_M / (2.0 * distance))
        result[f"{distance:g}m"] = round(
            width * angle / math.radians(hfov_deg), 2
        )
    return result


def gpu_memory_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(output.splitlines()[0].strip())
    except Exception:
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, np.float64), q))


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    from xrc_rebuilt.competition_robot import (
        BLUE_TRENCH_START_TRANSLATION,
        BLUE_TRENCH_START_YAW_DEG,
        CAMERA_BASELINE_NAMES,
        CAMERA_TWO_VIEW_NAMES,
        CAMERA_PRIM_PATHS,
        CAMERA_RESOLUTION,
        CAMERA_RIG,
        CompetitionRobotController,
        ROBOT_ROOT_PATH,
    )

    camera_names = CAMERA_TWO_VIEW_NAMES if args.two_camera else CAMERA_BASELINE_NAMES
    rig_name = "two_camera_ablation" if args.two_camera else "baseline_3cam"
    output_dir = args.out / rig_name
    output_dir.mkdir(parents=True, exist_ok=True)

    app = SimulationApp(
        {
            "headless": True,
            "width": CAMERA_RESOLUTION[0],
            "height": CAMERA_RESOLUTION[1],
            "multi_gpu": False,
        }
    )
    manifest: dict[str, object] = {
        "rig": rig_name,
        "camera_names": list(camera_names),
        "resolution": list(CAMERA_RESOLUTION),
        "rate_hz": 10,
        "max_fuel": args.max_fuel,
        "camera_specs": {
            name: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in CAMERA_RIG[name].items()
            }
            for name in camera_names
        },
        "captures": [],
        "performance": {},
    }
    try:
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.sensors.camera import Camera

        from xrc_rebuilt.isaac_scene import SceneBuilder

        usd_context = omni.usd.get_context()
        usd_context.new_stage()
        stats = SceneBuilder(
            usd_context.get_stage(),
            max_fuel=args.max_fuel,
            articulated_robot=True,
        ).build()
        sim = SimulationContext(
            physics_dt=1 / 60,
            rendering_dt=1 / 60,
            stage_units_in_meters=1.0,
        )
        sim.reset()

        robot = SingleArticulation(ROBOT_ROOT_PATH)
        robot.initialize()
        fuel = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        fuel.initialize()
        controller = CompetitionRobotController()
        controller.initialize(robot)

        cameras: dict[str, Camera] = {}
        for name in camera_names:
            camera = Camera(
                prim_path=CAMERA_PRIM_PATHS[name],
                resolution=CAMERA_RESOLUTION,
            )
            camera.initialize()
            cameras[name] = camera

        def wait_for_camera_frames(context: str) -> dict[str, np.ndarray]:
            """Wait for every RTX render product to publish a real frame.

            Isaac can return allocated, correctly shaped, but all-black buffers
            while several render products are starting.  A fixed eight-frame
            delay happened to work for two cameras and raced with the optional
            third camera.  Gate captures on content instead so benchmark output
            can never silently contain startup frames.
            """

            latest: dict[str, np.ndarray] = {}
            for frame_index in range(1, CAMERA_READY_MAX_FRAMES + 1):
                sim.step(render=True)
                latest = {
                    name: np.asarray(camera.get_rgba()).copy()
                    for name, camera in cameras.items()
                }
                ready = all(
                    image.size > 0
                    and image.ndim == 3
                    and image.shape[-1] >= 3
                    and float(image[..., :3].std()) > CAMERA_READY_STD_THRESHOLD
                    for image in latest.values()
                )
                if ready:
                    if frame_index > 1:
                        print(
                            f"CAMERA_READY context={context} frames={frame_index}",
                            flush=True,
                        )
                    return latest
            diagnostics = {
                name: {
                    "shape": list(image.shape),
                    "rgb_std": (
                        round(float(image[..., :3].std()), 3)
                        if image.size and image.ndim == 3
                        else None
                    ),
                }
                for name, image in latest.items()
            }
            raise RuntimeError(
                f"camera render products did not become ready for {context}: "
                f"{json.dumps(diagnostics)}"
            )

        # Force all render products through their asynchronous startup before
        # any articulation motion or benchmark timing.
        wait_for_camera_frames("initialization")

        def set_robot_pose(position: tuple[float, float, float], yaw_deg: float) -> None:
            robot.set_world_pose(
                position=np.asarray(position, np.float32),
                orientation=yaw_quaternion(yaw_deg),
            )
            robot.set_linear_velocity(np.zeros(3, np.float32))
            robot.set_angular_velocity(np.zeros(3, np.float32))
            for _ in range(5):
                controller.drive(0.0, 0.0, 0.0)
                sim.step(render=False)

        def set_compact(compact: bool) -> None:
            controller.set_storage_extended(not compact)
            target_phase = "COMPACT" if compact else "DEPLOYED"
            for _ in range(180):
                controller.step_mechanisms(1 / 60)
                controller.drive(0.0, 0.0, 0.0)
                sim.step(render=False)
                if controller.mechanism_phase == target_phase:
                    # Give the physical joints time to settle onto their targets.
                    for _ in range(20):
                        controller.drive(0.0, 0.0, 0.0)
                        sim.step(render=False)
                    break

        def place_calibration_balls() -> None:
            positions: list[list[float]] = []
            # Robot is placed at the origin with yaw=0 for this capture.
            for distance in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0):
                positions.append([distance, -0.06, 0.077])
                positions.append([-distance, 0.06, 0.077])
            count = min(len(positions), args.max_fuel)
            indices = np.arange(count, dtype=np.int32)
            fuel.set_world_poses(
                positions=np.asarray(positions[:count], np.float32),
                indices=indices,
            )
            fuel.set_linear_velocities(
                np.zeros((count, 3), np.float32), indices=indices
            )

        def capture(label: str) -> None:
            # Let RTX and camera annotators update at the new articulation pose,
            # then reject asynchronous startup/invalid buffers.
            for _ in range(7):
                sim.step(render=True)
            frames = wait_for_camera_frames(label)
            for name, camera in cameras.items():
                rgba = frames[name]
                if rgba.size == 0:
                    raise RuntimeError(f"empty camera image: {label}/{name}")
                target = output_dir / f"{label}__{name}.png"
                rgb = save_png(target, rgba)
                spec = CAMERA_RIG[name]
                entry = {
                    "scenario": label,
                    "camera": name,
                    "path": str(target),
                    "shape": list(rgb.shape),
                    "metrics": image_metrics(rgb),
                    "expected_ball_diameter_px": expected_ball_pixels(
                        CAMERA_RESOLUTION[0], float(spec["hfov_deg"])
                    ),
                }
                manifest["captures"].append(entry)
                print(
                    "CAMERA_CAPTURE "
                    + json.dumps(
                        {
                            "scenario": label,
                            "camera": name,
                            **entry["metrics"],
                        }
                    ),
                    flush=True,
                )

        # Deployed field views.
        set_compact(False)
        set_robot_pose((0.0, 0.0, 0.01), 0.0)
        place_calibration_balls()
        capture("calibration_deployed")

        # Formal match spawn: intake looks out to neutral, rear camera looks at
        # the alliance HUB through the trench.
        set_robot_pose(BLUE_TRENCH_START_TRANSLATION, BLUE_TRENCH_START_YAW_DEG)
        capture("spawn_deployed")

        # Representative neutral-zone scoring pose.  Chassis +X points north,
        # hence rear -X points toward the blue HUB.
        set_robot_pose((0.0, -1.2, 0.01), 90.0)
        capture("hub_range_deployed")

        # Compact under the real trench roof validates camera self-occlusion and
        # the physical camera envelope at the accepted match start.
        set_robot_pose(BLUE_TRENCH_START_TRANSLATION, BLUE_TRENCH_START_YAW_DEG)
        set_compact(True)
        capture("trench_compact")

        # Ten-Hz camera loop: five physics-only frames followed by one rendered
        # frame.  This reports the actual policy-transition wall time.
        no_render_ms: list[float] = []
        render_ms: list[float] = []
        transition_ms: list[float] = []
        for _ in range(max(1, args.benchmark_transitions)):
            transition_start = time.perf_counter()
            for _ in range(POLICY_ACTION_REPEAT - 1):
                start = time.perf_counter()
                controller.drive(0.0, 0.0, 0.0)
                sim.step(render=False)
                no_render_ms.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            controller.drive(0.0, 0.0, 0.0)
            sim.step(render=True)
            render_ms.append((time.perf_counter() - start) * 1000.0)
            transition_ms.append((time.perf_counter() - transition_start) * 1000.0)

        performance = {
            "policy_transitions": len(transition_ms),
            "policy_transitions_per_s": round(
                1000.0 / statistics.mean(transition_ms), 3
            ),
            "physics_steps_per_s": round(
                POLICY_ACTION_REPEAT
                * 1000.0
                / statistics.mean(transition_ms),
                3,
            ),
            "transition_ms_mean": round(statistics.mean(transition_ms), 3),
            "transition_ms_p50": round(percentile(transition_ms, 50), 3),
            "transition_ms_p95": round(percentile(transition_ms, 95), 3),
            "render_step_ms_mean": round(statistics.mean(render_ms), 3),
            "render_step_ms_p95": round(percentile(render_ms, 95), 3),
            "physics_only_step_ms_mean": round(statistics.mean(no_render_ms), 3),
            "gpu_memory_used_mb": gpu_memory_used_mb(),
        }
        manifest["performance"] = performance
        manifest["scene_stats"] = stats
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("CAMERA_RIG_RESULT " + json.dumps(performance), flush=True)
        sim.stop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
