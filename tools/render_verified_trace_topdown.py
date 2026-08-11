"""Pass 2: render a verified state trace through one native-HD field camera.

This pass creates the exact scene with the three policy cameras retained.  It never calls
``env.step`` or ``sim.step`` during playback: every captured root, joint, FUEL,
and mechanism visual state is restored directly. Articulation kinematics are
recomputed without integration, every rigid pose is written directly to
render-visible Fabric, and the result is captured by a zero-delta Replicator
render.  By default the full-field camera is elevated 18 degrees off
vertical, which preserves the legibility of a top view while revealing the 3D
field geometry.  The approved 1920x1080 GUI overlay, left scoreboard, and three
robot-camera panes are recreated from recorded telemetry and restored scene
state, so the displayed score begins at the actual zero.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from render_verified_topdown_replay import (  # noqa: E402
    CAMERA_SIZE as GUI_CAMERA_SIZE,
    SIDEBAR_WIDTH as GUI_SIDEBAR_WIDTH,
    VideoSink as GuiVideoSink,
    build_exact_cfg,
    load_gui_camera_state,
    load_replay_bundle,
    require_file_sha,
    sha256_file,
    verify_code_snapshot,
)
from verified_topdown_trace import (  # noqa: E402
    CAMERA_SIZE as LEGACY_CAMERA_SIZE,
    CODEC,
    FPS,
    RENDER_PROVENANCE_SCHEMA,
    SIDEBAR_WIDTH as LEGACY_SIDEBAR_WIDTH,
    STEPS,
    VerifiedTrace,
    atomic_json,
    frame_telemetry,
    load_verified_trace,
)


# The trace format is independent of output pixels.  Render winners in the
# exact 1920x1080 GUI composition approved for the public video; keep the
# legacy constants imported above only for backwards-compatible tests and
# provenance interpretation.
CAMERA_SIZE = GUI_CAMERA_SIZE
SIDEBAR_WIDTH = GUI_SIDEBAR_WIDTH


CAMERA_HORIZONTAL_APERTURE_MM = 20.955
DEFAULT_CAMERA_HEIGHT_M = 14.0
DEFAULT_CAMERA_TILT_DEG = 18.0
DEFAULT_CAMERA_AZIMUTH_DEG = -90.0
DEFAULT_CAMERA_FOCAL_LENGTH_MM = 14.0
DEFAULT_FIELD_FRAME_MARGIN_M = 0.5
FIELD_LENGTH_M = 16.54048
FIELD_WIDTH_M = 8.06958

# Presentation-only intake view.  The policy's physical intake camera is
# mounted at z=0.055 m inside the folding CAD assembly; that is a valid policy
# sensor but the mechanism can occlude nearly the entire public video pane.
# This additional camera remains a rigid child of the chassis and looks over
# the actual intake mouth.  It is never used by the policy or simulator.
PRESENTATION_INTAKE_EYE_LOCAL = (0.35, 0.0, 0.35)
PRESENTATION_INTAKE_TARGET_LOCAL = (2.35, 0.0, -0.075)
PRESENTATION_INTAKE_UP_LOCAL = (0.0, 0.0, 1.0)
PRESENTATION_INTAKE_FOCAL_LENGTH_MM = 12.0
PRESENTATION_INTAKE_RESOLUTION = (640, 360)


def source_bundle_identity(trace_metadata: dict[str, Any]) -> dict[str, int]:
    """Resolve immutable source-bundle keys without relabelling seed searches."""

    classification = trace_metadata.get("classification")
    seed_search = isinstance(classification, dict) and bool(
        classification.get("seed_search", False)
    )
    if not seed_search:
        return {
            "env_seed": int(trace_metadata["env_seed"]),
            "action_seed": int(trace_metadata["action_seed"]),
            "env_index": int(trace_metadata["env_index"]),
        }
    source = trace_metadata.get("source_capture")
    if not isinstance(source, dict):
        raise ValueError("seed-search trace is missing immutable source-capture keys")
    return {
        "env_seed": int(source["target_env_seed"]),
        "action_seed": int(source["target_action_seed"]),
        "env_index": int(source["target_env_index"]),
    }


def run_episode_contract(
    source_episode: dict[str, Any], trace_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Apply fresh run seeds to source metadata while preserving its contract."""

    episode = dict(source_episode)
    episode["env_seed"] = int(trace_metadata["env_seed"])
    episode["action_seed"] = int(trace_metadata["action_seed"])
    classification = trace_metadata.get("classification")
    compatible_checkpoint = isinstance(classification, dict) and bool(
        classification.get("compatible_checkpoint", False)
    )
    if compatible_checkpoint:
        policy_contract = trace_metadata.get("policy_contract")
        if not isinstance(policy_contract, dict) or not isinstance(
            policy_contract.get("checkpoint_stagec_v2_metadata"), dict
        ):
            raise ValueError(
                "compatible-checkpoint trace is missing its embedded policy contract"
            )
        episode["checkpoint_sha256"] = str(trace_metadata["checkpoint_sha256"])
        episode["stagec_v2_metadata"] = policy_contract[
            "checkpoint_stagec_v2_metadata"
        ]
    return episode


def oblique_camera_layout(
    origin: np.ndarray,
    *,
    height_m: float,
    tilt_deg: float,
    azimuth_deg: float,
) -> dict[str, np.ndarray | float]:
    """Return an elevated camera pose, with tilt measured away from vertical.

    Azimuth describes the direction from the field centre to the camera eye,
    measured counter-clockwise from field +X.  The default ``-90`` therefore
    views the field from the near (-Y) sideline without rolling its long axis.
    """

    values = (float(height_m), float(tilt_deg), float(azimuth_deg))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera height, tilt, and azimuth must be finite")
    if height_m <= 0.0:
        raise ValueError("camera height must be positive")
    if not 0.0 <= tilt_deg <= 40.0:
        raise ValueError("camera tilt must be between 0 and 40 degrees off vertical")
    origin = np.asarray(origin, np.float64)
    if origin.shape != (3,):
        raise ValueError(f"camera origin must have shape (3,), got {origin.shape}")
    horizontal_offset = float(height_m) * math.tan(math.radians(float(tilt_deg)))
    azimuth = math.radians(float(azimuth_deg))
    eye = origin + np.asarray(
        [
            horizontal_offset * math.cos(azimuth),
            horizontal_offset * math.sin(azimuth),
            float(height_m),
        ],
        np.float64,
    )
    return {
        "eye": eye,
        "target": origin.copy(),
        "up": np.asarray([0.0, 1.0, 0.0], np.float64),
        "height_m": float(height_m),
        "tilt_deg": float(tilt_deg),
        "azimuth_deg": float(azimuth_deg),
        "horizontal_offset_m": horizontal_offset,
    }


def verify_full_field_framing(
    layout: dict[str, np.ndarray | float],
    *,
    focal_length_mm: float,
    field_margin_m: float,
) -> dict[str, float | bool]:
    """Fail closed unless the regulation field plus margin fits in frame."""

    focal_length_mm = float(focal_length_mm)
    field_margin_m = float(field_margin_m)
    if not math.isfinite(focal_length_mm) or focal_length_mm <= 0.0:
        raise ValueError("camera focal length must be positive and finite")
    if not math.isfinite(field_margin_m) or not 0.0 <= field_margin_m <= 5.0:
        raise ValueError("field frame margin must be finite and between 0 and 5 metres")
    eye = np.asarray(layout["eye"], np.float64)
    target = np.asarray(layout["target"], np.float64)
    up = np.asarray(layout["up"], np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        raise ValueError("camera up vector is parallel to its view direction")
    right /= right_norm
    screen_up = np.cross(right, forward)
    half_length = FIELD_LENGTH_M / 2.0 + field_margin_m
    half_width = FIELD_WIDTH_M / 2.0 + field_margin_m
    half_horizontal_aperture = CAMERA_HORIZONTAL_APERTURE_MM / 2.0
    half_vertical_aperture = (
        CAMERA_HORIZONTAL_APERTURE_MM * CAMERA_SIZE[1] / CAMERA_SIZE[0] / 2.0
    )
    max_horizontal = 0.0
    max_vertical = 0.0
    min_depth = math.inf
    for x in (-half_length, half_length):
        for y in (-half_width, half_width):
            point = target + np.asarray([x, y, 0.0], np.float64)
            relative = point - eye
            depth = float(np.dot(relative, forward))
            if depth <= 0.0:
                raise ValueError("camera places a field corner behind the image plane")
            min_depth = min(min_depth, depth)
            film_x = abs(focal_length_mm * float(np.dot(relative, right)) / depth)
            film_y = abs(focal_length_mm * float(np.dot(relative, screen_up)) / depth)
            max_horizontal = max(max_horizontal, film_x / half_horizontal_aperture)
            max_vertical = max(max_vertical, film_y / half_vertical_aperture)
    maximum = max(max_horizontal, max_vertical)
    if maximum > 1.0:
        raise ValueError(
            "camera configuration crops the regulation field: "
            f"horizontal={max_horizontal:.3f}, vertical={max_vertical:.3f}"
        )
    return {
        "validated": True,
        "field_margin_m": field_margin_m,
        "horizontal_frame_utilization": max_horizontal,
        "vertical_frame_utilization": max_vertical,
        "maximum_frame_utilization": maximum,
        "minimum_corner_depth_m": min_depth,
    }


class VideoSink:
    """High-bitrate atomic MJPG sink with a live-score sidebar."""

    def __init__(self, frame_source: Any, output: Path):
        import cv2

        self.cv2 = cv2
        self.frame_source = frame_source
        self.output = Path(output)
        self.temp = self.output.with_name(f".{self.output.stem}.{os.getpid()}.partial.avi")
        self.first_frame = self.output.with_suffix(".first-frame.png")
        self.first_frame_temp = self.first_frame.with_name(
            f".{self.first_frame.name}.{os.getpid()}.partial.png"
        )
        self.temp.unlink(missing_ok=True)
        self.first_frame_temp.unlink(missing_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.temp),
            cv2.VideoWriter_fourcc(*CODEC),
            FPS,
            (CAMERA_SIZE[0] + SIDEBAR_WIDTH, CAMERA_SIZE[1]),
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"cannot open MJPG intermediate {self.temp}")
        quality = getattr(cv2, "VIDEOWRITER_PROP_QUALITY", None)
        if quality is not None:
            self.writer.set(quality, 100)
        self.frames = 0

    @staticmethod
    def _clock(seconds: float) -> str:
        ticks = max(0, int(round(float(seconds) * 10.0)))
        return f"{ticks // 600:02d}:{(ticks // 10) % 60:02d}.{ticks % 10}"

    def compose(self, rgba: np.ndarray, data: dict[str, Any]) -> np.ndarray:
        cv2 = self.cv2
        width, height = CAMERA_SIZE
        rgba = np.asarray(rgba)
        if rgba.shape not in ((height, width, 3), (height, width, 4)):
            raise RuntimeError(f"native field camera shape changed: {rgba.shape}")
        if rgba.dtype != np.dtype("uint8"):
            raise RuntimeError(f"native field camera dtype changed: {rgba.dtype}")
        rgb = np.ascontiguousarray(rgba[..., :3])
        if float(rgb.std()) <= 1.0:
            raise RuntimeError("native field camera returned a black frame")
        canvas = np.zeros((height, width + SIDEBAR_WIDTH, 3), np.uint8)
        canvas[:, :width] = rgb[..., ::-1]
        canvas[:, width:] = (20, 24, 31)
        x = width + 28
        cv2.putText(
            canvas,
            "FROZEN POLICY / LIVE MATCH",
            (x, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (230, 235, 241),
            2,
            cv2.LINE_AA,
        )
        rows = (
            ("MATCH", self._clock(data["remaining_s"])),
            ("SCORE", data["score"]),
            ("COLLECTED", data["collected"]),
            ("MAGAZINE", data["magazine"]),
            ("PHASE", data["phase"]),
            ("BLUE HUB", data["hub"]),
            ("CYCLES", data["cycles"]),
        )
        y = 104
        for label, value in rows:
            cv2.putText(
                canvas,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (142, 155, 173),
                1,
                cv2.LINE_AA,
            )
            color = (93, 220, 143) if label in ("SCORE", "BLUE HUB") else (235, 238, 242)
            if label == "BLUE HUB" and value == "INACTIVE":
                color = (94, 104, 235)
            cv2.putText(
                canvas,
                str(value),
                (x, y + 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                color,
                2,
                cv2.LINE_AA,
            )
            y += 82
        return canvas

    def write(self, data: dict[str, Any]) -> None:
        canvas = self.compose(np.asarray(self.frame_source()), data)
        if self.frames == 0 and not self.cv2.imwrite(str(self.first_frame_temp), canvas):
            raise RuntimeError(f"cannot write first-frame QA image {self.first_frame_temp}")
        self.writer.write(canvas)
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def abort(self) -> None:
        self.close()
        self.temp.unlink(missing_ok=True)
        self.first_frame_temp.unlink(missing_ok=True)


def restore_visual_state(trace: VerifiedTrace, index: int, slot: Any) -> None:
    """Restore exactly the state needed by USD/RTX without advancing physics."""

    arrays = trace.arrays
    slot.articulation.set_world_pose(
        arrays["robot_position"][index], arrays["robot_orientation_wxyz"][index]
    )
    slot.articulation.set_joint_positions(arrays["robot_joint_position"][index])
    slot.articulation.set_joint_velocities(arrays["robot_joint_velocity"][index])
    slot.fuel.set_world_poses(
        positions=arrays["fuel_position"][index],
        orientations=arrays["fuel_orientation_wxyz"][index],
    )
    controller = slot.controller
    (
        controller.storage_position,
        controller.container_extension,
        controller.intake_extension,
    ) = map(float, arrays["mechanism"][index])
    controller._apply_mechanism_visuals()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def publish_visual_state_to_fabric(
    trace: VerifiedTrace,
    index: int,
    slot: Any,
    combined_visual: Any,
    body_count: int,
) -> dict[str, float]:
    """Write the restored rigid state directly to render-visible Fabric.

    Isaac's articulation and rigid-body setters target PhysX tensors while a
    zero-delta Replicator render reads Fabric.  At fixed simulation time those
    stores are not automatically bridged, so publish every robot link and FUEL
    body explicitly after PhysX has recomputed articulation kinematics.
    """

    body_names = tuple(str(name) for name in slot.articulation._b.body_names)
    raw_links = _as_numpy(
        slot.articulation._b._physics_view.get_link_transforms()
    ).astype(np.float32, copy=False)
    robot_count = int(slot.articulation._b.count)
    body_count = len(body_names)
    if raw_links.shape == (robot_count, body_count, 7):
        links = raw_links[int(slot.index)]
    elif raw_links.shape == (robot_count * body_count, 7):
        links = raw_links.reshape(robot_count, body_count, 7)[int(slot.index)]
    else:
        raise RuntimeError(
            "articulation link-transform shape changed: "
            f"{raw_links.shape} != {(robot_count, body_count, 7)}"
        )
    if not np.isfinite(links).all():
        raise RuntimeError("articulation link transforms contain non-finite values")
    link_positions = np.ascontiguousarray(links[:, :3])
    # PhysX tensor order is XYZW; Isaac's XFormPrim API accepts WXYZ.
    link_orientations = np.ascontiguousarray(links[:, [6, 3, 4, 5]])
    fuel_positions = np.ascontiguousarray(
        trace.arrays["fuel_position"][index] + slot.fuel._origin[None, :],
        dtype=np.float32,
    )
    fuel_orientations = np.ascontiguousarray(
        trace.arrays["fuel_orientation_wxyz"][index], dtype=np.float32
    )
    positions = np.ascontiguousarray(
        np.concatenate((link_positions, fuel_positions), axis=0), dtype=np.float32
    )
    orientations = np.ascontiguousarray(
        np.concatenate((link_orientations, fuel_orientations), axis=0),
        dtype=np.float32,
    )
    expected_count = int(body_count) + int(slot.fuel.count)
    if positions.shape != (expected_count, 3) or orientations.shape != (
        expected_count,
        4,
    ):
        raise RuntimeError(
            "combined Fabric pose shape changed: "
            f"positions={positions.shape}, orientations={orientations.shape}, "
            f"expected_count={expected_count}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise RuntimeError("combined Fabric pose contains non-finite values")

    # This must be the final Fabric hierarchy operation before Replicator.
    # Each XFormPrim set/get begins with update_world_xforms(); separate robot
    # and FUEL writes (or a readback afterward) can therefore recompute and
    # overwrite part of the just-authored hierarchy. One combined write keeps
    # the captured state atomic.
    combined_visual.set_world_poses(
        positions=positions,
        orientations=orientations,
        usd=False,
    )
    return {"maximum_publish_input_error": 0.0}


def synchronize_visual_state(
    sim: Any,
    fixed_time: float,
    *,
    context: str,
    publish_to_fabric: Any,
) -> dict[str, float]:
    """Recompute articulation links and publish them without stepping physics."""

    physics_view = getattr(sim, "physics_sim_view", None)
    if physics_view is None:
        raise RuntimeError("visual-state synchronization lacks a physics simulation view")
    # The public simulation frontend is NumPy even though PhysX runs on GPU,
    # so invoke the articulation refresh explicitly.
    physics_view.update_articulations_kinematic()
    errors = dict(publish_to_fabric())
    current_time = float(sim.current_time)
    if current_time != float(fixed_time):
        raise RuntimeError(
            f"visual-state synchronization advanced simulation time during {context}: "
            f"{current_time} != {fixed_time}"
        )
    return errors


def read_policy_camera_frames(
    env: Any, env_index: int, *, allow_black: bool = False
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Read the three live robot cameras as CHW uint8 panels.

    The verified trace stores world state rather than pre-rendered pixels.  The
    camera prims are attached to the restored robot articulation, so a
    zero-delta RTX update recreates the intake, shooter, and navigation views
    without advancing physics.
    """

    names = tuple(getattr(env, "camera_names", ()))
    if len(names) != 3:
        raise RuntimeError(f"approved GUI requires exactly three policy cameras, got {names!r}")
    frames: list[np.ndarray] = []
    black_indices: list[int] = []
    for panel_index, name in enumerate(names):
        camera = env.cameras.get((int(env_index), name))
        if camera is None:
            raise RuntimeError(f"missing policy camera for env {env_index}: {name}")
        rgba = np.asarray(camera.get_rgba())
        if rgba.shape not in ((360, 640, 3), (360, 640, 4)) or rgba.dtype != np.uint8:
            raise RuntimeError(
                f"policy camera {name} shape/dtype changed: {rgba.shape}/{rgba.dtype}"
            )
        # The exact robot-mounted cameras render at 640x360. The approved
        # lower panes are 160x90, so deterministically decimate by four rather
        # than creating a second Isaac camera resolution or using a
        # platform-dependent image resize.
        rgb = np.ascontiguousarray(rgba[::4, ::4, :3])
        if rgb.shape != (90, 160, 3):
            raise RuntimeError(f"policy camera {name} downsample changed: {rgb.shape}")
        if float(rgb.std()) <= 1.0 and not allow_black:
            raise RuntimeError(f"policy camera {name} returned a black frame")
        if float(rgb.std()) <= 1.0:
            black_indices.append(panel_index)
        frames.append(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))))
    result = np.stack(frames, axis=0)
    if result.shape != (3, 3, 90, 160):
        raise RuntimeError(f"approved policy-panel tensor changed: {result.shape}")
    return result, tuple(black_indices)


def policy_camera_frames(env: Any, env_index: int) -> np.ndarray:
    """Compatibility wrapper returning only fully valid policy-camera panes."""

    return read_policy_camera_frames(env, env_index)[0]


def create_presentation_intake_camera(
    *,
    stage: Any,
    slot: Any,
    usd_geom: Any,
    gf: Any,
    camera_type: Any,
) -> tuple[Any, dict[str, Any]]:
    """Create a truthful chassis-mounted forward/intake presentation view.

    The camera is an extra render product.  It replaces only the pixel source
    used for the GUI's Intake pane; it does not replace any policy input,
    restore/modify world state, call physics, or advance simulation time.
    """

    path = (
        f"{slot.controller.usd_root_path}/chassis"
        "/Sensors/Cameras/PresentationIntake/OpticalCamera"
    )
    prim = usd_geom.Camera.Define(stage, path)
    prim.CreateHorizontalApertureAttr(CAMERA_HORIZONTAL_APERTURE_MM)
    prim.CreateVerticalApertureAttr(
        CAMERA_HORIZONTAL_APERTURE_MM
        * PRESENTATION_INTAKE_RESOLUTION[1]
        / PRESENTATION_INTAKE_RESOLUTION[0]
    )
    prim.CreateFocalLengthAttr(PRESENTATION_INTAKE_FOCAL_LENGTH_MM)
    prim.CreateClippingRangeAttr(gf.Vec2f(0.04, 100.0))
    view = gf.Matrix4d()
    view.SetLookAt(
        gf.Vec3d(*PRESENTATION_INTAKE_EYE_LOCAL),
        gf.Vec3d(*PRESENTATION_INTAKE_TARGET_LOCAL),
        gf.Vec3d(*PRESENTATION_INTAKE_UP_LOCAL),
    )
    usd_geom.Xformable(prim).AddTransformOp().Set(view.GetInverse())
    camera = camera_type(prim_path=path, resolution=PRESENTATION_INTAKE_RESOLUTION)
    camera.initialize()
    return camera, {
        "classification": "presentation_only_chassis_mounted_forward_intake_view",
        "prim_path": path,
        "parent_frame": "robot_chassis",
        "eye_local_xyz_m": list(PRESENTATION_INTAKE_EYE_LOCAL),
        "target_local_xyz_m": list(PRESENTATION_INTAKE_TARGET_LOCAL),
        "up_local_xyz": list(PRESENTATION_INTAKE_UP_LOCAL),
        "focal_length_mm": PRESENTATION_INTAKE_FOCAL_LENGTH_MM,
        "horizontal_aperture_mm": CAMERA_HORIZONTAL_APERTURE_MM,
        "resolution": list(PRESENTATION_INTAKE_RESOLUTION),
        "policy_input_changed": False,
        "world_state_changed": False,
        "physics_steps": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("FRC_POLICY_SPEED_SCALE", "1.0") != "1.0":
        raise RuntimeError("FRC_POLICY_SPEED_SCALE must be exactly 1.0")
    checkpoint_sha = require_file_sha(
        args.checkpoint, args.expected_checkpoint_sha256, "policy checkpoint"
    )
    prefix_sha = require_file_sha(
        args.prefix_checkpoint,
        args.expected_prefix_checkpoint_sha256,
        "prefix checkpoint",
    )
    template_sha = require_file_sha(args.template, args.expected_template_sha256, "template")
    code = verify_code_snapshot(
        args.code_root, args.code_archive, args.expected_code_archive_sha256
    )
    camera_state = camera_state_sha256 = None
    if args.camera_state_json is not None:
        camera_state, camera_state_sha256 = load_gui_camera_state(args.camera_state_json)
    trace = load_verified_trace(
        args.trace,
        expected_trace_sha256=args.expected_trace_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    source_identity = source_bundle_identity(trace.metadata)
    source_checkpoint_sha = (
        args.expected_source_checkpoint_sha256
        or args.expected_checkpoint_sha256
    )
    bundle = load_replay_bundle(
        args.bundle,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_checkpoint_sha256=source_checkpoint_sha,
        expected_env_seed=source_identity["env_seed"],
        expected_action_seed=source_identity["action_seed"],
        expected_env_index=source_identity["env_index"],
    )
    exact_matches = {
        "prefix checkpoint": (prefix_sha, trace.metadata["prefix_checkpoint_sha256"]),
        "template": (template_sha, trace.metadata["template_sha256"]),
        "code archive": (code["archive_sha256"], trace.metadata["code_archive_sha256"]),
        "source bundle": (bundle.sha256, trace.metadata["bundle_sha256"]),
    }
    for label, (actual, recorded) in exact_matches.items():
        if actual != recorded:
            raise ValueError(f"{label} custody differs from verified trace: {actual} != {recorded}")
    if args.output.suffix.lower() != ".avi":
        raise ValueError("--output must end in .avi")
    first_frame = args.output.with_suffix(".first-frame.png")
    for path in (args.output, args.provenance_out, first_frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")

    sys.path.insert(0, str((args.code_root / "src").resolve()))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    env = None
    sink = None
    render_product = None
    rgb_annotator = None
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.prims import XFormPrim
        from isaacsim.sensors.camera import Camera as IsaacCamera
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        cfg = build_exact_cfg(
            VecEnvCfg,
            template=args.template,
            episode=run_episode_contract(bundle.episode, trace.metadata),
        )
        # Keep the exact three robot-mounted camera products so the approved
        # GUI's intake/shooter/navigation panes are recreated from the same
        # restored world state as the main field view.
        cfg.cameras = True
        env = VecCompetitionEnv(cfg)
        if not bool(getattr(env, "_camera_ready", False)):
            raise RuntimeError("offline renderer policy cameras did not become ready")
        if len(env.cameras) != int(cfg.num_envs) * 3:
            raise RuntimeError(
                f"offline renderer expected three cameras per env, got {len(env.cameras)}"
            )
        slot = env.slots[int(trace.metadata["env_index"])]
        stage = omni.usd.get_context().get_stage()
        robot_root = str(slot.controller.usd_root_path).rstrip("/")
        rigid_paths_by_name: dict[str, str] = {}
        duplicate_rigid_names: set[str] = set()
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not path.startswith(robot_root + "/"):
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            name = str(prim.GetName())
            if name in rigid_paths_by_name:
                duplicate_rigid_names.add(name)
            rigid_paths_by_name[name] = path
        body_names = tuple(str(name) for name in slot.articulation._b.body_names)
        missing_body_names = sorted(set(body_names) - set(rigid_paths_by_name))
        duplicate_body_names = sorted(set(body_names) & duplicate_rigid_names)
        if missing_body_names or duplicate_body_names:
            raise RuntimeError(
                "cannot build an exact articulation-link Fabric view: "
                f"missing={missing_body_names}, duplicates={duplicate_body_names}"
            )
        link_paths = [rigid_paths_by_name[name] for name in body_names]
        if len(link_paths) != len(set(link_paths)):
            raise RuntimeError("articulation-link Fabric paths are not unique")
        fuel_paths = [
            str(slot.fuel._b.prim_paths[int(index)]) for index in slot.fuel._all
        ]
        if len(fuel_paths) != int(slot.fuel.count) or len(fuel_paths) != len(
            set(fuel_paths)
        ):
            raise RuntimeError("selected FUEL Fabric paths are incomplete or duplicated")
        combined_visual = XFormPrim(
            link_paths + fuel_paths,
            name=f"verified_trace_rigid_state_env_{slot.index}",
            reset_xform_properties=False,
            usd=False,
        )
        presentation_intake = None
        presentation_intake_meta = None
        if args.presentation_intake_view:
            presentation_intake, presentation_intake_meta = (
                create_presentation_intake_camera(
                    stage=stage,
                    slot=slot,
                    usd_geom=UsdGeom,
                    gf=Gf,
                    camera_type=IsaacCamera,
                )
            )
            # This dictionary is read only by policy_camera_frames below.  No
            # inference is run in this offline render pass.
            env.cameras[(slot.index, "intake")] = presentation_intake
        renderer_joint_names = [str(value) for value in slot.articulation.dof_names]
        if renderer_joint_names != trace.metadata["joint_names"]:
            raise RuntimeError("renderer articulation joint order differs from trace")
        if int(slot.fuel.count) != int(trace.metadata["fuel_count"]):
            raise RuntimeError("renderer FUEL count differs from trace")

        origin = np.asarray(env.env_origins[slot.index], np.float32)
        camera_path = f"/World/VerifiedTraceObliqueCamera_env_{slot.index}"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        if camera_state is None:
            camera = oblique_camera_layout(
                origin,
                height_m=args.camera_height_m,
                tilt_deg=args.camera_tilt_deg,
                azimuth_deg=args.camera_azimuth_deg,
            )
            focal_length_mm = float(args.camera_focal_length_mm)
            horizontal_aperture_mm = CAMERA_HORIZONTAL_APERTURE_MM
            vertical_aperture_mm = (
                CAMERA_HORIZONTAL_APERTURE_MM * CAMERA_SIZE[1] / CAMERA_SIZE[0]
            )
            horizontal_aperture_offset_mm = 0.0
            vertical_aperture_offset_mm = 0.0
            camera_exposure = 0.0
            clipping_range = np.asarray([0.1, 1000.0], np.float64)
            framing = verify_full_field_framing(
                camera,
                focal_length_mm=focal_length_mm,
                field_margin_m=args.camera_field_margin_m,
            )
        else:
            # The interactive GUI saved coordinates relative to its field at
            # world origin.  Translate eye and target by the selected vector
            # environment origin without altering the approved orientation.
            camera = {
                "eye": origin.astype(np.float64)
                + np.asarray(camera_state["eye_xyz"], np.float64),
                "target": origin.astype(np.float64)
                + np.asarray(camera_state["target_xyz"], np.float64),
                "up": np.asarray(camera_state["up_xyz"], np.float64),
            }
            focal_length_mm = float(camera_state["focal_length_mm"])
            horizontal_aperture_mm = float(camera_state["horizontal_aperture_mm"])
            vertical_aperture_mm = float(camera_state["vertical_aperture_mm"])
            horizontal_aperture_offset_mm = float(
                camera_state.get("horizontal_aperture_offset_mm", 0.0)
            )
            vertical_aperture_offset_mm = float(
                camera_state.get("vertical_aperture_offset_mm", 0.0)
            )
            camera_exposure = float(camera_state.get("exposure", 0.0))
            clipping_range = np.asarray(camera_state["clipping_range"], np.float64)
            framing = {
                "validated": True,
                "source": "exact_saved_gui_camera_state",
                "viewport_resolution": list(camera_state["viewport_resolution"]),
            }
        camera_prim.CreateFocalLengthAttr(focal_length_mm)
        camera_prim.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        camera_prim.CreateVerticalApertureAttr(vertical_aperture_mm)
        camera_prim.CreateHorizontalApertureOffsetAttr(horizontal_aperture_offset_mm)
        camera_prim.CreateVerticalApertureOffsetAttr(vertical_aperture_offset_mm)
        camera_prim.CreateExposureAttr(camera_exposure)
        camera_prim.CreateClippingRangeAttr(
            Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1]))
        )
        view = Gf.Matrix4d()
        view.SetLookAt(
            Gf.Vec3d(*map(float, camera["eye"])),
            Gf.Vec3d(*map(float, camera["target"])),
            Gf.Vec3d(*map(float, camera["up"])),
        )
        UsdGeom.Xformable(camera_prim).AddTransformOp().Set(view.GetInverse())
        dome = UsdLux.DomeLight.Define(stage, "/World/VideoGuiDome")
        dome.CreateIntensityAttr(850.0)
        dome.CreateColorAttr(Gf.Vec3f(0.76, 0.82, 0.95))
        key = UsdLux.DistantLight.Define(stage, "/World/VideoGuiKey")
        key.CreateIntensityAttr(2600.0)
        key.CreateAngleAttr(0.8)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(42.0, -28.0, -24.0))
        render_product = rep.create.render_product(camera_path, CAMERA_SIZE)
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])

        # Flush the restored tensor state into Fabric/RTX before warming the
        # render products. A Replicator zero-delta step alone does not perform
        # this synchronization and would leave every visual at frame zero.
        restore_visual_state(trace, 0, slot)
        fixed_time = float(env.sim.current_time)
        fabric_publish_input_error_max = 0.0
        fabric_errors = synchronize_visual_state(
            env.sim,
            fixed_time,
            context="camera warmup",
            publish_to_fabric=lambda: publish_visual_state_to_fabric(
                trace, 0, slot, combined_visual, len(body_names)
            ),
        )
        fabric_publish_input_error_max = max(fabric_errors.values())
        ready = False
        warmup_calls = 0
        for _ in range(120):
            rep.orchestrator.step(delta_time=0.0, rt_subframes=8, pause_timeline=False)
            warmup_calls += 1
            if float(env.sim.current_time) != fixed_time:
                raise RuntimeError("zero-delta camera warmup advanced simulation time")
            rgba = np.asarray(rgb_annotator.get_data())
            intake_ready = True
            if presentation_intake is not None:
                intake_rgba = np.asarray(presentation_intake.get_rgba())
                intake_ready = bool(
                    intake_rgba.shape[:2]
                    == (
                        PRESENTATION_INTAKE_RESOLUTION[1],
                        PRESENTATION_INTAKE_RESOLUTION[0],
                    )
                    and float(intake_rgba[..., :3].std()) > 1.0
                )
            if (
                rgba.shape[:2] == (CAMERA_SIZE[1], CAMERA_SIZE[0])
                and float(rgba[..., :3].std()) > 1.0
                and intake_ready
            ):
                ready = True
                break
        if not ready:
            raise RuntimeError("native field camera did not become ready at zero delta")

        sink = GuiVideoSink(rgb_annotator.get_data, args.output)
        first = last = None
        policy_camera_retry_calls = 0
        policy_camera_fallback_frames = 0
        policy_camera_fallback_streak = 0
        policy_camera_fallback_max_streak = 0
        previous_policy_frames = None
        if args.qa_frame_index is not None:
            qa_start = int(args.qa_frame_index)
            qa_stop = min(STEPS, qa_start + int(args.qa_frame_count))
            indices = range(qa_start, qa_stop)
            expected_frames = qa_stop - qa_start
        else:
            indices = range(STEPS)
            expected_frames = STEPS
        for index in indices:
            restore_visual_state(trace, index, slot)
            fabric_errors = synchronize_visual_state(
                env.sim,
                fixed_time,
                context=f"playback frame {index}",
                publish_to_fabric=lambda index=index: publish_visual_state_to_fabric(
                    trace, index, slot, combined_visual, len(body_names)
                ),
            )
            fabric_publish_input_error_max = max(
                fabric_publish_input_error_max, max(fabric_errors.values())
            )
            rep.orchestrator.step(delta_time=0.0, rt_subframes=1, pause_timeline=False)
            if float(env.sim.current_time) != fixed_time:
                raise RuntimeError(f"offline render advanced simulation time at frame {index}")
            policy_frames = None
            black_indices: tuple[int, ...] = ()
            for camera_attempt in range(9):
                policy_frames, black_indices = read_policy_camera_frames(
                    env, slot.index, allow_black=True
                )
                if not black_indices:
                    break
                if camera_attempt < 8:
                    # Camera render products can lag a direct Fabric teleport
                    # by one Kit update. Retry the exact same restored state at
                    # zero delta; never advance physics and never emit/drop a
                    # video frame while an inset is transiently black.
                    rep.orchestrator.step(
                        delta_time=0.0, rt_subframes=1, pause_timeline=False
                    )
                    policy_camera_retry_calls += 1
                    if float(env.sim.current_time) != fixed_time:
                        raise RuntimeError(
                            "policy-camera retry advanced simulation time at "
                            f"frame {index}"
                        )
            if policy_frames is None:
                raise RuntimeError(f"policy cameras unavailable at frame {index}")
            if black_indices:
                if previous_policy_frames is None:
                    raise RuntimeError(
                        "policy cameras were black before any valid frame at "
                        f"frame {index}: {black_indices}"
                    )
                policy_camera_fallback_streak += 1
                policy_camera_fallback_max_streak = max(
                    policy_camera_fallback_max_streak,
                    policy_camera_fallback_streak,
                )
                if policy_camera_fallback_streak > 3:
                    raise RuntimeError(
                        "policy camera blackout persisted beyond three frames at "
                        f"frame {index}: {black_indices}"
                    )
                policy_frames = policy_frames.copy()
                for panel_index in black_indices:
                    policy_frames[panel_index] = previous_policy_frames[panel_index]
                policy_camera_fallback_frames += 1
            else:
                policy_camera_fallback_streak = 0
            previous_policy_frames = policy_frames.copy()
            data = frame_telemetry(trace, index)
            if index == 0 and int(data["score"]) != 0:
                raise RuntimeError("rendered replay would reveal a nonzero score at the start")
            sink.write(data, policy_frames)
            first = data if first is None else first
            last = data
            if index % 100 == 0:
                print(
                    f"VERIFIED_TRACE_RENDER_PROGRESS frame={index + 1}/{STEPS} "
                    f"score={data['score']}",
                    flush=True,
                )
        if sink.frames != expected_frames:
            raise RuntimeError(
                f"rendered frame count mismatch: {sink.frames}/{expected_frames}"
            )
        sink.close()
        if not sink.temp.is_file() or sink.temp.stat().st_size <= 0:
            raise RuntimeError("MJPG renderer produced an empty intermediate")
        if not sink.first_frame_path.is_file():
            raise RuntimeError("renderer did not produce first-frame QA image")
        result = {
            "schema": RENDER_PROVENANCE_SCHEMA,
            "classification": {
                "type": "offline_verified_visual_state_render",
                "policy_or_archived_actions_executed_in_render_pass": False,
                "physics_steps_during_playback": 0,
                "zero_delta_render_calls": (
                    expected_frames + warmup_calls + policy_camera_retry_calls
                ),
                "explicit_articulation_kinematic_sync_calls": expected_frames + 1,
                "direct_fabric_visual_publish_calls": expected_frames + 1,
                "direct_fabric_articulation_links_per_call": len(body_names),
                "direct_fabric_fuel_bodies_per_call": int(slot.fuel.count),
                "maximum_fabric_publish_input_error": (
                    fabric_publish_input_error_max
                ),
                "zero_delta_camera_warmup_calls": warmup_calls,
                "zero_delta_playback_calls": expected_frames,
                "zero_delta_policy_camera_retry_calls": policy_camera_retry_calls,
                "policy_camera_last_valid_fallback_frames": (
                    policy_camera_fallback_frames
                ),
                "policy_camera_last_valid_fallback_max_streak": (
                    policy_camera_fallback_max_streak
                ),
                "policy_camera_last_valid_fallback_limit_frames": 3,
                "policy_camera_render_products": len(env.cameras),
                "presentation_intake_render_products": int(
                    presentation_intake is not None
                ),
                "qa_single_frame": (
                    args.qa_frame_index is not None and expected_frames == 1
                ),
                "qa_frame_start": (
                    int(args.qa_frame_index)
                    if args.qa_frame_index is not None
                    else None
                ),
                "qa_frame_count": (
                    expected_frames if args.qa_frame_index is not None else None
                ),
                "overhead_render_products": 1,
                "strict_nadir_render_products": 0,
                "oblique_full_field_render_products": 1,
            },
            "trace": {
                "path": str(trace.path),
                "sha256": trace.sha256,
                "live_terminal_score": int(trace.metadata["live_terminal_score"]),
            },
            "custody": {
                "checkpoint": {
                    "path": str(args.checkpoint.resolve()),
                    "sha256": checkpoint_sha,
                },
                "prefix_checkpoint": {
                    "path": str(args.prefix_checkpoint.resolve()),
                    "sha256": prefix_sha,
                },
                "bundle": {"path": str(bundle.path), "sha256": bundle.sha256},
                "source_bundle_checkpoint_sha256": source_checkpoint_sha.lower(),
                "template": {"path": str(args.template.resolve()), "sha256": template_sha},
                "exact_code": {
                    "root": str(args.code_root.resolve()),
                    "archive": str(args.code_archive.resolve()),
                    **code,
                },
            },
            "video": {
                "path": str(args.output.resolve()),
                "sha256": sha256_file(sink.temp),
                "bytes": sink.temp.stat().st_size,
                "codec": CODEC,
                "high_bitrate_intermediate": True,
                "fps": FPS,
                "frames": expected_frames,
                "duration_s": expected_frames / FPS,
                "camera_resolution": list(CAMERA_SIZE),
                "sidebar_width": SIDEBAR_WIDTH,
                "output_resolution": [CAMERA_SIZE[0] + SIDEBAR_WIDTH, CAMERA_SIZE[1]],
                "native_simulator_pixels": True,
                "camera_projection": (
                    "exact_saved_gui_perspective"
                    if camera_state is not None
                    else "elevated_oblique_full_field"
                ),
                "camera_eye_xyz": [float(value) for value in camera["eye"]],
                "camera_target_xyz": [float(value) for value in camera["target"]],
                "camera_up_xyz": [float(value) for value in camera["up"]],
                "camera_height_m": (
                    float(camera["height_m"]) if "height_m" in camera else None
                ),
                "camera_tilt_from_vertical_deg": (
                    float(camera["tilt_deg"]) if "tilt_deg" in camera else None
                ),
                "camera_azimuth_deg": (
                    float(camera["azimuth_deg"]) if "azimuth_deg" in camera else None
                ),
                "camera_horizontal_offset_m": (
                    float(camera["horizontal_offset_m"])
                    if "horizontal_offset_m" in camera
                    else None
                ),
                "camera_focal_length_mm": focal_length_mm,
                "camera_horizontal_aperture_mm": horizontal_aperture_mm,
                "camera_vertical_aperture_mm": vertical_aperture_mm,
                "camera_horizontal_aperture_offset_mm": horizontal_aperture_offset_mm,
                "camera_vertical_aperture_offset_mm": vertical_aperture_offset_mm,
                "camera_exposure": camera_exposure,
                "camera_clipping_range": clipping_range.tolist(),
                "camera_state_source": (
                    {
                        "path": str(args.camera_state_json.resolve()),
                        "sha256": camera_state_sha256,
                        "schema": camera_state["schema"],
                    }
                    if camera_state is not None
                    else None
                ),
                "full_field_framing": framing,
                "policy_panel_source_resolution": [160, 90],
                "policy_panel_count": 3,
                "presentation_intake_camera": presentation_intake_meta,
                "lighting": {
                    "dome_intensity": 850.0,
                    "dome_color": [0.76, 0.82, 0.95],
                    "distant_intensity": 2600.0,
                    "distant_angle_deg": 0.8,
                    "distant_rotation_xyz_deg": [42.0, -28.0, -24.0],
                },
            },
            "first_frame": first,
            "last_frame": last,
            "renderer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
        }
        provenance_ready = args.provenance_out.with_name(
            f".{args.provenance_out.name}.{os.getpid()}.ready"
        )
        atomic_json(provenance_ready, result)
        os.replace(sink.temp, args.output)
        os.replace(provenance_ready, args.provenance_out)
        print("VERIFIED_TRACE_RENDER_DONE " + json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if sink is not None:
            sink.abort()
        if rgb_annotator is not None:
            try:
                if render_product is not None:
                    rgb_annotator.detach([render_product])
            except Exception:
                pass
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
        if env is not None:
            env.close()
        app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-trace-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--expected-source-checkpoint-sha256",
        default=None,
        help=(
            "checkpoint SHA declared by the immutable source bundle when the "
            "rendered trace used a compatible checkpoint"
        ),
    )
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-prefix-checkpoint-sha256", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--expected-code-archive-sha256", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, default=None)
    parser.add_argument(
        "--camera-state-json",
        type=Path,
        default=None,
        help=(
            "exact 456-FUEL GUI viewport state saved with P; when supplied it "
            "takes precedence over height/tilt/azimuth camera arguments"
        ),
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT_M,
        help="camera height above field centre (default: %(default)s m)",
    )
    parser.add_argument(
        "--camera-tilt-deg",
        type=float,
        default=DEFAULT_CAMERA_TILT_DEG,
        help="camera tilt away from vertical; 0 is exact top-down (default: %(default)s)",
    )
    parser.add_argument(
        "--camera-azimuth-deg",
        type=float,
        default=DEFAULT_CAMERA_AZIMUTH_DEG,
        help="eye direction from field centre, CCW from field +X (default: %(default)s)",
    )
    parser.add_argument(
        "--camera-focal-length-mm",
        type=float,
        default=DEFAULT_CAMERA_FOCAL_LENGTH_MM,
        help="USD camera focal length (default: %(default)s mm)",
    )
    parser.add_argument(
        "--camera-field-margin-m",
        type=float,
        default=DEFAULT_FIELD_FRAME_MARGIN_M,
        help="extra field-edge margin required to remain visible (default: %(default)s m)",
    )
    parser.add_argument(
        "--presentation-intake-view",
        action="store_true",
        help=(
            "use an additional chassis-mounted forward/intake camera only for "
            "the public GUI pane; policy inputs and world state are unchanged"
        ),
    )
    parser.add_argument(
        "--qa-frame-index",
        type=int,
        default=None,
        help="render only this verified trace frame for visual QA",
    )
    parser.add_argument(
        "--qa-frame-count",
        type=int,
        default=1,
        help="with --qa-frame-index, render this many consecutive QA frames",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.qa_frame_index is not None and not 0 <= args.qa_frame_index < STEPS:
        raise SystemExit(f"--qa-frame-index must be in [0, {STEPS - 1}]")
    if args.qa_frame_count < 1:
        raise SystemExit("--qa-frame-count must be positive")
    if args.qa_frame_index is None and args.qa_frame_count != 1:
        raise SystemExit("--qa-frame-count requires --qa-frame-index")
    # Validate the requested view before starting Isaac Sim.  Framing is
    # translation-invariant, so field centre at the origin is sufficient here.
    if args.camera_state_json is None:
        camera = oblique_camera_layout(
            np.zeros(3, np.float64),
            height_m=args.camera_height_m,
            tilt_deg=args.camera_tilt_deg,
            azimuth_deg=args.camera_azimuth_deg,
        )
        verify_full_field_framing(
            camera,
            focal_length_mm=args.camera_focal_length_mm,
            field_margin_m=args.camera_field_margin_m,
        )
    else:
        load_gui_camera_state(args.camera_state_json)
    if args.provenance_out is None:
        args.provenance_out = args.output.with_suffix(".provenance.json")
    outputs = {
        args.output.resolve(),
        args.output.with_suffix(".first-frame.png").resolve(),
        args.provenance_out.resolve(),
    }
    protected = {
        args.trace.resolve(),
        args.bundle.resolve(),
        args.checkpoint.resolve(),
        args.prefix_checkpoint.resolve(),
        args.code_archive.resolve(),
        args.template.resolve(),
    }
    if args.camera_state_json is not None:
        protected.add(args.camera_state_json.resolve())
    if len(outputs) != 3 or outputs & protected:
        raise SystemExit("render outputs must be distinct from every custody input")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
