"""Complete Legacy Robot robot model for the Isaac REBUILT scene.

Builds the robot from ``assets/fresh_xrc/robot/robot_spec.json`` (produced by
``tools/extract_robot_spec.py``) as a PhysX articulation:

- chassis root link carrying the full xRC visual mesh and all solid colliders
  except the six drive-wheel spheres,
- six wheel links (sphere colliders, r=0.0808 m) on velocity-driven revolute
  joints, torque-limited so full speed matches the xRC-measured ~3.5 m/s,
- functional ball systems mirroring xRC's own trigger-volume design
  (``ballshooting_v2`` zones): an intake capture volume, eight visible
  preload/indexer positions, and the exact legacy Robot_RapidUS shooter law.

xRC drives its robots with per-wheel torque + scripted friction forces; the
revolute-drive articulation reproduces the same behaviour with real PhysX
contacts, which is what the RL transfer needs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "assets" / "fresh_xrc" / "robot" / "robot_spec.json"

# xRC-measured full-stick kinematics (see memory/docs: forward ~3.5 m/s).
# On flat carpet the wheels track the command with no slip (lane probe
# 2026-07-02: commanded 3.68 -> measured 3.68), so command the target directly.
MAX_DRIVE_SPEED_MPS = 3.5
# Differential scale so full turn stick yields xRC's ~100 deg/s yaw.
# Calibrated sweep (runs/drive_calibration.json): yaw_dps ~= 610 * scale.
TURN_SPEED_SCALE = 0.165
# Per-wheel drive torque limit: 32.91 kg reaching 3.5 m/s in ~1 s across six
# wheels of r=0.0808 m, with headroom for bump climbs.
WHEEL_DRIVE_TORQUE_NM = 4.0
WHEEL_DRIVE_DAMPING = 15.0
MAGAZINE_PRELOADS = 8
# Hopper capacity: the extraction exposed 8 indexer/preload markers, but the
# physical hopper holds many more.  We stack the 8-slot pattern into layers so
# the robot can carry a full hopper of FUEL (visible, co-moving).
HOPPER_CAPACITY = 40
HOPPER_LAYER_SPACING_M = 0.16  # > ball diameter (0.152 m)
SHOOTER_COOLDOWN_S = 0.07  # GenericIndexerMulti cooldown_period
INTAKE_PULL_SPEED_MPS = 4.0  # ballshooting_v2 'speed' on the intake zones
# reference-style wide shooter: release a horizontal row of balls per shot.
# The shot LAW is unchanged (aim/speed/pitch); this only spreads the release
# and adds the turret visual.  1 == the pristine single-ball behaviour.
SHOOTER_BARRELS = 3
SHOOTER_BARREL_SPACING_M = 0.16  # > ball diameter (0.152 m) so they don't overlap

# Exact values decoded from the pristine xRC v20.2b Robot_RapidUS prefab.
# DisAim travels 0..100.  Robot_RapidUS maps that fraction to 6..11 m/s and
# rotates the shooter -20 degrees about its pivot.  The markball's authored
# -local-X axis already points 72.9228 degrees upward at aim=0.
SHOOTER_MIN_SPEED_MPS = 6.0
SHOOTER_MAX_SPEED_MPS = 11.0
SHOOTER_BASE_PITCH_DEG = 70.355276
SHOOTER_TRAVEL_DEG = 20.0
SHOOTER_PIVOT_LOCAL = np.array([-0.3836273, -0.0050704, 0.4718809], dtype=np.float32)
SHOOTER_MUZZLE_LOCAL_AIM0 = np.array([-0.3035871, -0.0049403, 0.5762653], dtype=np.float32)
SHOOTER_DIRECTION_LOCAL_AIM0 = np.array(
    [0.33508985, 0.02713615, 0.94179532], dtype=np.float32
)

# Upper xRC MarkBall trigger centers.  A physical shot is aimed through this
# opening and then falls through the real HUB geometry to the scorer below.
HUB_MARKBALL_TARGETS = {
    "red": np.array([0.0199, 3.6874, 1.3646], dtype=np.float32),
    "blue": np.array([-0.0199, -3.6874, 1.3646], dtype=np.float32),
}
AUTO_ALIGN_TOLERANCE_DEG = 0.25
# Direct-shot range envelope, MEASURED by tools/sweep_envelope.py in PhysX
# (runs/shot_envelope.json): both HUBs score ~100% at 2.25-2.75 m and on-axis
# out to ~4 m; beyond that only favourable approach angles score.  The gate is
# opened to the reliable region so the robot shoots wherever a shot scores,
# not an arbitrary narrow band.
CALIBRATED_RANGE_MIN_M = 2.3
CALIBRATED_RANGE_MAX_M = 4.0

# Phase B fire gates.  These are xRC-derived physical minimums for a clean shot;
# they are never inflated to slow the demo (assistant_NEXT_PLAN Phase B2).
FIRE_MAX_SPEED_MPS = 0.08     # chassis horizontal speed at the instant of feed
FIRE_MAX_YAW_RATE_DPS = 3.0   # chassis yaw rate at the instant of feed


class ShooterState(str, Enum):
    """Explicit shooter/indexer states (assistant_NEXT_PLAN Phase B2).

    ``str`` mixin so the value serialises directly into GUI/JSON status.
    """

    IDLE = "IDLE"
    ACQUIRE_TARGET = "ACQUIRE_TARGET"
    TURNING = "TURNING"
    BRAKING = "BRAKING"
    READY = "READY"
    FEEDING = "FEEDING"
    COOLDOWN = "COOLDOWN"
    BLOCKED = "BLOCKED"


# Phase C2: solid xRC colliders that carry no enabled MeshRenderer.  These are
# the intake guides and hopper/body containment walls that read as "missing
# panels"; they are visualised as translucent polycarbonate (see
# docs/legacy_BODYWORK_AUDIT.md).  They are NOT trigger volumes.
CONTAINMENT_SUBTREES = ("Physics[7]/IntakeHopper[5]", "Physics[7]/Body[6]")
# Translucent polycarbonate appearance for the containment panels (RGBA).
CONTAINMENT_PANEL_RGBA = (0.62, 0.74, 0.86, 0.24)
# reference-style turret look (additive visual over the exact legacy shooter).
TURRET_METAL_RGBA = (0.74, 0.76, 0.80, 1.0)   # brushed aluminium
TURRET_BARREL_RGBA = (0.06, 0.06, 0.07, 1.0)  # black shooter barrels


def is_containment_collider_path(path: str) -> bool:
    """True for the solid IntakeHopper/Body colliders visualised as panels."""
    return any(subtree in path for subtree in CONTAINMENT_SUBTREES)


def containment_collider_paths(components: list[dict[str, Any]]) -> list[str]:
    """Enumerate the solid containment collider owner paths from components."""
    paths: list[str] = []
    for component in components:
        if component["source"]["type"] not in {"BoxCollider", "SphereCollider", "CapsuleCollider"}:
            continue
        values, path = component["values"], component["owner_path"]
        if values.get("m_IsTrigger") or not values.get("m_Enabled", True):
            continue
        if is_containment_collider_path(path):
            paths.append(path)
    return paths


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shooter_exit_speed(aim: float) -> float:
    """Exact legacy Robot_RapidUS law: power = 6 + 5*aim m/s."""
    fraction = float(np.clip(aim, 0.0, 1.0))
    return SHOOTER_MIN_SPEED_MPS + fraction * (
        SHOOTER_MAX_SPEED_MPS - SHOOTER_MIN_SPEED_MPS
    )


def shooter_elevation_rad(aim: float) -> float:
    """World-independent exit pitch including the prefab's fixed mounting."""
    fraction = float(np.clip(aim, 0.0, 1.0))
    return math.radians(SHOOTER_BASE_PITCH_DEG - SHOOTER_TRAVEL_DEG * fraction)


def shooter_muzzle_local(aim: float) -> np.ndarray:
    """Muzzle center after rotating the physical shooter about its pivot."""
    angle = math.radians(SHOOTER_TRAVEL_DEG * float(np.clip(aim, 0.0, 1.0)))
    c, s = math.cos(angle), math.sin(angle)
    pitch_down = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return SHOOTER_PIVOT_LOCAL + pitch_down @ (
        SHOOTER_MUZZLE_LOCAL_AIM0 - SHOOTER_PIVOT_LOCAL
    )


def shooter_direction_local(aim: float) -> np.ndarray:
    """Exact dynamic -local-X push axis of xRC's BS Outtake (3)."""
    angle = math.radians(SHOOTER_TRAVEL_DEG * float(np.clip(aim, 0.0, 1.0)))
    c, s = math.cos(angle), math.sin(angle)
    pitch_down = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    direction = pitch_down @ SHOOTER_DIRECTION_LOCAL_AIM0
    return direction / np.linalg.norm(direction)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_rotation(yaw: float) -> np.ndarray:
    """World Z-axis rotation matrix (chassis frame -> world) for a heading."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def solve_shot_geometry(
    position: np.ndarray, target: np.ndarray, samples: int = 600, gravity: float = 9.81
) -> dict[str, Any]:
    """Exact xRC coupled-law ballistic sweep to a HUB MarkBall opening.

    Pure function (no articulation): the single source of truth for both the
    live auto-aim (``RobotController.solve_auto_aim``) and the offline
    ``shot_planner.solve_direct_shot``.  Sweeps the one xRC aim parameter,
    resolving the muzzle's small lateral offset with two fixed-point
    iterations, and returns the candidate with least |vertical error| at the
    target, including muzzle origin, exit direction, and desired chassis yaw.
    """
    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    best: dict[str, Any] | None = None
    for aim in np.linspace(0.0, 1.0, samples):
        direction_local = shooter_direction_local(float(aim))
        local_azimuth = math.atan2(float(direction_local[1]), float(direction_local[0]))
        desired_yaw = (
            math.atan2(float(target[1] - position[1]), float(target[0] - position[0]))
            - local_azimuth
        )
        origin = position
        for _ in range(2):
            rotation = yaw_rotation(desired_yaw)
            origin = position + rotation @ shooter_muzzle_local(float(aim))
            bearing = math.atan2(float(target[1] - origin[1]), float(target[0] - origin[0]))
            desired_yaw = bearing - local_azimuth
        rotation = yaw_rotation(desired_yaw)
        origin = position + rotation @ shooter_muzzle_local(float(aim))
        direction = rotation @ direction_local
        speed = shooter_exit_speed(float(aim))
        horizontal_speed = speed * float(np.linalg.norm(direction[:2]))
        horizontal_range = float(np.linalg.norm(target[:2] - origin[:2]))
        if horizontal_speed < 1e-6:
            continue
        flight_time = horizontal_range / horizontal_speed
        predicted_z = float(
            origin[2] + speed * direction[2] * flight_time - 0.5 * gravity * flight_time * flight_time
        )
        error = predicted_z - float(target[2])
        candidate = {
            "aim": float(aim),
            "speed_mps": float(speed),
            "pitch_deg": math.degrees(math.asin(float(direction_local[2]))),
            "desired_yaw_rad": wrap_angle(desired_yaw),
            "muzzle_origin": origin.astype(np.float32),
            "exit_direction": direction.astype(np.float32),
            "range_m": horizontal_range,
            "flight_time_s": flight_time,
            "vertical_error_m": error,
        }
        if best is None or abs(error) < abs(float(best["vertical_error_m"])):
            best = candidate
    assert best is not None
    return best


@dataclass
class ZoneBox:
    """Oriented box in the chassis frame (Isaac axes)."""

    center: np.ndarray
    axes: np.ndarray  # 3x3, columns are box axes in chassis frame
    half: np.ndarray

    @classmethod
    def from_spec(cls, zone: dict[str, Any]) -> "ZoneBox":
        return cls(
            center=np.asarray(zone["center_isaac"], dtype=np.float32),
            axes=np.asarray(zone["axes_isaac_columns"], dtype=np.float32),
            half=np.asarray(zone["half_extents_m"], dtype=np.float32),
        )

    def contains(self, points_chassis: np.ndarray, margin: float = 0.0) -> np.ndarray:
        local = (np.atleast_2d(points_chassis) - self.center) @ self.axes
        return np.all(np.abs(local) <= self.half + margin, axis=1)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class RobotArticulationBuilder:
    """Authors the articulated Legacy Robot under ``root_path``.

    Reuses the caller's (SceneBuilder's) mesh/material helpers so visuals and
    colliders keep the exact conventions of the rest of the scene.
    """

    def __init__(
        self,
        scene_builder: Any,
        spec: dict[str, Any] | None = None,
        root_path: str = "/World/Robot/LegacyRobot",
        spawn_translation: tuple[float, float, float] = (1.52, -5.55, 0.01),
        spawn_yaw_deg: float = 90.0,
    ):
        self.sb = scene_builder
        self.spec = spec or load_spec()
        self.root_path = root_path
        self.spawn = spawn_translation
        self.spawn_yaw_deg = spawn_yaw_deg
        self.chassis_path = f"{root_path}/chassis"

    # -- geometry from the raw extraction (same logic the compound build used) --
    def _visual_and_collider_geometry(self):
        from xrc_rebuilt import isaac_scene as scene_mod

        robot_dir = scene_mod.ROBOT_DIR
        hierarchy = json.loads((robot_dir / "hierarchy.json").read_text(encoding="utf-8"))
        components = json.loads((robot_dir / "components.json").read_text(encoding="utf-8"))
        mesh_catalog = json.loads((robot_dir / "mesh_catalog.json").read_text(encoding="utf-8"))
        mesh_files = {item["source"]["key"]: robot_dir / item["file"] for item in mesh_catalog}
        enabled = {
            item["owner_path"]: bool(item["values"].get("m_Enabled", True))
            for item in components
            if item["source"]["type"] == "MeshRenderer"
        }
        world_by_path = {
            item["hierarchy_path"]: np.asarray(
                item["world_prefab_space"]["matrix_row_major"], dtype=np.float32
            )
            for item in hierarchy
        }
        origin = np.asarray(self.spec["frame"]["origin_prefab_unity"], dtype=np.float32)

        visual_blocks: list[np.ndarray] = []
        for item in hierarchy:
            visual = item.get("visual") or {}
            mesh_ref = visual.get("mesh")
            path = item["hierarchy_path"]
            if not item.get("active", True) or not mesh_ref or not enabled.get(path, True):
                continue
            if "/physics" in path.lower() or "/preload" in path.lower():
                continue
            mesh_file = mesh_files.get(mesh_ref["key"])
            if mesh_file is None:
                continue
            mesh = json.loads(mesh_file.read_text(encoding="utf-8"))
            vertices = np.asarray(mesh["vertices"], dtype=np.float32)
            indices = np.asarray(
                [tri for submesh in mesh["submesh_triangles"] for tri in submesh],
                dtype=np.int32,
            )
            if indices.size:
                transformed = scene_mod.transform_points(vertices, world_by_path[path])
                visual_blocks.append(transformed[indices])

        collider_blocks: list[dict[str, Any]] = []
        for component in components:
            kind = component["source"]["type"]
            if kind not in {"BoxCollider", "SphereCollider", "CapsuleCollider"}:
                continue
            values, path = component["values"], component["owner_path"]
            if values.get("m_IsTrigger") or not values.get("m_Enabled", True):
                continue
            if path not in world_by_path:
                continue
            # Drive wheels become their own articulation links.
            if kind == "SphereCollider" and "/wheel" in path:
                continue
            center = values.get("m_Center", {"x": 0, "y": 0, "z": 0})
            center_v = np.array([center["x"], center["y"], center["z"]], dtype=np.float32)
            if kind == "BoxCollider":
                size = values["m_Size"]
                half = np.array([size["x"], size["y"], size["z"]], dtype=np.float32) * 0.5
                points = (
                    np.array(
                        [
                            [x, y, z]
                            for x in (-half[0], half[0])
                            for y in (-half[1], half[1])
                            for z in (-half[2], half[2])
                        ],
                        dtype=np.float32,
                    )
                    + center_v
                )
                faces = np.array(
                    [
                        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
                    ],
                    dtype=np.int32,
                )
                local_triangles = points[faces]
            elif kind == "SphereCollider":
                local_triangles = self.sb._shape_sphere(float(values["m_Radius"])) + center_v
            else:
                local_triangles = self.sb._shape_capsule(
                    float(values["m_Radius"]),
                    float(values["m_Height"]),
                    int(values["m_Direction"]),
                ) + center_v
            world_triangles = scene_mod.transform_points(
                local_triangles.reshape(-1, 3), world_by_path[path]
            ).reshape(-1, 3, 3)
            collider_blocks.append({
                "path": path,
                "triangles": scene_mod.unity_to_usd(world_triangles - origin),
                "containment": is_containment_collider_path(path),
            })

        raw_visual = np.concatenate(visual_blocks)
        visual_triangles = scene_mod.unity_to_usd(raw_visual - origin)
        return visual_triangles, collider_blocks

    @staticmethod
    def _box_tris(center: tuple[float, float, float], half: tuple[float, float, float]) -> np.ndarray:
        cx, cy, cz = center
        hx, hy, hz = half
        pts = np.array(
            [[x, y, z] for x in (cx - hx, cx + hx) for y in (cy - hy, cy + hy) for z in (cz - hz, cz + hz)],
            dtype=np.float32,
        )
        faces = np.array(
            [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
            dtype=np.int32,
        )
        return pts[faces]

    def _build_turret_visual(self) -> int:
        """Additive reference-style wide turret over the exact legacy shooter.

        Pure visual (no collision, no physics change): a brushed-aluminium base
        and hood framing a horizontal row of ``SHOOTER_BARRELS`` black barrels,
        centred on the extracted muzzle, so the wide multi-ball shooter reads
        clearly.  Coordinates are chassis-local Isaac axes (x fwd, y left, z up),
        matching shooter_muzzle_local.  Toggle with SceneBuilder.show_turret.
        """
        sb = self.sb
        if not getattr(sb, "show_turret", True):
            return 0
        metal = sb._preview_material("turret_metal", TURRET_METAL_RGBA)
        barrel = sb._preview_material("turret_barrel", TURRET_BARREL_RGBA)
        group = f"{self.chassis_path}/Turret254"

        def add(name: str, tris: np.ndarray, color: tuple, material: Any) -> None:
            prim = sb._mesh(f"{group}/{name}", tris, color)
            sb.UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        parts = 0
        add("Base", self._box_tris((-0.36, 0.0, 0.50), (0.11, 0.30, 0.05)), TURRET_METAL_RGBA, metal)
        add("Hood", self._box_tris((-0.30, 0.0, 0.70), (0.09, 0.30, 0.03)), TURRET_METAL_RGBA, metal)
        add("SideL", self._box_tris((-0.31, 0.29, 0.60), (0.10, 0.02, 0.11)), TURRET_METAL_RGBA, metal)
        add("SideR", self._box_tris((-0.31, -0.29, 0.60), (0.10, 0.02, 0.11)), TURRET_METAL_RGBA, metal)
        parts += 4
        for k in range(SHOOTER_BARRELS):
            offset = (k - (SHOOTER_BARRELS - 1) / 2.0) * SHOOTER_BARREL_SPACING_M
            add(f"Barrel_{k}", self._box_tris((-0.29, offset, 0.61), (0.055, 0.06, 0.10)),
                TURRET_BARREL_RGBA, barrel)
            parts += 1
        return parts

    def build(self) -> dict[str, Any]:
        from pxr import Gf, UsdGeom, UsdPhysics

        sb = self.sb
        stage = sb.stage
        stats: dict[str, Any] = {}

        root = UsdGeom.Xform.Define(stage, self.root_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*self.spawn))
        half_yaw = math.radians(self.spawn_yaw_deg) * 0.5
        root.AddOrientOp().Set(
            Gf.Quatf(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        )
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        physx_articulation = sb.PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        physx_articulation.CreateEnabledSelfCollisionsAttr(False)
        physx_articulation.CreateSolverPositionIterationCountAttr(16)
        physx_articulation.CreateSolverVelocityIterationCountAttr(4)

        # ---- chassis link ----------------------------------------------------
        chassis = UsdGeom.Xform.Define(stage, self.chassis_path)
        chassis_prim = chassis.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(chassis_prim)
        mass_api = UsdPhysics.MassAPI.Apply(chassis_prim)
        wheel_mass = 1.0
        chassis_mass = float(self.spec["total_mass_kg"]) - 6.0 * wheel_mass
        mass_api.CreateMassAttr(chassis_mass)
        com_unity = self.spec["drive_params"]["center_of_mass_unity_local"]
        # Unity (x, y, z) -> Isaac (x, -z, y)
        mass_api.CreateCenterOfMassAttr(
            Gf.Vec3f(float(com_unity[0]), -float(com_unity[2]), float(com_unity[1]) + 0.15)
        )
        sb.PhysxSchema.PhysxRigidBodyAPI.Apply(chassis_prim)

        visual_triangles, collider_blocks = self._visual_and_collider_geometry()
        visual_prim = sb._mesh(
            f"{self.chassis_path}/Visual", visual_triangles, sb_palette_robot()
        )
        sb.UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(
            sb._preview_material("robot", sb_palette_robot())
        )
        show_panels = getattr(sb, "show_panels", True)
        panel_material = sb._preview_material("containment_panel", CONTAINMENT_PANEL_RGBA)
        panel_count = 0
        for index, entry in enumerate(collider_blocks):
            block = entry["triangles"]
            collider_prim = sb._mesh(
                f"{self.chassis_path}/Colliders/C{index:02d}",
                block,
                (0.1, 0.8, 0.2, 0.16),
                collision=True,
            )
            UsdPhysics.MeshCollisionAPI.Apply(collider_prim).CreateApproximationAttr(
                "convexHull"
            )
            collider_prim.GetAttribute("visibility").Set(
                "inherited" if sb.debug_colliders else "invisible"
            )
            # Phase C2: visualise the solid containment colliders as translucent
            # polycarbonate panels.  Reuse the EXACT collider geometry; do NOT add
            # a second collision shape -- the collider above already provides it.
            if entry["containment"]:
                panel_prim = sb._mesh(
                    f"{self.chassis_path}/ContainmentPanels/Panel_{panel_count:02d}",
                    block,
                    CONTAINMENT_PANEL_RGBA,
                )
                sb.UsdShade.MaterialBindingAPI.Apply(panel_prim).Bind(panel_material)
                panel_prim.GetAttribute("visibility").Set(
                    "inherited" if show_panels else "invisible"
                )
                panel_count += 1
        stats["robot_chassis_colliders"] = len(collider_blocks)
        stats["robot_containment_panels"] = panel_count
        stats["robot_visual_triangles"] = len(visual_triangles)
        stats["robot_turret_parts"] = self._build_turret_visual()

        # ---- wheel material (rubber on carpet) -------------------------------
        wheel_material = sb.UsdShade.Material.Define(
            stage, "/World/PhysicsMaterials/WheelRubber"
        )
        wheel_material_api = UsdPhysics.MaterialAPI.Apply(wheel_material.GetPrim())
        wheel_material_api.CreateStaticFrictionAttr(1.1)
        wheel_material_api.CreateDynamicFrictionAttr(1.0)
        wheel_material_api.CreateRestitutionAttr(0.0)

        # ---- wheel links + revolute drives -----------------------------------
        self.wheel_joint_paths: list[str] = []
        self.wheel_sides: list[str] = []
        for wheel in self.spec["wheels"]:
            name = wheel["name"].replace("wheel", "")
            center = np.asarray(wheel["center_isaac"], dtype=np.float32)
            radius = float(wheel["radius_m"])
            link_path = f"{self.root_path}/wheel{name}"
            link = UsdGeom.Xform.Define(stage, link_path)
            # local to the (already translated + yawed) articulation root
            link.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
            link_prim = link.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(link_prim)
            UsdPhysics.MassAPI.Apply(link_prim).CreateMassAttr(wheel_mass)
            sb.PhysxSchema.PhysxRigidBodyAPI.Apply(link_prim)
            sphere = UsdGeom.Sphere.Define(stage, f"{link_path}/collider")
            sphere.CreateRadiusAttr(radius)
            sphere.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                [Gf.Vec3f(0.05, 0.05, 0.06)]
            )
            sphere_prim = sphere.GetPrim()
            UsdPhysics.CollisionAPI.Apply(sphere_prim)
            binding = sb.UsdShade.MaterialBindingAPI.Apply(sphere_prim)
            binding.Bind(wheel_material, materialPurpose="physics")
            sphere_prim.GetAttribute("visibility").Set("invisible")

            joint_path = f"{link_path}/drive_{name}"
            joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([self.chassis_path])
            joint.CreateBody1Rel().SetTargets([link_path])
            joint.CreateAxisAttr("Y")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, center)))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateDampingAttr(WHEEL_DRIVE_DAMPING)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateMaxForceAttr(WHEEL_DRIVE_TORQUE_NM)
            self.wheel_joint_paths.append(joint_path)
            self.wheel_sides.append(wheel["side"])

        stats["robot_wheels"] = len(self.spec["wheels"])
        stats["robot_mass_kg"] = float(self.spec["total_mass_kg"])
        stats["robot_articulated"] = True
        return stats


def sb_palette_robot() -> tuple[float, float, float, float]:
    from xrc_rebuilt.isaac_scene import PALETTE

    return PALETTE["robot"]


@dataclass
class ShooterInputs:
    """Everything the state machine needs to decide one control tick.

    Populated from the live sim by ``RobotController.update`` and directly by
    unit tests, so the firing logic is validated without launching Isaac.
    """

    magazine_count: int
    hub_active: bool
    shot_valid: bool
    yaw_error_deg: float
    chassis_speed_mps: float
    yaw_rate_dps: float
    muzzle_clear: bool
    blocked_reason: str = ""


class ShooterStateMachine:
    """Explicit shooter/indexer state machine (assistant_NEXT_PLAN Phase B).

    Transitions::

        IDLE -> ACQUIRE_TARGET -> TURNING -> BRAKING -> READY
        READY -> FEEDING -> COOLDOWN -> READY
        any -> BLOCKED (invalid route/range/obstacle/no ball/inactive HUB)
        any -> IDLE   (no fire request / emergency stop)

    The three xRC control modes share one latch set:

    * one-button shoot -> ``request_single`` queues exactly one feed;
    * continuous fire  -> ``set_continuous`` latches until toggled off/empty;
    * hold-to-fire      -> ``press_hold`` / ``release_hold`` (Button Y); release
      stops the request on the very next tick.

    ``tick`` never releases a ball itself; it returns ``should_feed`` and the
    caller performs the physical feed.  This keeps the GUI and RL paths on the
    identical decision logic (boundary #8).
    """

    def __init__(
        self,
        cooldown_s: float = SHOOTER_COOLDOWN_S,
        yaw_tolerance_deg: float = AUTO_ALIGN_TOLERANCE_DEG,
        max_speed_mps: float = FIRE_MAX_SPEED_MPS,
        max_yaw_rate_dps: float = FIRE_MAX_YAW_RATE_DPS,
    ):
        self.cooldown_s = float(cooldown_s)
        self.yaw_tolerance_deg = float(yaw_tolerance_deg)
        self.max_speed_mps = float(max_speed_mps)
        self.max_yaw_rate_dps = float(max_yaw_rate_dps)
        # request latches
        self.continuous = False
        self.auto_align = True
        self.emergency_stop = False
        self._hold = False
        self._queued_single = False
        # runtime
        self.state = ShooterState.IDLE
        self.last_feed_time = -1.0
        self.feeds = 0
        self.feed_times: list[float] = []
        self.last_blocked_reason = ""

    # -- request API (GUI buttons / RL macros) --------------------------------
    def request_single(self) -> None:
        """ONE-BUTTON SHOOT: queue exactly one shot; fires when the gate opens."""
        self._queued_single = True
        self.auto_align = True

    def set_continuous(self, on: bool) -> None:
        self.continuous = bool(on)
        if on:
            self.auto_align = True

    def toggle_continuous(self) -> None:
        self.set_continuous(not self.continuous)

    def press_hold(self) -> None:
        """Hold-to-fire down (xRC Button Y press)."""
        self._hold = True
        self.auto_align = True

    def release_hold(self) -> None:
        """Hold-to-fire up: request stops within one tick."""
        self._hold = False

    def toggle_auto_align(self) -> None:
        self.auto_align = not self.auto_align

    def set_emergency_stop(self, on: bool) -> None:
        self.emergency_stop = bool(on)

    def toggle_emergency_stop(self) -> None:
        self.emergency_stop = not self.emergency_stop

    def cancel(self) -> None:
        """Drop every fire request (not the emergency stop)."""
        self._queued_single = False
        self.continuous = False
        self._hold = False

    # -- queries --------------------------------------------------------------
    @property
    def wants_fire(self) -> bool:
        return (
            self._queued_single or self.continuous or self._hold
        ) and not self.emergency_stop

    def request_mode(self) -> str:
        if self.emergency_stop:
            return "ESTOP"
        if self._hold:
            return "HOLD"
        if self.continuous:
            return "CONTINUOUS"
        if self._queued_single:
            return "SINGLE"
        return "IDLE"

    def actual_rate_hz(self, now_s: float, window_s: float = 2.0) -> float:
        recent = [t for t in self.feed_times if now_s - t <= window_s]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 1e-9 else 0.0

    # -- core tick ------------------------------------------------------------
    def tick(self, inputs: ShooterInputs, now_s: float) -> dict[str, Any]:
        want = self.wants_fire
        # We run the aim pipeline whenever firing is requested or target lock is
        # engaged; otherwise the drivetrain is free for navigation.
        active = (want or self.auto_align) and not self.emergency_stop
        should_feed = False
        reason = ""

        if self.emergency_stop:
            self.state = ShooterState.IDLE
            reason = "emergency stop"
        elif not active:
            self.state = ShooterState.IDLE
        elif inputs.magazine_count <= 0:
            self.state = ShooterState.BLOCKED
            reason = "magazine empty"
        elif not inputs.hub_active:
            self.state = ShooterState.BLOCKED
            reason = "target HUB inactive"
        elif not inputs.shot_valid:
            self.state = ShooterState.BLOCKED
            reason = inputs.blocked_reason or "no feasible shot"
        else:
            aligned = abs(inputs.yaw_error_deg) <= self.yaw_tolerance_deg
            settled = (
                inputs.chassis_speed_mps <= self.max_speed_mps
                and abs(inputs.yaw_rate_dps) <= self.max_yaw_rate_dps
            )
            cooled = (now_s - self.last_feed_time) >= self.cooldown_s
            if self.state in (ShooterState.IDLE, ShooterState.BLOCKED):
                # one explicit acquisition tick before turning/braking
                self.state = ShooterState.ACQUIRE_TARGET
            elif not aligned:
                self.state = ShooterState.TURNING
            elif not settled:
                self.state = ShooterState.BRAKING
            elif not want:
                # aimed and settled but no active fire request -> armed
                self.state = ShooterState.READY
            elif not (cooled and inputs.muzzle_clear):
                self.state = ShooterState.COOLDOWN
            else:
                self.state = ShooterState.FEEDING
                should_feed = True
                self.last_feed_time = now_s
                self.feeds += 1
                self.feed_times.append(now_s)
                if self._queued_single:
                    self._queued_single = False

        self.last_blocked_reason = reason
        return {
            "state": self.state.value,
            "should_feed": should_feed,
            "request_mode": self.request_mode(),
            "blocked_reason": reason,
            "magazine_count": int(inputs.magazine_count),
            "feeds": self.feeds,
            "fire_rate_hz": round(self.actual_rate_hz(now_s), 3),
        }


class RobotController:
    """Arcade drive + ball systems for the articulated robot at runtime.

    Wraps an ``isaacsim.core.prims.SingleArticulation`` and a fuel
    ``RigidPrim`` view.  All ball handling mirrors xRC's trigger-volume
    behaviour: FUEL captured by the intake zones occupies the extracted
    physical preload positions inside the robot; firing releases the oldest
    ball at the moving muzzle with the exact Robot_RapidUS exit law.
    """

    def __init__(self, spec: dict[str, Any] | None = None):
        self.spec = spec or load_spec()
        self.articulation: Any = None
        self._wheel_indices_left: list[int] = []
        self._wheel_indices_right: list[int] = []
        self.wheel_radius = float(self.spec["wheels"][0]["radius_m"])
        zones = self.spec["zones"]
        self.intake_zones = [
            ZoneBox.from_spec(z) for z in zones if z["kind"] == "intake"
        ]
        # Feed lane that must be clear of the previous ball before the next feed
        # (xRC renderer-disabled keep-clear + muzzle trigger volumes).
        self.feed_keep_clear_zones = [
            ZoneBox.from_spec(z) for z in zones if z["kind"] in ("keep_clear", "muzzle")
        ]
        # Base = the exact 8 extracted indexer/preload positions; stack copies
        # upward into a full hopper of HOPPER_CAPACITY visible co-moving slots.
        base_slots = np.asarray(self.spec["preloads_isaac"], dtype=np.float32)
        layers = int(math.ceil(HOPPER_CAPACITY / max(1, len(base_slots))))
        self.preload_slots = np.concatenate(
            [base_slots + np.array([0.0, 0.0, k * HOPPER_LAYER_SPACING_M], dtype=np.float32)
             for k in range(layers)]
        )[:HOPPER_CAPACITY]
        self.magazine: list[int] = []
        self.pen_reserved: set[int] = set()
        self.last_shot_time = -1.0
        self.intake_on = False
        self.shots_fired = 0
        self.balls_collected = 0
        self.auto_aim_alliance: str | None = None
        self.last_aim_solution: dict[str, float | str | bool] = {}
        self._last_solve_position: np.ndarray | None = None
        self.state_machine = ShooterStateMachine(
            cooldown_s=float(self.spec.get("indexer_cooldown_s", SHOOTER_COOLDOWN_S))
        )
        # fired balls still inside the feed keep-clear volume
        self._muzzle_watch: set[int] = set()
        # balls released per shot (1 = pristine single-ball; >1 = wide turret)
        self.barrels = 1
        # drive-follow: latch into brake+rotate once near the firing pose
        self._arrival_latched = False

    # -- articulation hookup -------------------------------------------------
    def initialize(self, articulation: Any) -> None:
        """Hook up a started isaacsim SingleArticulation and set drive gains
        in runtime (radian) units, sidestepping USD's degree-based authoring."""
        self.articulation = articulation
        names = list(articulation.dof_names)
        # Robot frame: +x forward, +y left, +z up.  xRC's "L"-named wheels sit
        # at NEGATIVE y (the extraction mirrors handedness), so classify sides
        # by actual y position, not by name.
        y_by_name = {
            "drive_" + w["name"].replace("wheel", ""): float(w["center_isaac"][1])
            for w in self.spec["wheels"]
        }
        self._wheel_indices_left = [
            i for i, n in enumerate(names) if y_by_name.get(n, 0.0) > 0.0
        ]
        self._wheel_indices_right = [
            i for i, n in enumerate(names) if y_by_name.get(n, 0.0) < 0.0
        ]
        assert len(self._wheel_indices_left) == 3 and len(self._wheel_indices_right) == 3, names
        count = len(names)
        controller = articulation.get_articulation_controller()
        controller.set_gains(
            kps=np.zeros(count, dtype=np.float32),
            kds=np.full(count, WHEEL_DRIVE_DAMPING, dtype=np.float32),
        )
        try:
            controller.set_max_efforts(
                np.full(count, WHEEL_DRIVE_TORQUE_NM, dtype=np.float32)
            )
        except AttributeError:
            articulation.set_max_efforts(
                np.full(count, WHEEL_DRIVE_TORQUE_NM, dtype=np.float32)
            )

    # -- drive ----------------------------------------------------------------
    def drive(self, forward: float, turn: float) -> None:
        """Arcade drive; forward/turn in [-1, 1]."""
        from isaacsim.core.utils.types import ArticulationAction

        forward = float(np.clip(forward, -1.0, 1.0))
        turn = float(np.clip(turn, -1.0, 1.0)) * TURN_SPEED_SCALE
        # +turn = counter-clockwise (+yaw): left side slows, right side speeds up
        left = np.clip(forward - turn, -1.0, 1.0) * MAX_DRIVE_SPEED_MPS
        right = np.clip(forward + turn, -1.0, 1.0) * MAX_DRIVE_SPEED_MPS
        omega = np.zeros(len(self.articulation.dof_names), dtype=np.float32)
        for index in self._wheel_indices_left:
            omega[index] = left / self.wheel_radius
        for index in self._wheel_indices_right:
            omega[index] = right / self.wheel_radius
        self.articulation.apply_action(ArticulationAction(joint_velocities=omega))

    def chassis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = self.articulation.get_world_pose()
        to_np = lambda v: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return to_np(position).astype(np.float32), to_np(orientation).astype(np.float32)

    def chassis_yaw(self) -> float:
        _, orientation = self.chassis_pose()
        w, x, y, z = (float(v) for v in orientation)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # -- ball systems ----------------------------------------------------------
    def preload(self, fuel_view: Any, count: int = MAGAZINE_PRELOADS) -> None:
        """Place the first ``count`` FUEL bodies in visible xRC preload slots."""
        indices = list(range(min(count, fuel_view.count, len(self.preload_slots))))
        self.magazine = indices[:]
        self.pen_reserved = set(indices)
        self.sync_magazine(fuel_view)

    def sync_magazine(self, fuel_view: Any) -> None:
        """Keep stored balls visible and rigidly co-moving in their real slots."""
        if not self.magazine or self.articulation is None:
            return
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        slots = self.preload_slots[: len(self.magazine)]
        positions = position[None, :] + slots @ rotation.T
        index_array = np.asarray(self.magazine, dtype=np.int32)
        fuel_view.set_world_poses(positions=positions, indices=index_array)
        fuel_view.set_linear_velocities(
            np.zeros((len(index_array), 3), dtype=np.float32), indices=index_array
        )
        fuel_view.set_angular_velocities(
            np.zeros((len(index_array), 3), dtype=np.float32), indices=index_array
        )

    def step_intake(self, fuel_view: Any, hub_pending: set[int]) -> int:
        """Capture balls whose centers are inside any intake zone.

        Stored balls are re-synchronized to their visible physical slots.
        """
        self.sync_magazine(fuel_view)
        if not self.intake_on or self.articulation is None:
            return 0
        # The real robot becomes physically full when all eight extracted
        # preload locations are occupied; do not hide additional balls.
        if len(self.magazine) >= len(self.preload_slots):
            return 0
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        raw_positions, _ = fuel_view.get_world_poses()
        to_np = lambda v: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        ball_positions = to_np(raw_positions).astype(np.float32)
        # chassis-frame coordinates of every ball
        local = (ball_positions - position) @ rotation  # rotation columns = chassis axes
        captured = 0
        candidate_mask = np.zeros(len(ball_positions), dtype=bool)
        for zone in self.intake_zones:
            candidate_mask |= zone.contains(local, margin=0.076)
        for index in np.nonzero(candidate_mask)[0]:
            index = int(index)
            if index in self.pen_reserved or index in hub_pending:
                continue
            self.magazine.append(index)
            self.pen_reserved.add(index)
            captured += 1
            self.balls_collected += 1
            if len(self.magazine) >= len(self.preload_slots):
                break
        self.sync_magazine(fuel_view)
        return captured

    def _hub_target(self, alliance: str | None = None) -> tuple[str, np.ndarray]:
        if alliance in HUB_MARKBALL_TARGETS:
            return str(alliance), HUB_MARKBALL_TARGETS[str(alliance)].copy()
        position, _ = self.chassis_pose()
        selected = min(
            HUB_MARKBALL_TARGETS,
            key=lambda name: float(np.linalg.norm(HUB_MARKBALL_TARGETS[name][:2] - position[:2])),
        )
        return selected, HUB_MARKBALL_TARGETS[selected].copy()

    @staticmethod
    def _yaw_rotation(yaw: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def solve_auto_aim(self, alliance: str | None = None) -> dict[str, float | str | bool]:
        """Solve the one xRC shooter parameter and chassis heading to a HUB.

        The search uses the exact coupled speed/pitch law.  It does not invent
        an independent hood or flywheel control that legacy does not have.
        """
        position, _ = self.chassis_pose()
        selected, target = self._hub_target(alliance)
        if (
            self._last_solve_position is not None
            and self.last_aim_solution.get("alliance") == selected
            and float(np.linalg.norm(position - self._last_solve_position)) < 0.005
        ):
            cached = dict(self.last_aim_solution)
            yaw_error = wrap_angle(float(cached["desired_yaw_rad"]) - self.chassis_yaw())
            cached["yaw_error_deg"] = math.degrees(yaw_error)
            cached["aligned"] = abs(math.degrees(yaw_error)) <= AUTO_ALIGN_TOLERANCE_DEG
            self.last_aim_solution = cached
            return cached
        geometry = solve_shot_geometry(position, target)
        best: dict[str, float | str | bool] = {
            "alliance": selected,
            "aim": float(geometry["aim"]),
            "speed_mps": float(geometry["speed_mps"]),
            "pitch_deg": float(geometry["pitch_deg"]),
            "desired_yaw_rad": float(geometry["desired_yaw_rad"]),
            "range_m": float(geometry["range_m"]),
            "flight_time_s": float(geometry["flight_time_s"]),
            "vertical_error_m": float(geometry["vertical_error_m"]),
            "valid": abs(float(geometry["vertical_error_m"])) <= 0.025,
        }
        range_m = float(best["range_m"])
        on_open_side = (
            selected == "red" and float(position[1]) > float(target[1]) + 0.45
        ) or (
            selected == "blue" and float(position[1]) < float(target[1]) - 0.45
        )
        in_calibrated_range = CALIBRATED_RANGE_MIN_M <= range_m <= CALIBRATED_RANGE_MAX_M
        best["open_side"] = on_open_side
        best["in_calibrated_range"] = in_calibrated_range
        if not on_open_side:
            best["blocked_reason"] = "HUB structure blocks this side"
        elif not in_calibrated_range:
            best["blocked_reason"] = (
                f"move into {CALIBRATED_RANGE_MIN_M:.1f}-{CALIBRATED_RANGE_MAX_M:.1f} m range"
            )
        elif abs(float(best["vertical_error_m"])) > 0.025:
            best["blocked_reason"] = "no xRC trajectory within shooter limits"
        else:
            best["blocked_reason"] = ""
        best["valid"] = bool(
            abs(float(best["vertical_error_m"])) <= 0.025
            and on_open_side
            and in_calibrated_range
        )
        yaw_error = wrap_angle(float(best["desired_yaw_rad"]) - self.chassis_yaw())
        best["yaw_error_deg"] = math.degrees(yaw_error)
        best["aligned"] = abs(math.degrees(yaw_error)) <= AUTO_ALIGN_TOLERANCE_DEG
        self._last_solve_position = position.copy()
        self.last_aim_solution = best
        return best

    def auto_align(self, alliance: str | None = None) -> dict[str, float | str | bool]:
        """Rotate the six-wheel chassis onto the selected HUB bearing."""
        solution = self.solve_auto_aim(alliance)
        error = math.radians(float(solution["yaw_error_deg"]))
        if bool(solution["aligned"]):
            self.drive(0.0, 0.0)
        else:
            turn = float(np.clip(3.5 * error, -1.0, 1.0))
            # Wheel static friction otherwise leaves the robot stranded about
            # half a degree short of target.  The small floor command moves
            # roughly 0.16 degree per 20 ms control tick, below our gate.
            if abs(turn) < 0.08:
                turn = math.copysign(0.08, turn)
            self.drive(0.0, turn)
        return solution

    def fire_auto(self, fuel_view: Any, now_s: float, alliance: str | None = None) -> int | None:
        """Fire only when the physical xRC solution is valid and aligned."""
        solution = self.solve_auto_aim(alliance)
        if not bool(solution["valid"]) or not bool(solution["aligned"]):
            return None
        try:
            linear = self.articulation.get_linear_velocity()
            angular = self.articulation.get_angular_velocity()
            if hasattr(linear, "detach"):
                linear = linear.detach().cpu().numpy()
                angular = angular.detach().cpu().numpy()
            if float(np.linalg.norm(np.asarray(linear)[:2])) > 0.08:
                return None
            if abs(float(np.asarray(angular)[2])) > math.radians(3.0):
                return None
        except (AttributeError, RuntimeError):
            pass
        return self.fire(fuel_view, aim=float(solution["aim"]), now_s=now_s)

    def _chassis_velocity(self) -> np.ndarray:
        try:
            v = self.articulation.get_linear_velocity()
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return np.asarray(v, dtype=np.float32)
        except (AttributeError, RuntimeError, TypeError):
            return np.zeros(3, dtype=np.float32)

    def _release_ball(self, fuel_view: Any, index: int, muzzle_world: np.ndarray, velocity: np.ndarray) -> None:
        i = np.asarray([index], dtype=np.int32)
        fuel_view.set_world_poses(positions=muzzle_world[None, :].astype(np.float32), indices=i)
        fuel_view.set_linear_velocities(velocity[None, :].astype(np.float32), indices=i)
        fuel_view.set_angular_velocities(np.zeros((1, 3), dtype=np.float32), indices=i)

    def fire(self, fuel_view: Any, aim: float, now_s: float) -> int | None:
        """Release the oldest magazine ball at the muzzle. Returns ball index."""
        if not self.magazine or now_s - self.last_shot_time < SHOOTER_COOLDOWN_S:
            return None
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        index = self.magazine.pop(0)
        self.pen_reserved.discard(index)
        direction_world = rotation @ shooter_direction_local(aim)
        # spawn one ball-diameter past the muzzle so the shot never starts
        # interpenetrating the shooter colliders
        muzzle_world = position + rotation @ shooter_muzzle_local(aim) + direction_world * 0.09
        velocity = direction_world * shooter_exit_speed(aim) + self._chassis_velocity()
        self._release_ball(fuel_view, index, muzzle_world, velocity)
        self.last_shot_time = now_s
        self.shots_fired += 1
        return index

    def fire_volley(self, fuel_view: Any, aim: float, now_s: float) -> list[int]:
        """Release a horizontal row of up to ``self.barrels`` balls (reference
        style wide shooter).  Same shot law as ``fire``; the row is spread
        perpendicular to the shot direction in the horizontal plane.  Returns
        the fired ball indices (empty if empty magazine or within cooldown).
        """
        if not self.magazine or now_s - self.last_shot_time < SHOOTER_COOLDOWN_S:
            return []
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        direction_world = rotation @ shooter_direction_local(aim)
        muzzle_center = position + rotation @ shooter_muzzle_local(aim) + direction_world * 0.09
        velocity = direction_world * shooter_exit_speed(aim) + self._chassis_velocity()
        horizontal = np.cross(direction_world, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        norm = float(np.linalg.norm(horizontal))
        horizontal = horizontal / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        count = min(int(self.barrels), len(self.magazine))
        offsets = (np.arange(count) - (count - 1) / 2.0) * SHOOTER_BARREL_SPACING_M
        fired: list[int] = []
        for k in range(count):
            index = self.magazine.pop(0)
            self.pen_reserved.discard(index)
            self._release_ball(fuel_view, index, muzzle_center + horizontal * float(offsets[k]), velocity)
            fired.append(index)
        self.last_shot_time = now_s
        self.shots_fired += count
        return fired

    # -- FSM-driven aim + fire pipeline (shared by GUI and RL) ----------------
    def _sense_velocity(self) -> tuple[float, float]:
        """Chassis horizontal speed (m/s) and yaw rate (deg/s) from the sim."""
        try:
            linear = self.articulation.get_linear_velocity()
            angular = self.articulation.get_angular_velocity()
            to_np = lambda v: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            linear = np.asarray(to_np(linear), dtype=np.float32)
            angular = np.asarray(to_np(angular), dtype=np.float32)
            return float(np.linalg.norm(linear[:2])), math.degrees(abs(float(angular[2])))
        except (AttributeError, RuntimeError, TypeError, IndexError):
            return 0.0, 0.0

    def _update_muzzle_watch(self, fuel_view: Any) -> bool:
        """Drop fired balls once they leave the feed keep-clear volume.

        Returns True when the feed lane is clear for the next ball.
        """
        if not self._muzzle_watch or self.articulation is None:
            return True
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        raw, _ = fuel_view.get_world_poses()
        to_np = lambda v: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        balls = np.asarray(to_np(raw), dtype=np.float32)
        still: set[int] = set()
        for index in self._muzzle_watch:
            if index >= len(balls):
                continue
            local = (balls[index] - position) @ rotation
            if any(
                zone.contains(local[None, :], margin=0.01)[0]
                for zone in self.feed_keep_clear_zones
            ):
                still.add(index)
        self._muzzle_watch = still
        return not self._muzzle_watch

    def update(
        self,
        fuel_view: Any,
        now_s: float,
        alliance: str | None = None,
        hub_active: bool = True,
        allow_drive: bool = True,
    ) -> dict[str, Any]:
        """One control tick: solve aim, run the FSM, drive, and feed if ready.

        This is the single code path the GUI and RL both use so their
        drivetrain/aim/fire behaviour is identical (boundary #8).
        """
        muzzle_clear = self._update_muzzle_watch(fuel_view)
        solution = self.solve_auto_aim(alliance)
        speed, yaw_rate = self._sense_velocity()
        inputs = ShooterInputs(
            magazine_count=len(self.magazine),
            hub_active=bool(hub_active),
            shot_valid=bool(solution["valid"]),
            yaw_error_deg=float(solution["yaw_error_deg"]),
            chassis_speed_mps=speed,
            yaw_rate_dps=yaw_rate,
            muzzle_clear=muzzle_clear,
            blocked_reason=str(solution.get("blocked_reason", "")),
        )
        status = self.state_machine.tick(inputs, now_s)
        state = status["state"]
        if allow_drive:
            if state in (ShooterState.TURNING.value, ShooterState.ACQUIRE_TARGET.value):
                error = math.radians(inputs.yaw_error_deg)
                turn = float(np.clip(3.5 * error, -1.0, 1.0))
                if 0.0 < abs(turn) < 0.08:
                    turn = math.copysign(0.08, turn)
                self.drive(0.0, turn)
            elif state != ShooterState.IDLE.value:
                # BRAKING / READY / COOLDOWN / FEEDING / BLOCKED -> hold position
                self.drive(0.0, 0.0)
        fired_index = None
        if status["should_feed"]:
            if self.barrels > 1:
                fired = self.fire_volley(fuel_view, aim=float(solution["aim"]), now_s=now_s)
                self._muzzle_watch.update(fired)
                fired_index = fired[0] if fired else None
                status["fired_indices"] = fired
            else:
                fired_index = self.fire(fuel_view, aim=float(solution["aim"]), now_s=now_s)
                if fired_index is not None:
                    self._muzzle_watch.add(fired_index)
                status["fired_indices"] = [fired_index] if fired_index is not None else []
        status["fired_index"] = fired_index
        status["solution"] = solution
        return status

    def begin_follow(self) -> None:
        """Reset the drive-follow state before starting a new path."""
        self._arrival_latched = False

    def follow(self, path: list[tuple[float, float]], arrival_yaw: float,
               allow_drive: bool = True) -> dict[str, Any]:
        """Drive one differential pure-pursuit tick along ``path`` toward the
        firing pose, then brake and rotate in place to ``arrival_yaw``.  Once
        within 1.5x the arrival tolerance it latches into orient-only (braking
        translation) so it settles instead of orbiting the goal.  Returns the
        command dict (phase == 'arrived' when in position).  Shared by GUI/RL.
        """
        from xrc_rebuilt.shot_planner import ARRIVE_TOL_M, ARRIVE_YAW_TOL_DEG, pursuit_command

        position, _ = self.chassis_pose()
        x, y, yaw = float(position[0]), float(position[1]), self.chassis_yaw()
        dist_goal = math.hypot(path[-1][0] - x, path[-1][1] - y)
        if dist_goal <= ARRIVE_TOL_M * 1.5:
            self._arrival_latched = True

        if self._arrival_latched:
            yaw_error = wrap_angle(float(arrival_yaw) - yaw)
            if abs(math.degrees(yaw_error)) <= ARRIVE_YAW_TOL_DEG:
                if allow_drive:
                    self.drive(0.0, 0.0)
                return {"phase": "arrived", "dist_goal": dist_goal,
                        "yaw_err_deg": math.degrees(yaw_error), "forward": 0.0, "turn": 0.0}
            turn = float(np.clip(2.0 * yaw_error, -1.0, 1.0))
            turn = turn if abs(turn) >= 0.08 else math.copysign(0.08, turn)
            if allow_drive:
                self.drive(0.0, turn)  # brake translation, rotate in place
            return {"phase": "orienting", "dist_goal": dist_goal,
                    "yaw_err_deg": math.degrees(yaw_error), "forward": 0.0, "turn": turn}

        command = pursuit_command((x, y, yaw), path, float(arrival_yaw))
        if allow_drive:
            self.drive(float(command["forward"]), float(command["turn"]))
        return command

    def stats(self) -> dict[str, Any]:
        return {
            "magazine": len(self.magazine),
            "balls_collected": self.balls_collected,
            "shots_fired": self.shots_fired,
            "shooter_state": self.state_machine.state.value,
            "fire_mode": self.state_machine.request_mode(),
            "feeds": self.state_machine.feeds,
            "auto_aim": self.last_aim_solution,
        }
