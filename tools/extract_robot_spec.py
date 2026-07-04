#!/usr/bin/env python3
"""Distill the raw Legacy Robot extraction into a runtime robot spec.

Reads ``assets/fresh_xrc/robot/{hierarchy,components,mesh_catalog}.json`` and
writes ``assets/fresh_xrc/robot/robot_spec.json`` containing everything the
Isaac robot builder and the RL environment need:

- the chassis frame (same origin convention as ``SceneBuilder.build_robot``:
  visual xz-midpoint, visual y-min in Unity prefab space),
- the six drive wheels (contact centers + effective radii, left/right side),
- the four suspension spring joints (recorded for reference),
- every ball-system trigger volume (intake, outtake, indexer, markball,
  keepClear) as an oriented box in the Isaac chassis frame,
- the eight PRELOAD positions,
- Robot_RapidUS drive/shooter parameters that are genuine prefab
  configuration (runtime-state fields are dropped).

Run with any Python that has numpy:
    C:/il/venv/Scripts/python.exe tools/extract_robot_spec.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROBOT_DIR = PROJECT / "assets" / "fresh_xrc" / "robot"

# Unity LH/Y-up -> Isaac RH/Z-up used across this repo: (x, y, z) -> (x, -z, y)
UNITY_TO_USD = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def load(name: str):
    return json.loads((ROBOT_DIR / name).read_text(encoding="utf-8"))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ matrix.T)[:, :3]


def visual_origin(hierarchy, components, mesh_files) -> np.ndarray:
    """Replicate SceneBuilder.build_robot's origin: xz-mid, y-min of visuals."""
    enabled = {
        item["owner_path"]: bool(item["values"].get("m_Enabled", True))
        for item in components
        if item["source"]["type"] == "MeshRenderer"
    }
    world = {
        item["hierarchy_path"]: np.asarray(
            item["world_prefab_space"]["matrix_row_major"], dtype=np.float64
        )
        for item in hierarchy
    }
    blocks = []
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
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        indices = np.asarray(
            [tri for submesh in mesh["submesh_triangles"] for tri in submesh],
            dtype=np.int64,
        )
        if indices.size:
            blocks.append(transform_points(vertices, world[path])[indices])
    points = np.concatenate(blocks).reshape(-1, 3)
    return np.array(
        [
            (points[:, 0].min() + points[:, 0].max()) * 0.5,
            points[:, 1].min(),
            (points[:, 2].min() + points[:, 2].max()) * 0.5,
        ],
        dtype=np.float64,
    )


def to_isaac(unity_points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(unity_points) - origin) @ UNITY_TO_USD.T


def matrix_scale(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix[:3, :3], axis=0)


def main() -> None:
    hierarchy = load("hierarchy.json")
    components = load("components.json")
    mesh_catalog = load("mesh_catalog.json")
    mesh_files = {item["source"]["key"]: ROBOT_DIR / item["file"] for item in mesh_catalog}
    world = {
        item["hierarchy_path"]: np.asarray(
            item["world_prefab_space"]["matrix_row_major"], dtype=np.float64
        )
        for item in hierarchy
    }
    origin = visual_origin(hierarchy, components, mesh_files)

    spec: dict = {
        "source": "assets/fresh_xrc/robot (extract_robot_spec.py)",
        "frame": {
            "convention": "Isaac RH/Z-up chassis frame; origin = visual xz-mid / y-min",
            "origin_prefab_unity": origin.tolist(),
        },
    }

    # --- wheels: the six SphereCollider drive wheels under NotUpdated/ -------
    wheels = []
    for component in components:
        path = component["owner_path"]
        if component["source"]["type"] != "SphereCollider" or "/wheel" not in path:
            continue
        values = component["values"]
        matrix = world[path]
        center_local = values.get("m_Center", {"x": 0, "y": 0, "z": 0})
        center_unity = transform_points(
            np.array([[center_local["x"], center_local["y"], center_local["z"]]]),
            matrix,
        )[0]
        radius = float(values["m_Radius"]) * float(matrix_scale(matrix).max())
        name = path.rsplit("/", 1)[-1].split("[")[0]  # e.g. wheelBR
        wheels.append(
            {
                "name": name,
                "center_isaac": np.round(to_isaac(center_unity, origin)[0], 6).tolist(),
                "radius_m": round(radius, 6),
            }
        )
    wheels.sort(key=lambda w: w["name"])
    # side classification: +y in Isaac chassis frame is one side, -y the other;
    # verified below against the L/R suffix in the xRC wheel names.
    for wheel in wheels:
        wheel["side"] = "left" if wheel["name"].endswith("L") else "right"
    spec["wheels"] = wheels

    # --- rigid bodies (mass bookkeeping) -------------------------------------
    bodies = []
    for component in components:
        if component["source"]["type"] != "Rigidbody":
            continue
        bodies.append(
            {
                "path": component["owner_path"],
                "mass_kg": float(component["values"].get("m_Mass", 0.0)),
            }
        )
    spec["rigidbodies"] = bodies
    spec["total_mass_kg"] = round(sum(body["mass_kg"] for body in bodies), 4)

    # --- suspension springs (reference only in v1) ----------------------------
    springs = []
    for component in components:
        if component["source"]["type"] != "ConfigurableJoint":
            continue
        path = component["owner_path"]
        if "WheelSpring" not in path:
            continue
        values = component["values"]
        springs.append(
            {
                "path": path,
                "linear_limit_m": float(values["m_LinearLimit"]["limit"]),
                "spring": float(values["m_LinearLimitSpring"]["spring"]),
                "damper": float(values["m_LinearLimitSpring"]["damper"]),
                "y_drive": values.get("m_YDrive"),
            }
        )
    spec["wheel_springs"] = springs

    # --- ball-system trigger volumes ------------------------------------------
    def zone_kind(path: str) -> str | None:
        lowered = path.lower()
        if "bs intake" in lowered:
            return "intake"
        if "bs outtake (3)" in lowered:
            return "shooter_feed"
        if "bs outtake" in lowered:
            return "transfer"
        if "markball" in lowered:
            return "muzzle"
        if "/indexer" in lowered:
            return "indexer"
        if "keepclear" in lowered:
            return "keep_clear"
        return None

    zones = []
    for component in components:
        if component["source"]["type"] != "BoxCollider":
            continue
        values, path = component["values"], component["owner_path"]
        kind = zone_kind(path)
        if kind is None or not values.get("m_IsTrigger"):
            continue
        matrix = world[path]
        center_local = values.get("m_Center", {"x": 0, "y": 0, "z": 0})
        size_local = values["m_Size"]
        center_unity = transform_points(
            np.array([[center_local["x"], center_local["y"], center_local["z"]]]),
            matrix,
        )[0]
        # oriented box: world axis directions and world half extents
        axes_unity = matrix[:3, :3] / matrix_scale(matrix)[None, :]
        half_unity = (
            np.array([size_local["x"], size_local["y"], size_local["z"]]) * 0.5
        ) * matrix_scale(matrix)
        axes_isaac = UNITY_TO_USD @ axes_unity
        zones.append(
            {
                "kind": kind,
                "path": path,
                "enabled": bool(values.get("m_Enabled", True)),
                "center_isaac": np.round(to_isaac(center_unity, origin)[0], 6).tolist(),
                "axes_isaac_columns": np.round(axes_isaac, 6).tolist(),
                "half_extents_m": np.round(half_unity, 6).tolist(),
                "bs_speed_mps": float(component.get("_speed", 0.0)) or None,
            }
        )
    # attach ballshooting_v2 speeds by owner path
    bs_by_path = {
        c["owner_path"]: c["values"]
        for c in components
        if c["source"]["type"] == "MonoBehaviour"
        and c["script"]["full_name"] == "ballshooting_v2"
    }
    for zone in zones:
        values = bs_by_path.get(zone["path"])
        if values:
            zone["bs_speed_mps"] = float(values.get("speed", 0.0)) * float(
                values.get("speedMultiplier", 1.0)
            )
            zone["bs_use_force"] = bool(values.get("use_force"))
            zone["bs_force_divider"] = float(values.get("force_divider", 1.0))
            zone["bs_hard_stop"] = bool(values.get("hard_stop"))
            zone["bs_disabled_by_default"] = bool(values.get("disable"))
    spec["zones"] = zones

    # --- preloads ---------------------------------------------------------------
    preloads = []
    for item in hierarchy:
        path = item["hierarchy_path"]
        if "/PRELOADS" in path and "/PRELOAD " in path:
            matrix = np.asarray(
                item["world_prefab_space"]["matrix_row_major"], dtype=np.float64
            )
            preloads.append(
                np.round(to_isaac(matrix[:3, 3], origin)[0], 6).tolist()
            )
    spec["preloads_isaac"] = preloads

    # --- Robot_RapidUS configuration -------------------------------------------
    rapid = next(
        c["values"]
        for c in components
        if c["source"]["type"] == "MonoBehaviour"
        and c["script"]["full_name"] == "Robot_RapidUS"
    )
    keep = (
        "use_new_algorithm",
        "friction_static",
        "friction_speed",
        "orth_torque_scaler",
        "orth_friction_static",
        "orth_friction_speed",
        "min_ang_velocity",
        "apply_friction_clamp",
        "friction_clamp_safety",
        "TankMotorScaler",
        "SixWheelMotorScaler",
        "turn_priority",
        "rot_inertia_scaler",
        "is_FRC",
    )
    spec["drive_params"] = {key: rapid.get(key) for key in keep}
    spec["drive_params"]["center_of_mass_unity_local"] = [
        rapid["centerOfMass"]["x"],
        rapid["centerOfMass"]["y"],
        rapid["centerOfMass"]["z"],
    ]

    # Shooter law and final accelerator axis decoded from the live legacy prefab.
    spec["shooter"] = {
        "power_law": "exit_speed_mps = 6 + 5*aim  (aim in [0,1])",
        "angle_law": "BS Outtake (3) pitch_deg = 70.355276 - 20*aim",
        "notes": "Rotate the muzzle and final accelerator about the physical shooter pivot.",
    }

    # indexer cooldown from GenericIndexerMulti
    indexer = next(
        (
            c["values"]
            for c in components
            if c["source"]["type"] == "MonoBehaviour"
            and c["script"]["full_name"] == "GenericIndexerMulti"
        ),
        None,
    )
    if indexer:
        spec["indexer_cooldown_s"] = float(indexer.get("cooldown_period", 0.07))

    out = ROBOT_DIR / "robot_spec.json"
    out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(f"origin_prefab_unity = {np.round(origin, 4).tolist()}")
    for wheel in wheels:
        print(
            f"  {wheel['name']:8s} side={wheel['side']:5s} r={wheel['radius_m']:.4f} "
            f"center_isaac={wheel['center_isaac']}"
        )
    print(f"zones: {[(z['kind'], z['path'][-24:]) for z in zones]}")
    print(f"preloads: {len(preloads)}  total_mass={spec['total_mass_kg']} kg")


if __name__ == "__main__":
    main()
