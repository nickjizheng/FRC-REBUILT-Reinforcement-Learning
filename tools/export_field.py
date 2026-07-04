"""Export the pristine xRC REBUILT scenes into an Isaac-friendly archive.

The exporter reads the original Unity player data directly.  It preserves the
rendered triangle geometry, static-batch slices, world-space colliders, rigid
bodies, joints, triggers, and named gameplay scripts.  No previous project
artifacts are inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler


def ptr_id(value: Any) -> int | None:
    return getattr(value, "path_id", None) or getattr(value, "m_PathID", None)


def ptr_file(value: Any) -> int:
    return int(getattr(value, "file_id", 0) or getattr(value, "m_FileID", 0) or 0)


def vec3(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([value.get("x", 0), value.get("y", 0), value.get("z", 0)], dtype=np.float64)
    return np.array([getattr(value, "x", 0), getattr(value, "y", 0), getattr(value, "z", 0)], dtype=np.float64)


def quat(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([value.get("x", 0), value.get("y", 0), value.get("z", 0), value.get("w", 1)], dtype=np.float64)
    return np.array([getattr(value, "x", 0), getattr(value, "y", 0), getattr(value, "z", 0), getattr(value, "w", 1)], dtype=np.float64)


def quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float64,
    )


def local_matrix(transform: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quat_matrix(quat(transform.m_LocalRotation)) * vec3(transform.m_LocalScale)
    result[:3, 3] = vec3(transform.m_LocalPosition)
    return result


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def decompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = matrix[:3, 3].copy()
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    safe = np.where(scale < 1e-12, 1, scale)
    rotation = matrix[:3, :3] / safe
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
        scale[0] *= -1
    return position, rotation, scale


def matrix_quat(rotation: np.ndarray) -> list[float]:
    # Stable matrix -> xyzw conversion.
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        values = ((rotation[2, 1]-rotation[1, 2])/s, (rotation[0, 2]-rotation[2, 0])/s, (rotation[1, 0]-rotation[0, 1])/s, 0.25*s)
    else:
        i = int(np.argmax(np.diag(rotation)))
        if i == 0:
            s = math.sqrt(1 + rotation[0,0]-rotation[1,1]-rotation[2,2]) * 2
            values = (0.25*s, (rotation[0,1]+rotation[1,0])/s, (rotation[0,2]+rotation[2,0])/s, (rotation[2,1]-rotation[1,2])/s)
        elif i == 1:
            s = math.sqrt(1 + rotation[1,1]-rotation[0,0]-rotation[2,2]) * 2
            values = ((rotation[0,1]+rotation[1,0])/s, 0.25*s, (rotation[1,2]+rotation[2,1])/s, (rotation[0,2]-rotation[2,0])/s)
        else:
            s = math.sqrt(1 + rotation[2,2]-rotation[0,0]-rotation[1,1]) * 2
            values = ((rotation[0,2]+rotation[2,0])/s, (rotation[1,2]+rotation[2,1])/s, 0.25*s, (rotation[1,0]-rotation[0,1])/s)
    return [round(float(v), 8) for v in values]


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    path_id = ptr_id(value)
    if path_id is not None:
        return {"file_id": ptr_file(value), "path_id": path_id}
    if hasattr(value, "__dict__"):
        return {str(k): json_value(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(path: str) -> str:
    text = path.lower()
    if text.startswith("centergoal"):
        if "/physics/net" in text or "/visuals/" in text and text.endswith("/net"):
            return "hub_net"
        return "hub"
    if text.startswith("ramp"):
        return "bump"
    if text.startswith(("biggate", "movinggate")):
        return "trench"
    if text.startswith("hang"):
        return "tower"
    if "visiontag" in text:
        return "apriltag"
    if text.startswith(("warehouse", "wharehouse", "depot")):
        return "source"
    if text.startswith("ballreturn"):
        return "return"
    if "tapeline" in text or "floortile" in text or text.endswith("/floor"):
        return "floor"
    if "wall" in text or "playerstation" in text:
        return "wall"
    return "field"


class Scene:
    def __init__(self, path: Path):
        self.path = path
        self.env = UnityPy.load(str(path))
        self.raw = {obj.path_id: obj for obj in self.env.objects}
        self.types = {obj.path_id: obj.type.name for obj in self.env.objects}
        self.gos: dict[int, Any] = {}
        self.transforms: dict[int, Any] = {}
        self.go_transform: dict[int, int] = {}
        self.transform_go: dict[int, int] = {}
        self.components: dict[int, list[tuple[str, int]]] = {}
        self._world: dict[int, np.ndarray] = {}
        self._path: dict[int, str] = {}
        self._mesh: dict[tuple[int, int], tuple[Any, np.ndarray, list[Any]]] = {}
        for obj in self.env.objects:
            try:
                if obj.type.name == "GameObject":
                    go = obj.read()
                    self.gos[obj.path_id] = go
                    items = []
                    for item in getattr(go, "m_Component", []) or []:
                        pointer = getattr(item, "component", item)
                        item_id = ptr_id(pointer)
                        if item_id is not None:
                            items.append((self.types.get(item_id, "Unknown"), item_id))
                    self.components[obj.path_id] = items
                elif obj.type.name == "Transform":
                    transform = obj.read()
                    self.transforms[obj.path_id] = transform
                    go_id = ptr_id(transform.m_GameObject)
                    if go_id is not None:
                        self.go_transform[go_id] = obj.path_id
                        self.transform_go[obj.path_id] = go_id
            except Exception:
                continue

    def name(self, go_id: int | None) -> str:
        return str(getattr(self.gos.get(go_id), "m_Name", "?"))

    def world(self, transform_id: int) -> np.ndarray:
        if transform_id not in self._world:
            result = local_matrix(self.transforms[transform_id])
            parent = ptr_id(self.transforms[transform_id].m_Father)
            if parent in self.transforms:
                result = self.world(parent) @ result
            self._world[transform_id] = result
        return self._world[transform_id]

    def path_for(self, transform_id: int) -> str:
        if transform_id not in self._path:
            names, current, seen = [], transform_id, set()
            while current in self.transforms and current not in seen:
                seen.add(current)
                names.append(self.name(self.transform_go.get(current)))
                current = ptr_id(self.transforms[current].m_Father)
            self._path[transform_id] = "/".join(reversed(names))
        return self._path[transform_id]

    def component(self, go_id: int, kind: str) -> int | None:
        return next((component_id for component_kind, component_id in self.components.get(go_id, []) if component_kind == kind), None)

    def mesh(self, pointer: Any) -> tuple[Any, np.ndarray, list[Any]]:
        key = (ptr_file(pointer), int(ptr_id(pointer) or 0))
        if key not in self._mesh:
            mesh = pointer.read()
            handler = MeshHandler(mesh)
            handler.process()
            self._mesh[key] = (mesh, np.asarray(handler.m_Vertices, dtype=np.float32), handler.get_triangles())
        return self._mesh[key]


def material_color(renderer: Any) -> list[float]:
    for pointer in getattr(renderer, "m_Materials", []) or []:
        try:
            material = pointer.read()
            saved = getattr(material, "m_SavedProperties", None)
            colors = dict(getattr(saved, "m_Colors", []) or [])
            for key in ("_BaseColor", "_Color"):
                if key in colors:
                    color = colors[key]
                    return [float(getattr(color, axis, default)) for axis, default in (("r", .7), ("g", .7), ("b", .7), ("a", 1))]
        except Exception:
            continue
    return [.7, .7, .7, 1]


def export_visuals(scene: Scene) -> tuple[np.ndarray, list[dict[str, Any]]]:
    blocks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    triangle_cursor = 0
    for go_id, transform_id in scene.go_transform.items():
        filter_id = scene.component(go_id, "MeshFilter")
        renderer_id = scene.component(go_id, "MeshRenderer")
        if filter_id is None or renderer_id is None:
            continue
        try:
            mesh_filter = scene.raw[filter_id].read()
            renderer = scene.raw[renderer_id].read()
            mesh, vertices, submeshes = scene.mesh(mesh_filter.m_Mesh)
        except Exception:
            continue
        static = getattr(renderer, "m_StaticBatchInfo", None)
        first = int(getattr(static, "firstSubMesh", 0) or 0)
        count = int(getattr(static, "subMeshCount", 0) or 0)
        indices = range(first, min(first + count, len(submeshes))) if count else range(len(submeshes))
        world_vertices = vertices if count else transform_points(vertices, scene.world(transform_id).astype(np.float32))
        local_blocks = []
        for submesh_index in indices:
            triangles = np.asarray(submeshes[submesh_index], dtype=np.int32)
            if triangles.size:
                local_blocks.append(world_vertices[triangles])
        if not local_blocks:
            continue
        block = np.concatenate(local_blocks).astype(np.float32)
        blocks.append(block)
        path = scene.path_for(transform_id)
        records.append({
            "level": scene.path.name,
            "path": path,
            "category": category(path),
            "mesh": str(getattr(mesh, "m_Name", "?")),
            "enabled": bool(getattr(renderer, "m_Enabled", True)),
            "active_self": bool(getattr(scene.gos[go_id], "m_IsActive", True)),
            "triangle_start": triangle_cursor,
            "triangle_count": len(block),
            "rgba": material_color(renderer),
            "static_batch": {"first_submesh": first, "submesh_count": count},
        })
        triangle_cursor += len(block)
    return (np.concatenate(blocks) if blocks else np.zeros((0, 3, 3), dtype=np.float32), records)


def export_physics(scene: Scene) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    colliders, mesh_blocks, bodies, joints, scripts = [], [], [], [], []
    mesh_cursor = 0
    collider_types = {"BoxCollider", "SphereCollider", "CapsuleCollider", "MeshCollider"}
    joint_types = {"HingeJoint", "ConfigurableJoint", "FixedJoint", "SpringJoint", "CharacterJoint"}
    for go_id, components in scene.components.items():
        transform_id = scene.go_transform.get(go_id)
        if transform_id is None:
            continue
        world = scene.world(transform_id)
        position, rotation, scale = decompose(world)
        path = scene.path_for(transform_id)
        for kind, component_id in components:
            if kind in collider_types:
                try:
                    value = scene.raw[component_id].read()
                except Exception:
                    continue
                center_local = vec3(getattr(value, "m_Center", None))
                center_world = transform_points(center_local.reshape(1, 3), world)[0]
                record: dict[str, Any] = {
                    "level": scene.path.name, "path": path, "component_id": component_id, "type": kind,
                    "enabled": bool(getattr(value, "m_Enabled", True)), "trigger": bool(getattr(value, "m_IsTrigger", False)),
                    "center": center_world.tolist(), "rotation_xyzw": matrix_quat(rotation), "scale": scale.tolist(),
                }
                if kind == "BoxCollider":
                    record["size"] = (vec3(value.m_Size) * np.abs(scale)).tolist()
                elif kind == "SphereCollider":
                    record["radius"] = float(value.m_Radius * np.max(np.abs(scale)))
                elif kind == "CapsuleCollider":
                    record.update(radius=float(value.m_Radius * max(abs(scale[0]), abs(scale[2]))), height=float(value.m_Height * abs(scale[int(value.m_Direction)])), direction=int(value.m_Direction))
                elif kind == "MeshCollider":
                    try:
                        mesh, vertices, submeshes = scene.mesh(value.m_Mesh)
                        transformed = transform_points(vertices, world.astype(np.float32))
                        tris = [transformed[np.asarray(items, dtype=np.int32)] for items in submeshes if len(items)]
                        block = np.concatenate(tris).astype(np.float32) if tris else np.zeros((0, 3, 3), dtype=np.float32)
                        mesh_blocks.append(block)
                        record.update(mesh_name=str(getattr(mesh, "m_Name", "?")), triangle_start=mesh_cursor, triangle_count=len(block), convex=bool(getattr(value, "m_Convex", False)))
                        mesh_cursor += len(block)
                    except Exception as error:
                        record["mesh_error"] = str(error)
                colliders.append(record)
            elif kind == "Rigidbody":
                try:
                    value = scene.raw[component_id].read()
                    linear_damping = getattr(value, "m_LinearDamping", None)
                    if linear_damping is None:
                        linear_damping = getattr(value, "m_Drag", 0) or 0
                    angular_damping = getattr(value, "m_AngularDamping", None)
                    if angular_damping is None:
                        angular_damping = getattr(value, "m_AngularDrag", 0) or 0
                    bodies.append({"level": scene.path.name, "path": path, "component_id": component_id, "mass": float(value.m_Mass), "linear_damping": float(linear_damping), "angular_damping": float(angular_damping), "use_gravity": bool(value.m_UseGravity), "kinematic": bool(value.m_IsKinematic), "constraints": int(value.m_Constraints), "center_of_mass": vec3(value.m_CenterOfMass).tolist(), "inertia_tensor": vec3(value.m_InertiaTensor).tolist(), "inertia_rotation_xyzw": quat(value.m_InertiaRotation).tolist()})
                except Exception:
                    continue
            elif kind in joint_types:
                try:
                    value = scene.raw[component_id].read()
                    joints.append({"level": scene.path.name, "path": path, "component_id": component_id, "type": kind, "values": json_value(vars(value))})
                except Exception:
                    continue
            elif kind == "MonoBehaviour":
                try:
                    raw = scene.raw[component_id]
                    value = raw.read(check_read=False)
                    script_name = str(getattr(value.m_Script.read(), "m_Name", "?"))
                    if script_name in {"WallBallReturn", "ReleaseBallDetector", "ballshooting_v2", "FlagGameElement", "MarkBall", "OutpostReloader", "GE_Counter"}:
                        try:
                            values = raw.read_typetree(check_read=False)
                        except Exception:
                            values = vars(value)
                        scripts.append({"level": scene.path.name, "path": path, "component_id": component_id, "script": script_name, "values": json_value(values)})
                except Exception:
                    continue
    mesh_triangles = np.concatenate(mesh_blocks) if mesh_blocks else np.zeros((0, 3, 3), dtype=np.float32)
    return colliders, [mesh_triangles], bodies, joints, scripts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [Scene(args.data_dir / name) for name in ("level74", "level75", "level76")]
    render_triangles, visuals = export_visuals(scenes[0])
    all_colliders: list[dict[str, Any]] = []
    collider_meshes: list[np.ndarray] = []
    bodies: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    scripts: list[dict[str, Any]] = []
    for scene in scenes:
        scene_colliders, scene_meshes, scene_bodies, scene_joints, scene_scripts = export_physics(scene)
        # Mesh triangle ranges were local to each level; offset before append.
        offset = sum(len(block) for block in collider_meshes)
        for collider in scene_colliders:
            if "triangle_start" in collider:
                collider["triangle_start"] += offset
        all_colliders.extend(scene_colliders)
        collider_meshes.extend(scene_meshes)
        bodies.extend(scene_bodies)
        joints.extend(scene_joints)
        scripts.extend(scene_scripts)
    collider_triangles = np.concatenate(collider_meshes) if collider_meshes else np.zeros((0, 3, 3), dtype=np.float32)

    np.savez_compressed(args.out_dir / "field_meshes.npz", render_triangles=render_triangles, collider_triangles=collider_triangles)
    for name, value in (("visuals", visuals), ("colliders", all_colliders), ("rigidbodies", bodies), ("joints", joints), ("gameplay_scripts", scripts)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

    manifest = {
        "source": str(args.data_dir.resolve()),
        "source_sha256": {name: sha256(args.data_dir / name) for name in ("level74", "level75", "level76")},
        "unity_version": "6000.4.11f1",
        "render_triangles": len(render_triangles),
        "visual_instances": len(visuals),
        "visual_categories": dict(Counter(item["category"] for item in visuals)),
        "colliders": len(all_colliders),
        "collider_types": dict(Counter(item["type"] for item in all_colliders)),
        "triggers": sum(item["trigger"] for item in all_colliders),
        "rigidbodies": len(bodies),
        "joints": len(joints),
        "gameplay_scripts": dict(Counter(item["script"] for item in scripts)),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
