#!/usr/bin/env python3
"""Fresh, deterministic UnityPy export of the exact ``Legacy Robot`` prefab.

This extractor intentionally reads only the pristine xRC installation.  It does
not consume any earlier extraction.  The export is simulator-agnostic JSON plus
PNG textures so downstream physics and rendering code can make its own choices.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler
from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator
from UnityPy.helpers.TypeTreeHelper import TypeTreeConfig, read_value
from UnityPy.streams import EndianBinaryReader


PROJECT = Path(__file__).resolve().parents[1]
XRC_DATA = Path(
    os.environ.get(
        "XRC_DATA_DIR",
        r"C:\Users\nickj\Desktop\xrc-vision-driver\vendor\xrc\app\xRC Simulator_Data",
    )
)
SOURCE = XRC_DATA / "resources.assets"
OUT = PROJECT / "assets" / "fresh_xrc" / "robot"
ROOT_NAME = "Legacy Robot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "unnamed"


def object_key(reader: Any) -> str:
    return f"{reader.assets_file.name}:{reader.path_id}"


def object_ref(reader: Any) -> dict[str, Any]:
    return {
        "key": object_key(reader),
        "asset_file": reader.assets_file.name,
        "path_id": reader.path_id,
        "type": reader.type.name,
    }


def jsonable(value: Any) -> Any:
    """Losslessly map UnityPy values into JSON-friendly structures."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"non_finite_float": repr(value)}
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii"), "byte_count": len(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    # attrs/dataclass-like Unity structures are slot based; annotations provide
    # their stable serialized field names.
    annotations: dict[str, Any] = {}
    for cls in reversed(type(value).__mro__):
        annotations.update(getattr(cls, "__annotations__", {}))
    if annotations:
        return {
            name: jsonable(getattr(value, name))
            for name in annotations
            if hasattr(value, name) and name not in {"object_reader", "assetsfile"}
        }
    if hasattr(value, "__dict__"):
        return {
            str(name): jsonable(item)
            for name, item in vars(value).items()
            if name not in {"object_reader", "assetsfile"} and not name.startswith("_")
        }
    return str(value)


def vec3(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def quat(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def mat_identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def trs_matrix(position: list[float], rotation: list[float], scale: list[float]) -> list[list[float]]:
    x, y, z, w = rotation
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    sx, sy, sz = scale
    return [
        [(1 - 2 * (y * y + z * z)) * sx, (2 * (x * y - z * w)) * sy, (2 * (x * z + y * w)) * sz, position[0]],
        [(2 * (x * y + z * w)) * sx, (1 - 2 * (x * x + z * z)) * sy, (2 * (y * z - x * w)) * sz, position[1]],
        [(2 * (x * z - y * w)) * sx, (2 * (y * z + x * w)) * sy, (1 - 2 * (x * x + y * y)) * sz, position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_summary(matrix: list[list[float]]) -> dict[str, Any]:
    scale = [math.sqrt(sum(matrix[row][col] ** 2 for row in range(3))) for col in range(3)]
    return {
        "matrix_row_major": matrix,
        "position": [matrix[0][3], matrix[1][3], matrix[2][3]],
        "axis_scale": scale,
        "note": "The matrix is authoritative; axis_scale is descriptive and does not discard possible shear.",
    }


def write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=None if compact else 2, separators=(",", ":") if compact else None)
        stream.write("\n")


def read_partial_monobehaviour(reader: Any, node: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Decode a valid prefix if Unity's generated schema no longer fits the blob.

    A few xRC editor-only ProBuilder scripts contain managed-reference records
    UnityPy cannot resolve, and Robot_RapidUS was serialized with a prior script
    layout.  This preserves every safely decoded value, the exact raw remainder,
    and the first incompatible field instead of guessing.
    """
    raw = reader.get_raw_data()
    stream = EndianBinaryReader(raw, endian=reader.reader.endian)
    config = TypeTreeConfig(True, reader.assets_file, False)
    values: dict[str, Any] = {}
    for child in node.m_Children:
        start = stream.Position
        try:
            values[child.m_Name] = read_value(child, stream, config)
        except Exception as exc:  # schema mismatch: do not consume uncertain bytes
            return values, {
                "failed_field": child.m_Name,
                "failed_type": child.m_Type,
                "byte_offset": start,
                "error": f"{type(exc).__name__}: {exc}",
                "remaining_raw_base64": base64.b64encode(raw[start:]).decode("ascii"),
                "remaining_byte_count": len(raw) - start,
            }
    if stream.Position != len(raw):
        return values, {
            "failed_field": None,
            "failed_type": None,
            "byte_offset": stream.Position,
            "error": "Generated schema ended before the serialized blob.",
            "remaining_raw_base64": base64.b64encode(raw[stream.Position:]).decode("ascii"),
            "remaining_byte_count": len(raw) - stream.Position,
        }
    return values, None


def mono_payload(reader: Any) -> dict[str, Any]:
    raw = reader.get_raw_data()
    payload: dict[str, Any] = {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_byte_count": len(raw),
    }
    try:
        head = reader.parse_monobehaviour_head()
        script = head.m_Script.deref_parse_as_object()
        namespace = script.m_Namespace or ""
        full_name = f"{namespace}.{script.m_ClassName}" if namespace else script.m_ClassName
        payload["script"] = {
            "class": script.m_ClassName,
            "namespace": namespace,
            "full_name": full_name,
            "assembly": script.m_AssemblyName,
            "mono_script": object_ref(script.object_reader),
        }
    except Exception as exc:
        payload["script_error"] = f"{type(exc).__name__}: {exc}"
        payload["raw_base64"] = base64.b64encode(raw).decode("ascii")
        return payload
    try:
        payload["values"] = jsonable(reader.read_typetree())
        payload["decode_status"] = "complete"
    except Exception as exc:
        node = reader.generate_monobehaviour_node()
        values, failure = read_partial_monobehaviour(reader, node)
        payload["values"] = jsonable(values)
        payload["decode_status"] = "safe_prefix_plus_raw_remainder"
        payload["decode_error"] = f"{type(exc).__name__}: {exc}"
        payload["schema_field_count"] = len(node.m_Children)
        payload["undecoded"] = failure
    return payload


def iter_tree(root_transform: Any) -> Iterable[tuple[Any, Any | None, int, str]]:
    stack = [(root_transform, None, 0, ROOT_NAME)]
    while stack:
        transform, parent, sibling_index, path = stack.pop()
        yield transform, parent, sibling_index, path
        children = [pointer.read() for pointer in transform.m_Children]
        child_names = [child.m_GameObject.read().m_Name for child in children]
        for index in reversed(range(len(children))):
            child = children[index]
            segment = f"{child_names[index]}[{index}]"
            stack.append((child, transform, index, f"{path}/{segment}"))


def export_mesh(reader: Any) -> dict[str, Any]:
    mesh = reader.read()
    handler = MeshHandler(mesh)
    handler.process()
    triangles = [[list(face) for face in submesh] for submesh in handler.get_triangles()]
    geometry = {
        "source": object_ref(reader),
        "name": mesh.m_Name,
        "vertex_count": handler.m_VertexCount,
        "vertices": handler.m_Vertices or [],
        "normals": handler.m_Normals or [],
        "tangents": handler.m_Tangents or [],
        "colors": handler.m_Colors or [],
        "uv": [getattr(handler, f"m_UV{i}") or [] for i in range(8)],
        "bone_indices": handler.m_BoneIndices or [],
        "bone_weights": handler.m_BoneWeights or [],
        "index_buffer": handler.m_IndexBuffer or [],
        "submesh_triangles": triangles,
        "submeshes": jsonable(mesh.m_SubMeshes),
        "local_aabb": jsonable(mesh.m_LocalAABB),
    }
    filename = f"{clean_name(reader.assets_file.name)}_{reader.path_id}_{clean_name(mesh.m_Name)}.json"
    write_json(OUT / "meshes" / filename, geometry, compact=True)
    return {
        "source": object_ref(reader),
        "name": mesh.m_Name,
        "file": f"meshes/{filename}",
        "vertex_count": handler.m_VertexCount,
        "triangle_count": sum(len(part) for part in triangles),
        "submesh_count": len(triangles),
    }


def export_material(reader: Any, texture_readers: dict[str, Any]) -> dict[str, Any]:
    material = reader.read()
    props = material.m_SavedProperties
    texture_slots = []
    for slot, env in props.m_TexEnvs:
        texture_ref = None
        if env.m_Texture.path_id:
            try:
                texture_reader = env.m_Texture.deref()
                texture_readers[object_key(texture_reader)] = texture_reader
                texture_ref = object_ref(texture_reader)
            except Exception as exc:
                texture_ref = {"file_id": env.m_Texture.file_id, "path_id": env.m_Texture.path_id, "error": str(exc)}
        texture_slots.append(
            {
                "slot": slot,
                "texture": texture_ref,
                "scale": jsonable(env.m_Scale),
                "offset": jsonable(env.m_Offset),
            }
        )
    try:
        shader_reader = material.m_Shader.deref()
        shader = object_ref(shader_reader)
        shader["name"] = getattr(shader_reader.read(), "m_Name", "")
    except Exception as exc:
        shader = {"file_id": material.m_Shader.file_id, "path_id": material.m_Shader.path_id, "error": str(exc)}
    data = {
        "source": object_ref(reader),
        "name": material.m_Name,
        "shader": shader,
        "render_queue": material.m_CustomRenderQueue,
        "string_tags": jsonable(material.stringTagMap),
        "disabled_shader_passes": jsonable(material.disabledShaderPasses),
        "colors": jsonable(props.m_Colors),
        "floats": jsonable(props.m_Floats),
        "ints": jsonable(props.m_Ints),
        "textures": texture_slots,
        "serialized": jsonable(reader.read_typetree()),
    }
    filename = f"{clean_name(reader.assets_file.name)}_{reader.path_id}_{clean_name(material.m_Name)}.json"
    write_json(OUT / "materials" / filename, data)
    return {"source": object_ref(reader), "name": material.m_Name, "file": f"materials/{filename}"}


def export_texture(reader: Any) -> dict[str, Any]:
    texture = reader.read()
    filename = f"{clean_name(reader.assets_file.name)}_{reader.path_id}_{clean_name(texture.m_Name)}.png"
    raw = reader.get_raw_data()
    entry = {
        "source": object_ref(reader),
        "name": texture.m_Name,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_byte_count": len(raw),
    }
    try:
        image = texture.image
        image.save(OUT / "textures" / filename, "PNG")
        entry.update({"file": f"textures/{filename}", "width": image.width, "height": image.height, "mode": image.mode})
    except Exception as exc:
        raw_filename = filename.removesuffix(".png") + ".serialized.bin"
        (OUT / "textures" / raw_filename).write_bytes(raw)
        entry["raw_file"] = f"textures/{raw_filename}"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Pristine source not found: {SOURCE}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for directory in (OUT, OUT / "meshes", OUT / "materials", OUT / "textures"):
        directory.mkdir(parents=True, exist_ok=True)

    environment = UnityPy.load(str(SOURCE))
    first_file = next(iter(environment.files.values()))
    generator = TypeTreeGenerator(first_file.unity_version)
    generator.load_local_dll_folder(str(XRC_DATA / "Managed"))
    environment.typetree_generator = generator

    matching = []
    for reader in environment.objects:
        if reader.type.name != "GameObject":
            continue
        try:
            if reader.read().m_Name == ROOT_NAME:
                matching.append(reader)
        except Exception:
            pass
    if len(matching) != 1:
        raise RuntimeError(f"Expected exactly one {ROOT_NAME!r}; found {len(matching)}")
    root_reader = matching[0]
    root_game_object = root_reader.read()
    root_transform = next(
        component.component.read()
        for component in root_game_object.m_Component
        if component.component.type.name in {"Transform", "RectTransform"}
    )

    hierarchy: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    component_counts: Counter[str] = Counter()
    mesh_readers: dict[str, Any] = {}
    material_readers: dict[str, Any] = {}
    world_by_transform: dict[int, list[list[float]]] = {}

    for transform, parent, sibling_index, path in iter_tree(root_transform):
        game_object = transform.m_GameObject.read()
        local_position = vec3(transform.m_LocalPosition)
        local_rotation = quat(transform.m_LocalRotation)
        local_scale = vec3(transform.m_LocalScale)
        local_matrix = trs_matrix(local_position, local_rotation, local_scale)
        parent_matrix = world_by_transform.get(parent.object_reader.path_id, mat_identity()) if parent else mat_identity()
        world_matrix = mat_mul(parent_matrix, local_matrix)
        world_by_transform[transform.object_reader.path_id] = world_matrix
        component_refs = []
        visual: dict[str, Any] = {}
        for item in game_object.m_Component:
            reader = item.component.deref()
            component_type = reader.type.name
            component_counts[component_type] += 1
            component_refs.append(object_ref(reader))
            record = {
                "source": object_ref(reader),
                "owner_game_object": object_ref(game_object.object_reader),
                "owner_path": path,
            }
            if component_type == "MonoBehaviour":
                record.update(mono_payload(reader))
            else:
                record["values"] = jsonable(reader.read_typetree())
            components.append(record)
            if component_type == "MeshFilter":
                pointer = reader.read().m_Mesh
                if pointer.path_id:
                    mesh_reader = pointer.deref()
                    mesh_readers[object_key(mesh_reader)] = mesh_reader
                    visual["mesh"] = object_ref(mesh_reader)
            elif component_type == "MeshRenderer":
                material_refs = []
                for pointer in reader.read().m_Materials:
                    if pointer.path_id:
                        material_reader = pointer.deref()
                        material_readers[object_key(material_reader)] = material_reader
                        material_refs.append(object_ref(material_reader))
                    else:
                        material_refs.append(None)
                visual["materials"] = material_refs
        hierarchy.append(
            {
                "game_object": object_ref(game_object.object_reader),
                "name": game_object.m_Name,
                "hierarchy_path": path,
                "sibling_index": sibling_index,
                "tag": getattr(game_object, "m_TagString", None),
                "layer": game_object.m_Layer,
                "active": bool(game_object.m_IsActive),
                "transform": object_ref(transform.object_reader),
                "parent_transform": object_ref(parent.object_reader) if parent else None,
                "children": [object_ref(pointer.deref()) for pointer in transform.m_Children],
                "local": {
                    "position": local_position,
                    "rotation_xyzw": local_rotation,
                    "scale": local_scale,
                    "matrix_row_major": local_matrix,
                },
                "world_prefab_space": matrix_summary(world_matrix),
                "components": component_refs,
                "visual": visual or None,
            }
        )

    mesh_catalog = [export_mesh(reader) for _, reader in sorted(mesh_readers.items())]
    texture_readers: dict[str, Any] = {}
    material_catalog = [
        export_material(reader, texture_readers) for _, reader in sorted(material_readers.items())
    ]
    texture_catalog = [export_texture(reader) for _, reader in sorted(texture_readers.items())]

    write_json(OUT / "hierarchy.json", hierarchy)
    write_json(OUT / "components.json", components)
    write_json(OUT / "mesh_catalog.json", mesh_catalog)
    write_json(OUT / "material_catalog.json", material_catalog)
    write_json(OUT / "texture_catalog.json", texture_catalog)

    complete_monos = sum(
        1 for component in components if component["source"]["type"] == "MonoBehaviour" and component.get("decode_status") == "complete"
    )
    partial_monos = sum(
        1
        for component in components
        if component["source"]["type"] == "MonoBehaviour" and component.get("decode_status") != "complete"
    )
    source_paths = [
        SOURCE,
        XRC_DATA / "resources.assets.resS",
        XRC_DATA / "globalgamemanagers.assets",
        XRC_DATA / "sharedassets0.assets",
        XRC_DATA / "sharedassets0.assets.resS",
        XRC_DATA / "Managed" / "Assembly-CSharp.dll",
        XRC_DATA / "Managed" / "Unity.ProBuilder.dll",
        XRC_DATA / "Managed" / "Unity.TextMeshPro.dll",
    ]
    provenance = {
        "extractor": "tools/export_legacy.py",
        "root_name": ROOT_NAME,
        "root": object_ref(root_reader),
        "unitypy_version": UnityPy.__version__,
        "unity_version": first_file.unity_version,
        "source_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
            if path.is_file()
        ],
        "counts": {
            "descendant_game_objects_including_root": len(hierarchy),
            "components": len(components),
            "components_by_type": dict(sorted(component_counts.items())),
            "unique_meshes": len(mesh_catalog),
            "mesh_vertices": sum(item["vertex_count"] for item in mesh_catalog),
            "mesh_triangles": sum(item["triangle_count"] for item in mesh_catalog),
            "unique_materials": len(material_catalog),
            "unique_textures": len(texture_catalog),
            "monobehaviours_complete": complete_monos,
            "monobehaviours_safe_prefix_plus_raw_remainder": partial_monos,
        },
        "coordinate_convention": {
            "source": "Unity left-handed coordinates, +Y up, quaternion [x,y,z,w]",
            "matrices": "row-major 4x4 matrices multiplying column vectors",
            "world_scope": "Prefab asset space; root local transform is included.",
        },
    }
    write_json(OUT / "provenance.json", provenance)

    # Hash the final artifacts after provenance itself is written.  The inventory
    # intentionally excludes itself to avoid a recursive hash definition.
    inventory = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "file_inventory.json":
            inventory.append(
                {"file": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_json(OUT / "file_inventory.json", inventory)
    print(json.dumps(provenance["counts"], indent=2))


if __name__ == "__main__":
    main()
