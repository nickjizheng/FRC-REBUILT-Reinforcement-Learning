#!/usr/bin/env python3
"""Headlessly convert the competition robot's large STEP assembly into a light sim mesh.

The original SolidWorks AP203 file is intentionally never copied or modified.
OpenCascade imports and tessellates it without creating any desktop/GDI CAD
windows, then trimesh merges duplicate vertices and fast-simplification reduces
the display mesh to a bounded triangle count.

Coordinate convention used by the runtime:
  STEP X -> robot +Y (left)
  STEP Y -> robot -X (the long intake end is robot-forward)
  STEP Z -> robot +Z (up)

Outputs:
  assets/robot_runtime/visual_mesh.npz  float32 vertices + int32 triangle indices
  assets/robot_runtime/visual_mesh.glb  portable inspection copy
  assets/robot_runtime/cad_metadata.json hashes, transforms, bounds and mesh counts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Desktop" / "Reference Robotapril2519.STEP"
DEFAULT_OUTPUT = PROJECT / "assets" / "robot"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_topology(shape, kind) -> int:
    from OCP.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, kind)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def load_step(path: Path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OpenCascade could not read {path}: {status}")
    roots = int(reader.NbRootsForTransfer())
    if roots < 1:
        raise RuntimeError(f"STEP contains no transferable roots: {path}")
    reader.TransferRoots()
    return reader.OneShape(), roots


def shape_bounds(shape) -> tuple[float, float, float, float, float, float]:
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    return tuple(float(value) for value in box.Get())


def tessellate(shape, linear_deflection_mm: float, angular_deflection_rad: float):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    mesher = BRepMesh_IncrementalMesh(
        shape,
        float(linear_deflection_mm),
        False,
        float(angular_deflection_rad),
        True,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade tessellation did not complete")

    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    vertex_offset = 0
    skipped_faces = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            skipped_faces += 1
            explorer.Next()
            continue
        transform = location.Transformation()
        vertices = np.empty((triangulation.NbNodes(), 3), dtype=np.float64)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(transform)
            vertices[index - 1] = (point.X(), point.Y(), point.Z())
        faces = np.empty((triangulation.NbTriangles(), 3), dtype=np.int32)
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for index in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(index).Get()
            faces[index - 1] = (
                (a - 1, c - 1, b - 1) if reversed_face else (a - 1, b - 1, c - 1)
            )
        vertex_blocks.append(vertices)
        face_blocks.append(faces + vertex_offset)
        vertex_offset += len(vertices)
        explorer.Next()

    if not face_blocks:
        raise RuntimeError("STEP tessellation produced no triangles")
    return np.concatenate(vertex_blocks), np.concatenate(face_blocks), skipped_faces


def step_to_robot(vertices_mm: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Convert mm STEP axes and center the wheelbase-sized body footprint."""
    cad_min = vertices_mm.min(axis=0)
    cad_max = vertices_mm.max(axis=0)

    # The export is approximately symmetric laterally. Along STEP Y the intake
    # creates a long negative-side overhang; use the middle of the full envelope
    # as a deterministic first-pass origin and record it for visual adjustment.
    origin_x_mm = 0.5 * (cad_min[0] + cad_max[0])
    origin_y_mm = 0.5 * (cad_min[1] + cad_max[1])
    ground_z_mm = cad_min[2]
    relative = vertices_mm - np.array([origin_x_mm, origin_y_mm, ground_z_mm])
    robot = np.column_stack((-relative[:, 1], relative[:, 0], relative[:, 2])) * 0.001
    return robot.astype(np.float64), {
        "cad_origin_x_mm": float(origin_x_mm),
        "cad_origin_y_mm": float(origin_y_mm),
        "cad_ground_z_mm": float(ground_z_mm),
    }


def clean_and_simplify(vertices: np.ndarray, faces: np.ndarray, target_faces: int):
    import fast_simplification
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=False)
    mesh.remove_unreferenced_vertices()
    source_faces = len(mesh.faces)
    if source_faces > target_faces:
        vertices_out, faces_out = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=int(target_faces),
            agg=5.0,
            verbose=False,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices_out,
            faces=faces_out,
            process=True,
            validate=False,
        )
        mesh.remove_unreferenced_vertices()
    return mesh, source_faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--linear-deflection-mm", type=float, default=1.5)
    parser.add_argument("--angular-deflection-rad", type=float, default=0.35)
    parser.add_argument("--target-faces", type=int, default=180_000)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source_hash = sha256_file(source)
    print(f"robot_STEP source={source} bytes={source.stat().st_size} sha256={source_hash}", flush=True)
    shape, roots = load_step(source)
    imported_s = time.perf_counter() - started
    print(f"robot_STEP imported roots={roots} seconds={imported_s:.2f}", flush=True)

    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

    solid_count = count_topology(shape, TopAbs_SOLID)
    brep_face_count = count_topology(shape, TopAbs_FACE)
    bounds_mm = shape_bounds(shape)
    print(
        f"robot_STEP topology solids={solid_count} brep_faces={brep_face_count} bounds_mm={bounds_mm}",
        flush=True,
    )

    tess_started = time.perf_counter()
    raw_vertices, raw_faces, skipped_faces = tessellate(
        shape, args.linear_deflection_mm, args.angular_deflection_rad
    )
    print(
        f"robot_STEP tessellated vertices={len(raw_vertices)} triangles={len(raw_faces)} "
        f"skipped_faces={skipped_faces} seconds={time.perf_counter()-tess_started:.2f}",
        flush=True,
    )

    robot_vertices, transform = step_to_robot(raw_vertices)
    simplify_started = time.perf_counter()
    mesh, merged_face_count = clean_and_simplify(
        robot_vertices, raw_faces, max(10_000, args.target_faces)
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    bounds_robot_m = np.concatenate((vertices.min(axis=0), vertices.max(axis=0))).tolist()
    print(
        f"robot_STEP optimized merged_triangles={merged_face_count} vertices={len(vertices)} "
        f"triangles={len(faces)} seconds={time.perf_counter()-simplify_started:.2f}",
        flush=True,
    )

    npz_path = output / "visual_mesh.npz"
    glb_path = output / "visual_mesh.glb"
    np.savez_compressed(npz_path, vertices=vertices, faces=faces)
    mesh.export(glb_path)

    metadata = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "step_schema": "AP203 CONFIG_CONTROL_DESIGN",
        "step_authoring": "SolidWorks 2025 / SwSTEP 2.0",
        "roots": roots,
        "solid_count": solid_count,
        "brep_face_count": brep_face_count,
        "bounds_cad_mm": list(bounds_mm),
        "axis_mapping": {"robot_x": "-STEP_Y", "robot_y": "+STEP_X", "robot_z": "+STEP_Z"},
        "origin": transform,
        "linear_deflection_mm": args.linear_deflection_mm,
        "angular_deflection_rad": args.angular_deflection_rad,
        "raw_tessellated_vertices": len(raw_vertices),
        "raw_tessellated_triangles": len(raw_faces),
        "merged_triangles": merged_face_count,
        "optimized_vertices": len(vertices),
        "optimized_triangles": len(faces),
        "bounds_robot_m": bounds_robot_m,
        "npz": str(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "glb": str(glb_path),
        "glb_bytes": glb_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "cad_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("robot_STEP_DONE " + json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
