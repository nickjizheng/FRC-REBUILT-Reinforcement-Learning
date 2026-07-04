#!/usr/bin/env python3
"""Split the exact reference eDrawings export into independently animated groups."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SCRATCH = (
    Path.home() / "AppData/Local/Temp/assistant/C--Users-nickj-Desktop-xrc-rebuilt-robot-rl"
    / "44cbb605-f319-4ed0-8722-92dc466c60e6/scratchpad"
)
OUT = PROJECT / "assets/robot_runtime/mechanisms"
OX, OY, GROUND_Z = 0.14102080464363098, -0.0027344077825546265, 0.025993309915065765


def load_triangles(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.triangles, dtype=np.float32)


def triangle_hash(triangles: np.ndarray) -> np.ndarray:
    quantized = np.rint(np.asarray(triangles, dtype=np.float64) * 1.0e4).astype(np.int64)
    vertex_hash = (
        (quantized[:, :, 0] * 73856093)
        ^ (quantized[:, :, 1] * 19349663)
        ^ (quantized[:, :, 2] * 83492791)
    )
    vertex_hash.sort(axis=1)
    return (
        (vertex_hash[:, 0] * 1000003)
        ^ (vertex_hash[:, 1] * 9176)
        ^ (vertex_hash[:, 2] * 6361)
    )


def robot_frame(vertices: np.ndarray) -> np.ndarray:
    source = np.asarray(vertices, dtype=np.float64)
    return np.column_stack(
        (-source[:, 1] - OX, source[:, 0] - OY, source[:, 2] - GROUND_Z)
    )


def save_group(name: str, triangles: np.ndarray, target_faces: int) -> dict[str, object]:
    vertices = robot_frame(triangles.reshape(-1, 3))
    faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=False)
    mesh.remove_unreferenced_vertices()
    source_faces = len(mesh.faces)
    if source_faces > target_faces:
        vertices, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=target_faces,
            agg=5.0,
            verbose=False,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=False)
        mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", vertices=vertices, faces=faces)
    return {
        "name": name,
        "source_faces": source_faces,
        "runtime_faces": len(faces),
        "runtime_vertices": len(vertices),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
    }


def save_robot_mesh(name: str, mesh: trimesh.Trimesh, target_faces: int) -> dict[str, object]:
    """Save an already robot-frame mesh, optionally simplifying it."""
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    source_faces = len(mesh.faces)
    if source_faces > target_faces:
        vertices, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=target_faces,
            agg=5.0,
            verbose=False,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=False)
        mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    np.savez_compressed(OUT / f"{name}.npz", vertices=vertices, faces=faces)
    return {
        "name": name,
        "source_faces": source_faces,
        "runtime_faces": len(faces),
        "runtime_vertices": len(vertices),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
    }


def load_robot_mesh(name: str) -> trimesh.Trimesh:
    part = np.load(OUT / f"{name}.npz")
    return trimesh.Trimesh(part["vertices"], part["faces"], process=False)


def combine_components(components: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not components:
        raise ValueError("cannot combine an empty component list")
    return trimesh.util.concatenate(components)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    sources = {
        "full": args.scratch / "reference_robot_full.stl",
        "full_hopper": args.scratch / "Full_Hopper.stl",
        "intake": args.scratch / "Intake.stl",
        "intake_power": args.scratch / "Intake_Power.stl",
        "hopper_base": args.scratch / "Hopper_Base_group.stl",
        "hopper_horizontal": args.scratch / "Hopper_Horizontal_group.stl",
        "hopper_vertical": args.scratch / "Hopper_Vertical_group.stl",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing mechanism exports: " + ", ".join(missing))

    full = load_triangles(sources["full"])
    # Subtract EVERY separately-rendered group from the static chassis, not just
    # the full-hopper envelope.  full_hopper misses ~21k top-board faces (its
    # tessellation of that region differs from the full model), so those faces
    # used to survive in static as a frozen replica of the moving top container.
    # The per-group exports hash-match the full model there, so unioning them in
    # removes the duplicate cleanly.
    remove_hashes = np.unique(
        np.concatenate(
            [triangle_hash(load_triangles(sources["full_hopper"])),
             triangle_hash(load_triangles(sources["intake"])),
             triangle_hash(load_triangles(sources["intake_power"])),
             triangle_hash(load_triangles(sources["hopper_base"])),
             triangle_hash(load_triangles(sources["hopper_horizontal"])),
             triangle_hash(load_triangles(sources["hopper_vertical"]))]
        )
    )
    static_mask = ~np.isin(triangle_hash(full), remove_hashes)

    # --- Geometric touch-ups the per-assembly STL exports get wrong ---
    from scipy.spatial import cKDTree

    cen = robot_frame(full.reshape(-1, 3)).reshape(-1, 3, 3).mean(axis=1)
    nrm = np.cross(full[:, 1] - full[:, 0], full[:, 2] - full[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    # robot frame is (x,y,z) -> (-y,x,z), so the robot-frame Y normal is STL nx.
    robot_ny = nrm[:, 0]

    # Four near-zero-thickness horizontal remnants are duplicate sheet edges
    # left by mismatched STL tessellation.  Remove them before any mechanism
    # reassignment so they cannot migrate into the intake or hopper groups.
    split_sliver_mask = (
        static_mask
        & (np.abs(np.abs(cen[:, 1]) - 0.347) < 0.007)
        & ((np.abs(cen[:, 2] - 0.187) < 0.007) | (np.abs(cen[:, 2] - 0.533) < 0.007))
        & (cen[:, 0] > -0.10) & (cen[:, 0] < 0.52)
    )
    static_mask &= ~split_sliver_mask

    def _grp_centroids(name: str) -> np.ndarray:
        tri = load_triangles(sources[name])
        return robot_frame(tri.reshape(-1, 3)).reshape(-1, 3, 3).mean(axis=1)

    # (1) Some top-container faces survive in static because their tessellation
    # matches no group export's hash.  Drop static faces up high that sit right
    # on the moving vertical group (<30mm) - the frozen top-board fragments.
    vert_tree = cKDTree(_grp_centroids("hopper_vertical"))
    top_cand = np.where(static_mask & (cen[:, 2] > 0.55))[0]
    d_top, _ = vert_tree.query(cen[top_cand])
    static_mask[top_cand[d_top < 0.030]] = False

    # (2) The two outer side boards belong to the moving front container but the
    # export left them in static, so they stayed put while the middle board
    # retracted.  Reassign the outward-facing side-wall faces that are connected
    # (<50mm) to hopper_horizontal so they retract with it.
    hh_tree = cKDTree(_grp_centroids("hopper_horizontal"))
    side_cand = np.where(
        static_mask
        & (np.abs(robot_ny) > 0.7)
        & (np.abs(cen[:, 1]) >= 0.31)
        & (cen[:, 0] > 0.03) & (cen[:, 0] < 0.52)
        & (cen[:, 2] > 0.14) & (cen[:, 2] < 0.55)
    )[0]
    d_side, _ = hh_tree.query(cen[side_cand])
    side_mask = np.zeros(len(full), dtype=bool)
    side_mask[side_cand[d_side < 0.050]] = True
    static_mask &= ~side_mask
    side_boards = full[side_mask]

    # (3) The folding intake mechanism has many faces the export left in static,
    # so those parts never swung with the rest of the intake.  Reassign the
    # static faces coincident (<10mm) with the intake group inside its envelope.
    intake_tree = cKDTree(
        np.concatenate(
            [_grp_centroids("intake"), _grp_centroids("intake_power")], axis=0
        )
    )
    intake_cand = np.where(
        static_mask
        & (cen[:, 0] > -0.03) & (cen[:, 0] < 0.55)
        & (cen[:, 2] > 0.03) & (cen[:, 2] < 0.26)
        & (np.abs(cen[:, 1]) < 0.37)
    )[0]
    d_int, _ = intake_tree.query(cen[intake_cand])
    intake_mask = np.zeros(len(full), dtype=bool)
    intake_mask[intake_cand[d_int < 0.018]] = True

    # The low cross-robot roller and several front linkages are separate intake
    # components, not children of either exported intake subassembly.  Move the
    # complete low/front envelope instead of leaving a frozen roller ahead of
    # the bumper.  The fixed chassis crossbar ends at x<0.217 and is excluded.
    intake_front_mask = (
        static_mask
        & (cen[:, 0] > 0.217) & (cen[:, 0] < 0.52)
        & (cen[:, 2] > 0.025) & (cen[:, 2] < 0.27)
        & (np.abs(cen[:, 1]) < 0.37)
    )
    intake_mask |= intake_front_mask
    static_mask &= ~intake_mask
    intake_extra = full[intake_mask]

    print(
        f"reassigned: top-frag dropped, side_boards={int(side_mask.sum())}, "
        f"intake_extra={int(intake_mask.sum())}, slivers={int(split_sliver_mask.sum())}"
    )

    records = [save_group("static_chassis", full[static_mask], 850_000)]
    for name, target in (("hopper_base", 150_000), ("hopper_vertical", 100_000)):
        records.append(save_group(name, load_triangles(sources[name]), target))
    # intake keeps its exported faces plus the reassigned folding parts.
    intake_triangles = np.concatenate(
        [
            load_triangles(sources["intake"]),
            load_triangles(sources["intake_power"]),
            intake_extra,
        ],
        axis=0,
    )
    records.append(save_group("intake", intake_triangles, 200_000))
    # hopper_horizontal keeps its exported faces plus the reassigned side boards.
    hopper_horizontal_triangles = np.concatenate(
        [load_triangles(sources["hopper_horizontal"]), side_boards], axis=0
    )
    records.append(save_group("hopper_horizontal", hopper_horizontal_triangles, 60_000))

    # Component-level correction after tessellation:
    # - the two broad, thin triangular feeder plates were included by the
    #   Intake Power export but are fixed internal structure, not part of the
    #   folding intake.  Keep them on HopperBase so they cannot rotate through
    #   the real rollers;
    # - the high rear shooter crown is the only remaining structure above the
    #   trench envelope.  Isolate complete connected CAD components so a real
    #   retract joint can lower the exact geometry (no visual clipping trick).
    intake_mesh = load_robot_mesh("intake")
    intake_parts = list(intake_mesh.split(only_watertight=False))
    fixed_intake_parts = [
        part for part in intake_parts
        if (
            # Central triangular plates, bearings and power hardware stay on the
            # fixed hopper.  Do not also classify the broad x=0.02..0.50 wire
            # frame as fixed: its long curved run belongs to the folding intake.
            part.bounds[1, 0] < 0.215
            and abs(float(part.centroid[1])) < 0.25
            and part.bounds[1, 2] < 0.235
        )
    ]
    fixed_intake_ids = {id(part) for part in fixed_intake_parts}
    moving_intake_parts = [part for part in intake_parts if id(part) not in fixed_intake_ids]
    hopper_base_mesh = combine_components(
        [load_robot_mesh("hopper_base"), *fixed_intake_parts]
    )

    static_mesh = load_robot_mesh("static_chassis")
    static_parts = list(static_mesh.split(only_watertight=False))
    shooter_parts = [
        part for part in static_parts
        if part.bounds[1, 2] > 0.535 and part.bounds[0, 0] < -0.15
    ]
    shooter_ids = {id(part) for part in shooter_parts}
    fixed_static_parts = [part for part in static_parts if id(part) not in shooter_ids]

    replacements = {
        "static_chassis": save_robot_mesh(
            "static_chassis", combine_components(fixed_static_parts), 850_000
        ),
        "intake": save_robot_mesh(
            "intake", combine_components(moving_intake_parts), 200_000
        ),
        "hopper_base": save_robot_mesh("hopper_base", hopper_base_mesh, 180_000),
        "shooter_retract": save_robot_mesh(
            "shooter_retract", combine_components(shooter_parts), 140_000
        ),
    }
    records = [replacements.get(record["name"], record) for record in records]
    records.append(replacements["shooter_retract"])
    print(
        f"component correction: fixed_intake={sum(len(p.faces) for p in fixed_intake_parts)}, "
        f"shooter_retract={sum(len(p.faces) for p in shooter_parts)}"
    )

    metadata = {
        "source_files": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in sources.items()
        },
        "static_raw_faces": int(np.count_nonzero(static_mask)),
        "removed_raw_faces": int(np.count_nonzero(~static_mask)),
        "groups": records,
        "transform": {"robot_xyz": "(-source_y-OX, source_x-OY, source_z-GROUND_Z)"},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
