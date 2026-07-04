#!/usr/bin/env python3
"""Phase C1 bodywork audit of the pristine Legacy Robot prefab.

Reads the extracted ground truth (``assets/fresh_xrc/robot``) and produces, for
every object under the containment-relevant subtrees, a machine-readable record
of: hierarchy path, active state (self + effective), MeshRenderer presence and
enabled state, mesh key, world transform, material(s), and every collider's
type / trigger flag / enabled flag / dimensions / world AABB.

Outputs:
  runs/bodywork_audit.json     full per-object table + summary
  docs/legacy_BODYWORK_AUDIT.md  human-readable tables and the key finding

No Isaac required.  Run:  py -3.12 tools/audit_bodywork.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROBOT = PROJECT / "assets" / "fresh_xrc" / "robot"

# Containment-relevant subtrees named in assistant_NEXT_PLAN Phase C1.
ROOTS = [
    "Legacy Robot/Body[0]/Visuals[6]/MainBody[3]",
    "Legacy Robot/Body[0]/Visuals[6]/MainBody[3]/Intake[4]",
    "Legacy Robot/Body[0]/Visuals[6]/MainBody[3]/Hopper[5]",
    "Legacy Robot/Body[0]/Visuals[6]/Basket[7]",
    "Legacy Robot/Body[0]/Physics[7]/IntakeHopper[5]",
    "Legacy Robot/Body[0]/Physics[7]/Body[6]",
]
COLLIDER_TYPES = {"BoxCollider", "SphereCollider", "CapsuleCollider"}


def unity_to_usd(p: np.ndarray) -> list[float]:
    return [float(p[0]), float(-p[2]), float(p[1])]


def transform(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (homogeneous @ mat.T)[:, :3]


def collider_corners(kind: str, values: dict) -> np.ndarray:
    center = values.get("m_Center", {"x": 0, "y": 0, "z": 0})
    c = np.array([center["x"], center["y"], center["z"]], float)
    if kind == "BoxCollider":
        s = values["m_Size"]
        half = np.array([s["x"], s["y"], s["z"]], float) * 0.5
    elif kind == "SphereCollider":
        r = float(values["m_Radius"])
        half = np.array([r, r, r], float)
    else:  # CapsuleCollider
        r = float(values["m_Radius"])
        h = float(values["m_Height"]) * 0.5
        direction = int(values.get("m_Direction", 1))
        half = np.array([r, r, r], float)
        half[direction] = max(r, h)
    return np.array(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1])
         for z in (-half[2], half[2])], float
    ) + c


def main() -> None:
    hierarchy = json.loads((ROBOT / "hierarchy.json").read_text(encoding="utf-8"))
    components = json.loads((ROBOT / "components.json").read_text(encoding="utf-8"))
    materials = json.loads((ROBOT / "material_catalog.json").read_text(encoding="utf-8"))
    mat_name = {int(m["source"]["path_id"]): m["name"] for m in materials}

    active_self = {it["hierarchy_path"]: bool(it.get("active", True)) for it in hierarchy}

    def active_effective(path: str) -> bool:
        segments = path.split("/")
        for depth in range(1, len(segments) + 1):
            prefix = "/".join(segments[:depth])
            if prefix in active_self and not active_self[prefix]:
                return False
        return True

    comps_by_owner: dict[str, list[dict]] = {}
    for comp in components:
        comps_by_owner.setdefault(comp["owner_path"], []).append(comp)

    records: list[dict] = []
    for it in hierarchy:
        path = it["hierarchy_path"]
        under = [r for r in ROOTS if path == r or path.startswith(r + "/")]
        if not under:
            continue
        owned = comps_by_owner.get(path, [])
        renderer = next((c for c in owned if c["source"]["type"] == "MeshRenderer"), None)
        mat_names = []
        if renderer:
            for m in renderer["values"].get("m_Materials", []):
                pid = int(m.get("m_PathID", 0))
                if pid:
                    mat_names.append(mat_name.get(pid, f"pathid:{pid}"))
        visual = it.get("visual") or {}
        mesh_key = (visual.get("mesh") or {}).get("key")
        mat = np.array(it["world_prefab_space"]["matrix_row_major"], float)
        world_pos = np.array([mat[0][3], mat[1][3], mat[2][3]], float)

        colliders = []
        for comp in owned:
            kind = comp["source"]["type"]
            if kind not in COLLIDER_TYPES:
                continue
            values = comp["values"]
            corners_world = transform(mat, collider_corners(kind, values))
            lo, hi = corners_world.min(0), corners_world.max(0)
            entry = {
                "type": kind,
                "is_trigger": bool(values.get("m_IsTrigger", False)),
                "enabled": bool(values.get("m_Enabled", True)),
                "center_local": values.get("m_Center"),
                "world_aabb_min_unity": [round(float(v), 5) for v in lo],
                "world_aabb_max_unity": [round(float(v), 5) for v in hi],
                "world_size_m": [round(float(v), 5) for v in (hi - lo)],
            }
            if kind == "BoxCollider":
                entry["size"] = values.get("m_Size")
            else:
                entry["radius"] = values.get("m_Radius")
                if kind == "CapsuleCollider":
                    entry["height"] = values.get("m_Height")
                    entry["direction"] = values.get("m_Direction")
            colliders.append(entry)

        records.append({
            "path": path,
            "roots": under,
            "active_self": active_self.get(path, True),
            "active_effective": active_effective(path),
            "has_renderer": renderer is not None,
            "renderer_enabled": bool(renderer["values"].get("m_Enabled")) if renderer else False,
            "materials": mat_names,
            "mesh_key": mesh_key,
            "world_pos_unity": [round(float(v), 5) for v in world_pos],
            "world_pos_isaac": [round(v, 5) for v in unity_to_usd(world_pos)],
            "colliders": colliders,
        })

    # ---- global claims the plan asks us to verify --------------------------
    all_renderers = [c for c in components if c["source"]["type"] == "MeshRenderer"]
    active_enabled_renderers = [
        c for c in all_renderers
        if c["values"].get("m_Enabled") and active_effective(c["owner_path"])
        and (next((h for h in hierarchy if h["hierarchy_path"] == c["owner_path"]), {}).get("visual") or {}).get("mesh")
    ]
    exported_like = [
        c for c in active_enabled_renderers
        if "/physics" not in c["owner_path"].lower() and "/preload" not in c["owner_path"].lower()
    ]
    excluded = [c for c in active_enabled_renderers if c not in exported_like]

    def solid_colliders(root: str) -> list[dict]:
        out = []
        for rec in records:
            if root not in rec["roots"]:
                continue
            for col in rec["colliders"]:
                if not col["is_trigger"] and col["enabled"]:
                    out.append({"path": rec["path"], "renderer_enabled": rec["renderer_enabled"], **col})
        return out

    containment = {
        "IntakeHopper[5]": solid_colliders("Legacy Robot/Body[0]/Physics[7]/IntakeHopper[5]"),
        "Body[6]": solid_colliders("Legacy Robot/Body[0]/Physics[7]/Body[6]"),
    }

    summary = {
        "roots": ROOTS,
        "objects_audited": len(records),
        "active_enabled_meshrenderers_total": len(active_enabled_renderers),
        "of_which_visual_exported": len(exported_like),
        "of_which_excluded_physics_or_preload": len(excluded),
        "containment_solid_colliders": {k: len(v) for k, v in containment.items()},
    }

    out = {"summary": summary, "containment_solid_colliders": containment, "objects": records}
    (PROJECT / "runs" / "bodywork_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("BODYWORK_AUDIT", json.dumps(summary, indent=2))
    write_markdown(out)


def write_markdown(out: dict) -> None:
    s = out["summary"]
    lines = [
        "# Legacy Robot bodywork audit (Phase C1)",
        "",
        "Generated by `tools/audit_bodywork.py` from the pristine extracted prefab",
        "`assets/fresh_xrc/robot`. Full per-object data: `runs/bodywork_audit.json`.",
        "",
        "## Summary",
        "",
        f"- Objects audited under the C1 subtrees: **{s['objects_audited']}**",
        f"- Active + enabled MeshRenderers in the whole robot: **{s['active_enabled_meshrenderers_total']}**",
        f"  - exported as visuals (not under /physics or /preload): **{s['of_which_visual_exported']}**",
        f"  - excluded (physics/preload): **{s['of_which_excluded_physics_or_preload']}**",
        f"- Solid (non-trigger, enabled) colliders — the containment surfaces:",
        f"  - `Physics[7]/IntakeHopper[5]`: **{s['containment_solid_colliders']['IntakeHopper[5]']}**",
        f"  - `Physics[7]/Body[6]`: **{s['containment_solid_colliders']['Body[6]']}**",
        "",
        "## Key finding",
        "",
        "Every active MeshRenderer is already exported into the live scene, so no",
        "visible mesh is missing. The containment that reads as \"missing panels\"",
        "above the chassis and around the intake/hopper is represented in xRC as",
        "**renderer-disabled solid collider boxes** under `Physics[7]/IntakeHopper[5]`",
        "and `Physics[7]/Body[6]`. Phase C2 therefore visualises those exact solid",
        "colliders as translucent panels; it does **not** unhide trigger volumes",
        "(keep-clear/intake/indexer sensors) and does not invent new sheet metal.",
        "",
        "## Containment solid colliders (Phase C2 build set)",
        "",
        "World AABB and size are in xRC/Unity prefab metres.",
        "",
        "| subtree | path | type | trigger | world size (m) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for subtree, cols in out["containment_solid_colliders"].items():
        for col in cols:
            leaf = col["path"].split("/", 3)[-1]
            size = "×".join(f"{v:.3f}" for v in col["world_size_m"])
            lines.append(
                f"| {subtree} | `{leaf}` | {col['type']} | {col['is_trigger']} | {size} |"
            )
    lines += [
        "",
        "## All colliders under the audited subtrees",
        "",
        "| path | type | trigger | enabled | renderer_enabled | world size (m) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in out["objects"]:
        for col in rec["colliders"]:
            leaf = rec["path"].split("/", 2)[-1]
            size = "×".join(f"{v:.3f}" for v in col["world_size_m"])
            lines.append(
                f"| `{leaf}` | {col['type']} | {col['is_trigger']} | {col['enabled']} "
                f"| {rec['renderer_enabled']} | {size} |"
            )
    lines += [
        "",
        "## Phase C2 build status",
        "",
        "`RobotArticulationBuilder.build` (src/xrc_rebuilt/robot_model.py) emits a",
        "translucent polycarbonate panel for each of the 20 solid containment",
        "colliders under `/World/Robot/LegacyRobot/chassis/ContainmentPanels/Panel_NN`,",
        "reusing the **exact** collider triangles. No second collision shape is added",
        "(the enabled BoxCollider already provides the physics), and no trigger volume",
        "is rendered. Panels are on by default; `run_sim.py --no-panels` hides them and",
        "`--debug-colliders` shows the physics colliders, as independent toggles.",
        "`tests/test_bodywork.py` locks the panel selection to the 20 colliders above.",
        "",
        "## Phase C3 verification (pending GPU window)",
        "",
        "Deferred to a coordinated Isaac window; run headless-capture from:",
        "",
        "- pristine (no panels): `run_sim.py --no-panels`",
        "- with containment panels: `run_sim.py` (default)",
        "",
        "then compare front/both-sides/rear/overhead/intake/shooter views against the",
        "pristine xRC legacy, and run the ball-clip / abrupt-maneuver / bump-trench",
        "checks. Physics containment already exists (the colliders are live in the",
        "baseline); C3 confirms the panels are visually faithful and nothing snags.",
    ]
    (PROJECT / "docs" / "legacy_BODYWORK_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote docs/legacy_BODYWORK_AUDIT.md")


if __name__ == "__main__":
    main()
