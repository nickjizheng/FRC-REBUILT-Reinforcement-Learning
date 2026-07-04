#!/usr/bin/env python
"""Pure-numpy clearance analysis of the HUB exit chutes (no Isaac needed).

Rebuilds the exact static collision triangle soup that SceneBuilder uses,
then measures signed clearance (distance-to-mesh minus ball radius) around
the HubRouter release poses, per alliance and exit lane.  Grid-searches the
(y, z) plane at each lane x for the widest-clearance release point.

Run:  /c/il/venv/Scripts/python.exe tools/analyze_hub_exit_geometry.py
Writes runs/hub_exit_geometry.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xrc_rebuilt.isaac_scene import box_triangles, unity_to_usd  # noqa: E402

BALL_R = 0.076
EXIT_X = (-0.4002, -0.1284, 0.1343, 0.3877)
EXIT_Y = {"red": 3.4706, "blue": -3.4706}
RELEASE_Z = 1.02


def load_collision_triangles() -> np.ndarray:
    field = PROJECT_ROOT / "assets" / "fresh_xrc" / "field"
    collider_data = np.load(field / "field_meshes.npz")["collider_triangles"]
    colliders = json.loads((field / "colliders.json").read_text(encoding="utf-8"))
    blocks = []
    for c in colliders:
        if c.get("trigger") or not c.get("enabled", True):
            continue
        if c["type"] == "BoxCollider":
            blocks.append(box_triangles(c))
        elif c["type"] == "MeshCollider" and c.get("triangle_count", 0):
            s, n = int(c["triangle_start"]), int(c["triangle_count"])
            blocks.append(collider_data[s:s + n])
    return unity_to_usd(np.concatenate(blocks))


def point_triangle_distances(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact point-to-triangle distance for every triangle (vectorized)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(vb + vc == 0, 1.0, vb + vc)
    # region tests -> closest point
    closest = np.empty_like(a)
    # vertex A
    m = (d1 <= 0) & (d2 <= 0)
    closest[m] = a[m]
    # vertex B
    m2 = (~m) & (d3 >= 0) & (d4 <= d3)
    closest[m2] = b[m2]
    done = m | m2
    # edge AB
    m3 = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where(d1 - d3 == 0, 0.0, d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3))
    closest[m3] = a[m3] + t[m3, None] * ab[m3]
    done |= m3
    # vertex C
    m4 = (~done) & (d6 >= 0) & (d5 <= d6)
    closest[m4] = c[m4]
    done |= m4
    # edge AC
    m5 = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where(d2 - d6 == 0, 0.0, d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6))
    closest[m5] = a[m5] + t[m5, None] * ac[m5]
    done |= m5
    # edge BC
    m6 = (~done) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = np.where((d4 - d3) + (d5 - d6) == 0, 0.0,
                 (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1.0, (d4 - d3) + (d5 - d6)))
    closest[m6] = b[m6] + t[m6, None] * (c[m6] - b[m6])
    done |= m6
    # interior
    mi = ~done
    v = vb / denom
    w = vc / denom
    closest[mi] = a[mi] + v[mi, None] * ab[mi] + w[mi, None] * ac[mi]
    return np.linalg.norm(p - closest, axis=-1)


def clearance(p, tris) -> float:
    """min distance from point to mesh, minus ball radius (negative = overlap)."""
    return float(point_triangle_distances(np.asarray(p, dtype=np.float64), tris).min() - BALL_R)


def main() -> None:
    tris_all = load_collision_triangles().astype(np.float64)
    cent = tris_all.mean(axis=1)
    out = {"ball_radius_m": BALL_R, "release_z_m": RELEASE_Z, "alliances": {}}
    for alliance, ey in EXIT_Y.items():
        sign = 1.0 if ey > 0 else -1.0
        near = tris_all[
            (np.abs(cent[:, 0]) < 1.2)
            & (cent[:, 1] * sign > 2.5)
            & (cent[:, 1] * sign < 4.5)
            & (cent[:, 2] > 0.3)
            & (cent[:, 2] < 2.0)
        ]
        lanes = {}
        for k, x in enumerate(EXIT_X):
            release = (x, ey, RELEASE_Z)
            rel_clear = clearance(release, near)
            # grid search: y from just outside the exit down to the sensor band,
            # z spanning the chute
            ys = sign * np.arange(3.05, 3.55, 0.02)
            zs = np.arange(0.80, 1.14, 0.01)
            grid = np.full((len(ys), len(zs)), -1.0)
            for i, y in enumerate(ys):
                for j, z in enumerate(zs):
                    grid[i, j] = clearance((x, y, z), near)
            best = np.unravel_index(np.argmax(grid), grid.shape)
            # best clearance at each y along the exit path (corridor profile)
            corridor = [
                {
                    "y": round(float(ys[i]), 3),
                    "best_z": round(float(zs[int(np.argmax(grid[i]))]), 3),
                    "clearance_mm": round(float(grid[i].max() * 1000.0), 2),
                }
                for i in range(len(ys))
            ]
            lanes[str(k)] = {
                "x": x,
                "release_pose": list(release),
                "release_clearance_mm": round(rel_clear * 1000.0, 2),
                "best_point": {
                    "y": round(float(ys[best[0]]), 3),
                    "z": round(float(zs[best[1]]), 3),
                    "clearance_mm": round(float(grid[best]) * 1000.0, 2),
                },
                "corridor_profile": corridor,
            }
            print(
                f"{alliance} lane {k} (x={x:+.4f}): release clearance "
                f"{rel_clear*1000:+.1f} mm | best point y={ys[best[0]]:+.3f} "
                f"z={zs[best[1]]:.3f} -> {grid[best]*1000:+.1f} mm"
            )
        out["alliances"][alliance] = lanes

    dest = PROJECT_ROOT / "runs" / "hub_exit_geometry.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", dest)


if __name__ == "__main__":
    main()
