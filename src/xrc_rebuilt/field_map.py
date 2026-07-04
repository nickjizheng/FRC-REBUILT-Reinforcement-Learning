"""Static field occupancy map + A* path planner (assistant_NEXT_PLAN Phase D4).

Builds a 2-D occupancy grid from the exact extracted field colliders so the
drive-to-shoot planner never routes a straight line through structures.  Only
static structural colliders that intersect the chassis height band become
obstacles; the floor and the traversable BUMPS/TRENCHES are excluded (the plan
requires the robot to cross those).  FUEL spheres are dynamic and excluded.

Pure Python + numpy; no Isaac.  The PhysX drive-follow that confirms a planned
path is actually drivable is a separate, deferred GPU step.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELD_DIR = PROJECT_ROOT / "assets" / "fresh_xrc" / "field"

# Chassis body height band (Isaac metres). Obstacles must intersect this to
# block the footprint; the floor (~0) and low bump ramps fall below it.
CHASSIS_Z_BAND = (0.12, 0.60)
# Traversable / non-obstacle path tokens.
TRAVERSABLE_TOKENS = ("bump", "trench", "ramp", "floor", "fuel")
ROBOT_FOOTPRINT_RADIUS_M = 0.52  # ~circumscribed legacy footprint (real clearance)


def _obstacle_rects() -> list[tuple[float, float, float, float]]:
    """World-XY (Isaac) axis-aligned rectangles of static structural colliders."""
    from xrc_rebuilt.isaac_scene import box_triangles, unity_to_usd

    colliders = json.loads((FIELD_DIR / "colliders.json").read_text(encoding="utf-8"))
    mesh_tris = np.load(FIELD_DIR / "field_meshes.npz")["collider_triangles"]
    rects: list[tuple[float, float, float, float]] = []
    for col in colliders:
        if col.get("trigger") or not col.get("enabled", True):
            continue
        if col["type"] not in ("BoxCollider", "MeshCollider", "CapsuleCollider"):
            continue
        path = col.get("path", "").lower()
        if any(tok in path for tok in TRAVERSABLE_TOKENS):
            continue
        if col["type"] == "BoxCollider":
            world = unity_to_usd(box_triangles(col).reshape(-1, 3))
        elif col["type"] == "MeshCollider" and col.get("triangle_count", 0):
            start, count = int(col["triangle_start"]), int(col["triangle_count"])
            world = unity_to_usd(mesh_tris[start:start + count].reshape(-1, 3))
        else:  # capsule: approximate by its center +/- radius box
            center = unity_to_usd(np.asarray(col["center"], dtype=np.float32))
            r = float(col.get("radius", 0.1)) * float(max(abs(v) for v in col.get("scale", [1, 1, 1])))
            world = center + np.array(
                [[dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)],
                dtype=np.float32,
            )
        z_lo, z_hi = float(world[:, 2].min()), float(world[:, 2].max())
        if z_hi < CHASSIS_Z_BAND[0] or z_lo > CHASSIS_Z_BAND[1]:
            continue
        rects.append((float(world[:, 0].min()), float(world[:, 0].max()),
                      float(world[:, 1].min()), float(world[:, 1].max())))
    return rects


class OccupancyGrid:
    def __init__(
        self,
        resolution: float = 0.08,
        robot_radius: float = ROBOT_FOOTPRINT_RADIUS_M,
        bounds: tuple[float, float, float, float] = (-4.4, 4.4, -9.5, 9.5),
    ):
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.x0, self.x1, self.y0, self.y1 = bounds
        self.nx = int(math.ceil((self.x1 - self.x0) / resolution))
        self.ny = int(math.ceil((self.y1 - self.y0) / resolution))
        self.occupied = np.zeros((self.nx, self.ny), dtype=bool)
        self._rects = _obstacle_rects()
        self._rasterize()

    def _rasterize(self) -> None:
        pad = self.robot_radius
        for xmin, xmax, ymin, ymax in self._rects:
            ix0 = max(0, self._ix(xmin - pad))
            ix1 = min(self.nx - 1, self._ix(xmax + pad))
            iy0 = max(0, self._iy(ymin - pad))
            iy1 = min(self.ny - 1, self._iy(ymax + pad))
            if ix0 <= ix1 and iy0 <= iy1:
                self.occupied[ix0:ix1 + 1, iy0:iy1 + 1] = True

    def _ix(self, x: float) -> int:
        return int((x - self.x0) / self.resolution)

    def _iy(self, y: float) -> int:
        return int((y - self.y0) / self.resolution)

    def cell_center(self, ix: int, iy: int) -> tuple[float, float]:
        return (self.x0 + (ix + 0.5) * self.resolution, self.y0 + (iy + 0.5) * self.resolution)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def is_free(self, x: float, y: float) -> bool:
        ix, iy = self._ix(x), self._iy(y)
        return self.in_bounds(ix, iy) and not self.occupied[ix, iy]

    def occupied_fraction(self) -> float:
        return float(self.occupied.mean())


_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def plan_path(
    grid: OccupancyGrid, start: tuple[float, float], goal: tuple[float, float]
) -> list[tuple[float, float]] | None:
    """8-connected A* over the occupancy grid. Returns world waypoints or None.

    A grid A* is an acceptable first planner (the plan allows A*/Hybrid-A*
    over SE(2)); it never crosses an occupied cell, so it never tunnels through
    a structure. Heading is resolved by the drive-follow / firing pose.
    """
    s = (grid._ix(start[0]), grid._iy(start[1]))
    g = (grid._ix(goal[0]), grid._iy(goal[1]))
    if not (grid.in_bounds(*s) and grid.in_bounds(*g)):
        return None
    if grid.occupied[s] or grid.occupied[g]:
        return None
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost: dict[tuple[int, int], float] = {s: 0.0}

    def h(c: tuple[int, int]) -> float:
        return math.hypot(c[0] - g[0], c[1] - g[1])

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == g:
            cells = [cur]
            while cur in came:
                cur = came[cur]
                cells.append(cur)
            cells.reverse()
            return [grid.cell_center(ix, iy) for ix, iy in cells]
        for dx, dy, step in _NEIGHBORS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not grid.in_bounds(*nxt) or grid.occupied[nxt]:
                continue
            new_cost = cost[cur] + step
            if new_cost < cost.get(nxt, math.inf):
                cost[nxt] = new_cost
                came[nxt] = cur
                heapq.heappush(open_heap, (new_cost + h(nxt), nxt))
    return None


def _line_of_sight(grid: OccupancyGrid, a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if the straight segment a->b stays in free (inflated) cells."""
    steps = max(1, int(math.dist(a, b) / (grid.resolution * 0.5)))
    for i in range(steps + 1):
        t = i / steps
        if not grid.is_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
            return False
    return True


def simplify_path(
    grid: OccupancyGrid, path: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """String-pull an 8-connected A* path into long straight segments so a
    pure-pursuit follower flows smoothly instead of zig-zagging into corners.
    """
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_of_sight(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def candidate_firing_positions(
    hub_target_xy: tuple[float, float],
    open_side_sign: float,
    range_lo: float = 2.55,
    range_hi: float = 2.95,
    n_range: int = 3,
    n_bearing: int = 13,
) -> list[tuple[float, float]]:
    """Positions on the open-side ring(s) around a HUB within calibrated range.

    ``open_side_sign``: blue HUB is scored from -y (sign -1), red from +y (+1).
    """
    tx, ty = hub_target_xy
    out: list[tuple[float, float]] = []
    for r in np.linspace(range_lo, range_hi, n_range):
        # bearings spanning the open half-plane (+/-70 deg off the axis)
        for bearing in np.linspace(-1.22, 1.22, n_bearing):
            x = tx + r * math.sin(bearing)
            y = ty + open_side_sign * r * math.cos(bearing)
            out.append((float(x), float(y)))
    return out
