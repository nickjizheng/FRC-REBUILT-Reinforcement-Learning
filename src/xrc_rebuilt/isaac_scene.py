"""Build and run the high-fidelity xRC REBUILT Isaac scene."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELD_DIR = PROJECT_ROOT / "assets" / "fresh_xrc" / "field"
ROBOT_DIR = PROJECT_ROOT / "assets" / "fresh_xrc" / "robot"


PALETTE = {
    "floor": (0.10, 0.12, 0.15, 1.0),
    "wall": (0.62, 0.70, 0.76, 0.68),
    "field": (0.38, 0.42, 0.46, 1.0),
    "return": (0.90, 0.55, 0.06, 1.0),
    "hub_red": (0.72, 0.055, 0.045, 1.0),
    "hub_blue": (0.04, 0.24, 0.82, 1.0),
    "hub": (0.35, 0.38, 0.42, 1.0),
    "hub_net": (0.035, 0.035, 0.04, 0.78),
    "source_red": (0.62, 0.06, 0.05, 1.0),
    "source_blue": (0.04, 0.18, 0.66, 1.0),
    "source": (0.52, 0.46, 0.24, 1.0),
    "trench_red": (0.56, 0.05, 0.045, 1.0),
    "trench_blue": (0.035, 0.16, 0.60, 1.0),
    "trench": (0.28, 0.31, 0.34, 1.0),
    "tower_red": (0.72, 0.06, 0.05, 1.0),
    "tower_blue": (0.04, 0.22, 0.76, 1.0),
    "tower": (0.42, 0.44, 0.46, 1.0),
    "bump": (0.95, 0.28, 0.025, 1.0),
    "apriltag": (0.92, 0.92, 0.88, 1.0),
    "robot": (0.025, 0.028, 0.034, 1.0),
}

# Extracted directly from xRC level74.  The source scene has four overlapping
# Lit/Unlit rectangular DMX bars around each HUB's top angles.  Keeping these
# values here makes the runtime switch materials without approximating either
# the original mesh or its colors.
XRC_HUB_LIGHT_COLORS = {
    "red": (0.7450980544, 0.5220759511, 0.4745098054, 0.9098039269),
    "blue": (0.3230000138, 0.7370000482, 1.0, 0.9098039269),
    "unlit": (0.5019608140, 0.4760000110, 0.4760000110, 0.9098039269),
    "white": (1.0, 1.0, 1.0, 1.0),
}
HUB_WARNING_PULSE_HZ = 2.0
HUB_WHITE_CHASE_HZ = 2.0


class HubRouter:
    """Runtime equivalent of xRC's scorer + four ``bs_out`` accelerators."""

    HUBS = {
        "red": {"sensor_y": 3.269, "exit_y": 3.4706, "direction_y": -1.0},
        "blue": {"sensor_y": -3.269, "exit_y": -3.4706, "direction_y": 1.0},
    }
    EXIT_X = (-0.4002, -0.1284, 0.1343, 0.3877)
    # Minimum time between two FUEL dropping out of the SAME exit chute, so a big
    # volley leaves the hub one ball at a time at a realistic rate (no jam).  With
    # four chutes this drains ~13 balls/s, faster than any shooter can feed them.
    EXIT_INTERVAL_MIN_S = 0.22
    EXIT_INTERVAL_MAX_S = 0.42
    EXIT_POSITION_JITTER_M = 0.018

    def __init__(self, view: Any, count: int, seed: int = 2026):
        from xrc_rebuilt.rules import (
            MATCH_DURATION_S,
            fuel_score_is_eligible,
            hub_active_at,
            sample_hub_exit,
            sample_hub_routing_delay,
        )

        self.view = view
        self.count = count
        self.rng = random.Random(seed)
        self.sample_exit = sample_hub_exit
        self.sample_delay = sample_hub_routing_delay
        self._fuel_score_is_eligible = fuel_score_is_eligible
        self._hub_active_at = hub_active_at
        self._match_duration_s = float(MATCH_DURATION_S)
        # official scoring context, maintained by the run loop each tick
        self.match_first_inactive: str | None = None
        self.sandbox = False
        self.pending: dict[int, tuple[float, str, int, float]] = {}
        self.exit_free_at: dict[tuple[str, int], float] = {}
        self.blocked_until_clear: set[int] = set()
        # released balls watched until they physically clear the hub; a ball
        # wedged in an exit chute (validated ~2.6% of releases) is re-released
        self.released_watch: dict[int, tuple[float, str]] = {}
        self.released = 0
        self.re_released = 0
        self.detected = 0
        self.crowd_rescued = 0
        # REBUILT hub state: the hub always physically routes FUEL, but only
        # FUEL sensed while that hub is ACTIVE earns score (2026 manual:
        # "FUEL scored in an inactive HUB will not earn any points").
        self.active = {"red": True, "blue": True}
        self.scored = {"red": 0, "blue": 0}
        self.delays: list[float] = []
        self.release_intervals: list[float] = []
        self.demo_indices: list[int] = []

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def inject_demo(self) -> None:
        if self.count < 2:
            return
        self.demo_indices = [self.count - 2, self.count - 1]
        positions = np.asarray([[0.0, self.HUBS["red"]["sensor_y"], 0.98], [0.0, self.HUBS["blue"]["sensor_y"], 0.98]], dtype=np.float32)
        indices = np.asarray(self.demo_indices, dtype=np.int32)
        self.view.set_world_poses(positions=positions, indices=indices)
        self.view.set_linear_velocities(np.zeros((2, 3), dtype=np.float32), indices=indices)

    def step(self, elapsed_s: float) -> None:
        positions, _ = self.view.get_world_poses()
        positions_np = self._numpy(positions)
        velocities_np = self._numpy(self.view.get_linear_velocities())
        self.blocked_until_clear = {
            index for index in self.blocked_until_clear
            if not (float(positions_np[index, 2]) < 0.40 or abs(float(positions_np[index, 1])) < 2.70)
        }
        # Anti-pile crowd rescue.  The wide continuous stream aims ~0.42 m
        # outboard of the scorer sensor, so several FUEL can occupy the funnel
        # mouth at once; a churning pile there never drops below the settled
        # speed gate, and new arrivals bounce off it and out.  Anything buried
        # at the bottom of a crowd has physically scored - vacuum the deepest
        # excess (keep at most one 3-wide volley resident) regardless of speed.
        crowd_capture: dict[int, str] = {}
        for alliance, hub in self.HUBS.items():
            mask = (
                (np.abs(positions_np[:, 0]) <= 0.90)
                & (np.abs(positions_np[:, 1] - hub["sensor_y"]) <= 0.65)
                & (positions_np[:, 2] >= 0.40)
                & (positions_np[:, 2] <= 1.60)
            )
            candidates = [
                int(index) for index in np.flatnonzero(mask)
                if int(index) not in self.pending
                and int(index) not in self.blocked_until_clear
            ]
            if len(candidates) > 3:
                candidates.sort(key=lambda i: float(positions_np[i, 2]))
                for index in candidates[: len(candidates) - 3]:
                    crowd_capture[index] = alliance
        for index, position in enumerate(positions_np):
            if index in self.pending or index in self.blocked_until_clear:
                continue
            speed = float(np.linalg.norm(velocities_np[index]))
            for alliance, hub in self.HUBS.items():
                dy = abs(float(position[1]) - hub["sensor_y"])
                # Primary capture window spans the whole funnel mouth including
                # the aim-point strip at the back rim, PLUS a rescue for any
                # ball moving slowly anywhere in the wider hub volume.  The
                # rescue speed gate is loose (0.85 m/s) so jostling FUEL in a
                # crowd still counts as arrived - only clean fly-throughs
                # (apex speeds >1 m/s) stay airborne.
                at_sensor = abs(float(position[0])) <= 0.70 and dy <= 0.55 and 0.45 <= float(position[2]) <= 1.42
                settled = abs(float(position[0])) <= 0.90 and dy <= 0.65 and float(position[2]) >= 0.40 and speed < 0.85
                if at_sensor or settled or crowd_capture.get(index) == alliance:
                    if not (at_sensor or settled):
                        self.crowd_rescued += 1
                    delay = float(self.sample_delay(self.rng))
                    exit_index = int(self.sample_exit(self.rng))
                    self.pending[index] = (elapsed_s + delay, alliance, exit_index, delay)
                    self.detected += 1
                    if self._score_eligible(alliance, elapsed_s):
                        self.scored[alliance] += 1
                    holding = np.asarray([[9.0 + index * 0.002, 0.0, -2.0]], dtype=np.float32)
                    indices = np.asarray([index], dtype=np.int32)
                    self.view.set_world_poses(positions=holding, indices=indices)
                    self.view.set_linear_velocities(np.zeros((1, 3), dtype=np.float32), indices=indices)
                    break
        ready = sorted(
            ((index, item) for index, item in self.pending.items() if item[0] <= elapsed_s),
            key=lambda kv: kv[1][0],
        )
        for index, (_, alliance, exit_index, delay) in ready:
            chute = (alliance, exit_index)
            free_at = self.exit_free_at.get(chute, 0.0)
            if elapsed_s < free_at:
                # This chute is still clearing its previous ball - hold this one
                # so FUEL drops out one at a time at a realistic speed, never
                # jamming several into a chute at once.
                self.pending[index] = (free_at, alliance, exit_index, delay)
                continue
            self._release(index, alliance, exit_index)
            interval = self.rng.uniform(
                self.EXIT_INTERVAL_MIN_S, self.EXIT_INTERVAL_MAX_S
            )
            self.exit_free_at[chute] = elapsed_s + interval
            self.release_intervals.append(interval)
            self.released += 1
            self.delays.append(delay)
            self.released_watch[index] = (elapsed_s, alliance)
            del self.pending[index]
        self._unstick_watchdog(elapsed_s, positions_np)

    def _score_eligible(self, alliance: str, elapsed_s: float) -> bool:
        """Official REBUILT scoring windows for a FUEL sensed at elapsed_s.

        Active phases count, PLUS the manual's 3-second scoring-assessment
        grace after every deactivation (shift changes and the final buzzer),
        via rules.fuel_score_is_eligible.  Sandbox free play always counts.
        """
        if self.sandbox:
            return True
        try:
            if self.match_first_inactive is None:
                # pre-decision phases (AUTO/TRANSITION): both hubs active
                return bool(
                    self._hub_active_at(
                        alliance, min(elapsed_s, self._match_duration_s - 1e-3)
                    )
                )
            return bool(
                self._fuel_score_is_eligible(
                    alliance, elapsed_s, self.match_first_inactive
                )
            )
        except Exception:
            return bool(self.active.get(alliance, True))

    def _release(self, index: int, alliance: str, exit_index: int) -> None:
        hub = self.HUBS[alliance]
        jitter = self.EXIT_POSITION_JITTER_M
        position = np.asarray(
            [[
                self.EXIT_X[exit_index] + self.rng.uniform(-jitter, jitter),
                hub["exit_y"] + self.rng.uniform(-0.010, 0.010),
                1.02 + self.rng.uniform(-0.014, 0.014),
            ]],
            dtype=np.float32,
        )
        # The xRC trigger drives local -X at 1 m/s; in field coordinates the
        # dominant component points toward the neutral zone.  Real chute contact
        # adds small lateral, speed and drop variation.
        velocity = np.asarray(
            [[
                self.rng.uniform(-0.16, 0.16),
                hub["direction_y"] * self.rng.uniform(0.84, 1.16),
                self.rng.uniform(-0.14, -0.035),
            ]],
            dtype=np.float32,
        )
        indices = np.asarray([index], dtype=np.int32)
        self.view.set_world_poses(positions=position, indices=indices)
        self.view.set_linear_velocities(velocity, indices=indices)
        # residual spin from the holding pen otherwise kicks the ball into the
        # chute lip (hub_validation: wedged balls showed 1.4-2.9 m/s deviations)
        spin = np.asarray(
            [[self.rng.uniform(-2.0, 2.0) for _ in range(3)]],
            dtype=np.float32,
        )
        self.view.set_angular_velocities(spin, indices=indices)
        self.blocked_until_clear.add(index)

    def _unstick_watchdog(self, elapsed_s: float, positions_np: np.ndarray) -> None:
        """Re-release balls that wedged in an exit chute instead of rolling out."""
        stale = [
            (index, alliance)
            for index, (release_time, alliance) in self.released_watch.items()
            if elapsed_s - release_time > 2.0
        ]
        if not stale:
            return
        velocities = self._numpy(self.view.get_linear_velocities())
        for index, alliance in stale:
            position = positions_np[index]
            cleared = float(position[2]) < 0.3 or abs(float(position[1])) < 2.7
            if cleared:
                del self.released_watch[index]
                continue
            if float(np.linalg.norm(velocities[index])) < 0.05:
                self._release(index, alliance, int(self.sample_exit(self.rng)))
                self.re_released += 1
                self.released_watch[index] = (elapsed_s, alliance)

    def stats(self) -> dict[str, Any]:
        return {
            "hub_detected": self.detected,
            "hub_released": self.released,
            "hub_re_released": self.re_released,
            "hub_crowd_rescued": self.crowd_rescued,
            "hub_scored_red": self.scored["red"],
            "hub_scored_blue": self.scored["blue"],
            "hub_pending": len(self.pending),
            "hub_mean_route_delay_s": float(np.mean(self.delays)) if self.delays else None,
            "hub_max_route_delay_s": float(max(self.delays)) if self.delays else None,
            "hub_mean_release_interval_s": (
                float(np.mean(self.release_intervals))
                if self.release_intervals else None
            ),
        }


def unity_to_usd(points: np.ndarray) -> np.ndarray:
    """Unity LH/Y-up -> Isaac RH/Z-up: (x,y,z) -> (x,-z,y)."""
    result = np.empty_like(points, dtype=np.float32)
    result[..., 0] = points[..., 0]
    result[..., 1] = -points[..., 2]
    result[..., 2] = points[..., 1]
    return result


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homogeneous @ np.asarray(matrix, dtype=np.float32).T)[:, :3]


def quat_matrix_xyzw(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
        ], dtype=np.float32,
    )


def box_triangles(collider: dict[str, Any]) -> np.ndarray:
    center = np.asarray(collider["center"], dtype=np.float32)
    half = np.asarray(collider["size"], dtype=np.float32) * 0.5
    local = np.array(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])],
        dtype=np.float32,
    )
    vertices = local @ quat_matrix_xyzw(collider["rotation_xyzw"]).T + center
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ], dtype=np.int32,
    )
    return vertices[faces]


def visual_group(item: dict[str, Any]) -> str:
    group = item["category"]
    path = item["path"].lower()
    if group in {"hub", "source", "trench", "tower"}:
        if "red" in path:
            return f"{group}_red"
        if "blue" in path:
            return f"{group}_blue"
    return group


def is_original_visible(item: dict[str, Any]) -> bool:
    path = item["path"].lower()
    if not item.get("enabled", True) or not item.get("active_self", True):
        return False
    if "/physics/" in path or path.endswith("/physics"):
        return False
    return not any(token in path for token in ("robot_ref", "camera_ref", "sizingcube", "respawn", "reflection", "probevolume", "urpsettings"))


class SceneBuilder:
    def __init__(
        self,
        stage: Any,
        debug_colliders: bool = False,
        max_fuel: int = 456,
        articulated_robot: bool = True,
        show_panels: bool = True,
        show_turret: bool = True,
    ):
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

        self.Gf, self.PhysxSchema, self.Sdf = Gf, PhysxSchema, Sdf
        self.UsdGeom, self.UsdPhysics, self.UsdShade, self.Vt = UsdGeom, UsdPhysics, UsdShade, Vt
        self.UsdLux = UsdLux
        self.stage = stage
        self.debug_colliders = debug_colliders
        self.max_fuel = max_fuel
        self.articulated_robot = articulated_robot
        self.show_panels = show_panels
        self.show_turret = show_turret
        self.ball_prims: list[Any] = []
        self.robot_prim: Any | None = None
        self.robot_root_path: str | None = None
        self.hub_light_prims: dict[str, dict[str, list[Any]]] = {}
        self.hub_light_shaders: dict[str, list[Any]] = {}
        self.hub_light_state: dict[str, str] = {}
        self.stats: dict[str, int] = {}

    def _mesh(self, path: str, triangles: np.ndarray, color: tuple[float, float, float, float], collision: bool = False) -> Any:
        UsdGeom, UsdPhysics, Gf, Vt = self.UsdGeom, self.UsdPhysics, self.Gf, self.Vt
        triangles = np.asarray(triangles, dtype=np.float32)
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        flat = triangles.reshape(-1, 3)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(flat))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(flat), dtype=np.int32)))
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color[:3])])
        mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(color[3])])
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            self.PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        return mesh.GetPrim()

    def _preview_material(self, key: str, color: tuple[float, float, float, float]) -> Any:
        UsdShade, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        material = UsdShade.Material.Define(self.stage, f"/World/Looks/{key}")
        shader = UsdShade.Shader.Define(self.stage, f"/World/Looks/{key}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.52)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _hub_light_material(
        self,
        key: str,
        color: tuple[float, float, float, float],
        *,
        emissive: bool,
    ) -> tuple[Any, Any]:
        """Create one independently animated xRC DMX-bar material."""

        UsdShade, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        path = f"/World/Looks/{key}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color[:3])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.18)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color[:3]) if emissive else Gf.Vec3f(0.0, 0.0, 0.0)
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material, shader

    @staticmethod
    def _hub_light_source_items(
        visuals: list[dict[str, Any]], alliance: str, state: str
    ) -> list[dict[str, Any]]:
        token = f"CenterGoal {alliance.title()}/Visuals/Lights/{state}/"
        return [
            item
            for item in visuals
            if item.get("enabled", True)
            and item.get("active_self", True)
            and item.get("path", "").startswith(token)
        ]

    def build_hub_lights(
        self, triangles: np.ndarray, visuals: list[dict[str, Any]]
    ) -> None:
        """Author xRC's exact four Lit and four Unlit meshes per HUB."""

        total_triangles = 0
        for alliance in ("red", "blue"):
            self.hub_light_prims[alliance] = {"lit": [], "unlit": []}
            self.hub_light_shaders[alliance] = []
            for state in ("Lit", "Unlit"):
                source_items = self._hub_light_source_items(visuals, alliance, state)
                # Sort around the HUB perimeter so the transition white chase
                # advances spatially rather than following Unity object names.
                blocks = []
                hub_y = 3.65 if alliance == "red" else -3.65
                for item in source_items:
                    start, count = int(item["triangle_start"]), int(item["triangle_count"])
                    block = unity_to_usd(triangles[start:start + count])
                    center = block.reshape(-1, 3).mean(axis=0)
                    angle = math.atan2(float(center[1]) - hub_y, float(center[0]))
                    blocks.append((angle, block, item))
                blocks.sort(key=lambda value: value[0])
                key = state.lower()
                for index, (_, block, item) in enumerate(blocks):
                    source_color = tuple(float(value) for value in item["rgba"])
                    expected = XRC_HUB_LIGHT_COLORS[
                        alliance if state == "Lit" else "unlit"
                    ]
                    # Use the per-object source value when present, while the
                    # extracted constants above document the canonical colors.
                    color = source_color if len(source_color) == 4 else expected
                    prim = self._mesh(
                        f"/World/HubLights/{alliance}/{state}_{index}", block, color
                    )
                    material, shader = self._hub_light_material(
                        f"hub_light_{alliance}_{key}_{index}",
                        color,
                        emissive=(state == "Lit"),
                    )
                    self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                    self.hub_light_prims[alliance][key].append(prim)
                    if state == "Lit":
                        self.hub_light_shaders[alliance].append(shader)
                    total_triangles += len(block)
        self.stats["hub_light_bars"] = sum(
            len(parts["lit"]) for parts in self.hub_light_prims.values()
        )
        self.stats["hub_light_source_triangles"] = total_triangles
        self.update_hub_lights(0.0, None)

    def _set_hub_light_shader(
        self, shader: Any, color: tuple[float, float, float, float], brightness: float
    ) -> None:
        brightness = float(np.clip(brightness, 0.0, 1.0))
        rgb = np.asarray(color[:3], dtype=np.float32)
        diffuse = rgb * (0.18 + 0.82 * brightness)
        emissive = rgb * (3.5 * brightness)
        shader.GetInput("diffuseColor").Set(self.Gf.Vec3f(*map(float, diffuse)))
        shader.GetInput("emissiveColor").Set(self.Gf.Vec3f(*map(float, emissive)))

    def update_hub_lights(
        self,
        elapsed_s: float,
        first_inactive: str | None,
    ) -> None:
        """Animate HUB bars from the official match state and clock."""

        from xrc_rebuilt.rules import HubLightState, hub_light_state_at

        for alliance in ("red", "blue"):
            state = hub_light_state_at(alliance, max(0.0, elapsed_s), first_inactive)
            self.hub_light_state[alliance] = state.value
            lit = self.hub_light_prims.get(alliance, {}).get("lit", [])
            unlit = self.hub_light_prims.get(alliance, {}).get("unlit", [])
            shaders = self.hub_light_shaders.get(alliance, [])
            is_off = state is HubLightState.INACTIVE
            for prim in lit:
                prim.GetAttribute("visibility").Set("invisible" if is_off else "inherited")
            for prim in unlit:
                prim.GetAttribute("visibility").Set("inherited" if is_off else "invisible")
            if is_off:
                continue

            base_color = XRC_HUB_LIGHT_COLORS[alliance]
            if state is HubLightState.POST_MATCH_WHITE:
                for shader in shaders:
                    self._set_hub_light_shader(
                        shader, XRC_HUB_LIGHT_COLORS["white"], 1.0
                    )
            elif state is HubLightState.WARNING:
                # Smooth xRC-style pulse; never fully black, so the alliance
                # identity remains legible between peaks.
                phase = 0.5 + 0.5 * math.sin(
                    2.0 * math.pi * HUB_WARNING_PULSE_HZ * elapsed_s
                )
                brightness = 0.12 + 0.88 * phase
                for shader in shaders:
                    self._set_hub_light_shader(shader, base_color, brightness)
            elif state is HubLightState.TRANSITION_CHASE:
                chase = int(elapsed_s * HUB_WHITE_CHASE_HZ * max(1, len(shaders)))
                chase %= max(1, len(shaders))
                for index, shader in enumerate(shaders):
                    color = (
                        XRC_HUB_LIGHT_COLORS["white"]
                        if index == chase
                        else base_color
                    )
                    self._set_hub_light_shader(shader, color, 1.0)
            else:
                for shader in shaders:
                    self._set_hub_light_shader(shader, base_color, 1.0)

    def build_lighting(self) -> None:
        UsdLux, Gf = self.UsdLux, self.Gf
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(850)
        dome.CreateColorAttr(Gf.Vec3f(0.76, 0.82, 0.95))
        key = UsdLux.DistantLight.Define(self.stage, "/World/Lights/Key")
        key.CreateIntensityAttr(2600)
        key.CreateAngleAttr(0.8)
        key.AddRotateXYZOp().Set(Gf.Vec3f(42, -28, -24))

    def build_field_visuals(self) -> None:
        data = np.load(FIELD_DIR / "field_meshes.npz")
        triangles = data["render_triangles"]
        visuals = json.loads((FIELD_DIR / "visuals.json").read_text(encoding="utf-8"))
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for item in visuals:
            # HUB lights are rebuilt separately from their exact source blocks
            # so Lit/Unlit runtime states are not baked into the static HUB.
            if "/visuals/lights/" in item.get("path", "").lower():
                continue
            if is_original_visible(item):
                start, count = int(item["triangle_start"]), int(item["triangle_count"])
                grouped[visual_group(item)].append(triangles[start:start+count])
        for group, blocks in grouped.items():
            color = PALETTE.get(group, PALETTE["field"])
            prim = self._mesh(f"/World/FieldVisual/{group}", unity_to_usd(np.concatenate(blocks)), color)
            self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._preview_material(group, color))
        self.build_hub_lights(triangles, visuals)
        self.stats["visual_groups"] = len(grouped)
        self.stats["visible_triangles"] = sum(sum(len(block) for block in blocks) for blocks in grouped.values())

    def build_field_collisions(self) -> None:
        collider_data = np.load(FIELD_DIR / "field_meshes.npz")["collider_triangles"]
        colliders = json.loads((FIELD_DIR / "colliders.json").read_text(encoding="utf-8"))
        blocks: list[np.ndarray] = []
        for collider in colliders:
            if collider.get("trigger") or not collider.get("enabled", True):
                continue
            if collider["type"] == "BoxCollider":
                blocks.append(box_triangles(collider))
            elif collider["type"] == "MeshCollider" and collider.get("triangle_count", 0):
                start, count = int(collider["triangle_start"]), int(collider["triangle_count"])
                blocks.append(collider_data[start:start+count])
        if blocks:
            self._mesh("/World/FieldCollision", unity_to_usd(np.concatenate(blocks)), (0.1, 0.8, 0.15, 0.15), collision=True)
        if self.debug_colliders and blocks:
            self.stage.GetPrimAtPath("/World/FieldCollision").GetAttribute("visibility").Set("inherited")
        else:
            self.stage.GetPrimAtPath("/World/FieldCollision").GetAttribute("visibility").Set("invisible")
        self.stats["static_collision_triangles"] = sum(map(len, blocks))

    def _physics_material(self) -> Any:
        material = self.UsdShade.Material.Define(self.stage, "/World/PhysicsMaterials/FuelFoam")
        api = self.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(1.2)
        api.CreateDynamicFrictionAttr(0.9)
        api.CreateRestitutionAttr(0.08)
        # FRC FUEL is compliant foam.  The former 600 kN/m contact plus friction
        # 10/5 made a millimetre of intentional roller squeeze launch a 0.08 kg
        # ball across the field.  Damped 40 kN/m contact preserves solid
        # collision while allowing realistic roller grip and packed-net give.
        try:
            physx = self.PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            physx.CreateCompliantContactStiffnessAttr(40000.0)
            physx.CreateCompliantContactDampingAttr(1200.0)
        except Exception as exc:  # pragma: no cover - schema availability guard
            print(f"compliant fuel contacts unavailable: {exc}", flush=True)
        return material

    def build_fuel(self) -> None:
        UsdGeom, UsdPhysics, Gf = self.UsdGeom, self.UsdPhysics, self.Gf
        colliders = json.loads((FIELD_DIR / "colliders.json").read_text(encoding="utf-8"))
        candidates = [item for item in colliders if item["type"] == "SphereCollider" and item["level"] in {"level75", "level76"} and "fuel" in item["path"].lower()]
        material = self._physics_material()
        fuel_look = self._preview_material("fuel", (0.94, 0.69, 0.035, 1.0))
        for index, item in enumerate(candidates[: self.max_fuel]):
            sphere = UsdGeom.Sphere.Define(self.stage, f"/World/Fuel/Fuel_{index:03d}")
            sphere.CreateRadiusAttr(float(item["radius"]))
            sphere.AddTranslateOp().Set(Gf.Vec3d(*map(float, unity_to_usd(np.asarray(item["center"], dtype=np.float32)))))
            sphere.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(0.94, 0.69, 0.035)])
            prim = sphere.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateRigidBodyEnabledAttr(True)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(0.08)
            fuel_body = self.PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            fuel_body.CreateSolverPositionIterationCountAttr(4)
            fuel_body.CreateSolverVelocityIterationCountAttr(1)
            # let idle field FUEL sleep quickly so it drops out of the solver
            fuel_body.CreateSleepThresholdAttr(0.02)
            fuel_body.CreateStabilizationThresholdAttr(0.02)
            binding = self.UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(fuel_look)
            binding.Bind(material, materialPurpose="physics")
            self.ball_prims.append(prim)
        self.stats["fuel_bodies"] = len(self.ball_prims)
        # Formal preload sourcing (manual 6.3.3): our eight come from the
        # 48-ball neutral staging grid ("a FUEL not pre-loaded in a ROBOT is
        # staged in the NEUTRAL ZONE") - the other 40 stay staged mid-field.
        # Take the two full grid columns nearest our spawn so the remaining
        # stack stays rectangular; depot/row/warehouse FUEL is never touched.
        preload_pool = [
            (index, item)
            for index, item in enumerate(candidates[: self.max_fuel])
            if "preload" in item["path"].lower()
        ]
        preload_pool.sort(key=lambda pair: -float(pair[1]["center"][0]))
        self.preload_fuel_indices = [index for index, _ in preload_pool[:8]]
        self.stats["fuel_preload_pool"] = len(preload_pool)
        # xRC level75 contributes 408 field bodies and level76 contributes 48
        # preload bodies.  Loading eight into this robot moves existing bodies;
        # it does not create or delete FUEL.
        self.stats["fuel_total_xrc"] = len(self.ball_prims)
        staged_robot_count = min(8, len(self.ball_prims))
        self.stats["fuel_robot_preloaded"] = staged_robot_count
        self.stats["fuel_field_at_start"] = (
            len(self.ball_prims) - staged_robot_count
        )

    @staticmethod
    def _shape_sphere(radius: float, rings: int = 8, segments: int = 12) -> np.ndarray:
        vertices = []
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for segment in range(segments):
                theta = 2 * math.pi * segment / segments
                vertices.append((radius * math.sin(phi) * math.cos(theta), radius * math.cos(phi), radius * math.sin(phi) * math.sin(theta)))
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = []
        for ring in range(rings):
            for segment in range(segments):
                nxt = (segment + 1) % segments
                a, b = ring * segments + segment, ring * segments + nxt
                c, d = (ring + 1) * segments + segment, (ring + 1) * segments + nxt
                faces.extend(((a, c, d), (a, d, b)))
        return vertices[np.asarray(faces, dtype=np.int32)]

    @staticmethod
    def _shape_capsule(radius: float, height: float, direction: int, rings: int = 10, segments: int = 12) -> np.ndarray:
        # Convex surface; the collider is later tagged as a PhysX convex hull.
        half_line = max(0.0, height * 0.5 - radius)
        vertices = []
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            axial = radius * math.cos(phi) + (half_line if ring < rings / 2 else -half_line)
            radial = radius * math.sin(phi)
            for segment in range(segments):
                theta = 2 * math.pi * segment / segments
                point = [radial * math.cos(theta), axial, radial * math.sin(theta)]
                if direction == 0:
                    point = [point[1], point[0], point[2]]
                elif direction == 2:
                    point = [point[0], point[2], point[1]]
                vertices.append(point)
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = []
        for ring in range(rings):
            for segment in range(segments):
                nxt = (segment + 1) % segments
                a, b = ring * segments + segment, ring * segments + nxt
                c, d = (ring + 1) * segments + segment, (ring + 1) * segments + nxt
                faces.extend(((a, c, d), (a, d, b)))
        return vertices[np.asarray(faces, dtype=np.int32)]

    def build_robot(self) -> None:
        if self.articulated_robot:
            from xrc_rebuilt.competition_robot import CompetitionRobotArticulationBuilder

            articulation_builder = CompetitionRobotArticulationBuilder(self)
            self.stats.update(articulation_builder.build())
            self.robot_root_path = articulation_builder.root_path
            self.robot_prim = self.stage.GetPrimAtPath(articulation_builder.chassis_path)
            return
        if not (ROBOT_DIR / "hierarchy.json").exists():
            return
        hierarchy = json.loads((ROBOT_DIR / "hierarchy.json").read_text(encoding="utf-8"))
        components = json.loads((ROBOT_DIR / "components.json").read_text(encoding="utf-8"))
        mesh_catalog = json.loads((ROBOT_DIR / "mesh_catalog.json").read_text(encoding="utf-8"))
        mesh_files = {item["source"]["key"]: ROBOT_DIR / item["file"] for item in mesh_catalog}
        enabled = {
            item["owner_path"]: bool(item["values"].get("m_Enabled", True))
            for item in components if item["source"]["type"] == "MeshRenderer"
        }
        world_by_path = {
            item["hierarchy_path"]: np.asarray(item["world_prefab_space"]["matrix_row_major"], dtype=np.float32)
            for item in hierarchy
        }
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
            indices = np.asarray([tri for submesh in mesh["submesh_triangles"] for tri in submesh], dtype=np.int32)
            if indices.size:
                transformed = transform_points(vertices, world_by_path[path])
                visual_blocks.append(transformed[indices])
        if not visual_blocks:
            return
        raw_visual = np.concatenate(visual_blocks)
        all_points = raw_visual.reshape(-1, 3)
        origin = np.array([(all_points[:, 0].min() + all_points[:, 0].max()) * 0.5, all_points[:, 1].min(), (all_points[:, 2].min() + all_points[:, 2].max()) * 0.5], dtype=np.float32)
        robot = self.UsdGeom.Xform.Define(self.stage, "/World/Robot/LegacyRobot")
        # Spawn on the +X BUMP lane so the live demo exercises ramp physics.
        robot.AddTranslateOp().Set(self.Gf.Vec3d(1.52, -5.55, 0.035))
        prim = robot.GetPrim()
        body = self.UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateVelocityAttr(self.Gf.Vec3f(0.0, 0.95, 0.0))
        self.UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(32.91)
        self.PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(10)
        visual_triangles = unity_to_usd(raw_visual - origin)
        visual_prim = self._mesh("/World/Robot/LegacyRobot/Visual", visual_triangles, PALETTE["robot"])
        self.UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(self._preview_material("robot", PALETTE["robot"]))

        collider_count = 0
        for component in components:
            kind = component["source"]["type"]
            if kind not in {"BoxCollider", "SphereCollider", "CapsuleCollider"}:
                continue
            values, path = component["values"], component["owner_path"]
            if values.get("m_IsTrigger") or not values.get("m_Enabled", True) or path not in world_by_path:
                continue
            center = values.get("m_Center", {"x": 0, "y": 0, "z": 0})
            center_v = np.array([center["x"], center["y"], center["z"]], dtype=np.float32)
            if kind == "BoxCollider":
                size = values["m_Size"]
                half = np.array([size["x"], size["y"], size["z"]], dtype=np.float32) * 0.5
                points = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])], dtype=np.float32) + center_v
                faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]], dtype=np.int32)
                local_triangles = points[faces]
            elif kind == "SphereCollider":
                local_triangles = self._shape_sphere(float(values["m_Radius"])) + center_v
            else:
                local_triangles = self._shape_capsule(float(values["m_Radius"]), float(values["m_Height"]), int(values["m_Direction"])) + center_v
            world_triangles = transform_points(local_triangles.reshape(-1, 3), world_by_path[path]).reshape(-1, 3, 3)
            local_usd = unity_to_usd(world_triangles - origin)
            collider_prim = self._mesh(f"/World/Robot/LegacyRobot/Colliders/C{collider_count:02d}", local_usd, (0.1, 0.8, 0.2, 0.16), collision=True)
            self.UsdPhysics.MeshCollisionAPI.Apply(collider_prim).CreateApproximationAttr("convexHull")
            collider_prim.GetAttribute("visibility").Set("inherited" if self.debug_colliders else "invisible")
            collider_count += 1
        self.robot_prim = prim
        self.stats.update(robot_visual_triangles=len(visual_triangles), robot_colliders=collider_count, robot_mass_kg=32.91)

    def build_physics_scene(self) -> None:
        UsdPhysics, PhysxSchema, Gf = self.UsdPhysics, self.PhysxSchema, self.Gf
        scene = UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        # Scene CCD flag ON, but only bodies that opt in pay for sweeps: the
        # robot chassis + mechanism links enable per-body CCD (they cross
        # several centimetres per 60 Hz step and the detailed colliders keep
        # 4 mm contact envelopes, so hard wall/hub hits used to interpenetrate
        # or tunnel).  All 456 FUEL stay non-CCD - the old full-field sweep
        # cost does not return.
        physx.CreateEnableCCDAttr(True)
        physx.CreateEnableEnhancedDeterminismAttr(False)
        # 60 Hz (was 250): 250 steps/s could not run real-time with the ball
        # field.  The GUI SimulationContext uses a matching 1/60 dt.
        physx.CreateTimeStepsPerSecondAttr(60)
        # GPU dynamics at 60 Hz: parallel-solve the ball field on the RTX GPU.
        # (At 250 Hz the per-step CPU<->GPU sync dominated; at 60 Hz there are 4x
        # fewer syncs, so the GPU parallel solve wins for the many-body field.)
        physx.CreateEnableGPUDynamicsAttr(True)
        physx.CreateBroadphaseTypeAttr("GPU")
        physx.CreateGpuFoundLostPairsCapacityAttr(2097152)
        physx.CreateGpuTotalAggregatePairsCapacityAttr(2097152)
        physx.CreateGpuMaxRigidContactCountAttr(2097152)
        physx.CreateGpuMaxRigidPatchCountAttr(327680)

    def build(self) -> dict[str, int]:
        self.UsdGeom.SetStageUpAxis(self.stage, self.UsdGeom.Tokens.z)
        self.UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.UsdGeom.Xform.Define(self.stage, "/World")
        self.UsdGeom.Xform.Define(self.stage, "/World/FieldVisual")
        self.UsdGeom.Xform.Define(self.stage, "/World/Fuel")
        self.UsdGeom.Xform.Define(self.stage, "/World/Robot")
        self.build_physics_scene()
        self.build_lighting()
        self.build_field_visuals()
        self.build_field_collisions()
        self.build_fuel()
        self.build_robot()
        return self.stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pristine-xRC REBUILT Isaac scene")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 keeps the interactive viewer open")
    parser.add_argument("--max-fuel", type=int, default=456, help="dynamic field FUEL bodies (full field = 456)")
    parser.add_argument("--debug-colliders", action="store_true")
    parser.add_argument("--no-panels", action="store_true", help="legacy fallback option")
    parser.add_argument("--no-turret", action="store_true", help="legacy fallback option")
    parser.add_argument("--no-autopilot", action="store_true", help="leave Competition Robot stationary for manual inspection")
    parser.add_argument(
        "--hub-demo",
        action="store_true",
        help="showcase only: seed two FUEL into the HUB sensor volumes "
        "(formal matches start with every ball at its official spot)",
    )
    parser.add_argument("--compound-robot", action="store_true", help="use the legacy single-rigid-body robot instead of the articulation")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional USD export path (omitted by default to avoid an ~80 MB cache)",
    )
    parser.add_argument("--stats", type=Path, default=PROJECT_ROOT / "runs" / "scene_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Prevent Kit from interpreting this script's --frames/--max-fuel flags.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": args.headless, "width": 1600, "height": 900, "multi_gpu": False}
    )
    print("XRC_REBUILT_APP_READY", app.is_running(), flush=True)
    # Fabric keeps all 456 moving FUEL transforms on the fast scene delegate.
    # Onboard GUI views use GPU viewports directly (not CPU RGB annotators), so
    # they are compatible with updateToUsd=False and avoid a per-frame readback.
    if not args.headless:
        try:
            import carb
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("omni.physx.fabric")
            _fab = carb.settings.get_settings()
            _fab.set_bool("/app/useFabricSceneDelegate", True)
            _fab.set_bool("/physics/updateToUsd", False)
            _fab.set_bool("/physics/updateVelocitiesToUsd", False)
            _fab.set_bool("/physics/fabricUpdateTransformations", True)
            print("XRC_REBUILT_FABRIC on (GPU camera viewports)", flush=True)
        except Exception as _fab_err:  # noqa: BLE001
            print("XRC_REBUILT_FABRIC_FAILED", repr(_fab_err), flush=True)
    try:
        import omni.timeline
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.viewports import set_camera_view

        context = omni.usd.get_context()
        print("XRC_REBUILT_BUILD_START", flush=True)
        context.new_stage()
        stage = context.get_stage()
        builder = SceneBuilder(
            stage,
            debug_colliders=args.debug_colliders,
            max_fuel=args.max_fuel,
            articulated_robot=not args.compound_robot,
            show_panels=not args.no_panels,
            show_turret=not args.no_turret,
        )
        stats = builder.build()
        # Start close enough to inspect the visible magazine and shot arc while
        # retaining the blue HUB, net, bump and trench in the same live view.
        set_camera_view(
            eye=np.array([7.2, -10.2, 4.6]),
            target=np.array([0.0, -3.9, 0.65]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        # Camera navigation remains available, but simulation objects cannot be
        # selected with click, drag-box, or the viewport context menu.
        viewport_selection_guard = None
        viewport_context_menu_guard = None
        if not args.headless:
            try:
                from omni.kit.viewport.utility import (
                    disable_context_menu,
                    disable_selection,
                    get_active_viewport,
                )

                viewport_api = get_active_viewport()
                if viewport_api is not None:
                    viewport_selection_guard = disable_selection(
                        viewport_api, disable_click=True
                    )
                    viewport_context_menu_guard = disable_context_menu(viewport_api)
                context.get_selection().clear_selected_prim_paths()
                print("XRC_REBUILT_VIEWPORT_SELECTION disabled", flush=True)
            except Exception as selection_error:
                print(
                    f"XRC_REBUILT_VIEWPORT_SELECTION_FAILED {selection_error!r}",
                    flush=True,
                )
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            stage.GetRootLayer().Export(str(args.save))
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(
            json.dumps({**stats, "usd": str(args.save) if args.save else None}, indent=2),
            encoding="utf-8",
        )
        print("XRC_REBUILT_SCENE", json.dumps(stats), flush=True)

        controls = {
            "hold": False,             # SPACE / dashboard button: auto aim & shoot
            "ferry": False,            # F: lob FUEL back toward our own zone
            "intake": False,           # I: intake rollers on/off (off at start)
            "storage_extended": False, # C or N: extend / compact (start compact)
            "hub": "blue",             # fixed alliance HUB for this blue robot
            "auto_target": False,      # never switch to the opponent's red HUB
            "first_inactive": None,    # decided at AUTO end from real AUTO FUEL
            "auto_fuel": None,         # AUTO snapshot {"red": n, "blue": n}
        }
        match_ref = {"t0": 0.0}  # mutable so the dashboard RESET button can restart
        last_status: dict[str, Any] = {}
        target_reason = "auto"
        status_window = None
        status_labels: dict[str, Any] = {}
        gui_camera_viewports: dict[str, Any] = {}
        # Live inspection viewports are deliberately smaller than the policy
        # observations. Training still receives full 640x360 frames.
        GUI_CAMERA_RENDER = (320, 180)
        if not args.headless:
            import omni.ui as ui

            ACCENT = 0xFF3CB0F0

            def toggle_intake() -> None:
                controls["intake"] = not controls["intake"]

            def toggle_storage() -> None:
                extending = not controls["storage_extended"]
                controls["storage_extended"] = extending
                controls["intake"] = extending

            def hold_press(*_args) -> None:
                controls["hold"] = True

            def hold_release(*_args) -> None:
                controls["hold"] = False

            def ferry_press(*_args) -> None:
                controls["ferry"] = True

            def ferry_release(*_args) -> None:
                controls["ferry"] = False

            def reset_match() -> None:
                # restart the REBUILT match clock and per-match scoring
                match_ref["t0"] = sim.current_time
                controls["first_inactive"] = None
                controls["auto_fuel"] = None
                router.scored = {"red": 0, "blue": 0}

            def declutter_isaac_ui() -> None:
                """Hide Isaac's default panels: viewport + dashboard only."""
                try:
                    import omni.kit.mainwindow

                    omni.kit.mainwindow.get_main_window().get_main_menu_bar().visible = False
                except Exception:
                    pass
                keep = {"Competition Robot"}
                protected = {"DockSpace", "Status Bar", "Debug"}
                try:
                    for window in ui.Workspace.get_windows():
                        title = (getattr(window, "title", "") or "").strip()
                        if (
                            not title
                            or title in keep
                            or title in protected
                            or title.startswith("Viewport")
                        ):
                            continue
                        try:
                            window.visible = False
                        except Exception:
                            pass
                except Exception:
                    pass

            status_window = ui.Window(
                "Competition Robot",
                width=430,
                height=560,
                visible=True,
                dockPreference=ui.DockPreference.LEFT,
            )
            with status_window.frame:
                with ui.VStack(spacing=7):
                    ui.Label("REBUILT  |  Competition Robot", style={"font_size": 22, "color": ACCENT})
                    status_labels["clock"] = ui.Label(
                        "AUTO  0:20", style={"font_size": 40, "color": 0xFFFFFFFF}
                    )
                    status_labels["phase"] = ui.Label(
                        "AUTO - both HUBS active", style={"font_size": 16}
                    )
                    with ui.HStack(spacing=10, height=24):
                        status_labels["hub_red"] = ui.Label(
                            "RED HUB - ACTIVE", style={"font_size": 16, "color": 0xFF35D435}
                        )
                        status_labels["hub_blue"] = ui.Label(
                            "BLUE HUB - ACTIVE", style={"font_size": 16, "color": 0xFF35D435}
                        )
                    status_labels["score"] = ui.Label(
                        "FUEL scored   RED 0   BLUE 0", style={"font_size": 22}
                    )
                    status_labels["fuel_inventory"] = ui.Label(
                        f"xRC FUEL total {stats.get('fuel_total_xrc', 0)}  |  "
                        f"field {stats.get('fuel_field_at_start', 0)} + "
                        f"robot {stats.get('fuel_robot_preloaded', 0)}",
                        style={"font_size": 16},
                    )
                    ui.Spacer(height=6)
                    status_labels["robot"] = ui.Label("Robot COMPACT | intake OFF | hopper 8")
                    status_labels["aim"] = ui.Label("Auto aim idle - hold SPACE to aim & shoot")
                    ui.Spacer(height=6)
                    ui.Label("CONTROLS", style={"color": ACCENT, "font_size": 16})
                    ui.Label("Drive: W/A/S/D translate  |  LEFT/RIGHT turn  (UP/DOWN = fwd/back)")
                    ui.Label("C or N: compact / extend        I: intake on/off")
                    ui.Label("SPACE (hold): auto aim & shoot into the HUB")
                    ui.Label("F (hold): ferry FUEL back toward our zone (side lanes only)")
                    with ui.HStack(spacing=6, height=34):
                        ui.Button("COMPACT / EXTEND", clicked_fn=toggle_storage)
                        ui.Button("INTAKE ON/OFF", clicked_fn=toggle_intake)
                    with ui.HStack(spacing=6, height=34):
                        ui.Button(
                            "AUTO AIM + SHOOT (hold)",
                            mouse_pressed_fn=hold_press,
                            mouse_released_fn=hold_release,
                        )
                        ui.Button(
                            "FERRY (hold)",
                            mouse_pressed_fn=ferry_press,
                            mouse_released_fn=ferry_release,
                        )
                    ui.Button("RESET MATCH", clicked_fn=reset_match, height=30)
            declutter_isaac_ui()

        fuel_view = RigidPrim("/World/Fuel/Fuel_.*", reset_xform_properties=False)
        articulated = bool(stats.get("robot_articulated"))
        robot_view = None
        if not articulated:
            robot_view = RigidPrim("/World/Robot/LegacyRobot", reset_xform_properties=False)
        # 60 Hz physics for a real-time interactive GUI (headless validation keeps
        # its own 250 Hz SimulationContext for shot fidelity).
        sim = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        sim.reset()
        fuel_view.initialize()
        controller = None
        if articulated:
            from isaacsim.core.prims import SingleArticulation
            from xrc_rebuilt.competition_robot import (
                ROBOT_ROOT_PATH,
                STORAGE_LOWERED_POSITION,
                XRC_PRELOAD_COUNT,
                CompetitionRobotController,
            )

            robot = SingleArticulation(ROBOT_ROOT_PATH)
            robot.initialize()
            controller = CompetitionRobotController(alliance_lock="blue")
            controller.initialize(robot)
            # G303 STARTING CONFIGURATION: begin compact (mechanisms stowed
            # inside the frame) with the intake off; deploy after the start.
            controller.snap_storage_state(False)
            controller.intake_on = False
            controller.preload(
                fuel_view,
                count=XRC_PRELOAD_COUNT,
                indices=getattr(builder, "preload_fuel_indices", None),
            )
        else:
            robot_view.initialize()
        # Separate GPU-backed viewport windows avoid Replicator's host readback
        # and are safe with Fabric. They are explicitly shown because the
        # stripped-down quick layout otherwise leaves newly created viewports
        # behind the primary viewport window.
        if not args.headless:
            from omni.kit.viewport.utility import create_viewport_window
            from pxr import Sdf
            from xrc_rebuilt.competition_robot import (
                CAMERA_BASELINE_NAMES as _CAM_NAMES,
                CAMERA_PRIM_PATHS as _CAM_PATHS,
            )

            def _lock_camera_viewport(vp) -> None:
                """Make a camera viewport display-only: hide its 'Camera'
                manipulator layer so it cannot capture WASD/keyboard (or mouse
                fly) from the robot.  Returns nothing; safe to call repeatedly
                since the layer is created lazily on the first render."""
                try:
                    layer = vp._find_viewport_layer("Camera", "manipulator")
                    if layer is not None:
                        layer.visible = False
                except Exception:
                    pass

            for _index, _cam_name in enumerate(_CAM_NAMES):
                try:
                    _title = f"Viewport {_cam_name.title()}"
                    _x = 440 + _index * (GUI_CAMERA_RENDER[0] + 12)
                    _viewport = create_viewport_window(
                        name=_title,
                        width=GUI_CAMERA_RENDER[0],
                        height=GUI_CAMERA_RENDER[1],
                        position_x=_x,
                        position_y=680,
                        camera_path=Sdf.Path(_CAM_PATHS[_cam_name]),
                    )
                    _viewport.position_x = _x
                    _viewport.position_y = 680
                    _viewport.viewport_api.resolution = GUI_CAMERA_RENDER
                    _viewport.viewport_api.camera_path = Sdf.Path(
                        _CAM_PATHS[_cam_name]
                    )
                    _viewport.visible = True
                    ui.Workspace.show_window(_title, True)
                    _lock_camera_viewport(_viewport)  # first attempt (may be early)
                    gui_camera_viewports[_cam_name] = _viewport
                    print(
                        f"GUI_GPU_CAMERA_READY {_cam_name} "
                        f"{GUI_CAMERA_RENDER[0]}x{GUI_CAMERA_RENDER[1]}",
                        flush=True,
                    )
                except Exception as _cam_err:  # noqa: BLE001
                    print(f"GUI GPU camera {_cam_name} unavailable: {_cam_err!r}", flush=True)
        # Phase F helpers: prebuild the static field map once for fast target select.
        from xrc_rebuilt.rules import (
            AUTO_DURATION_S,
            MATCH_DURATION_S,
            SCORING_ASSESSMENT_GRACE_S,
            arena_timer_seconds,
            select_first_inactive_alliance,
        )
        from xrc_rebuilt.shot_planner import FieldState, MatchState, RobotState, select_target_hub

        planner_field = FieldState()
        planner_field.occupancy()

        # ---- manual keyboard teleop: WASD/arrows drive; Space=continuous; E=e-stop ----
        pressed_keys: set[Any] = set()
        kbd_sub = None
        keyboard_input = None
        if not args.headless:
            import carb.input
            import omni.appwindow

            keyboard_input = carb.input.KeyboardInput
            _press_types = (carb.input.KeyboardEventType.KEY_PRESS,
                            carb.input.KeyboardEventType.KEY_REPEAT)

            def _on_keyboard(event, *_):
                if event.type in _press_types:
                    pressed_keys.add(event.input)
                    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                        if event.input == keyboard_input.SPACE:
                            controls["hold"] = True
                        elif event.input == keyboard_input.F:
                            controls["ferry"] = True
                        elif event.input == keyboard_input.I:
                            controls["intake"] = not controls["intake"]
                        elif event.input in (keyboard_input.N, keyboard_input.C):
                            toggle_storage()
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    pressed_keys.discard(event.input)
                    if event.input == keyboard_input.SPACE:
                        controls["hold"] = False
                    elif event.input == keyboard_input.F:
                        controls["ferry"] = False
                return True

            kbd_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
                omni.appwindow.get_default_app_window().get_keyboard(), _on_keyboard
            )

        def manual_drive_command():
            """(forward, strafe, turn) in [-1,1], or None when idle."""
            if keyboard_input is None or not pressed_keys:
                return None
            k = keyboard_input
            forward = (1.0 if (k.W in pressed_keys or k.UP in pressed_keys) else 0.0) \
                - (1.0 if (k.S in pressed_keys or k.DOWN in pressed_keys) else 0.0)
            strafe = (1.0 if k.A in pressed_keys else 0.0) \
                - (1.0 if k.D in pressed_keys else 0.0)
            turn = (1.0 if k.LEFT in pressed_keys else 0.0) \
                - (1.0 if k.RIGHT in pressed_keys else 0.0)
            return (forward, strafe, turn) if (forward or strafe or turn) else None

        router = HubRouter(fuel_view, len(builder.ball_prims))
        if args.hub_demo:
            # non-formal showcase only: drops one FUEL through each hub
            router.inject_demo()
        fuel_start, _ = fuel_view.get_world_poses()
        if articulated:
            robot_start = controller.chassis_pose()[0][None, :]
        else:
            robot_start, _ = robot_view.get_world_poses()
        frame = 0
        drive_direction = 1.0  # compound-robot fallback only
        max_robot_height = 0.0
        hub_chip_state: dict[str, str] = {}
        match_ref["t0"] = sim.current_time  # reset() warm-start = 2 physics steps (8 ms)
        import time as _time
        _perf_wall0 = _time.perf_counter()
        _perf_sim0 = sim.current_time
        _perf = {"step": 0.0, "render": 0.0, "control": 0.0}
        while app.is_running() and (args.frames <= 0 or frame < args.frames):
            if not args.no_autopilot and not articulated and frame % 2 == 0:
                # legacy single-rigid-body fallback (--compound-robot)
                robot_position, _ = robot_view.get_world_poses()
                robot_position_np = robot_position.detach().cpu().numpy() if hasattr(robot_position, "detach") else np.asarray(robot_position)
                max_robot_height = max(max_robot_height, float(robot_position_np[0, 2]))
                if robot_position_np[0, 1] > 6.2:
                    drive_direction = -1.0
                elif robot_position_np[0, 1] < -6.2:
                    drive_direction = 1.0
                velocity = robot_view.get_linear_velocities()
                velocity_np = velocity.detach().cpu().numpy() if hasattr(velocity, "detach") else np.asarray(velocity).copy()
                velocity_np[:, 0] = np.clip((1.52 - robot_position_np[:, 0]) * 2.5, -0.8, 0.8)
                velocity_np[:, 1] = 1.05 * drive_direction
                robot_view.set_linear_velocities(velocity_np)
            # step(render=True) runs a full Kit update: with rendering_dt=1/60 and
            # TimeStepsPerSecond=250 it advances FOUR 0.004 s physics steps (16 ms),
            # not one -- a 1.75x clock skew in GUI mode (docs/TIMING_AUDIT.md BUG 1).
            # Always advance physics by exactly one dt and refresh the viewport with
            # sim.render(), which steps zero physics.
            _t = _time.perf_counter()
            sim.step(render=False)
            _perf["step"] += _time.perf_counter() - _t
            if not args.headless and frame % 2 == 0:
                _t = _time.perf_counter()
                sim.render()
                _perf["render"] += _time.perf_counter() - _t
            frame += 1
            if frame % 120 == 0:  # PERF report
                _wall = _time.perf_counter() - _perf_wall0
                _rtf = (sim.current_time - _perf_sim0) / _wall if _wall > 0 else 0.0
                print(
                    f"PERF rtf={_rtf:.2f} iters/s={frame/_wall:.0f} "
                    f"step={_perf['step']/_wall*100:.0f}% render={_perf['render']/_wall*100:.0f}% "
                    f"control={_perf['control']/_wall*100:.0f}%",
                    flush=True,
                )
            if frame % 2 == 0:
                elapsed_sim = sim.current_time - match_ref["t0"]
                # ---- REBUILT match flow (2026 manual): AUTO (both active) ->
                # TRANSITION (both active) -> SHIFT 1..4 alternating -> ENDGAME
                # (both active).  At AUTO end the higher AUTO scorer goes
                # INACTIVE first (tie: FMS random).  After the final buzzer the
                # field keeps playing as a SANDBOX until RESET MATCH.
                sandbox = elapsed_sim >= MATCH_DURATION_S
                # AUTO result assessed at 23 s, not 20 s: the manual gives a
                # 3-second scoring delay between AUTO and TELEOP, so AUTO
                # shots still in flight at the buzzer count toward who rests
                # first.  SHIFT 1 starts at 30 s - the decision still lands
                # well before it is needed.
                if controls["first_inactive"] is None and (
                    elapsed_sim >= AUTO_DURATION_S + SCORING_ASSESSMENT_GRACE_S
                ):
                    controls["auto_fuel"] = dict(router.scored)
                    controls["first_inactive"] = select_first_inactive_alliance(
                        router.scored["red"], router.scored["blue"], seed=2026
                    ).value
                match_state = MatchState(
                    min(elapsed_sim, MATCH_DURATION_S - 1e-3), controls["first_inactive"]
                )
                router.active = {
                    alliance: True if sandbox else match_state.hub_active(alliance)
                    for alliance in ("red", "blue")
                }
                router.sandbox = sandbox
                router.match_first_inactive = controls["first_inactive"]
                builder.update_hub_lights(
                    elapsed_sim, controls["first_inactive"]
                )
                router.step(elapsed_sim)
                # Dashboard mirrors the physical xRC light meshes, including
                # transition chase and the post-MATCH white assessment state.
                if not args.headless and frame % 6 == 0:
                    for alliance in ("red", "blue"):
                        hub_chip_state[alliance] = builder.hub_light_state[alliance]
                if articulated and not args.no_autopilot:
                    position, _ = controller.chassis_pose()
                    max_robot_height = max(max_robot_height, float(position[2]))
                    controller.intake_on = bool(controls["intake"])
                    controller.set_storage_extended(bool(controls["storage_extended"]))
                    controller.step_mechanisms(dt_s=2.0 / 60.0)
                    controller.step_intake(fuel_view, set(router.pending), dt_s=2.0 / 60.0)
                    # No scripted AUTO routine: the sim never drives or shoots
                    # by itself, but manual controls retain normal permissions.
                    auto_phase = (not sandbox) and elapsed_sim < AUTO_DURATION_S
                    sm = controller.state_machine
                    sm.set_continuous(False)
                    fire_request = bool(controls["hold"] or controls["ferry"])
                    sm.press_hold() if fire_request else sm.release_hold()
                    sm.set_emergency_stop(False)
                    sm.auto_align = fire_request
                    # ---- match-aware target selection (always automatic) ----
                    if controls["auto_target"] and frame % 250 == 0:
                        pos, _ = controller.chassis_pose()
                        if sandbox:
                            controls["hub"] = "red" if float(pos[1]) > 0.0 else "blue"
                            target_reason = "sandbox_nearest"
                        else:
                            picked, target_reason = select_target_hub(
                                RobotState(float(pos[0]), float(pos[1])),
                                match_state,
                                field_state=planner_field,
                                preferred=str(controls["hub"]),
                                drive_fallback=False,
                            )
                            if picked is not None:
                                controls["hub"] = picked
                    hub_active = True if sandbox else match_state.hub_active(str(controls["hub"]))
                    # Human teleop overrides autopilot driving; the shooter FSM
                    # still auto-aims/fires (its speed gate holds fire until you
                    # stop), so you drive anywhere and it scores when in range.
                    manual = manual_drive_command()
                    if manual is not None:
                        controller.drive(manual[0], manual[2], strafe=manual[1])
                    _tc = _time.perf_counter()
                    last_status = controller.update(
                        fuel_view,
                        now_s=elapsed_sim,
                        alliance=str(controls["hub"]),
                        hub_active=hub_active,
                        allow_drive=(manual is None),
                        fire_mode=(
                            "ferry"
                            if controls["ferry"] and not controls["hold"]
                            else "score"
                        ),
                    )
                    _perf["control"] += _time.perf_counter() - _tc
            if not args.headless and frame == 30:
                declutter_isaac_ui()  # catch panels that appear after startup
                # re-lock the camera viewports now that their lazily-created
                # 'Camera' manipulator layers exist, so WASD stays with the robot
                for _vp in gui_camera_viewports.values():
                    try:
                        _cl = _vp._find_viewport_layer("Camera", "manipulator")
                        if _cl is not None:
                            _cl.visible = False
                    except Exception:
                        pass
            if status_labels and frame % 15 == 0:
                elapsed = max(0.0, sim.current_time - match_ref["t0"])
                sandbox_ui = elapsed >= MATCH_DURATION_S
                ms = MatchState(
                    min(elapsed, MATCH_DURATION_S - 1e-3), controls["first_inactive"]
                )
                phase_display = {
                    "auto": "AUTO - manual controls enabled; no scripted actions",
                    "transition": "TRANSITION SHIFT - both HUBS active",
                    "shift_1": "SHIFT 1",
                    "shift_2": "SHIFT 2",
                    "shift_3": "SHIFT 3",
                    "shift_4": "SHIFT 4",
                    "endgame": "END GAME - both HUBS active",
                }
                if sandbox_ui:
                    status_labels["clock"].text = "0:00"
                    status_labels["phase"].text = (
                        "MATCH COMPLETE - sandbox free play (RESET MATCH to replay)"
                    )
                else:
                    arena = arena_timer_seconds(elapsed)
                    minutes, seconds = divmod(int(math.ceil(arena)), 60)
                    prefix = "AUTO" if ms.phase() == "auto" else "TELEOP"
                    status_labels["clock"].text = f"{prefix}  {minutes}:{seconds:02d}"
                    status_labels["phase"].text = phase_display.get(ms.phase(), ms.phase())
                chip_colors = {
                    "active": 0xFF35D435,
                    "warning": 0xFF3CB0F0,
                    "inactive": 0xFF3535E8,
                    "transition_chase": 0xFFFFFFFF,
                    "post_match_white": 0xFFFFFFFF,
                }
                for alliance in ("red", "blue"):
                    chip = hub_chip_state.get(alliance) or (
                        "active" if router.active.get(alliance, True) else "inactive"
                    )
                    label = status_labels[f"hub_{alliance}"]
                    label.text = f"{alliance.upper()} HUB - {chip.upper()}"
                    try:
                        label.style = {"font_size": 16, "color": chip_colors[chip]}
                    except Exception:
                        pass
                score_txt = (
                    f"FUEL scored   RED {router.scored['red']}   BLUE {router.scored['blue']}"
                )
                if controls["auto_fuel"]:
                    autos = controls["auto_fuel"]
                    resting = str(controls["first_inactive"] or "").upper()
                    score_txt += (
                        f"   (AUTO {autos['red']}-{autos['blue']}, {resting} rests first)"
                    )
                status_labels["score"].text = score_txt
                if controller is not None:
                    mode = "EXTENDED" if controls["storage_extended"] else "COMPACT"
                    status_labels["robot"].text = (
                        f"Robot {mode} | intake {'ON' if controls['intake'] else 'OFF'} | "
                        f"hopper {len(controller.magazine)} | fired {controller.shots_fired} | "
                        f"target {str(controls['hub']).upper()}"
                    )
                    solution = controller.last_aim_solution
                    st = last_status or {}
                    auto_ui = (not sandbox_ui) and elapsed < AUTO_DURATION_S
                    if (controls["hold"] or controls["ferry"]) and solution:
                        kind = (
                            "FERRY" if solution.get("mode") == "ferry" else "Auto aim"
                        )
                        if not bool(solution["valid"]):
                            aim_state = "BLOCKED: " + str(
                                solution.get("blocked_reason", "no shot")
                            )
                        elif bool(solution["aligned"]):
                            aim_state = "FIRING"
                        else:
                            aim_state = "LOCKING"
                        status_labels["aim"].text = (
                            f"{kind} {aim_state} | d={float(solution.get('distance_m', 0.0)):.2f} m"
                            f" | {st.get('state', 'IDLE')}"
                        )
                    elif auto_ui:
                        status_labels["aim"].text = (
                            "AUTO ready - manual controls enabled; no auto-fire"
                        )
                    else:
                        status_labels["aim"].text = (
                            "Idle - SPACE: aim & shoot | F: ferry to our zone"
                        )
        fuel_end, _ = fuel_view.get_world_poses()
        if articulated:
            robot_end = controller.chassis_pose()[0][None, :]
        else:
            robot_end, _ = robot_view.get_world_poses()
        fuel_velocity = fuel_view.get_linear_velocities()
        as_numpy = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        fuel_start_np, fuel_end_np = as_numpy(fuel_start), as_numpy(fuel_end)
        robot_start_np, robot_end_np = as_numpy(robot_start), as_numpy(robot_end)
        velocity_np = as_numpy(fuel_velocity)
        diagnostics = {
            "frames": frame,
            "sim_seconds": sim.current_time - match_ref["t0"],
            "fuel_count": len(fuel_end_np),
            "fuel_mean_displacement_m": float(np.linalg.norm(fuel_end_np - fuel_start_np, axis=1).mean()),
            "fuel_moved_over_5cm": int((np.linalg.norm(fuel_end_np - fuel_start_np, axis=1) > 0.05).sum()),
            "fuel_height_range_m": [float(fuel_end_np[:, 2].min()), float(fuel_end_np[:, 2].max())],
            "fuel_max_speed_mps": float(np.linalg.norm(velocity_np, axis=1).max()),
            "robot_displacement_m": float(np.linalg.norm(robot_end_np - robot_start_np)),
            "robot_max_height_m": max_robot_height,
            "robot_start": robot_start_np.tolist(),
            "robot_end": robot_end_np.tolist(),
            **router.stats(),
            **(controller.stats() if controller is not None else {}),
        }
        (args.stats.parent / "physics_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print("XRC_REBUILT_PHYSICS", json.dumps(diagnostics), flush=True)
        sim.stop()
    except BaseException as error:
        import traceback
        print("XRC_REBUILT_ERROR", repr(error), flush=True)
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
