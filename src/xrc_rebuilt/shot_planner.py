"""Direct-shot solver + global drive-to-shoot planner (Phase D1/D2/D4).

Two public APIs, per assistant_NEXT_PLAN Phase D1:

    solve_direct_shot(robot_state, hub, field_state) -> ShotPlan | BlockedReason
    plan_global_score(robot_state, hub, field_state) -> DriveAndShootPlan

``solve_direct_shot`` reuses the single source of truth for xRC ballistics
(``robot_model.solve_shot_geometry``): the exact coupled 6..11 m/s /
70.355..50.355 deg law and dynamic muzzle.  Analytic ballistics are used as a
seed; a versioned calibration artifact (keyed by field/robot/ball/dt/Isaac
hashes) refines the aim and reports measured uncertainty once a PhysX
calibration pass has been run on the *matching* scene.  Until then results are
flagged ``calibrated=False`` and clearance/uncertainty are analytic proxies.

``plan_global_score`` interprets "score from anywhere" as an end-to-end command:
fire directly if a robust direct shot exists, otherwise drive (collision-free
A* over the exact field map) to a verified open-side firing pose and shoot.  It
never returns a blind shot; if nothing is reachable it returns ``unreachable``.

Pure Python + numpy; no Isaac.  Drive-following and the PhysX shot calibration
are separate deferred GPU steps.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xrc_rebuilt.field_map import (
    OccupancyGrid,
    candidate_firing_positions,
    plan_path,
    simplify_path,
)
from xrc_rebuilt.robot_model import (
    CALIBRATED_RANGE_MAX_M,
    CALIBRATED_RANGE_MIN_M,
    HUB_MARKBALL_TARGETS,
    SPEC_PATH,
    solve_shot_geometry,
    wrap_angle,
)
from xrc_rebuilt.rules import ALLIANCE_SHIFT_PHASES, hub_active_at, phase_at

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELD_COLLIDERS = PROJECT_ROOT / "assets" / "fresh_xrc" / "field" / "colliders.json"
CALIBRATION_DIR = PROJECT_ROOT / "runs" / "calibration"

# Settled articulation-root height used for the analytic seed (Isaac-validated
# blue pose was solved at ~this chassis origin height).
CHASSIS_ORIGIN_Z = 0.20
# Direct-shot envelope, measured in PhysX (see robot_model + shot_envelope.json).
CALIBRATED_RANGE = (CALIBRATED_RANGE_MIN_M, CALIBRATED_RANGE_MAX_M)
OPEN_SIDE_MARGIN = 0.45
VERTICAL_TOL = 0.025
# Conservative shot dispersion used until a PhysX calibration is available.
ANALYTIC_UNCERTAINTY_M = 0.05
# xRC FUEL physics constants (isaac_scene.build_fuel / _physics_material).
BALL_MASS_KG = 0.08
BALL_RESTITUTION = 0.5
BALL_RADIUS_M = 0.076
PHYSICS_DT = 0.004
ISAAC_VERSION = "5.1.0.0"


# --------------------------------------------------------------------------- #
# state + plan dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class RobotState:
    x: float
    y: float
    yaw_rad: float = 0.0
    z: float = CHASSIS_ORIGIN_Z
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate_dps: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class FieldState:
    active_hubs: tuple[str, ...] = ("red", "blue")
    grid: OccupancyGrid | None = None

    def occupancy(self) -> OccupancyGrid:
        if self.grid is None:
            self.grid = OccupancyGrid()
        return self.grid


@dataclass
class ShotPlan:
    hub: str
    aim: float
    speed_mps: float
    pitch_deg: float
    muzzle_pose: tuple[float, float, float]
    exit_direction: tuple[float, float, float]
    desired_yaw_rad: float
    flight_time_s: float
    range_m: float
    vertical_error_m: float
    clearance_margin_m: float
    uncertainty_m: float
    calibrated: bool
    valid: bool = True
    reason: str = "direct"


@dataclass
class BlockedReason:
    hub: str
    reason: str  # unsafe_side | under_range | over_range | no_trajectory | obstructed
    range_m: float
    detail: str = ""
    valid: bool = False


@dataclass
class DriveAndShootPlan:
    hub: str
    firing_pose: tuple[float, float, float] | None  # (x, y, yaw)
    path: list[tuple[float, float]]
    arrival_yaw_rad: float
    braking_from_index: int
    shot: ShotPlan | None
    replan_policy: str
    valid: bool
    reason: str


# --------------------------------------------------------------------------- #
# versioned calibration artifact
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationKey:
    field_hash: str
    robot_hash: str
    ball_mass_kg: float
    ball_restitution: float
    ball_radius_m: float
    physics_dt: float
    isaac_version: str


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def current_calibration_key() -> CalibrationKey:
    return CalibrationKey(
        field_hash=_sha16(FIELD_COLLIDERS),
        robot_hash=_sha16(SPEC_PATH),
        ball_mass_kg=BALL_MASS_KG,
        ball_restitution=BALL_RESTITUTION,
        ball_radius_m=BALL_RADIUS_M,
        physics_dt=PHYSICS_DT,
        isaac_version=ISAAC_VERSION,
    )


@dataclass
class ShotCalibration:
    """PhysX-refined aim corrections + measured uncertainty for ONE scene.

    Empty until a PhysX calibration pass populates it (deferred GPU step).
    ``matches`` guards against silently using a LUT from a different scene.
    """

    key: CalibrationKey
    aim_offset_by_hub: dict[str, float] = field(default_factory=dict)
    uncertainty_by_hub: dict[str, float] = field(default_factory=dict)

    def matches(self, key: CalibrationKey) -> bool:
        return asdict(self.key) == asdict(key)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "key": asdict(self.key),
                    "aim_offset_by_hub": self.aim_offset_by_hub,
                    "uncertainty_by_hub": self.uncertainty_by_hub,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ShotCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            key=CalibrationKey(**data["key"]),
            aim_offset_by_hub=data.get("aim_offset_by_hub", {}),
            uncertainty_by_hub=data.get("uncertainty_by_hub", {}),
        )


# --------------------------------------------------------------------------- #
# direct-shot solver
# --------------------------------------------------------------------------- #
def _open_side(hub: str, y: float, target_y: float) -> bool:
    if hub == "blue":
        return y < target_y - OPEN_SIDE_MARGIN
    return y > target_y + OPEN_SIDE_MARGIN


def solve_direct_shot(
    robot_state: RobotState,
    hub: str,
    field_state: FieldState | None = None,
    calibration: ShotCalibration | None = None,
    samples: int = 2001,
) -> ShotPlan | BlockedReason:
    """Best direct shot from ``robot_state`` to ``hub`` (or why it is blocked).

    ``samples`` is the aim-sweep resolution; use a coarse value for batch
    screening (planner candidates / feasibility map) and the full default for
    the shot actually taken.
    """
    if hub not in HUB_MARKBALL_TARGETS:
        raise ValueError(f"unknown hub {hub!r}")
    target = HUB_MARKBALL_TARGETS[hub]
    geometry = solve_shot_geometry(robot_state.position, target, samples=samples)
    range_m = float(geometry["range_m"])
    aim = float(geometry["aim"])
    uncertainty = ANALYTIC_UNCERTAINTY_M
    calibrated = False
    if calibration is not None and calibration.matches(current_calibration_key()):
        aim = float(np.clip(aim + calibration.aim_offset_by_hub.get(hub, 0.0), 0.0, 1.0))
        uncertainty = calibration.uncertainty_by_hub.get(hub, uncertainty)
        calibrated = True

    if not _open_side(hub, float(robot_state.y), float(target[1])):
        return BlockedReason(hub, "unsafe_side", range_m, "HUB structure blocks this side")
    if range_m < CALIBRATED_RANGE[0]:
        return BlockedReason(hub, "under_range", range_m, f"{range_m:.2f} m < {CALIBRATED_RANGE[0]} m")
    if range_m > CALIBRATED_RANGE[1]:
        return BlockedReason(hub, "over_range", range_m, f"{range_m:.2f} m > {CALIBRATED_RANGE[1]} m")
    if abs(float(geometry["vertical_error_m"])) > VERTICAL_TOL:
        return BlockedReason(hub, "no_trajectory", range_m, "no xRC trajectory within shooter limits")

    # Analytic clearance proxy = distance to the nearest envelope edge; true
    # swept-sphere clearance vs HUB/net/tower is computed by the PhysX
    # calibration pass (deferred) and stored on the calibration artifact.
    clearance = round(min(range_m - CALIBRATED_RANGE[0], CALIBRATED_RANGE[1] - range_m), 4)
    return ShotPlan(
        hub=hub,
        aim=aim,
        speed_mps=float(geometry["speed_mps"]),
        pitch_deg=float(geometry["pitch_deg"]),
        muzzle_pose=tuple(round(float(v), 5) for v in geometry["muzzle_origin"]),
        exit_direction=tuple(round(float(v), 5) for v in geometry["exit_direction"]),
        desired_yaw_rad=float(geometry["desired_yaw_rad"]),
        flight_time_s=float(geometry["flight_time_s"]),
        range_m=range_m,
        vertical_error_m=float(geometry["vertical_error_m"]),
        clearance_margin_m=clearance,
        uncertainty_m=uncertainty,
        calibrated=calibrated,
    )


# --------------------------------------------------------------------------- #
# global drive-to-shoot planner
# --------------------------------------------------------------------------- #
def _path_length(path: list[tuple[float, float]]) -> float:
    return float(sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)))


def _braking_index(path: list[tuple[float, float]], brake_dist: float = 0.5) -> int:
    total = 0.0
    for i in range(len(path) - 1, 0, -1):
        total += math.dist(path[i], path[i - 1])
        if total >= brake_dist:
            return i - 1
    return 0


def plan_global_score(
    robot_state: RobotState,
    hub: str,
    field_state: FieldState | None = None,
    calibration: ShotCalibration | None = None,
) -> DriveAndShootPlan:
    """End-to-end plan: fire directly if possible, else drive to a firing pose."""
    field_state = field_state or FieldState()
    grid = field_state.occupancy()

    direct = solve_direct_shot(robot_state, hub, field_state, calibration)
    if isinstance(direct, ShotPlan):
        return DriveAndShootPlan(
            hub=hub,
            firing_pose=(robot_state.x, robot_state.y, direct.desired_yaw_rad),
            path=[(robot_state.x, robot_state.y)],
            arrival_yaw_rad=direct.desired_yaw_rad,
            braking_from_index=0,
            shot=direct,
            replan_policy="none",
            valid=True,
            reason="direct",
        )

    target = HUB_MARKBALL_TARGETS[hub]
    open_sign = -1.0 if hub == "blue" else 1.0
    # Collect firing poses that are free and have a valid direct shot, then plan
    # a collision-free path to the closest ones first (A* is the expensive step).
    viable: list[tuple[float, tuple[float, float]]] = []
    for cx, cy in candidate_firing_positions((float(target[0]), float(target[1])), open_sign):
        if not grid.is_free(cx, cy):
            continue
        screen = solve_direct_shot(RobotState(cx, cy), hub, field_state, calibration, samples=401)
        if isinstance(screen, ShotPlan):
            viable.append((math.dist((robot_state.x, robot_state.y), (cx, cy)), (cx, cy)))
    viable.sort(key=lambda v: v[0])

    chosen: tuple[tuple[float, float], ShotPlan, list[tuple[float, float]]] | None = None
    for _, (cx, cy) in viable:
        path = plan_path(grid, (robot_state.x, robot_state.y), (cx, cy))
        if path is not None:
            # full-precision shot for the pose we actually commit to
            shot = solve_direct_shot(RobotState(cx, cy), hub, field_state, calibration)
            if isinstance(shot, ShotPlan):
                # dense A* path (smoothing regressed closed-loop tracking; kept as
                # a utility for future kinodynamic work, not applied here)
                chosen = ((cx, cy), shot, path)
                break

    if chosen is None:
        return DriveAndShootPlan(
            hub=hub, firing_pose=None, path=[], arrival_yaw_rad=0.0,
            braking_from_index=0, shot=None, replan_policy="none",
            valid=False, reason="unreachable",
        )
    (cx, cy), shot, path = chosen
    return DriveAndShootPlan(
        hub=hub,
        firing_pose=(cx, cy, shot.desired_yaw_rad),
        path=path,
        arrival_yaw_rad=shot.desired_yaw_rad,
        braking_from_index=_braking_index(path),
        shot=shot,
        replan_policy="replan_if_stalled",
        valid=True,
        reason="drive_to_shoot",
    )


# --------------------------------------------------------------------------- #
# differential pure-pursuit path follower (Phase D4 drive-follow)
# --------------------------------------------------------------------------- #
ARRIVE_TOL_M = 0.30    # firing pose has +/- tolerance; the solver re-aims on arrival
ARRIVE_YAW_TOL_DEG = 2.0
LOOKAHEAD_M = 0.35     # track the path tightly so corners are not cut into obstacles
TURN_IN_PLACE_RAD = 1.2  # only spin in place when the target is truly behind
SLOWDOWN_RADIUS_M = 0.8


def _lookahead_point(
    path: list[tuple[float, float]], pos: tuple[float, float], lookahead: float
) -> tuple[float, float]:
    """Point interpolated ALONG the path ~``lookahead`` from ``pos``.

    Interpolating within segments (not snapping to the next vertex) keeps
    tracking tight on smoothed paths whose vertices are metres apart, so the
    robot follows the route instead of beelining to a distant corner.
    """
    closest = min(range(len(path)), key=lambda i: math.dist(path[i], pos))
    for k in range(closest, len(path) - 1):
        a, b = path[k], path[k + 1]
        seg = math.dist(a, b)
        steps = max(1, int(seg / 0.05))
        for s in range(steps + 1):
            t = s / steps
            point = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if math.dist(point, pos) >= lookahead:
                return point
    return path[-1]


def pursuit_command(
    pose: tuple[float, float, float],
    path: list[tuple[float, float]],
    arrival_yaw: float,
    lookahead: float = LOOKAHEAD_M,
    arrive_tol: float = ARRIVE_TOL_M,
) -> dict[str, Any]:
    """One tick of differential pure-pursuit toward the end of ``path``, then
    orient to ``arrival_yaw``.  Returns forward/turn in [-1, 1] and a phase
    (driving / turning / orienting / arrived).  Pure function (no Isaac);
    ``RobotController.follow`` applies it to the differential drivetrain.
    """
    x, y, yaw = pose
    dist_goal = math.dist((x, y), path[-1])
    if dist_goal <= arrive_tol:
        yaw_err = wrap_angle(arrival_yaw - yaw)
        if abs(math.degrees(yaw_err)) <= ARRIVE_YAW_TOL_DEG:
            return {"forward": 0.0, "turn": 0.0, "phase": "arrived",
                    "dist_goal": dist_goal, "yaw_err_deg": math.degrees(yaw_err)}
        turn = float(np.clip(2.5 * yaw_err, -1.0, 1.0))
        turn = turn if abs(turn) >= 0.08 else math.copysign(0.08, turn)
        return {"forward": 0.0, "turn": turn, "phase": "orienting",
                "dist_goal": dist_goal, "yaw_err_deg": math.degrees(yaw_err)}
    tx, ty = _lookahead_point(path, (x, y), lookahead)
    heading_err = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
    if abs(heading_err) > TURN_IN_PLACE_RAD:
        turn = float(np.clip(1.6 * heading_err, -1.0, 1.0))
        turn = turn if abs(turn) >= 0.10 else math.copysign(0.10, turn)
        return {"forward": 0.0, "turn": turn, "phase": "turning",
                "dist_goal": dist_goal, "yaw_err_deg": math.degrees(heading_err)}
    # Arc toward the target: forward scales with heading alignment so the robot
    # slows through turns (never cutting corners into obstacles) and decelerates
    # to a stop near the goal (no speed floor -> settles instead of orbiting).
    alignment = max(0.0, math.cos(heading_err))
    slowdown = min(1.0, dist_goal / SLOWDOWN_RADIUS_M)
    return {
        "forward": float(np.clip(alignment * slowdown, 0.0, 1.0)),
        "turn": float(np.clip(1.5 * heading_err, -1.0, 1.0)),
        "phase": "driving", "dist_goal": dist_goal, "yaw_err_deg": math.degrees(heading_err),
    }


# --------------------------------------------------------------------------- #
# match-aware autonomous target selection (Phase F)
# --------------------------------------------------------------------------- #
@dataclass
class MatchState:
    """Live 160 s match state used to gate and select the target HUB."""

    elapsed_s: float
    first_inactive: str | None = None  # set at SHIFT 1 from AUTO FUEL results

    def phase(self) -> str:
        return phase_at(self.elapsed_s).value

    def hub_active(self, hub: str) -> bool:
        phase = phase_at(self.elapsed_s)
        if phase in ALLIANCE_SHIFT_PHASES and self.first_inactive is None:
            return True  # pre-determination: treat both HUBs as active
        return bool(hub_active_at(hub, self.elapsed_s, self.first_inactive))

    def active_hubs(self) -> list[str]:
        return [h for h in ("red", "blue") if self.hub_active(h)]


def select_target_hub(
    robot_state: RobotState,
    match_state: MatchState,
    field_state: FieldState | None = None,
    calibration: ShotCalibration | None = None,
    preferred: str | None = None,
    drive_fallback: bool = True,
) -> tuple[str | None, str]:
    """Pick the legal/active HUB to score in (not merely the nearest).

    Prefers an active HUB with a valid direct shot from the current pose,
    keeping ``preferred`` if it is still valid; else (when ``drive_fallback``)
    an active HUB reachable by driving; else ``(None, reason)``.  The caller
    re-runs this to switch HUBs the instant the selected one deactivates.
    ``drive_fallback=False`` keeps it to the fast direct screen for live use.
    """
    field_state = field_state or FieldState()
    active = match_state.active_hubs()
    if not active:
        return None, "no active HUB"
    direct = []
    for hub in active:
        result = solve_direct_shot(robot_state, hub, field_state, calibration, samples=401)
        if isinstance(result, ShotPlan):
            direct.append((result.range_m, hub))
    if direct:
        valid = [hub for _, hub in direct]
        if preferred in valid:
            return preferred, "direct"
        direct.sort()
        return direct[0][1], "direct"
    if not drive_fallback:
        return None, "no direct shot"
    for hub in active:
        if plan_global_score(robot_state, hub, field_state, calibration).valid:
            return hub, "drive_to_shoot"
    return None, "unreachable"


# --------------------------------------------------------------------------- #
# analytic full-field feasibility map (Phase D3 seed)
# --------------------------------------------------------------------------- #
def feasibility_map(
    field_state: FieldState | None = None,
    calibration: ShotCalibration | None = None,
    pos_step: float = 0.25,
    region: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Classify every legal floor cell for red and blue by best-heading direct
    shot.  This is the analytic seed; the robust randomized-physics evaluation
    (Phase D3) refines it in a PhysX pass.  Heading is solved per cell, so no
    yaw sweep is needed for direct feasibility.  ``region`` limits the sampled
    area (used by tests); defaults to the full legal floor.
    """
    field_state = field_state or FieldState()
    grid = field_state.occupancy()
    x0, x1, y0, y1 = region if region is not None else (grid.x0, grid.x1, grid.y0, grid.y1)
    cells: list[dict[str, Any]] = []
    counts = {"red": {}, "blue": {}}
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            if grid.is_free(x, y):
                cell: dict[str, Any] = {"x": round(x, 3), "y": round(y, 3)}
                for hub in ("red", "blue"):
                    result = solve_direct_shot(RobotState(x, y), hub, field_state, calibration, samples=401)
                    label = "direct" if isinstance(result, ShotPlan) else result.reason
                    cell[hub] = label
                    counts[hub][label] = counts[hub].get(label, 0) + 1
                cells.append(cell)
            y += pos_step
        x += pos_step
    return {
        "pos_step_m": pos_step,
        "calibrated": bool(calibration is not None),
        "counts": counts,
        "cells": cells,
    }
