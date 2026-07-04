"""Imported the competition robot CAD, swerve drivetrain, intake and shooter controller."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from xrc_rebuilt.calibrated_controls import ShooterSetpoint, CalibratedRobotControl, wrap_angle
from xrc_rebuilt.robot_model import (
    ShooterInputs,
    ShooterState,
    ShooterStateMachine,
    quat_wxyz_to_matrix,
)
from xrc_rebuilt.swerve import (
    MAX_WHEEL_SPEED_MPS,
    MODULE_ORDER,
    MODULE_POSITIONS,
    WHEEL_RADIUS_M,
    field_to_robot,
    inverse_kinematics,
    optimize_module,
)

PROJECT = Path(__file__).resolve().parents[2]
ASSET_DIR = PROJECT / "assets" / "robot_runtime"
VISUAL_MESH = ASSET_DIR / "visual_mesh.npz"
MECHANISM_ASSET_DIR = ASSET_DIR / "mechanisms"
SPLIT_CAD_ASSETS = {
    "static": MECHANISM_ASSET_DIR / "static_chassis.npz",
    "intake": MECHANISM_ASSET_DIR / "intake.npz",
    "hopper_base": MECHANISM_ASSET_DIR / "hopper_base.npz",
    "hopper_horizontal": MECHANISM_ASSET_DIR / "hopper_horizontal.npz",
    "hopper_vertical": MECHANISM_ASSET_DIR / "hopper_vertical.npz",
    "shooter_retract": MECHANISM_ASSET_DIR / "shooter_retract.npz",
}
ROBOT_ROOT_PATH = "/World/Robot/CompetitionRobot"
ROBOT_MASS_KG = 52.0
# The robot uses a continuous high-throughput path (a 5000 RPM tunnel
# feeding an independently driven 1700 RPM final roller), rather than stopping
# between tidy "volleys".  Keep the physically irregular 1/2/3-wide release,
# but make three-wide contacts common and leave only a one-frame settling window
# when the requested group has not quite reached the muzzle.
SHOOTER_COOLDOWN_S = 0.030
SHOOTER_BATCH_WAIT_S = 0.018
SHOOTER_VOLLEY_WEIGHTS = (0.10, 0.28, 0.62)  # one, two, three FUEL
# Match-footage release model for a full-width competition shooter. The
# real shooter is one full-width grip-tape drum (25 in between the side
# plates), so FUEL exits anywhere along the row - there are no lanes - and
# every ball carries its own small speed/direction error from foam
# compression on the drum.  In the air that reads as an even curtain across
# the whole shooter with no repeating shape.  Three FUEL cannot sit abreast
# in a 0.48 m row without falling back into lanes (0.152 m diameter forces
# fixed windows), and the real drum never releases them truly abreast anyway:
# laterally-close FUEL leave milliseconds apart.  So lateral spots use the
# minimum-gap slack construction with a SUB-diameter lateral floor, and
# lateral neighbours are staggered a step along the flight path; the pair
# guarantees a centre distance of sqrt(LAT_GAP^2 + (LON_STEP-LON_RAND)^2)
# = 0.155 m > 0.152 m, so spawns never interpenetrate while the row reads
# as one continuous random curtain.  Cadence is jittered off the metronome.
SHOOTER_ROW_HALF_WIDTH_M = 0.24        # drum half-width minus ball radius
SHOOTER_LAT_GAP_M = 0.118              # lateral floor between volley FUEL
SHOOTER_LON_STEP_M = 0.115             # flight-path stagger, lat-neighbours
SHOOTER_LON_RAND_M = 0.015             # extra per-ball stagger randomness
# The curtain look comes from the spatial ladder + cadence above; velocity
# error only costs accuracy.  Jitters are sized so the worst-case stack-up at
# 4 m is ~0.06 m range and, with the convergence term cancelling most of the
# lateral spawn offset over the flight, ~0.10 m lateral - well inside the
# hub mouth.
SHOOTER_SPEED_JITTER_FRAC = 0.007      # +/-0.7% exit speed per ball
SHOOTER_YAW_JITTER_DEG = 0.5           # per-ball lateral direction error
SHOOTER_PITCH_JITTER_DEG = 0.3         # per-ball vertical direction error
SHOOTER_CONVERGE_FRAC = 0.75           # lateral offset fraction aimed out
SHOOTER_CADENCE_JITTER = (0.8, 1.7)    # cadence randomness between volleys
# Cadence normalised PER BALL, not per volley: after an n-wide release the
# next window is n * SHOOTER_BALL_PERIOD_S (x jitter), so the stream carries
# the same balls-per-second whether the hopper is full (3-wides) or nearly
# empty (1/2-wides).  Without this the tail of the magazine fired sparse
# singles at the full triple cadence - 3x thinner in the air.
SHOOTER_BALL_PERIOD_S = 0.0125
# xRC's source prefab exposes eight real PRELOAD anchors.  There is deliberately
# no software capacity: after preload, every FUEL remains a dynamic PhysX body
# and the hopper's collision walls/jamming determine how many more will fit.
XRC_PRELOAD_COUNT = 8
INTAKE_PULL_SPEED_MPS = 4.0
INTAKE_INDEX_SPEED_MPS = 1.50
INTAKE_SIMULTANEOUS_LANES = 3
INTAKE_LATERAL_RELEASE_LIMIT_M = 0.265
HOPPER_PRESSURE_CAPACITY = 60
# 152 mm foam FUEL can locally flatten by roughly 6% in a packed hopper.
# Release still requires 140 mm centre clearance; PhysX/net contacts decide the
# final packing and the retention acceptance rejects any pressure ejection.
FUEL_RELEASE_CLEARANCE_M = 0.140
SHOOTER_LANES = 3
SHOOTER_LANE_SPACING_M = 0.16
DRIVER_SPEED_RATE = 0.7
DRIVER_ANGULAR_RATE = 0.95
MAX_ANGULAR_RATE_RAD_S = 0.75 * 2.0 * math.pi
# A keyboard is always 100% stick.  Preserve robot's real limits while scaling
# binary keys to a controllable equivalent joystick deflection.
KEYBOARD_TRANSLATION_SCALE = 0.70
KEYBOARD_TURN_SCALE = 0.65
DRIVER_ACCEL_LIMIT_MPS2 = 2.8
DRIVER_DECEL_LIMIT_MPS2 = 4.0
DRIVER_ANGULAR_ACCEL_LIMIT_RAD_S2 = 5.0
DRIVER_CONTROL_DT_S = 1.0 / 30.0
AUTO_ALIGN_TOLERANCE_DEG = 0.35
FIRE_MAX_YAW_RATE_DPS = 4.0
SCORE_FIRE_MAX_SPEED_MPS = 2.5
FERRY_AIM_TOLERANCE_DEG = 2.0
FERRY_FIRE_MAX_SPEED_MPS = 3.0
FERRY_FIRE_MAX_YAW_RATE_DPS = 18.0
SCORE_AIM_KP = 6.0
FERRY_AIM_KP = 8.0
AIM_YAW_DAMPING = 0.65
SCORE_AIM_MAX_OMEGA_RAD_S = 3.0
FERRY_AIM_MAX_OMEGA_RAD_S = 4.0

HUB_TARGETS = {
    "red": np.array([0.0199, 3.6874, 1.3646], dtype=np.float32),
    "blue": np.array([-0.0199, -3.6874, 1.3646], dtype=np.float32),
}
# Pull the landing point this far along the line of fire toward the robot:
# the raw xRC sensor target sits ~0.17 m outboard of the funnel centre
# (structure spans y 2.775..4.259 -> middle 3.517), so uncorrected streams
# read as landing at the back rim.
HUB_AIM_SHORT_BIAS_M = 0.18

# ---- Ferry / return lob (2026 meta: "ferry from the opponent zone") -------
# Measured hub structure bounds (assets/fresh_xrc field_meshes): footprint
# x +/-0.746 m, y 2.775..4.259 m per alliance, top rim z 3.059 m.  A return
# lob must clear a hub box or use a side lane; the solver samples the
# parabola across both hub footprints and refuses blocked lanes.
NEUTRAL_ZONE_HALF_Y_M = 2.775        # own zone starts past the hub band edge
FERRY_HUB_BLOCK_X_HALF_M = 0.95      # hub footprint + FUEL clearance
FERRY_HUB_BLOCK_Y_M = (2.775, 4.259)
FERRY_HUB_BLOCK_TOP_M = 3.26         # rim 3.059 + FUEL diameter margin
FERRY_PITCH_DEG = 55.0               # the big cross-field parabola
FERRY_SIDE_LANE_X_M = 2.55           # open corridor beside the hubs
FERRY_TARGET_Y_M = 4.35              # own-zone landing row (sign by alliance)
FERRY_TARGET_X_LIMIT_M = 3.35
FERRY_MIN_RANGE_M = 2.5
FERRY_MAX_EXIT_SPEED_MPS = 11.0
FERRY_MIN_EXIT_SPEED_MPS = 6.0
FERRY_MUZZLE_Z_M = 0.60
FERRY_LANDING_Z_M = 0.076

# The CAD side view establishes the shooter at robot -X; robot's command code
# likewise passes angleOfShooter=180 degrees.  Values are measured from the
# optimized CAD envelope and can later be refined from a component-level STEP.
MUZZLE_LOCAL = np.array([-0.46, 0.0, 0.55], dtype=np.float32)
# Shooter is robot -X, so the robot ground intake is on robot +X.  Captured
# balls follow this visible conveyor path at the xRC ballshooting_v2 speed
# rather than teleporting directly into a storage slot.
INTAKE_CENTER_LOCAL = np.array([0.52, 0.0, 0.11], dtype=np.float32)
INTAKE_HALF_EXTENTS = np.array([0.14, 0.35, 0.13], dtype=np.float32)
INTAKE_PATH_LOCAL = np.array(
    [
        [0.64, 0.0, 0.09],   # grab a ball off the floor in front of the bumper
        [0.54, 0.0, 0.105],  # feed it under the front board to the roller nip
        [0.40, 0.0, 0.135],  # cross the board while the whole ball is below it
        [0.40, 0.0, 0.340],  # lift above the highest rigid hopper roller
        [0.40, 0.0, 0.340],  # placeholder: high approach over the target column
        [0.40, 0.0, 0.320],  # placeholder: descend into the release-grid cell
    ],
    dtype=np.float32,
)
# Candidate indexer destinations inside the rigid hopper.  These are insertion
# targets, not pinned storage slots: after release every FUEL remains dynamic.
# Staggered close-packing: adjacent layers are 125 mm apart vertically and
# offset 80 mm in X, giving >148 mm centre distance for 152 mm foam FUEL.
# The sloped net has less headroom toward +X, so upper layers stop rearward.
# 16 + 12 + 12 + 8 hopper targets, plus eight turret targets = 56 total.
INTAKE_RELEASE_GRID_LOCAL = np.asarray(
    [
        [x, y, z]
        for z, x_positions in (
            (0.270, (-0.060, 0.100, 0.260, 0.400)),
            (0.395, (0.020, 0.180, 0.340)),
            (0.520, (-0.060, 0.100, 0.260)),
            (0.645, (0.020, 0.180)),
        )
        for x in x_positions
        for y in (-0.240, -0.080, 0.080, 0.240)
    ],
    dtype=np.float32,
)
# Real open-top container volume.  The floor sits just above the roller bed and
# the walls rise to the open top where the deformable net closes the lid, so the
# full bottom + top space stores FUEL (physics-limited, ~45 balls).
HOPPER_FLOOR_Z = 0.145
# This is the FIXED lower hopper wall.  The split upper sheets and net provide
# containment above it.  Raising this proxy to 0.70 m left a fixed 0.70 m wall
# in compact mode and blocked the 0.591 m xRC trench roof.
HOPPER_WALL_TOP_M = 0.50
HOPPER_MIN_LOCAL = np.array([-0.13, -0.34, HOPPER_FLOOR_Z], dtype=np.float32)
HOPPER_MAX_LOCAL = np.array([0.50, 0.34, 0.82], dtype=np.float32)
FEEDER_MAX_X_LOCAL = 0.18
# Fast tunnel transport keeps successive groups almost touching, matching the
# continuous-stream behaviour of the real reference roller floor/tunnel.  The launch
# velocity is still calculated independently by the calibrated shooter model.
FEEDER_CONVEY_SPEED_MPS = 3.0
# Eight-ball pre-shot area inside the turret: two rows of four, centred between
# the front/rear roller banks.  Shooter release remains a maximum three-ball
# volley; the fourth ball waits for the next roller index.
FEEDER_READY_LOCAL = np.asarray(
    [
        [-0.160, -0.2325, 0.215], [-0.160, -0.0775, 0.215],
        [-0.160, 0.0775, 0.215], [-0.160, 0.2325, 0.215],
        [-0.299, -0.2325, 0.371], [-0.299, -0.0775, 0.371],
        [-0.299, 0.0775, 0.371], [-0.299, 0.2325, 0.371],
    ],
    dtype=np.float32,
)

# 2026 FIRST R405: padding fills 2.5-5.75 in above the floor.  The real robot
# bumper is continuous, soft padding outside the whole frame.  It is modeled
# as removable front/rear C assemblies whose side returns meet at mid-frame;
# the intake deploys over/under it and the bumper itself is not a ramp.
# Bumper bottom.  Kept just below the FRC 2.5" zone so the skirt still limits FUEL
# from wedging under the frame, but NOT so low it catches the field ramps (whose
# tops reach ~0.093 m): a skirt at ~2 cm hung into the ramp face and blocked the
# wheels from climbing.  A flat skirt can't fully seal balls AND clear a ramp that
# is taller than a ball, so this favours ramp clearance; low bumper friction below
# lets it slide up ramps.
BUMPER_BOTTOM_M = 0.065
BUMPER_TOP_M = 5.75 * 0.0254
BUMPER_CENTER_Z_M = 0.5 * (BUMPER_BOTTOM_M + BUMPER_TOP_M)
BUMPER_HALF_HEIGHT_M = 0.5 * (BUMPER_TOP_M - BUMPER_BOTTOM_M)
# The red bumper padding wraps OUTSIDE the CAD 护栏 (hulan) bumper-mount wood,
# not the inner drivetrain base.  Measured 护栏 wood perimeter at bumper height
# in the mesh: X[-0.502,+0.217], Y[+/-0.368], centred X=-0.142 (~0.72x0.74 m,
# matching the standalone A/B护栏 0.734 m board length).  The bumper's inner
# face sits at that perimeter and the noodle extends BUMPER_DEPTH_M outward.
BUMPER_OUTER_HALF_M = 0.43
BUMPER_DEPTH_M = 0.062
BUMPER_CENTER_X_M = -0.142
BUMPER_BLUE_RGBA = (0.025, 0.16, 0.82, 1.0)

# WeidaiSub.cpp: 22.5:1 actuator, normalized positions 0.99 (StartStorage) and
# 0.02 (CloseStorage).  The 22.25 in xRC/FIRST trench opening is 0.565 m high.
STORAGE_EXTENDED_POSITION = 0.99
STORAGE_LOWERED_POSITION = 0.02
STORAGE_PIVOT_Z_M = 0.30
STORAGE_EXTENDED_TOP_M = 0.74
STORAGE_LOWERED_TOP_M = 0.51
STORAGE_TRAVEL_PER_S = 1.25
# Show ONLY the full reference CAD (which already includes the hopper/intake/shooter).
# The procedural overlay mechanisms (animated Weidai net, intake board, shooter
# cage) are hidden to avoid duplicate/overlapping structures.  Set True to bring
# the animated overlays back.
INCLUDE_PROCEDURAL_MECHANISMS = False
STORAGE_XFORM_PATH = f"{ROBOT_ROOT_PATH}/chassis/StorageNetMotion"
STORAGE_SIDE_NET_PATH = f"{STORAGE_XFORM_PATH}/WhiteSideNet"
STORAGE_TOP_NET_PATH = f"{STORAGE_XFORM_PATH}/BlackTopNet"
STORAGE_LINKAGE_PATH = f"{STORAGE_XFORM_PATH}/FourBarLinkage"
INTAKE_XFORM_PATH = f"{ROBOT_ROOT_PATH}/chassis/IntakeDeployMotion"
INTAKE_STOW_ANGLE_DEG = -72.0
CAD_MECHANISM_ROOT = f"{ROBOT_ROOT_PATH}/CADMechanisms"
CAD_INTAKE_PATH = f"{CAD_MECHANISM_ROOT}/Intake"
CAD_HOPPER_BASE_PATH = f"{ROBOT_ROOT_PATH}/chassis/CADMechanisms/HopperBase"
CAD_HORIZONTAL_PATH = f"{CAD_MECHANISM_ROOT}/HorizontalExtension"
CAD_VERTICAL_PATH = f"{CAD_MECHANISM_ROOT}/VerticalExtension"
CAD_SHOOTER_RETRACT_PATH = f"{CAD_MECHANISM_ROOT}/ShooterRetract"
NET_EXTENDED_COLLIDER_ROOT = f"{ROBOT_ROOT_PATH}/chassis/NetCollisionExtended"
NET_COMPACT_COLLIDER_ROOT = f"{ROBOT_ROOT_PATH}/chassis/NetCollisionCompact"
NET_VISUAL_ROOT = f"{ROBOT_ROOT_PATH}/chassis/NetMotion"
CAD_INTAKE_PIVOT = np.array([0.145, 0.0, 0.20], dtype=np.float32)
CAD_HORIZONTAL_RETRACT_M = 0.145
# Align the 0.7408 m upper assembly with the 0.5328 m lower-frame top.
CAD_VERTICAL_LOWER_M = 0.208
CAD_SHOOTER_LOWER_M = 0.140
# Preserve the staged order.  Two sequential 0.333 s strokes plus transition
# ticks produce a roughly 0.70 s compact/deploy program at 60 Hz.
INTAKE_EXTENSION_PER_S = 3.0
CONTAINER_EXTENSION_PER_S = 3.0

# Initial inspection pose beneath the blue TRENCH.  Local +X is the intake/nose;
# +90 degrees points it toward the NEUTRAL ZONE.  With the asymmetric bumper
# envelope, the nose remains behind the trench's neutral-side plan-view edge
# (-3.05 m).  The centre is deliberately shifted 0.150 m farther toward the
# blue alliance wall than the earlier inspection pose so the robot/upper-beam
# alignment matches the top-down reference.  The bumper still leaves about
# 0.21 m clearance to each side of the 1.2786 m clear opening.
BLUE_TRENCH_START_TRANSLATION = (3.3582, -3.8850, 0.01)
BLUE_TRENCH_START_YAW_DEG = 90.0
BLUE_TRENCH_NEUTRAL_EDGE_Y_M = -3.05
BLUE_TRENCH_ALLIANCE_EDGE_Y_M = -4.24
BLUE_TRENCH_CLEAR_X_MIN_M = 2.7189
BLUE_TRENCH_CLEAR_X_MAX_M = 3.9975
NET_EXTENDED_CORNERS = np.asarray(
    [
        [0.1892, -0.317, 0.727], [0.1892, 0.317, 0.727],
        [0.50, -0.35, 0.528], [0.50, 0.35, 0.528],
    ],
    dtype=np.float32,
)

# Convex collision proxies for the visible reference mechanisms.  The STEP-derived
# render meshes are intentionally not used directly as dynamic triangle-mesh
# colliders (PhysX cannot robustly collide moving concave meshes).  These boxes
# follow the measured CAD envelopes closely and are parented to the same moving
# links as their visual counterparts.
HOPPER_ROLLER_COLLIDERS = (
    ((0.016, 0.273, 0.016), (-0.262, 0.002, 0.137)),
    ((0.016, 0.273, 0.016), (-0.207, 0.002, 0.129)),
    ((0.016, 0.273, 0.016), (0.031, 0.002, 0.212)),
    ((0.016, 0.273, 0.016), (0.097, 0.002, 0.223)),
    ((0.016, 0.273, 0.016), (0.164, 0.002, 0.233)),
)
UPPER_SIDE_BOARD_COLLIDERS = (
    ((0.182, 0.010, 0.210), (0.006, -0.344, 0.527)),
    ((0.182, 0.010, 0.210), (0.006, 0.344, 0.527)),
)
FRONT_SIDE_POST_COLLIDERS = (
    ((0.014, 0.014, 0.153), (0.486, -0.331, 0.381)),
    ((0.014, 0.014, 0.153), (0.486, 0.336, 0.381)),
)
FEEDER_TUNNEL_COLLIDERS = (
    # Clearances below are measured for the 152 mm FUEL diameter and the two
    # four-wide FEEDER_READY_LOCAL rows.  The old proxies overlapped the bottom
    # row by 16 mm and each outer ball by 20.5 mm, turning a full feeder into a
    # PhysX spring that ejected balls through the rear of the turret.
    ((0.100, 0.340, 0.006), (-0.235, 0.0, 0.127)),  # thin fixed roller bed
    ((0.100, 0.300, 0.018), (-0.235, 0.0, 0.645)),  # real top roller/stop
    # x=-0.42 leaves the staged row clear on its +X side and the already
    # released muzzle spheres clear on its -X side.
    ((0.014, 0.340, 0.242), (-0.420, 0.0, 0.402)),  # closed rear stop
    ((0.100, 0.012, 0.242), (-0.235, -0.345, 0.402)),
    ((0.100, 0.012, 0.242), (-0.235, 0.345, 0.402)),
)
SHOOTER_CAGE_COLLIDERS = (
    ((0.120, 0.012, 0.129), (-0.333, -0.330, 0.514)),
    ((0.120, 0.012, 0.129), (-0.333, 0.330, 0.514)),
    # Top guard stops before the x=-0.46 muzzle.  The previous span began at
    # x=-0.453, so a 76 mm-radius FUEL overlapped it the instant it launched.
    ((0.070, 0.330, 0.012), (-0.285, 0.0, 0.631)),
)
# Low, inset belly pan: closes the under-chassis leak without extending into
# the +X intake mouth or the moving front rollers.
BELLY_PAN_COLLIDER = ((0.340, 0.330, 0.012), (-0.100, 0.0, 0.090))
TURRET_ROLLER_COLLIDERS = (
    ((0.016, 0.296, 0.016), (-0.344, 0.002, 0.190)),
    ((0.016, 0.296, 0.016), (-0.374, 0.002, 0.237)),
    ((0.016, 0.296, 0.016), (-0.389, 0.002, 0.290)),
    ((0.016, 0.296, 0.016), (-0.390, 0.002, 0.345)),
    ((0.016, 0.296, 0.016), (-0.390, 0.002, 0.400)),
    ((0.024, 0.300, 0.024), (-0.208, 0.009, 0.310)),
    ((0.007, 0.332, 0.007), (-0.218, 0.009, 0.394)),
    ((0.007, 0.330, 0.007), (-0.420, -0.004, 0.470)),
)


def _box_triangles(half: tuple[float, float, float], center=(0.0, 0.0, 0.0)) -> np.ndarray:
    hx, hy, hz = half
    vertices = np.array(
        [[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)],
        dtype=np.float32,
    ) + np.asarray(center, dtype=np.float32)
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int32,
    )
    return vertices[faces]


def storage_top_m(position: float) -> float:
    """Physical top of the folding net at a normalized Weidai position."""
    t = float(np.clip(
        (position - STORAGE_LOWERED_POSITION)
        / (STORAGE_EXTENDED_POSITION - STORAGE_LOWERED_POSITION),
        0.0,
        1.0,
    ))
    t = t * t * (3.0 - 2.0 * t)
    return STORAGE_LOWERED_TOP_M + t * (STORAGE_EXTENDED_TOP_M - STORAGE_LOWERED_TOP_M)


def _storage_curve_segments(position: float):
    """reference-style four-bar/net geometry at a normalized Weidai position."""
    t = float(np.clip(
        (position - STORAGE_LOWERED_POSITION)
        / (STORAGE_EXTENDED_POSITION - STORAGE_LOWERED_POSITION), 0.0, 1.0
    ))
    t = t * t * (3.0 - 2.0 * t)
    x0 = -0.08
    x1 = 0.28 + 0.22 * t
    z0 = 0.02
    z1 = 0.20 + 0.24 * t
    z_front = 0.08 + 0.04 * t

    def top_at(x: float) -> float:
        fraction = float(np.clip((x - x0) / max(1e-6, x1 - x0), 0.0, 1.0))
        return z1 + (z_front - z1) * fraction

    side = []
    for y in (-0.315, 0.315):
        side.extend([
            ((x0, y, z0), (x1, y, z0)), ((x0, y, z1), (x1, y, z_front)),
            ((x0, y, z0), (x0, y, z1)), ((x0, y, z1), (x1, y, z0)),
        ])
        for fraction in np.linspace(0.10, 0.90, 7):
            x = x0 + (x1 - x0) * float(fraction)
            top = top_at(x)
            side.append(((x, y, z0), (x, y, top)))
            x_next = min(x1, x + 0.12)
            side.append(((x, y, top), (x_next, y, z0)))

    top = []
    for x in np.linspace(x0, x1, 7):
        z = top_at(float(x))
        top.append(((float(x), -0.30, z), (float(x), 0.30, z)))
    for y in np.linspace(-0.30, 0.30, 7):
        top.append(((x0, float(y), z1), (x1, float(y), z_front)))

    # Two mirrored parallelogram linkages keep the upper roller frame nearly
    # horizontal while it rises, matching the new native reference assembly.
    upper_rear = (-0.02 + 0.01 * t, 0.10 + 0.34 * t)
    upper_front = (0.22 + 0.08 * t, 0.10 + 0.30 * t)
    linkage = []
    for y in (-0.33, 0.33):
        linkage.extend([
            ((-0.06, y, 0.02), (upper_rear[0], y, upper_rear[1])),
            ((0.22, y, 0.02), (upper_front[0], y, upper_front[1])),
            ((upper_rear[0], y, upper_rear[1]),
             (upper_front[0], y, upper_front[1])),
            ((-0.06, y, 0.02), (0.22, y, 0.02)),
        ])
    return side, top, linkage, t


def _four_corner_soft_net_point(
    corners: np.ndarray, u: float, v: float, *, center_sag_m: float = 0.145
) -> np.ndarray:
    """Point on the corner-supported net surface in robot-local coordinates."""
    anchors = np.asarray(corners, dtype=np.float32)
    if anchors.shape != (4, 3):
        raise ValueError("soft-net corners must have shape (4, 3)")
    rear_left, rear_right, front_left, front_right = anchors
    rear = rear_left * (1.0 - v) + rear_right * v
    front = front_left * (1.0 - v) + front_right * v
    result = rear * (1.0 - u) + front * u
    # Only the four knots are fixed.  The centre and unreinforced edge ropes
    # hang under gravity, matching the real storage net rather than a flat lid.
    center = math.sin(math.pi * u) * math.sin(math.pi * v)
    edge_u = math.sin(math.pi * u) * abs(2.0 * v - 1.0) ** 4
    edge_v = math.sin(math.pi * v) * abs(2.0 * u - 1.0) ** 4
    result = result.copy()
    result[2] -= center_sag_m * (center + 0.42 * (edge_u + edge_v))
    return result


def _four_corner_soft_net_segments(
    corners: np.ndarray,
    *,
    rope_count: int = 9,
    samples_per_rope: int = 13,
    center_sag_m: float = 0.145,
) -> tuple[list[tuple[tuple[float, float, float], tuple[float, float, float]]], np.ndarray]:
    """Build the hanging reference storage net with only its four corners anchored.

    ``corners`` is ordered rear-left, rear-right, front-left, front-right in
    robot coordinates.  The bilinear surface follows the slanted hopper mouth;
    gravity-like sag is zero at the four tie points, appreciable along the free
    edges, and greatest in the middle.  Returning ordinary curve segments keeps
    the net lightweight while making its flexible shape explicit in the GUI.
    """
    anchors = np.asarray(corners, dtype=np.float32)
    if anchors.shape != (4, 3):
        raise ValueError("soft-net corners must have shape (4, 3)")
    rope_count = max(3, int(rope_count))
    samples_per_rope = max(3, int(samples_per_rope))

    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    rope_parameters = np.linspace(0.0, 1.0, rope_count)
    samples = np.linspace(0.0, 1.0, samples_per_rope)
    # Two crossing rope families form the real open knotted mesh.  Each rope is
    # sampled into short pieces so the rendered curve visibly bends under load.
    for fixed_v in rope_parameters:
        points = [
            _four_corner_soft_net_point(anchors, float(u), float(fixed_v), center_sag_m=center_sag_m)
            for u in samples
        ]
        segments.extend(
            (tuple(float(value) for value in a), tuple(float(value) for value in b))
            for a, b in zip(points[:-1], points[1:])
        )
    for fixed_u in rope_parameters:
        points = [
            _four_corner_soft_net_point(anchors, float(fixed_u), float(v), center_sag_m=center_sag_m)
            for v in samples
        ]
        segments.extend(
            (tuple(float(value) for value in a), tuple(float(value) for value in b))
            for a, b in zip(points[:-1], points[1:])
        )
    return segments, anchors


def _four_corner_soft_net_collision_blocks(
    corners: np.ndarray,
    *,
    resolution: int = 6,
    thickness_m: float = 0.018,
    center_sag_m: float = 0.145,
) -> list[np.ndarray]:
    """Closed convex tiles forming a continuous solid collision net.

    The visible object remains an open rope mesh, but FUEL is much larger than
    its openings.  Adjacent thin prisms therefore model the rope contact as one
    watertight, curved containment surface that is safe on a moving rigid body.
    """
    resolution = max(2, int(resolution))
    half_t = 0.5 * float(thickness_m)
    params = np.linspace(0.0, 1.0, resolution + 1)
    blocks: list[np.ndarray] = []
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    for iu in range(resolution):
        for iv in range(resolution):
            u0, u1 = float(params[iu]), float(params[iu + 1])
            v0, v1 = float(params[iv]), float(params[iv + 1])
            surface = np.asarray(
                [
                    _four_corner_soft_net_point(corners, u0, v0, center_sag_m=center_sag_m),
                    _four_corner_soft_net_point(corners, u1, v0, center_sag_m=center_sag_m),
                    _four_corner_soft_net_point(corners, u1, v1, center_sag_m=center_sag_m),
                    _four_corner_soft_net_point(corners, u0, v1, center_sag_m=center_sag_m),
                ],
                dtype=np.float32,
            )
            lower, upper = surface.copy(), surface.copy()
            lower[:, 2] -= half_t
            upper[:, 2] += half_t
            vertices = np.concatenate([lower, upper], axis=0)
            blocks.append(vertices[faces])
    return blocks


def _hopper_slots() -> np.ndarray:
    """The eight xRC preloads begin in the two four-wide turret rows."""
    return FEEDER_READY_LOCAL.copy()


def _point_on_polyline(points: np.ndarray, distance_m: float) -> tuple[np.ndarray, bool]:
    """Return a point ``distance_m`` along a polyline and completion state."""
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    remaining = max(0.0, float(distance_m))
    for start, delta, length in zip(points[:-1], segments, lengths):
        if remaining <= float(length):
            fraction = remaining / max(1e-9, float(length))
            return start + delta * fraction, False
        remaining -= float(length)
    return points[-1].copy(), True


class CompetitionRobotArticulationBuilder:
    """Author the lightweight CAD visual plus a physical four-module swerve."""

    def __init__(
        self,
        scene_builder: Any,
        root_path: str = ROBOT_ROOT_PATH,
        # Start fully beneath the blue trench, compact, with the intake/nose
        # facing outward toward the neutral zone.  AUTO deliberately does
        # nothing; the driver can leave through the tunnel in TELEOP.
        spawn_translation: tuple[float, float, float] = BLUE_TRENCH_START_TRANSLATION,
        spawn_yaw_deg: float = BLUE_TRENCH_START_YAW_DEG,
    ):
        self.sb = scene_builder
        self.root_path = root_path
        self.chassis_path = f"{root_path}/chassis"
        self.spawn = spawn_translation
        self.spawn_yaw_deg = spawn_yaw_deg

    def build(self) -> dict[str, Any]:
        from pxr import Gf, UsdGeom, UsdPhysics

        sb, stage = self.sb, self.sb.stage
        if not VISUAL_MESH.is_file():
            raise FileNotFoundError(
                f"Missing optimized robot CAD: {VISUAL_MESH}; run tools/import_robot_step.py"
            )
        asset = np.load(VISUAL_MESH)
        vertices = np.asarray(asset["vertices"], dtype=np.float32)
        faces = np.asarray(asset["faces"], dtype=np.int32)
        use_split_cad = all(path.is_file() for path in SPLIT_CAD_ASSETS.values())
        # Use the FULL reference CAD as-is (Reference Robotapril2519.STEP -> visual_mesh).
        # The CAD already contains the hopper/intake/shooter, so the procedural
        # overlay mechanisms below are hidden (INCLUDE_PROCEDURAL_MECHANISMS)
        # to avoid the earlier duplicate-structure overlap.

        root = UsdGeom.Xform.Define(stage, self.root_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*self.spawn))
        half_yaw = math.radians(self.spawn_yaw_deg) * 0.5
        root.AddOrientOp().Set(Gf.Quatf(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)))
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        physx_art = sb.PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        physx_art.CreateEnabledSelfCollisionsAttr(False)
        physx_art.CreateSolverPositionIterationCountAttr(16)
        physx_art.CreateSolverVelocityIterationCountAttr(4)

        chassis = UsdGeom.Xform.Define(stage, self.chassis_path)
        chassis_prim = chassis.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(chassis_prim)
        mass = UsdPhysics.MassAPI.Apply(chassis_prim)
        mass.CreateMassAttr(ROBOT_MASS_KG - 6.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.03, 0.0, 0.18))
        chassis_rb = sb.PhysxSchema.PhysxRigidBodyAPI.Apply(chassis_prim)
        # Sweep-based CCD for the frame: at teleop speed the chassis crosses
        # ~5 cm per 60 Hz step while these detailed colliders keep 4 mm
        # contact envelopes (needed for the real 18-29 mm trench gaps), so
        # without CCD a hard drive into a wall/hub interpenetrated visibly
        # and thin field boards could be tunnelled outright.
        chassis_rb.CreateEnableCCDAttr(True)

        # The eDrawings master export is losslessly partitioned by original
        # assembly path, so the complete CAD remains visible while intake,
        # horizontal container and vertical container receive independent Xforms.
        cad_material = sb._preview_material("competition_robot", (0.62, 0.65, 0.69, 1.0))
        robot_visual_faces = len(faces)
        if use_split_cad:
            UsdGeom.Xform.Define(stage, CAD_MECHANISM_ROOT)
            group_paths = {
                "static": f"{self.chassis_path}/CADMechanisms/StaticChassis",
                "intake": CAD_INTAKE_PATH,
                "hopper_base": CAD_HOPPER_BASE_PATH,
                "hopper_horizontal": CAD_HORIZONTAL_PATH,
                "hopper_vertical": CAD_VERTICAL_PATH,
                "shooter_retract": CAD_SHOOTER_RETRACT_PATH,
            }
            for group, path in group_paths.items():
                xform = UsdGeom.Xform.Define(stage, path)
                if group == "intake":
                    xform.AddTranslateOp().Set(
                        Gf.Vec3d(*[float(value) for value in CAD_INTAKE_PIVOT])
                    )
                part = np.load(SPLIT_CAD_ASSETS[group])
                part_vertices = np.asarray(part["vertices"], dtype=np.float32)
                part_faces = np.asarray(part["faces"], dtype=np.int32)
                if group == "intake":
                    part_vertices = part_vertices - CAD_INTAKE_PIVOT
                part_visual = sb._mesh(
                    f"{path}/CAD", part_vertices[part_faces], (0.62, 0.65, 0.69, 1.0)
                )
                sb.UsdShade.MaterialBindingAPI.Apply(part_visual).Bind(cad_material)
                if group in (
                    "intake", "hopper_horizontal", "hopper_vertical", "shooter_retract"
                ):
                    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
                    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(0.75)
                    link_rb = sb.PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
                    # mechanism links ride the chassis at the same speeds
                    link_rb.CreateEnableCCDAttr(True)

            intake_joint = UsdPhysics.RevoluteJoint.Define(
                stage, f"{self.root_path}/joints/intake_fold"
            )
            intake_joint.CreateBody0Rel().SetTargets([self.chassis_path])
            intake_joint.CreateBody1Rel().SetTargets([CAD_INTAKE_PATH])
            intake_joint.CreateAxisAttr("Y")
            intake_joint.CreateLowerLimitAttr(INTAKE_STOW_ANGLE_DEG)
            intake_joint.CreateUpperLimitAttr(0.0)
            intake_joint.CreateLocalPos0Attr(
                Gf.Vec3f(*[float(value) for value in CAD_INTAKE_PIVOT])
            )
            intake_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            intake_drive = UsdPhysics.DriveAPI.Apply(intake_joint.GetPrim(), "angular")
            intake_drive.CreateTypeAttr("force")
            intake_drive.CreateTargetPositionAttr(0.0)
            intake_drive.CreateStiffnessAttr(500.0)
            intake_drive.CreateDampingAttr(50.0)
            intake_drive.CreateMaxForceAttr(180.0)
            sb.PhysxSchema.PhysxJointAPI.Apply(
                intake_joint.GetPrim()
            ).CreateMaxJointVelocityAttr(math.radians(150.0))

            for path, joint_name, axis, lower in (
                (CAD_HORIZONTAL_PATH, "horizontal_retract", "X", -CAD_HORIZONTAL_RETRACT_M),
                (CAD_VERTICAL_PATH, "vertical_lower", "Z", -CAD_VERTICAL_LOWER_M),
                (CAD_SHOOTER_RETRACT_PATH, "shooter_lower", "Z", -CAD_SHOOTER_LOWER_M),
            ):
                joint = UsdPhysics.PrismaticJoint.Define(
                    stage, f"{self.root_path}/joints/{joint_name}"
                )
                joint.CreateBody0Rel().SetTargets([self.chassis_path])
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateAxisAttr(axis)
                joint.CreateLowerLimitAttr(lower)
                joint.CreateUpperLimitAttr(0.0)
                joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                # Isaac's conservative default joint-speed cap left the visual
                # link ~63 mm behind its target during the compact animation.
                # The real linear actuators complete this short stroke quickly.
                sb.PhysxSchema.PhysxJointAPI.Apply(
                    joint.GetPrim()
                ).CreateMaxJointVelocityAttr(1.0)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
                drive.CreateTypeAttr("force")
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateStiffnessAttr(3200.0)
                drive.CreateDampingAttr(180.0)
                drive.CreateMaxForceAttr(2600.0)
            robot_visual_faces = sum(
                len(np.load(path)["faces"]) for path in SPLIT_CAD_ASSETS.values()
            )
        else:
            visual = sb._mesh(
                f"{self.chassis_path}/CADVisual", vertices[faces], (0.62, 0.65, 0.69, 1.0)
            )
            sb.UsdShade.MaterialBindingAPI.Apply(visual).Bind(cad_material)

        # Blue-alliance bumper: two removable C-shaped soft assemblies.
        # Each has one full front/back pad and two half-side returns; together
        # they wrap just outside the native CAD's rigid lower perimeter.
        rail_center = BUMPER_OUTER_HALF_M - BUMPER_DEPTH_M * 0.5
        cx = BUMPER_CENTER_X_M
        front_c_blocks = [
            _box_triangles(
                (BUMPER_DEPTH_M * 0.5, BUMPER_OUTER_HALF_M, BUMPER_HALF_HEIGHT_M),
                (rail_center + cx, 0.0, BUMPER_CENTER_Z_M),
            ),
            _box_triangles(
                (rail_center * 0.5, BUMPER_DEPTH_M * 0.5, BUMPER_HALF_HEIGHT_M),
                (rail_center * 0.5 + cx, rail_center, BUMPER_CENTER_Z_M),
            ),
            _box_triangles(
                (rail_center * 0.5, BUMPER_DEPTH_M * 0.5, BUMPER_HALF_HEIGHT_M),
                (rail_center * 0.5 + cx, -rail_center, BUMPER_CENTER_Z_M),
            ),
        ]
        rear_c_blocks = [
            _box_triangles(
                (BUMPER_DEPTH_M * 0.5, BUMPER_OUTER_HALF_M, BUMPER_HALF_HEIGHT_M),
                (-rail_center + cx, 0.0, BUMPER_CENTER_Z_M),
            ),
            _box_triangles(
                (rail_center * 0.5, BUMPER_DEPTH_M * 0.5, BUMPER_HALF_HEIGHT_M),
                (-rail_center * 0.5 + cx, rail_center, BUMPER_CENTER_Z_M),
            ),
            _box_triangles(
                (rail_center * 0.5, BUMPER_DEPTH_M * 0.5, BUMPER_HALF_HEIGHT_M),
                (-rail_center * 0.5 + cx, -rail_center, BUMPER_CENTER_Z_M),
            ),
        ]
        bumper_blocks = front_c_blocks + rear_c_blocks
        bumper_material = sb._preview_material("bumper_robot_blue", BUMPER_BLUE_RGBA)
        for index, blocks in enumerate((front_c_blocks, rear_c_blocks)):
            bumper_visual = sb._mesh(
                f"{self.chassis_path}/Bumper/{'Front' if index == 0 else 'Rear'}C",
                np.concatenate(blocks),
                BUMPER_BLUE_RGBA,
            )
            sb.UsdShade.MaterialBindingAPI.Apply(bumper_visual).Bind(bumper_material)

        # Folding storage cage/net.  Curve nodes are updated by a four-bar
        # kinematic model; there is no uniform scale shortcut.
        storage = UsdGeom.Xform.Define(stage, STORAGE_XFORM_PATH)
        storage.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, STORAGE_PIVOT_Z_M))
        white_net = sb._preview_material("competition_robot_net_white", (0.82, 0.84, 0.86, 1.0))
        black_net = sb._preview_material("competition_robot_net_black", (0.015, 0.018, 0.022, 1.0))
        silver = sb._preview_material("competition_robot_frame_silver", (0.68, 0.72, 0.75, 1.0))

        def net_curves(path: str, segments: list[tuple[tuple[float, float, float],
                                                        tuple[float, float, float]]],
                       width: float, material: Any, color: tuple[float, float, float]) -> None:
            curves = UsdGeom.BasisCurves.Define(stage, path)
            points = [Gf.Vec3f(*point) for segment in segments for point in segment]
            curves.CreateTypeAttr(UsdGeom.Tokens.linear)
            curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            curves.CreateCurveVertexCountsAttr([2] * len(segments))
            curves.CreatePointsAttr(points)
            curves.CreateWidthsAttr([width] * len(points))
            curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
            sb.UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(material)

        # These procedural storage overlays (white side net, black top net, silver
        # four-bar linkage) duplicate what the real CAD already contains.  When
        # the CAD is shown they must NOT be created at all: MakeInvisible on their
        # parent Xform does not hide BasisCurves children under PhysX Fabric
        # (inherited visibility is not flat-cached), so they leaked into the view
        # as stray white/silver curves near the base.
        if INCLUDE_PROCEDURAL_MECHANISMS:
            side_segments, top_segments, linkage_segments, _ = _storage_curve_segments(
                STORAGE_EXTENDED_POSITION
            )
            net_curves(
                STORAGE_SIDE_NET_PATH, side_segments, 0.007,
                white_net, (0.82, 0.84, 0.86),
            )
            net_curves(
                STORAGE_TOP_NET_PATH, top_segments, 0.009,
                black_net, (0.015, 0.018, 0.022),
            )
            net_curves(
                STORAGE_LINKAGE_PATH, linkage_segments, 0.018,
                silver, (0.68, 0.72, 0.75),
            )

        # The real reference net is not a rigid rectangular lid.  It is a loose,
        # knotted black rope net tied only to the four upper board corners.  Its
        # rear pair is higher than the front pair, and both the free edges and
        # centre hang visibly into the hopper (as in the supplied robot photos).
        # The four net knots belong to two different moving links, so the net
        # cannot inherit either link transform.  Keep it in chassis-local space
        # and update all ropes/knots from the two joint positions together.
        UsdGeom.Xform.Define(stage, NET_VISUAL_ROOT)
        net_corners = NET_EXTENDED_CORNERS.copy()
        top_net_segments, net_anchors = _four_corner_soft_net_segments(net_corners)
        net_curves(
            f"{NET_VISUAL_ROOT}/Ropes", top_net_segments, 0.0055,
            black_net, (0.018, 0.020, 0.024),
        )
        # Small knots make the four actual attachment points readable in the
        # viewport and guarantee that no edge appears to float off a board.
        for anchor_index, anchor in enumerate(net_anchors):
            knot = UsdGeom.Sphere.Define(
                stage, f"{NET_VISUAL_ROOT}/CornerKnot_{anchor_index}"
            )
            knot.CreateRadiusAttr(0.008)
            knot.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in anchor]))
            knot.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                [Gf.Vec3f(0.018, 0.020, 0.024)]
            )
            sb.UsdShade.MaterialBindingAPI.Apply(knot.GetPrim()).Bind(black_net)

        # Silver shooter cage overlay.  Another procedural duplicate of the real
        # CAD shooter - only build it with the procedural mechanisms, since a
        # BasisCurves overlay cannot be reliably hidden under Fabric.
        if INCLUDE_PROCEDURAL_MECHANISMS:
            shooter_frame: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
            for y in (-0.325, 0.325):
                shooter_frame.extend([
                    ((-0.47, y, 0.30), (-0.47, y, 0.69)),
                    ((-0.47, y, 0.69), (-0.08, y, 0.62)),
                    ((-0.08, y, 0.62), (-0.08, y, 0.30)),
                    ((-0.47, y, 0.30), (-0.08, y, 0.30)),
                    ((-0.47, y, 0.30), (-0.08, y, 0.62)),
                ])
            for x, z in [(-0.47, 0.30), (-0.47, 0.69), (-0.08, 0.30), (-0.08, 0.62)]:
                shooter_frame.append(((x, -0.325, z), (x, 0.325, z)))
            net_curves(
                f"{self.chassis_path}/ShooterMetalCage", shooter_frame, 0.018,
                silver, (0.68, 0.72, 0.75),
            )

        # The large intake board is part of the same square->triangle mode
        # change described by the real robot: it folds up/in when compact and
        # swings out underneath the Weidai when the storage/net extends.
        intake_motion = UsdGeom.Xform.Define(stage, INTAKE_XFORM_PATH)
        intake_motion.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.24))
        intake_motion.AddRotateYOp().Set(0.0)
        # Same as the storage nets: this procedural board+roller overlay duplicates
        # the real CAD intake and, because parent MakeInvisible does not flat-cache
        # to its children under Fabric, would otherwise leak into view.  Only build
        # it when the procedural mechanisms are actually requested.
        if INCLUDE_PROCEDURAL_MECHANISMS:
            intake_board = sb._mesh(
                f"{INTAKE_XFORM_PATH}/BlackIntakeBoard",
                _box_triangles((0.17, 0.29, 0.018), (0.17, 0.0, -0.115)),
                (0.018, 0.022, 0.028, 1.0),
            )
            sb.UsdShade.MaterialBindingAPI.Apply(intake_board).Bind(black_net)
            intake_roller = UsdGeom.Cylinder.Define(
                stage, f"{INTAKE_XFORM_PATH}/IntakeRoller"
            )
            intake_roller.CreateAxisAttr("Y")
            intake_roller.CreateRadiusAttr(0.045)
            intake_roller.CreateHeightAttr(0.62)
            intake_roller.AddTranslateOp().Set(Gf.Vec3d(0.31, 0.0, -0.155))
            intake_roller.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                [Gf.Vec3f(0.015, 0.018, 0.022)]
            )
            sb.UsdShade.MaterialBindingAPI.Apply(intake_roller.GetPrim()).Bind(black_net)

        # (Removed: 3 black "accent" rollers.  They were a procedural overlay on
        # the shooter whose typeless parent prim never took the MakeInvisible, so
        # they showed through as floating black rollers.  The silver CAD body
        # already contains the real shooter, so they are simply not created.)

        # Show only the full reference CAD: hide the procedural overlay mechanisms
        # (they duplicate hopper/intake/shooter that the CAD already contains).
        if not INCLUDE_PROCEDURAL_MECHANISMS:
            for _overlay in (
                STORAGE_XFORM_PATH, INTAKE_XFORM_PATH,
                f"{self.chassis_path}/ShooterMetalCage",
                f"{self.chassis_path}/BlackAccents",
            ):
                _prim = stage.GetPrimAtPath(_overlay)
                if _prim and _prim.IsValid():
                    UsdGeom.Imageable(_prim).MakeInvisible()

        # Chassis envelope plus a real open-top hopper.  Intake enters through
        # the central throat in the +X wall; balls then remain dynamic and stack,
        # collide and jam naturally instead of being pinned to software slots.
        # Rigid open-top ball container ("stomach"/Weidai).  FUEL is held purely
        # by these colliders that match the real hopper boards - there are no
        # velocity "walls".  Balls fill the whole volume from just above the
        # roller bed (HOPPER_FLOOR_Z) up to the open top, where the deformable
        # net forms the lid.  The +X front is split around the intake throat.
        fz = HOPPER_FLOOR_Z
        wtop = HOPPER_WALL_TOP_M
        wcz, whz = 0.5 * (fz + wtop), 0.5 * (wtop - fz)
        # The collision lid must be wider and flatter than the visible rope net:
        # widen the two rear corners to the full container width (the visual
        # trapezoid left a rear-corner gap FUEL squeezed through) and cut the sag
        # so it seals just under the wall tops instead of dipping into the load.
        net_collision_corners = net_corners.copy()
        net_collision_corners[:2, 1] = np.sign(net_collision_corners[:2, 1]) * 0.35
        net_collision_blocks = _four_corner_soft_net_collision_blocks(
            net_collision_corners, center_sag_m=0.045, thickness_m=0.030
        )
        compact_net_corners = net_collision_corners.copy()
        compact_net_corners[:2, 2] -= CAD_VERTICAL_LOWER_M
        compact_net_corners[2:, 0] -= CAD_HORIZONTAL_RETRACT_M
        compact_net_collision_blocks = _four_corner_soft_net_collision_blocks(
            compact_net_corners, center_sag_m=0.045, thickness_m=0.030
        )
        base_collision_blocks = [
            # floor just above the roller bed, ending behind the intake rollers.
            # Thicker slab so packed FUEL cannot squeeze through at the bottom.
            # End at x=0.32, leaving a 160 mm throat between the floor and
            # front board for one 152 mm FUEL to turn upward after the nip.
            _box_triangles((0.23, 0.36, 0.020), (0.09, 0.0, fz)),          # X[-0.14,0.32]
            # Upper part of the -X wall.  The lower opening feeds the rigid
            # three-wide staging tunnel; the tunnel itself has a closed rear,
            # floor, roof and sides, so FUEL cannot leak into the shooter CAD.
            _box_triangles((0.018, 0.36, 0.070), (-0.14, 0.0, 0.430)),
            # side walls (+/-Y): thickened and overlapped into floor/back/front so
            # the box cage has no seam gaps at the vertical corners.
            _box_triangles((0.33, 0.018, whz), (0.18, 0.352, wcz)),
            _box_triangles((0.33, 0.018, whz), (0.18, -0.352, wcz)),
            # front board (+X): fully SOLID - no opening.  Its bottom edge sits
            # just above the intake so stored FUEL never passes through the board.
            # Capped at z=0.54 so that, retracted, it stays under the 0.565 m
            # trench opening; the net lid (on the vertical link) seals above it.
            _box_triangles((0.018, 0.36, 0.5 * (0.54 - 0.22)), (0.50, 0.0, 0.5 * (0.54 + 0.22))),  # z[0.22,0.54]
            # ground intake: a horizontal roller pair right under the front board
            # and in front of the bumper.  FUEL is drawn up BETWEEN the two rigid
            # bottom rollers; the narrow nip stops stored balls falling back out.
            _box_triangles((0.02, 0.34, 0.028), (0.43, 0.0, 0.155)),       # rear bottom roller
            # Measured CAD roller centre.  The previous x=0.57 proxy extended
            # beyond the intake model and rotated up to z=0.618 when stowed,
            # becoming the invisible trench blocker.
            _box_triangles((0.014, 0.34, 0.018), (0.476, 0.0, 0.151)),
            # Solid upper machine/board behind the net.  The corrected net ties
            # to this board's back edge at x=0.1892; only the shorter opening in
            # front of that edge is flexible netting.
            _box_triangles((0.1646, 0.335, 0.010), (0.0246, 0.0, 0.727)),
        ]
        hopper_roller_start = len(base_collision_blocks)
        base_collision_blocks.extend(
            _box_triangles(half, center)
            for half, center in HOPPER_ROLLER_COLLIDERS
        )
        hopper_roller_stop = len(base_collision_blocks)
        upper_side_board_start = len(base_collision_blocks)
        base_collision_blocks.extend(
            _box_triangles(half, center)
            for half, center in UPPER_SIDE_BOARD_COLLIDERS
        )
        upper_side_board_stop = len(base_collision_blocks)
        front_side_post_start = len(base_collision_blocks)
        base_collision_blocks.extend(
            _box_triangles(half, center)
            for half, center in FRONT_SIDE_POST_COLLIDERS
        )
        front_side_post_stop = len(base_collision_blocks)
        feeder_tunnel_start = len(base_collision_blocks)
        base_collision_blocks.extend(
            _box_triangles(half, center)
            for half, center in FEEDER_TUNNEL_COLLIDERS
        )
        feeder_tunnel_stop = len(base_collision_blocks)
        shooter_cage_start = len(base_collision_blocks)
        base_collision_blocks.extend(
            _box_triangles(half, center)
            for half, center in SHOOTER_CAGE_COLLIDERS
        )
        shooter_cage_stop = len(base_collision_blocks)
        belly_pan_start = len(base_collision_blocks)
        base_collision_blocks.append(_box_triangles(*BELLY_PAN_COLLIDER))
        belly_pan_stop = len(base_collision_blocks)
        turret_roller_start = len(base_collision_blocks)
        base_collision_blocks.extend(
            _box_triangles(half, center)
            for half, center in TURRET_ROLLER_COLLIDERS
        )
        turret_roller_stop = len(base_collision_blocks)
        net_collision_start = len(base_collision_blocks)
        net_collision_stop = net_collision_start + len(net_collision_blocks)
        compact_net_start = net_collision_stop
        compact_net_stop = compact_net_start + len(compact_net_collision_blocks)
        collision_blocks = (
            base_collision_blocks + net_collision_blocks
            + compact_net_collision_blocks + bumper_blocks
        )
        chassis_collider_count = compact_net_stop
        UsdGeom.Xform.Define(stage, NET_EXTENDED_COLLIDER_ROOT)
        UsdGeom.Xform.Define(stage, NET_COMPACT_COLLIDER_ROOT)

        bumper_physics = sb.UsdShade.Material.Define(
            stage, "/World/PhysicsMaterials/CompetitionRobotSoftBumper"
        )
        bumper_api = UsdPhysics.MaterialAPI.Apply(bumper_physics.GetPrim())
        # Low friction so FUEL slides off the bumper skirt instead of gripping and
        # getting dragged/stuck; a little restitution nudges balls away from it.
        bumper_api.CreateStaticFrictionAttr(0.30)
        bumper_api.CreateDynamicFrictionAttr(0.22)
        bumper_api.CreateRestitutionAttr(0.10)
        net_physics = sb.UsdShade.Material.Define(
            stage, "/World/PhysicsMaterials/RobotStorageNet"
        )
        net_api = UsdPhysics.MaterialAPI.Apply(net_physics.GetPrim())
        net_api.CreateStaticFrictionAttr(0.38)
        net_api.CreateDynamicFrictionAttr(0.28)
        net_api.CreateRestitutionAttr(0.06)
        # Elastic net: soft compliant contact (~20x softer than the near-rigid
        # FUEL pile) so a ball pushing up SINKS into the net and the net springs
        # it back, instead of hitting a rigid lid.  The give also lets the pile
        # bulge the net upward, so a few more balls fit under it.
        try:
            net_physx = sb.PhysxSchema.PhysxMaterialAPI.Apply(net_physics.GetPrim())
            net_physx.CreateCompliantContactStiffnessAttr(12000.0)
            net_physx.CreateCompliantContactDampingAttr(250.0)
        except Exception as exc:  # pragma: no cover - schema availability guard
            print(f"compliant net contacts unavailable: {exc}", flush=True)
        for index, triangles in enumerate(collision_blocks):
            if index == 4:
                collider_path = f"{CAD_HORIZONTAL_PATH}/Colliders/FrontBoard"
            elif index == 1:
                # This rear closure already tops out at 0.50 m (trench-safe).
                # Lowering it swept directly through both staged FUEL rows and
                # hard-stopped the vertical link halfway through compact mode.
                collider_path = f"{CAD_HOPPER_BASE_PATH}/Colliders/Wall_{index}"
            elif index in (2, 3):
                # Lower side walls start at the roller bed (z=0.145) and are
                # fixed.  Moving them down made their bottom hit the field after
                # exactly 0.145 m, hard-stopping the upper mechanism early.
                collider_path = f"{CAD_HOPPER_BASE_PATH}/Colliders/Wall_{index}"
            elif index in (5, 6):
                collider_path = f"{CAD_INTAKE_PATH}/Colliders/Intake_{index}"
            elif index == 7:
                collider_path = f"{CAD_VERTICAL_PATH}/Colliders/UpperBoard"
            elif hopper_roller_start <= index < hopper_roller_stop:
                collider_path = (
                    f"{CAD_HOPPER_BASE_PATH}/Colliders/"
                    f"Roller_{index-hopper_roller_start:02d}"
                )
            elif upper_side_board_start <= index < upper_side_board_stop:
                # These are the two tall upper hopper sheets.  They ride the
                # vertical link and therefore lower below trench clearance in
                # compact mode instead of remaining as static invisible walls.
                collider_path = (
                    f"{CAD_VERTICAL_PATH}/Colliders/"
                    f"UpperSideBoard_{index-upper_side_board_start:02d}"
                )
            elif front_side_post_start <= index < front_side_post_stop:
                collider_path = (
                    f"{CAD_HORIZONTAL_PATH}/Colliders/"
                    f"FrontSidePost_{index-front_side_post_start:02d}"
                )
            elif feeder_tunnel_start <= index < feeder_tunnel_stop:
                feeder_index = index - feeder_tunnel_start
                if feeder_index == 0:
                    # The roller bed is chassis-fixed.  Parenting its z=0.143 m
                    # floor to a -0.140 m shooter stroke drove the collider
                    # below the field and lifted the whole robot in compact mode.
                    collider_path = (
                        f"{CAD_HOPPER_BASE_PATH}/Colliders/FeederTunnelFloor"
                    )
                else:
                    collider_path = (
                        f"{CAD_SHOOTER_RETRACT_PATH}/Colliders/"
                        f"FeederTunnel_{feeder_index:02d}"
                    )
            elif shooter_cage_start <= index < shooter_cage_stop:
                collider_path = (
                    f"{CAD_SHOOTER_RETRACT_PATH}/Colliders/"
                    f"ShooterCage_{index-shooter_cage_start:02d}"
                )
            elif belly_pan_start <= index < belly_pan_stop:
                collider_path = f"{self.chassis_path}/Colliders/BellyPan"
            elif turret_roller_start <= index < turret_roller_stop:
                collider_path = (
                    f"{self.chassis_path}/Colliders/"
                    f"TurretRoller_{index-turret_roller_start:02d}"
                )
            elif net_collision_start <= index < net_collision_stop:
                # The net lid also rides the vertical link so it drops below the
                # trench height when compact instead of staying up at 0.73 m.
                collider_path = f"{CAD_VERTICAL_PATH}/NetColliders/C{index-net_collision_start:02d}"
            elif compact_net_start <= index < compact_net_stop:
                collider_path = f"{NET_COMPACT_COLLIDER_ROOT}/C{index-compact_net_start:02d}"
            else:
                collider_path = f"{self.chassis_path}/Colliders/C{index:02d}"
            if index in (5, 6):
                triangles = triangles - CAD_INTAKE_PIVOT
            prim = sb._mesh(
                collider_path,
                triangles,
                (0.1, 0.8, 0.2, 0.12),
                collision=True,
            )
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            # The default ~20 mm contact envelope on both the robot and field
            # inflated a real 18-29 mm trench gap into an invisible collision.
            # Millimetre-scale offsets match these detailed rigid mechanisms
            # while retaining a positive rest separation (no tunnelling).
            physx_collision = sb.PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr(0.004)
            physx_collision.CreateRestOffsetAttr(0.001)
            if net_collision_start <= index < compact_net_stop:
                sb.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    net_physics, materialPurpose="physics"
                )
            elif index >= chassis_collider_count:
                sb.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    bumper_physics, materialPurpose="physics"
                )
            prim.GetAttribute("visibility").Set(
                "inherited" if sb.debug_colliders else "invisible"
            )
            if compact_net_start <= index < compact_net_stop:
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)

        wheel_material = sb.UsdShade.Material.Define(
            stage, "/World/PhysicsMaterials/CompetitionRobotWheelRubber"
        )
        wheel_api = UsdPhysics.MaterialAPI.Apply(wheel_material.GetPrim())
        wheel_api.CreateStaticFrictionAttr(1.05)
        wheel_api.CreateDynamicFrictionAttr(0.90)
        wheel_api.CreateRestitutionAttr(0.0)

        for name in MODULE_ORDER:
            mx, my = MODULE_POSITIONS[name]
            module_center = (float(mx), float(my), WHEEL_RADIUS_M)
            steer_path = f"{self.root_path}/module_{name}"
            steer = UsdGeom.Xform.Define(stage, steer_path)
            steer.AddTranslateOp().Set(Gf.Vec3d(*module_center))
            UsdPhysics.RigidBodyAPI.Apply(steer.GetPrim())
            UsdPhysics.MassAPI.Apply(steer.GetPrim()).CreateMassAttr(0.5)
            sb.PhysxSchema.PhysxRigidBodyAPI.Apply(steer.GetPrim())

            steer_joint = UsdPhysics.RevoluteJoint.Define(
                stage, f"{steer_path}/steer_{name}"
            )
            steer_joint.CreateBody0Rel().SetTargets([self.chassis_path])
            steer_joint.CreateBody1Rel().SetTargets([steer_path])
            steer_joint.CreateAxisAttr("Z")
            steer_joint.CreateLocalPos0Attr(Gf.Vec3f(*module_center))
            steer_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            steer_drive = UsdPhysics.DriveAPI.Apply(steer_joint.GetPrim(), "angular")
            steer_drive.CreateTypeAttr("force")
            steer_drive.CreateTargetPositionAttr(0.0)
            steer_drive.CreateStiffnessAttr(220.0)
            steer_drive.CreateDampingAttr(22.0)
            steer_drive.CreateMaxForceAttr(45.0)

            wheel_path = f"{self.root_path}/wheel_{name}"
            wheel = UsdGeom.Xform.Define(stage, wheel_path)
            wheel.AddTranslateOp().Set(Gf.Vec3d(*module_center))
            UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
            UsdPhysics.MassAPI.Apply(wheel.GetPrim()).CreateMassAttr(1.0)
            sb.PhysxSchema.PhysxRigidBodyAPI.Apply(wheel.GetPrim())
            cylinder = UsdGeom.Cylinder.Define(stage, f"{wheel_path}/collider")
            cylinder.CreateAxisAttr("Y")
            cylinder.CreateRadiusAttr(WHEEL_RADIUS_M)
            cylinder.CreateHeightAttr(0.045)
            cylinder.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                [Gf.Vec3f(0.035, 0.035, 0.04)]
            )
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            sb.UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(
                wheel_material, materialPurpose="physics"
            )
            cylinder.GetPrim().GetAttribute("visibility").Set("invisible")

            drive_joint = UsdPhysics.RevoluteJoint.Define(
                stage, f"{wheel_path}/drive_{name}"
            )
            drive_joint.CreateBody0Rel().SetTargets([steer_path])
            drive_joint.CreateBody1Rel().SetTargets([wheel_path])
            drive_joint.CreateAxisAttr("Y")
            drive_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            drive_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            drive = UsdPhysics.DriveAPI.Apply(drive_joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(9.0)
            drive.CreateMaxForceAttr(6.5)

        return {
            "robot_name": "Competition Robot (Reference Robot CAD)",
            "robot_visual_triangles": int(robot_visual_faces),
            "robot_split_cad_mechanisms": bool(use_split_cad),
            "robot_visual_vertices": int(len(vertices)),
            "robot_chassis_colliders": len(collision_blocks),
            "robot_bumper_assemblies": 2,
            "robot_bumper_collision_pads": len(bumper_blocks),
            "robot_bumper_zone_m": [BUMPER_BOTTOM_M, BUMPER_TOP_M],
            "robot_bumper_continuous": True,
            "robot_bumper_alliance": "blue",
            "robot_bumper_profile": "front and rear C-shaped external soft assemblies",
            "robot_storage_net": "folding triangular white side net + sloped black top net",
            "robot_storage_motion": "reference-style four-bar compact -> raised horizontal carriage; intake below",
            "robot_storage_height_m": [STORAGE_LOWERED_TOP_M, STORAGE_EXTENDED_TOP_M],
            "robot_trench_clearance_m": 22.25 * 0.0254,
            "robot_wheels": 4,
            "robot_mass_kg": ROBOT_MASS_KG,
            "robot_articulated": True,
            "robot_drive": "competition four-module swerve",
            "robot_control": "calibrated shooter + moving-aim vector compensation",
        }


class CompetitionRobotController:
    def __init__(self, alliance_lock: str | None = None):
        if alliance_lock is not None and alliance_lock not in HUB_TARGETS:
            raise ValueError(f"unknown alliance lock: {alliance_lock!r}")
        self.alliance_lock = alliance_lock
        self.articulation: Any = None
        self.control = CalibratedRobotControl()
        self.state_machine = ShooterStateMachine(
            cooldown_s=SHOOTER_COOLDOWN_S,
            yaw_tolerance_deg=AUTO_ALIGN_TOLERANCE_DEG,
            max_speed_mps=SCORE_FIRE_MAX_SPEED_MPS,
            max_yaw_rate_dps=FIRE_MAX_YAW_RATE_DPS,
        )
        self._steer_indices: dict[str, int] = {}
        self._drive_indices: dict[str, int] = {}
        self._mechanism_indices: dict[str, int] = {}
        self.preload_slots = _hopper_slots()
        self.magazine: list[int] = []
        self.captured_indices: set[int] = set()
        self.intake_transit: dict[int, int] = {}
        self.intake_lane_y: dict[int, float] = {}
        self.intake_release_target: dict[int, np.ndarray] = {}
        self.feeder_queue: list[int] = []
        self.pen_reserved: set[int] = set()
        self.intake_on = False
        self.last_shot_time = -1.0
        self.last_aim_solution: dict[str, Any] = {}
        self.shots_fired = 0
        self.balls_collected = 0
        self.barrels = SHOOTER_LANES
        self.volleys_fired = 0
        self.last_fired_indices: list[int] = []
        self._muzzle_watch: set[int] = set()
        self._last_module_angles = {name: 0.0 for name in MODULE_ORDER}
        self._driver_field_velocity = np.zeros(2, dtype=np.float32)
        self._driver_omega = 0.0
        self.retention_corrections = 0
        self.storage_position = STORAGE_EXTENDED_POSITION
        self.storage_target = STORAGE_EXTENDED_POSITION
        self.intake_extension = 1.0
        self.container_extension = 1.0
        self.mechanism_phase = "DEPLOYED"
        self.mechanism_transitions = 0
        self._net_collision_compact: bool | None = None
        self.compact_feeder_ready = True
        self._batch_wait_started: float | None = None
        self._volley_target_count: int | None = None
        self._shooter_rng = random.Random(2026)
        self._next_shot_gap_s = SHOOTER_COOLDOWN_S

    def initialize(self, articulation: Any) -> None:
        self.articulation = articulation
        names = list(articulation.dof_names)
        for module in MODULE_ORDER:
            self._steer_indices[module] = names.index(f"steer_{module}")
            self._drive_indices[module] = names.index(f"drive_{module}")
        for joint_name in (
            "intake_fold", "horizontal_retract", "vertical_lower", "shooter_lower"
        ):
            if joint_name in names:
                self._mechanism_indices[joint_name] = names.index(joint_name)
        count = len(names)
        kps = np.zeros(count, dtype=np.float32)
        kds = np.zeros(count, dtype=np.float32)
        for index in self._steer_indices.values():
            kps[index], kds[index] = 220.0, 22.0
        for index in self._drive_indices.values():
            kps[index], kds[index] = 0.0, 12.0
        for index in self._mechanism_indices.values():
            kps[index], kds[index] = 3200.0, 180.0
        controller = articulation.get_articulation_controller()
        controller.set_gains(kps=kps, kds=kds)
        efforts = np.full(count, 9.0, dtype=np.float32)
        for index in self._steer_indices.values():
            efforts[index] = 45.0
        for index in self._mechanism_indices.values():
            efforts[index] = 2600.0
        try:
            controller.set_max_efforts(efforts)
        except AttributeError:
            articulation.set_max_efforts(efforts)

    def snap_storage_state(self, extended: bool) -> None:
        """Set the initial mechanism pose before the first live physics frame.

        This is only for MATCH staging/reset.  Runtime C/N commands continue to
        use :meth:`step_mechanisms` and therefore retain their physical motion.
        """

        value = 1.0 if extended else 0.0
        self.storage_target = (
            STORAGE_EXTENDED_POSITION if extended else STORAGE_LOWERED_POSITION
        )
        self.storage_position = self.storage_target
        self.container_extension = value
        self.intake_extension = value
        self.intake_on = bool(extended)
        self.mechanism_phase = "DEPLOYED" if extended else "COMPACT"
        if self.articulation is not None and self._mechanism_indices:
            positions = self._np(self.articulation.get_joint_positions()).astype(
                np.float32, copy=True
            )
            targets = {
                "intake_fold": 0.0 if extended else math.radians(INTAKE_STOW_ANGLE_DEG),
                "horizontal_retract": 0.0 if extended else -CAD_HORIZONTAL_RETRACT_M,
                "vertical_lower": 0.0 if extended else -CAD_VERTICAL_LOWER_M,
                "shooter_lower": 0.0 if extended else -CAD_SHOOTER_LOWER_M,
            }
            for name, index in self._mechanism_indices.items():
                positions[index] = float(targets[name])
            self.articulation.set_joint_positions(positions)
            try:
                velocities = self._np(
                    self.articulation.get_joint_velocities()
                ).astype(np.float32, copy=True)
                for index in self._mechanism_indices.values():
                    velocities[index] = 0.0
                self.articulation.set_joint_velocities(velocities)
            except (AttributeError, RuntimeError):
                pass
        self._apply_mechanism_visuals()
        self._command_mechanism_joints()

    @staticmethod
    def _np(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def chassis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = self.articulation.get_world_pose()
        return self._np(position).astype(np.float32), self._np(orientation).astype(np.float32)

    def chassis_yaw(self) -> float:
        _, q = self.chassis_pose()
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def chassis_velocity(self) -> tuple[np.ndarray, float]:
        linear = self._np(self.articulation.get_linear_velocity()).astype(np.float32)
        angular = self._np(self.articulation.get_angular_velocity()).astype(np.float32)
        return linear, float(angular[2])

    @property
    def storage_top(self) -> float:
        return storage_top_m(self.storage_position)

    @property
    def intake_deployed(self) -> bool:
        return self.intake_extension >= 0.95 and self.container_extension >= 0.95

    def set_storage_extended(self, extended: bool) -> None:
        """Select the source-faithful staged deploy/compact sequence."""
        target = STORAGE_EXTENDED_POSITION if extended else STORAGE_LOWERED_POSITION
        if target != self.storage_target:
            self.mechanism_transitions += 1
            self.mechanism_phase = "CONTAINER_OPENING" if extended else "PREP_CLEARANCE"
            # Storage commands provide a one-time intake default.  The separate
            # intake button remains free to override it afterward.
            self.intake_on = bool(extended)
        self.storage_target = target

    def toggle_storage(self) -> None:
        self.set_storage_extended(self.storage_target < 0.5)

    def _apply_mechanism_visuals(self) -> None:
        """Drive the split full-CAD assemblies and their flexible net."""
        try:
            import omni.usd
            from pxr import Gf, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            # The rendered ropes deform continuously, while collision uses two
            # closed tiled surfaces.  Swap to the correctly shortened compact
            # surface during contraction; leaving the extended surface active
            # gives the invisible net the wrong +X footprint under the trench.
            compact_collision = self.container_extension < 0.5
            if compact_collision != self._net_collision_compact:
                for root_path, enabled in (
                    (f"{CAD_VERTICAL_PATH}/NetColliders", not compact_collision),
                    (NET_COMPACT_COLLIDER_ROOT, compact_collision),
                ):
                    root = stage.GetPrimAtPath(root_path)
                    if not root or not root.IsValid():
                        continue
                    for child in root.GetChildren():
                        enabled_attr = UsdPhysics.CollisionAPI(
                            child
                        ).GetCollisionEnabledAttr()
                        if enabled_attr:
                            enabled_attr.Set(bool(enabled))
                self._net_collision_compact = compact_collision
            side, top, linkage, t = _storage_curve_segments(self.storage_position)
            for path, segments in (
                (STORAGE_SIDE_NET_PATH, side),
                (STORAGE_TOP_NET_PATH, top),
                (STORAGE_LINKAGE_PATH, linkage),
            ):
                points = [Gf.Vec3f(*point) for segment in segments for point in segment]
                proc_prim = stage.GetPrimAtPath(path)
                if proc_prim and proc_prim.IsValid():
                    proc_prim.GetAttribute("points").Set(points)

            # Full CAD movement is driven by articulation joints (never by
            # rewriting collision Xforms, which invalidates PhysX tensor views).
            horizontal_offset = -CAD_HORIZONTAL_RETRACT_M * (1.0 - self.container_extension)
            vertical_offset = -CAD_VERTICAL_LOWER_M * (1.0 - self.container_extension)

            # Chassis-local net: high/rear knots follow the vertical link; low/
            # front knots follow the horizontal link.  No inherited transform or
            # inverse compensation means Fabric sees one unambiguous coordinate
            # system and cannot leave four stretched endpoint remnants.
            net_corners = NET_EXTENDED_CORNERS.copy()
            net_corners[:2, 2] += vertical_offset
            net_corners[2:, 0] += horizontal_offset
            net_segments, _ = _four_corner_soft_net_segments(net_corners)
            rope_path = f"{NET_VISUAL_ROOT}/Ropes"
            net_points = [
                Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
                for segment in net_segments for point in segment
            ]
            rope_prim = stage.GetPrimAtPath(rope_path)
            if rope_prim and rope_prim.IsValid():
                rope_prim.GetAttribute("points").Set(net_points)
            # PhysX Fabric renders curves from its own flatcache buffers and
            # ignores the plain USD point write above, so push the same geometry
            # through the runtime (usdrt) stage so the top edge actually deforms.
            try:
                from usdrt import Usd as _RtUsd, Vt as _RtVt, Gf as _RtGf, Sdf as _RtSdf

                _rt = _RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
                _rope = _rt.GetPrimAtPath(_RtSdf.Path(rope_path))
                if _rope and _rope.IsValid():
                    _pts_attr = _rope.GetAttribute("points")
                    if _pts_attr and _pts_attr.IsValid():
                        _pts_attr.Set(
                            _RtVt.Vec3fArray(
                                [
                                    _RtGf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                                    for segment in net_segments for p in segment
                                ]
                            )
                        )
                for index, corner in enumerate(net_corners):
                    _knot = _rt.GetPrimAtPath(
                        _RtSdf.Path(f"{NET_VISUAL_ROOT}/CornerKnot_{index}")
                    )
                    if _knot and _knot.IsValid():
                        _translate = _knot.GetAttribute("xformOp:translate")
                        if _translate and _translate.IsValid():
                            _translate.Set(
                                _RtGf.Vec3d(
                                    float(corner[0]), float(corner[1]), float(corner[2])
                                )
                            )
            except Exception as net_runtime_error:
                if not getattr(self, "_reported_net_runtime_error", False):
                    print(f"NET_RUNTIME_UPDATE_FAILED {net_runtime_error!r}", flush=True)
                    self._reported_net_runtime_error = True
            for index, corner in enumerate(net_corners):
                knot = stage.GetPrimAtPath(f"{NET_VISUAL_ROOT}/CornerKnot_{index}")
                if knot and knot.IsValid():
                    knot.GetAttribute("xformOp:translate").Set(
                        Gf.Vec3d(float(corner[0]), float(corner[1]), float(corner[2]))
                    )

            # Deployment sequence from the native assembly: the under-frame
            # intake starts moving first, then the paired links lift the upper
            # carriage.  Retraction follows the same collision-safe path back.
            stage.GetPrimAtPath(INTAKE_XFORM_PATH).GetAttribute("xformOp:rotateY").Set(
                INTAKE_STOW_ANGLE_DEG * (1.0 - self.intake_extension)
            )
        except (AttributeError, RuntimeError, ImportError):
            # Unit tests deliberately run without a live USD stage.
            pass

    def step_mechanisms(self, dt_s: float = 1.0 / 30.0) -> float:
        """Advance the robot clearance -> intake -> container sequence."""
        dt = max(0.0, float(dt_s))
        wants_extended = self.storage_target == STORAGE_EXTENDED_POSITION
        intake_step = INTAKE_EXTENSION_PER_S * dt
        container_step = CONTAINER_EXTENSION_PER_S * dt
        if wants_extended:
            if self.container_extension < 1.0:
                self.mechanism_phase = "CONTAINER_OPENING"
                self.container_extension = min(1.0, self.container_extension + container_step)
            elif self.intake_extension < 1.0:
                self.mechanism_phase = "INTAKE_DEPLOYING"
                self.intake_extension = min(1.0, self.intake_extension + intake_step)
            else:
                self.mechanism_phase = "DEPLOYED"
        else:
            # RobotContainer.cpp first opens the storage for clearance, then
            # retracts intake to 0.03, and only then closes Weidai to 0.02.
            if not self.compact_feeder_ready:
                # Do not sweep the retracting front board through loose FUEL.
                # Up to eight balls must first reach the rigid pre-shot tunnel;
                # a fuller robot must shoot down before trench mode is possible.
                self.mechanism_phase = "FEEDER_STAGING"
            elif self.container_extension < 1.0 and self.intake_extension > 0.0:
                self.mechanism_phase = "PREP_CLEARANCE"
                self.container_extension = min(1.0, self.container_extension + container_step)
            elif self.intake_extension > 0.0:
                self.mechanism_phase = "INTAKE_RETRACTING"
                self.intake_extension = max(0.0, self.intake_extension - intake_step)
            elif self.container_extension > 0.0:
                self.mechanism_phase = "CONTAINER_CLOSING"
                self.container_extension = max(0.0, self.container_extension - container_step)
            else:
                self.mechanism_phase = "COMPACT"
        self.storage_position = STORAGE_LOWERED_POSITION + self.container_extension * (
            STORAGE_EXTENDED_POSITION - STORAGE_LOWERED_POSITION
        )
        self._apply_mechanism_visuals()
        self._command_mechanism_joints()
        return self.storage_position

    def _command_mechanism_joints(self) -> None:
        """Send stable joint targets; never mutate live collider Xforms."""
        if self.articulation is None or not self._mechanism_indices:
            return
        try:
            from isaacsim.core.utils.types import ArticulationAction

            positions = np.full(len(self.articulation.dof_names), np.nan, dtype=np.float32)
            targets = {
                "intake_fold": math.radians(
                    INTAKE_STOW_ANGLE_DEG * (1.0 - self.intake_extension)
                ),
                "horizontal_retract": -CAD_HORIZONTAL_RETRACT_M
                * (1.0 - self.container_extension),
                "vertical_lower": -CAD_VERTICAL_LOWER_M
                * (1.0 - self.container_extension),
                "shooter_lower": -CAD_SHOOTER_LOWER_M
                * (1.0 - self.container_extension),
            }
            for name, index in self._mechanism_indices.items():
                positions[index] = float(targets[name])
            self.articulation.apply_action(ArticulationAction(joint_positions=positions))
        except (AttributeError, RuntimeError, ImportError):
            pass

    def drive_swerve(self, vx: float, vy: float, omega: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        states = inverse_kinematics(float(vx), float(vy), float(omega))
        positions = np.full(len(self.articulation.dof_names), np.nan, dtype=np.float32)
        velocities = np.full(len(self.articulation.dof_names), np.nan, dtype=np.float32)
        try:
            measured = self._np(self.articulation.get_joint_positions())
        except (AttributeError, RuntimeError):
            measured = None
        for module in MODULE_ORDER:
            steer_index = self._steer_indices[module]
            current = (
                float(measured[steer_index]) if measured is not None else self._last_module_angles[module]
            )
            state = optimize_module(states[module], current)
            positions[steer_index] = state.angle_rad
            velocities[self._drive_indices[module]] = state.speed_mps / WHEEL_RADIUS_M
            self._last_module_angles[module] = state.angle_rad
        self.articulation.apply_action(
            ArticulationAction(joint_positions=positions, joint_velocities=velocities)
        )

    def drive(self, forward: float, turn: float, strafe: float = 0.0) -> None:
        """robot default teleop: field-centric closed-loop swerve request."""
        target_field = np.array(
            [np.clip(forward, -1.0, 1.0), np.clip(strafe, -1.0, 1.0)],
            dtype=np.float32,
        ) * (MAX_WHEEL_SPEED_MPS * DRIVER_SPEED_RATE * KEYBOARD_TRANSLATION_SCALE)
        delta = target_field - self._driver_field_velocity
        limit = (
            DRIVER_DECEL_LIMIT_MPS2
            if np.linalg.norm(target_field) < np.linalg.norm(self._driver_field_velocity)
            else DRIVER_ACCEL_LIMIT_MPS2
        ) * DRIVER_CONTROL_DT_S
        norm = float(np.linalg.norm(delta))
        if norm > limit:
            delta *= limit / norm
        self._driver_field_velocity += delta

        target_omega = (
            float(np.clip(turn, -1.0, 1.0))
            * MAX_ANGULAR_RATE_RAD_S
            * DRIVER_ANGULAR_RATE
            * KEYBOARD_TURN_SCALE
        )
        omega_delta = float(np.clip(
            target_omega - self._driver_omega,
            -DRIVER_ANGULAR_ACCEL_LIMIT_RAD_S2 * DRIVER_CONTROL_DT_S,
            DRIVER_ANGULAR_ACCEL_LIMIT_RAD_S2 * DRIVER_CONTROL_DT_S,
        ))
        self._driver_omega += omega_delta
        vx_robot, vy_robot = field_to_robot(
            float(self._driver_field_velocity[0]),
            float(self._driver_field_velocity[1]),
            self.chassis_yaw(),
        )
        self.drive_swerve(
            vx_robot,
            vy_robot,
            self._driver_omega,
        )

    def preload(
        self,
        fuel_view: Any,
        count: int = XRC_PRELOAD_COUNT,
        start_index: int = 0,
        indices: list[int] | None = None,
    ) -> None:
        if indices:
            # Formal match: preload the designated "preloads/*" FUEL bodies so
            # the depot/row/warehouse layout stays exactly official.
            chosen = [
                int(i) for i in indices[:count] if 0 <= int(i) < fuel_view.count
            ][: len(self.preload_slots)]
            self.magazine = chosen
        else:
            count = min(count, fuel_view.count - start_index, len(self.preload_slots))
            self.magazine = list(range(start_index, start_index + max(0, count)))
        self.captured_indices = set(self.magazine)
        self.pen_reserved = set(self.magazine)
        self.intake_transit.clear()
        self.intake_lane_y.clear()
        self.intake_release_target.clear()
        self.feeder_queue.clear()
        self.compact_feeder_ready = not self.magazine
        if not self.magazine:
            return
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        positions = position + self.preload_slots[: len(self.magazine)] @ rotation.T
        indices = np.asarray(self.magazine, dtype=np.int32)
        zeros = np.zeros((len(indices), 3), dtype=np.float32)
        fuel_view.set_world_poses(positions=positions.astype(np.float32), indices=indices)
        fuel_view.set_linear_velocities(zeros, indices=indices)
        fuel_view.set_angular_velocities(zeros, indices=indices)

    def sync_magazine(self, fuel_view: Any) -> None:
        """Detect dynamic balls physically inside the hopper; never move them."""
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        raw, _ = fuel_view.get_world_poses()
        balls = self._np(raw).astype(np.float32)
        local = (balls - position) @ rotation
        # Detection uses the full swept envelope, so folding the flexible net
        # never makes an already captured rigid body disappear from inventory.
        inside = np.all((local >= HOPPER_MIN_LOCAL) & (local <= HOPPER_MAX_LOCAL), axis=1)
        detected = [int(i) for i in np.flatnonzero(inside) if int(i) not in self.intake_transit]
        self.captured_indices.update(detected)
        stored = [i for i in self.captured_indices if i not in self.intake_transit]
        previous_order = [i for i in self.magazine if i in stored]
        self.magazine = previous_order + [i for i in stored if i not in previous_order]
        self.pen_reserved = set(self.magazine) | set(self.intake_transit)

        # FUEL is contained purely by the rigid hopper colliders now - there are
        # no velocity "walls".  Captured balls settle physically anywhere inside
        # the machine, filling the whole bottom + top volume, and nothing is
        # teleported or clamped by code.

    def _sync_intake_transit(self, fuel_view: Any, dt_s: float) -> int:
        """Move captured balls through the xRC-style force-zone conveyor."""
        if not self.intake_transit:
            return 0
        robot_position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        chassis_linear, _ = self.chassis_velocity()
        raw, _ = fuel_view.get_world_poses()
        balls = self._np(raw).astype(np.float32)
        all_local = (balls - robot_position) @ rotation
        completed: list[int] = []
        release_final = len(INTAKE_PATH_LOCAL) - 1
        release_approach = release_final - 1
        for index, waypoint in list(self.intake_transit.items()):
            local = all_local[index]
            if waypoint >= release_approach:
                target = self.intake_release_target.get(index)
                if target is None:
                    target = self._choose_intake_release_target(
                        all_local, index, self.intake_lane_y.get(index, float(local[1]))
                    )
                    if target is None:
                        # Once all preferred indexer cells are occupied, a ball
                        # that has physically crossed the front board is already
                        # in the hopper.  Release it in place so the compliant
                        # foam/net contacts find a natural packed position.
                        # Balls still outside the throat remain a real jam.
                        if (
                            float(local[0]) <= 0.43
                            and abs(float(local[1])) <= 0.29
                            and float(local[2]) >= 0.24
                        ):
                            completed.append(index)
                        continue
                    self.intake_release_target[index] = target
                target = target.copy()
                if waypoint == release_approach:
                    # Traverse above the solid roller bank first, then descend
                    # vertically into the selected dynamic-pile cell.
                    target[2] = max(float(target[2]), 0.340)
            else:
                target = INTAKE_PATH_LOCAL[waypoint].copy()
                target[1] = self.intake_lane_y.get(index, float(local[1]))
            delta = target - local
            distance = float(np.linalg.norm(delta))
            if distance < 0.09:
                retargeted = False
                if waypoint == release_final:
                    # A blocked hopper entrance must stall the roller instead
                    # of position-driving another ball into the pile and
                    # pressurising the body.  A destination can become occupied
                    # after it was reserved because released FUEL remains fully
                    # dynamic, so first try another free indexer destination.
                    other_indices = [
                        i for i in range(len(balls))
                        if i != index and i not in self.intake_transit
                    ]
                    if other_indices:
                        clearance = np.linalg.norm(
                            all_local[np.asarray(other_indices, dtype=np.int32)] - target,
                            axis=1,
                        )
                        if float(clearance.min()) < FUEL_RELEASE_CLEARANCE_M:
                            replacement = self._choose_intake_release_target(
                                all_local,
                                index,
                                self.intake_lane_y.get(index, float(local[1])),
                            )
                            if replacement is None:
                                if (
                                    -0.10 <= float(local[0]) <= 0.43
                                    and abs(float(local[1])) <= 0.29
                                    and 0.24 <= float(local[2]) <= 0.66
                                ):
                                    # The ball has reached the physical pile
                                    # under the net; release it where contact
                                    # forces stopped it instead of pushing the
                                    # soft lid toward an idealized grid point.
                                    completed.append(index)
                                continue
                            self.intake_release_target[index] = replacement
                            target = replacement.copy()
                            delta = target - local
                            distance = float(np.linalg.norm(delta))
                            retargeted = True
                if not retargeted:
                    waypoint += 1
                    self.intake_transit[index] = waypoint
                    if waypoint >= len(INTAKE_PATH_LOCAL):
                        completed.append(index)
                        continue
                    if waypoint >= release_approach:
                        target = self.intake_release_target.get(index)
                        if target is None:
                            target = self._choose_intake_release_target(
                                all_local, index, self.intake_lane_y.get(index, float(local[1]))
                            )
                            if target is None:
                                continue
                            self.intake_release_target[index] = target
                        target = target.copy()
                        if waypoint == release_approach:
                            target[2] = max(float(target[2]), 0.340)
                    else:
                        target = INTAKE_PATH_LOCAL[waypoint].copy()
                        target[1] = self.intake_lane_y.get(index, float(local[1]))
                    delta = target - local
                    distance = float(np.linalg.norm(delta))
            # Position-drive the ball along the intake path so it is carried UP
            # between the two bottom rollers reliably.  A velocity-only pull is
            # fought by the rigid roller nip and stalls, so step the ball's pose
            # directly along the path (the path passes through the nip) and let it
            # go dynamic again once it is released among the stored FUEL.
            conveyor_speed = (
                INTAKE_INDEX_SPEED_MPS
                if waypoint >= release_approach
                else INTAKE_PULL_SPEED_MPS
            )
            step = min(distance, conveyor_speed * max(float(dt_s), 1e-3))
            new_local = local + delta / max(distance, 1e-6) * step
            new_world = robot_position + rotation @ new_local
            indices = np.asarray([index], dtype=np.int32)
            fuel_view.set_world_poses(positions=new_world[None, :].astype(np.float32), indices=indices)
            fuel_view.set_linear_velocities(chassis_linear[None, :].astype(np.float32), indices=indices)
        for index in completed:
            self.intake_transit.pop(index, None)
            self.intake_lane_y.pop(index, None)
            self.intake_release_target.pop(index, None)
            self.captured_indices.add(index)
            self.balls_collected += 1
            indices = np.asarray([index], dtype=np.int32)
            fuel_view.set_linear_velocities(
                chassis_linear[None, :].astype(np.float32), indices=indices
            )
        self.sync_magazine(fuel_view)
        return len(completed)

    def _choose_intake_release_target(
        self, all_local: np.ndarray, index: int, intake_y: float
    ) -> np.ndarray | None:
        """Choose a free low-to-high indexer destination without overlap."""
        occupied_indices = [
            i for i in range(len(all_local))
            if i != index and i not in self.intake_transit
        ]
        occupied = (
            all_local[np.asarray(occupied_indices, dtype=np.int32)]
            if occupied_indices
            else np.empty((0, 3), dtype=np.float32)
        )
        reserved = [
            target for other, target in self.intake_release_target.items()
            if other != index
        ]
        if reserved:
            occupied = np.concatenate(
                [occupied, np.asarray(reserved, dtype=np.float32)], axis=0
            )
        for height in np.unique(INTAKE_RELEASE_GRID_LOCAL[:, 2]):
            layer = INTAKE_RELEASE_GRID_LOCAL[
                np.isclose(INTAKE_RELEASE_GRID_LOCAL[:, 2], height)
            ]
            scored: list[tuple[float, np.ndarray]] = []
            for candidate in layer:
                clearance = (
                    float(np.linalg.norm(occupied - candidate, axis=1).min())
                    if len(occupied)
                    else 10.0
                )
                if clearance < FUEL_RELEASE_CLEARANCE_M:
                    continue
                # Stay near the entering lane when possible, then prefer deeper
                # (-X) locations so the intake mouth remains clear.
                score = clearance - 0.35 * abs(float(candidate[1]) - intake_y)
                score += 0.08 * (0.40 - float(candidate[0]))
                scored.append((score, candidate))
            if scored:
                return max(scored, key=lambda item: item[0])[1].copy()
        return None

    def _step_feeder(self, fuel_view: Any, dt_s: float) -> None:
        """Convey up to three stored balls into the rigid pre-shot tunnel."""
        self.feeder_queue = [i for i in self.feeder_queue if i in self.magazine]
        robot_position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        raw, _ = fuel_view.get_world_poses()
        balls = self._np(raw).astype(np.float32)
        local = (balls - robot_position) @ rotation
        available = [
            i for i in self.magazine
            if i not in self.intake_transit
        ]
        # Assign each open lane to its nearest ball.  Sorting only by X swapped
        # the four preloaded top-row balls with the bottom row, making all eight
        # cross through one another and the solid rollers.
        self.feeder_queue = []
        for lane in range(len(FEEDER_READY_LOCAL)):
            if not available:
                break
            nearest = min(
                available,
                key=lambda index: float(
                    np.linalg.norm(local[index] - FEEDER_READY_LOCAL[lane])
                ),
            )
            self.feeder_queue.append(nearest)
            available.remove(nearest)
        chassis_linear, _ = self.chassis_velocity()
        ready_count = 0
        for lane, index in enumerate(self.feeder_queue[: len(FEEDER_READY_LOCAL)]):
            final_target = FEEDER_READY_LOCAL[lane]
            point = local[index]
            final_distance = float(np.linalg.norm(final_target - point))
            if final_distance <= 0.11:
                ready_count += 1
                continue
            # Route around the rigid hopper rollers and through the low opening
            # under the rear wall.  A straight line to x=-0.30 intersects every
            # roller and used to eject balls back toward the intake.
            if float(point[0]) > -0.02:
                if (
                    float(point[2]) < 0.325
                    or abs(float(point[1] - final_target[1])) > 0.035
                ):
                    target = np.array(
                        [point[0], final_target[1], 0.340], dtype=np.float32
                    )
                else:
                    target = np.array(
                        [-0.04, final_target[1], 0.340], dtype=np.float32
                    )
            elif float(point[0]) > -0.20:
                if float(point[2]) > 0.300:
                    target = np.array(
                        [-0.06, final_target[1], 0.280], dtype=np.float32
                    )
                else:
                    target = np.array(
                        [-0.23, final_target[1], 0.280], dtype=np.float32
                    )
            else:
                target = final_target
            delta = target - point
            distance = float(np.linalg.norm(delta))
            if distance <= 0.018:
                continue
            # Tail boost: with <=9 FUEL left the rollers are unloaded (no
            # ball-on-ball drag), so restaging runs faster - this keeps the
            # last balls arriving at the ready slots quickly enough that the
            # air stream stays as dense as when the hopper was full.
            convey = FEEDER_CONVEY_SPEED_MPS * (1.6 if len(self.magazine) <= 9 else 1.0)
            step = min(distance, convey * max(float(dt_s), 1e-3))
            new_local = point + delta / max(distance, 1e-6) * step
            new_world = robot_position + rotation @ new_local
            indices = np.asarray([index], dtype=np.int32)
            fuel_view.set_world_poses(
                positions=new_world[None, :].astype(np.float32), indices=indices
            )
            fuel_view.set_linear_velocities(
                chassis_linear[None, :].astype(np.float32), indices=indices
            )
        required = min(len(self.magazine), len(FEEDER_READY_LOCAL))
        self.compact_feeder_ready = bool(
            len(self.magazine) <= len(FEEDER_READY_LOCAL)
            and ready_count >= required
        )

    def step_intake(
        self, fuel_view: Any, hub_pending: set[int], dt_s: float = 1.0 / 30.0
    ) -> int:
        self.sync_magazine(fuel_view)
        completed = self._sync_intake_transit(fuel_view, dt_s)
        self._step_feeder(fuel_view, dt_s)
        if (
            not self.intake_on
            or not self.intake_deployed
            or len(self.captured_indices | self.pen_reserved) >= HOPPER_PRESSURE_CAPACITY
        ):
            return completed
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        raw, _ = fuel_view.get_world_poses()
        balls = self._np(raw).astype(np.float32)
        local = (balls - position) @ rotation
        mask = np.all(np.abs(local - INTAKE_CENTER_LOCAL) <= INTAKE_HALF_EXTENTS + 0.076, axis=1)
        available_lanes = max(0, INTAKE_SIMULTANEOUS_LANES - len(self.intake_transit))
        if not available_lanes:
            return completed
        candidates = []
        for index in np.flatnonzero(mask):
            index = int(index)
            if index in self.pen_reserved or index in hub_pending:
                continue
            # Prefer balls already deepest under the horizontal roller row.
            candidates.append(index)
        candidates.sort(key=lambda index: float(local[index, 0]))
        for index in candidates[:available_lanes]:
            self.pen_reserved.add(index)
            self.intake_transit[index] = 0
            self.intake_lane_y[index] = float(
                np.clip(
                    local[index, 1],
                    -INTAKE_LATERAL_RELEASE_LIMIT_M,
                    INTAKE_LATERAL_RELEASE_LIMIT_M,
                )
            )
        return completed

    def _hub_target(self, alliance: str | None) -> tuple[str, np.ndarray]:
        if self.alliance_lock is not None:
            alliance = self.alliance_lock
        if alliance in HUB_TARGETS:
            return str(alliance), HUB_TARGETS[str(alliance)].copy()
        position, _ = self.chassis_pose()
        selected = min(HUB_TARGETS, key=lambda h: np.linalg.norm(HUB_TARGETS[h][:2] - position[:2]))
        return selected, HUB_TARGETS[selected].copy()

    def solve_auto_aim(self, alliance: str | None = None) -> dict[str, Any]:
        selected, target = self._hub_target(alliance)
        position, _ = self.chassis_pose()
        linear, _ = self.chassis_velocity()
        # Land at the funnel centre, not the deep sensor point: the raw xRC
        # target sits ~0.17 m outboard of the funnel middle, which read as a
        # stream biased to the back rim (落点靠后).  Shorten along the line
        # of fire so entry centres from either approach side.
        to_target = target[:2] - position[:2]
        horizontal = float(np.linalg.norm(to_target))
        if horizontal > 1e-6:
            target = target.copy()
            target[:2] -= to_target / horizontal * min(
                HUB_AIM_SHORT_BIAS_M, 0.4 * horizontal
            )
        setpoint = self.control.solve_shot(
            position[:2], linear[:2], target[:2], compensate_motion=True
        )
        yaw_error = wrap_angle(setpoint.chassis_yaw_rad - self.chassis_yaw())
        valid = not setpoint.clamped and 1.4 <= setpoint.distance_m <= 5.7
        result = {
            "alliance": selected,
            "distance_m": setpoint.distance_m,
            "speed_mps": setpoint.exit_speed_mps,
            "pitch_deg": setpoint.pitch_deg,
            "motor_target_rps": setpoint.motor_target_rps,
            "desired_yaw_rad": setpoint.chassis_yaw_rad,
            "shot_yaw_rad": setpoint.shot_yaw_rad,
            "flight_time_s": setpoint.flight_time_s,
            "yaw_error_deg": math.degrees(yaw_error),
            "aligned": abs(math.degrees(yaw_error)) <= AUTO_ALIGN_TOLERANCE_DEG,
            "valid": valid,
            "blocked_reason": "" if valid else "outside robot calibrated range 1.4-5.7 m",
            "setpoint": setpoint,
        }
        self.last_aim_solution = result
        return result

    def solve_ferry(self, alliance: str | None = None) -> dict[str, Any]:
        """Solve the long return lob that throws FUEL back toward our zone.

        The 2026 meta shot (reference: "ferry from the opponent zone").  Available
        in every match phase, but refused while the robot is inside its OWN
        court (nothing to return from there).  The hub structures block
        mid-field trajectories, so the solver tries the straight lane first,
        then the two side corridors, sampling the 55-degree parabola over
        both measured hub boxes.  Ranges beyond the flywheel envelope are
        shortened along the same bearing ("as far back as it can throw"),
        and the winning setpoint reuses the exact fire_setpoint stream.
        """
        requested = self.alliance_lock if self.alliance_lock is not None else alliance
        selected = requested if requested in HUB_TARGETS else "blue"
        own_sign = 1.0 if selected == "red" else -1.0
        position, _ = self.chassis_pose()
        x0, y0 = float(position[0]), float(position[1])
        theta = math.radians(FERRY_PITCH_DEG)
        drop = FERRY_LANDING_Z_M - FERRY_MUZZLE_Z_M

        def refuse(reason: str) -> dict[str, Any]:
            result = {
                "alliance": selected,
                "mode": "ferry",
                "valid": False,
                "aligned": False,
                "yaw_error_deg": 0.0,
                "distance_m": 0.0,
                "speed_mps": 0.0,
                "pitch_deg": FERRY_PITCH_DEG,
                "blocked_reason": reason,
                "setpoint": None,
                "target_xy": None,
            }
            self.last_aim_solution = result
            return result

        if y0 * own_sign > NEUTRAL_ZONE_HALF_Y_M:
            return refuse("in own court - ferry from neutral/opponent side")

        def speed_for_range(horizontal_range: float) -> float:
            denominator = 2.0 * math.cos(theta) ** 2 * (
                horizontal_range * math.tan(theta) - drop
            )
            return math.sqrt(
                9.81 * horizontal_range * horizontal_range / max(1e-6, denominator)
            )

        def max_range() -> float:
            v2 = FERRY_MAX_EXIT_SPEED_MPS ** 2
            sin2t = math.sin(2.0 * theta)
            disc = (v2 * sin2t) ** 2 - 8.0 * 9.81 * v2 * math.cos(theta) ** 2 * drop
            return (v2 * sin2t + math.sqrt(max(0.0, disc))) / (2.0 * 9.81)

        def blocked_by_hub(tx: float, ty: float, rng_m: float, v: float) -> bool:
            for hub_sign in (1.0, -1.0):
                y_band = sorted(
                    (hub_sign * FERRY_HUB_BLOCK_Y_M[0], hub_sign * FERRY_HUB_BLOCK_Y_M[1])
                )
                dy_total = ty - y0
                if abs(dy_total) < 1e-6:
                    continue
                f0 = (y_band[0] - y0) / dy_total
                f1 = (y_band[1] - y0) / dy_total
                f_lo = max(0.0, min(f0, f1))
                f_hi = min(1.0, max(f0, f1))
                if f_lo >= f_hi:
                    continue
                for k in range(7):
                    f = f_lo + (f_hi - f_lo) * k / 6.0
                    if abs(x0 + (tx - x0) * f) > FERRY_HUB_BLOCK_X_HALF_M:
                        continue
                    horizontal = f * rng_m
                    z = (
                        FERRY_MUZZLE_Z_M
                        + horizontal * math.tan(theta)
                        - 9.81 * horizontal * horizontal
                        / (2.0 * v * v * math.cos(theta) ** 2)
                    )
                    if z < FERRY_HUB_BLOCK_TOP_M:
                        return True
            return False

        candidates = [float(np.clip(x0, -FERRY_TARGET_X_LIMIT_M, FERRY_TARGET_X_LIMIT_M))]
        for lane in (FERRY_SIDE_LANE_X_M, -FERRY_SIDE_LANE_X_M):
            if all(abs(lane - existing) > 0.2 for existing in candidates):
                candidates.append(lane)
        candidates.sort(key=lambda c: abs(c - x0))
        reach = max_range()
        last_reason = "no clear ferry lane"
        for lane_x in candidates:
            tx, ty = lane_x, own_sign * FERRY_TARGET_Y_M
            rng_m = math.hypot(tx - x0, ty - y0)
            if rng_m < FERRY_MIN_RANGE_M:
                last_reason = "too close to own zone - just drive in"
                continue
            if rng_m > reach:
                scale = reach / rng_m
                tx = x0 + (tx - x0) * scale
                ty = y0 + (ty - y0) * scale
                rng_m = reach
            v = max(FERRY_MIN_EXIT_SPEED_MPS, speed_for_range(rng_m))
            if blocked_by_hub(tx, ty, rng_m, v):
                last_reason = "HUB blocks the return lob - move to a side lane"
                continue
            shot_yaw = math.atan2(ty - y0, tx - x0)
            chassis_yaw = wrap_angle(shot_yaw - self.control.mount_yaw_rad)
            yaw_error = wrap_angle(chassis_yaw - self.chassis_yaw())
            setpoint = ShooterSetpoint(
                distance_m=rng_m,
                exit_speed_mps=v,
                pitch_deg=FERRY_PITCH_DEG,
                motor_target_rps=float(
                    np.interp(
                        v, self.control.motor_speed_axis, self.control.motor_target_rps
                    )
                ),
                chassis_yaw_rad=chassis_yaw,
                shot_yaw_rad=shot_yaw,
                flight_time_s=rng_m / max(1e-6, v * math.cos(theta)),
                clamped=False,
            )
            result = {
                "alliance": selected,
                "mode": "ferry",
                "valid": True,
                "distance_m": rng_m,
                "speed_mps": v,
                "pitch_deg": FERRY_PITCH_DEG,
                "motor_target_rps": setpoint.motor_target_rps,
                "desired_yaw_rad": chassis_yaw,
                "shot_yaw_rad": shot_yaw,
                "flight_time_s": setpoint.flight_time_s,
                "yaw_error_deg": math.degrees(yaw_error),
                "aligned": abs(math.degrees(yaw_error)) <= FERRY_AIM_TOLERANCE_DEG,
                "blocked_reason": "",
                "setpoint": setpoint,
                "target_xy": (float(tx), float(ty)),
            }
            self.last_aim_solution = result
            return result
        return refuse(last_reason)

    def _release(self, fuel_view: Any, index: int, position: np.ndarray, velocity: np.ndarray) -> None:
        indices = np.asarray([index], dtype=np.int32)
        fuel_view.set_world_poses(positions=position[None, :].astype(np.float32), indices=indices)
        fuel_view.set_linear_velocities(velocity[None, :].astype(np.float32), indices=indices)
        fuel_view.set_angular_velocities(np.zeros((1, 3), dtype=np.float32), indices=indices)

    def fire_setpoint(self, fuel_view: Any, solution: dict[str, Any], now_s: float) -> int | None:
        self.last_fired_indices = []
        if not self.magazine or now_s - self.last_shot_time < self._next_shot_gap_s:
            return None
        position, orientation = self.chassis_pose()
        rotation = quat_wxyz_to_matrix(orientation)
        setpoint = solution["setpoint"]
        linear, _ = self.chassis_velocity()
        velocity = self.control.launch_velocity_world(setpoint, linear[:2])
        direction = velocity / max(1e-6, float(np.linalg.norm(velocity)))
        # Spawn just beyond the physical rear gate (roughly one FUEL radius
        # past the CAD muzzle).  At steep close-range pitches the old 90 mm
        # offset left the sphere's trailing edge touching the gate.
        muzzle_center = position + rotation @ MUZZLE_LOCAL + direction * 0.14
        lateral = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        raw, _ = fuel_view.get_world_poses()
        balls = self._np(raw).astype(np.float32)
        local = (balls[self.magazine] - position) @ rotation
        queued_ready = [
            index for lane, index in enumerate(self.feeder_queue)
            if index in self.magazine
            and float(np.linalg.norm(local[self.magazine.index(index)] - FEEDER_READY_LOCAL[lane])) < 0.11
        ]
        feeder_ready = (
            queued_ready
            if self.feeder_queue
            else [
                index for index, p in zip(self.magazine, local)
                if float(p[0]) <= FEEDER_MAX_X_LOCAL
            ]
        )
        max_count = min(int(self.barrels), len(self.magazine))
        if self._volley_target_count is None:
            counts = list(range(1, max_count + 1))
            self._volley_target_count = int(
                self._shooter_rng.choices(
                    counts,
                    weights=SHOOTER_VOLLEY_WEIGHTS[:max_count],
                    k=1,
                )[0]
            )
        desired_count = min(self._volley_target_count, max_count)
        if len(feeder_ready) < desired_count:
            if self._batch_wait_started is None:
                self._batch_wait_started = now_s
                return None
            if now_s - self._batch_wait_started < SHOOTER_BATCH_WAIT_S:
                return None
        self._batch_wait_started = None
        self._volley_target_count = None
        if feeder_ready:
            chosen = feeder_ready[: min(desired_count, len(feeder_ready))]
        else:
            # The large roller catches the closest single ball first; later
            # contacts can naturally form 2- or 3-ball groups.
            chosen = [self.magazine[int(np.argmin(local[:, 0]))]]
        count = len(chosen)
        # Full-width drum release (see SHOOTER_ROW_HALF_WIDTH_M block).
        # Lateral spots: slack construction (sorted uniform cuts + mandatory
        # gaps) so every draw succeeds and the whole row stays reachable.
        # Longitudinal: lat-neighbours get a flight-path step (direction
        # randomly flipped) - together a guaranteed >0.152 m centre distance.
        rng = self._shooter_rng
        speed = float(np.linalg.norm(velocity))
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # stagger axis orthogonal to the row, so the lat/lon pair distance is
        # the true 3D spawn distance whatever the aim direction
        lon_axis = direction - float(direction @ lateral) * lateral
        lon_axis = lon_axis / max(1e-6, float(np.linalg.norm(lon_axis)))
        slack = max(
            0.0,
            2.0 * SHOOTER_ROW_HALF_WIDTH_M - (count - 1) * SHOOTER_LAT_GAP_M,
        )
        cuts = sorted(rng.uniform(0.0, slack) for _ in range(count))
        lats = [
            -SHOOTER_ROW_HALF_WIDTH_M + cut + k * SHOOTER_LAT_GAP_M
            for k, cut in enumerate(cuts)
        ]
        ladder = list(range(count))
        if rng.random() < 0.5:
            ladder.reverse()
        placements = [
            (lat, step * SHOOTER_LON_STEP_M + rng.uniform(0.0, SHOOTER_LON_RAND_M))
            for lat, step in zip(lats, ladder)
        ]
        rng.shuffle(placements)
        for index, (lat, lon) in zip(chosen, placements):
            self.magazine.remove(index)
            self.pen_reserved.discard(index)
            self.captured_indices.discard(index)
            if index in self.feeder_queue:
                self.feeder_queue.remove(index)
            # per-ball drum error: tiny independent yaw/pitch/speed deviation,
            # plus a convergence term that steers out half the lateral spawn
            # offset over the flight so edge-of-row FUEL still enters the hub
            yaw = math.tan(math.radians(
                rng.uniform(-SHOOTER_YAW_JITTER_DEG, SHOOTER_YAW_JITTER_DEG)))
            yaw -= (
                SHOOTER_CONVERGE_FRAC * float(lat)
                * math.cos(math.radians(setpoint.pitch_deg))
                / max(1.0, float(setpoint.distance_m))
            )
            pitch = math.tan(math.radians(
                rng.uniform(-SHOOTER_PITCH_JITTER_DEG, SHOOTER_PITCH_JITTER_DEG)))
            ball_direction = direction + lateral * yaw + up * pitch
            ball_direction = ball_direction / max(1e-6, float(np.linalg.norm(ball_direction)))
            ball_speed = speed * (
                1.0 + rng.uniform(-SHOOTER_SPEED_JITTER_FRAC, SHOOTER_SPEED_JITTER_FRAC)
            )
            self._release(
                fuel_view,
                index,
                muzzle_center + lateral * float(lat) + lon_axis * float(lon),
                ball_direction * ball_speed,
            )
            self._muzzle_watch.add(index)
            self.last_fired_indices.append(index)
        self.last_shot_time = now_s
        self._next_shot_gap_s = (
            count * SHOOTER_BALL_PERIOD_S * rng.uniform(*SHOOTER_CADENCE_JITTER)
        )
        self.shots_fired += count
        self.volleys_fired += 1
        return self.last_fired_indices[0] if self.last_fired_indices else None

    def update(
        self,
        fuel_view: Any,
        now_s: float,
        alliance: str | None = None,
        hub_active: bool = True,
        allow_drive: bool = True,
        fire_mode: str = "score",
    ) -> dict[str, Any]:
        ferry_mode = fire_mode == "ferry"
        if ferry_mode:
            solution = self.solve_ferry(alliance)
        else:
            solution = self.solve_auto_aim(alliance)
        # Ferry lands in a broad zone and can safely use a much quicker gate.
        # HUB shots retain tight yaw accuracy while allowing the translational
        # speed already compensated by RealTimeAimDrive.
        self.state_machine.yaw_tolerance_deg = (
            FERRY_AIM_TOLERANCE_DEG if ferry_mode else AUTO_ALIGN_TOLERANCE_DEG
        )
        self.state_machine.max_speed_mps = (
            FERRY_FIRE_MAX_SPEED_MPS if ferry_mode else SCORE_FIRE_MAX_SPEED_MPS
        )
        self.state_machine.max_yaw_rate_dps = (
            FERRY_FIRE_MAX_YAW_RATE_DPS if ferry_mode else FIRE_MAX_YAW_RATE_DPS
        )
        linear, yaw_rate = self.chassis_velocity()
        # REBUILT: shooting at an INACTIVE hub is always allowed - the FUEL
        # simply earns no points (the router gates scoring).  The hub state
        # therefore never blocks the shooter FSM; ``hub_active`` is kept for
        # signature compatibility.
        inputs = ShooterInputs(
            magazine_count=len(self.magazine),
            hub_active=True,
            shot_valid=bool(solution["valid"]),
            yaw_error_deg=float(solution["yaw_error_deg"]),
            chassis_speed_mps=float(np.linalg.norm(linear[:2])),
            yaw_rate_dps=math.degrees(abs(yaw_rate)),
            muzzle_clear=(now_s - self.last_shot_time) >= SHOOTER_COOLDOWN_S,
            blocked_reason=str(solution["blocked_reason"]),
        )
        status = self.state_machine.tick(inputs, now_s)
        if allow_drive:
            if status["state"] in (ShooterState.ACQUIRE_TARGET.value, ShooterState.TURNING.value):
                error = math.radians(inputs.yaw_error_deg)
                kp = FERRY_AIM_KP if ferry_mode else SCORE_AIM_KP
                omega_limit = (
                    FERRY_AIM_MAX_OMEGA_RAD_S
                    if ferry_mode
                    else SCORE_AIM_MAX_OMEGA_RAD_S
                )
                omega = kp * error - AIM_YAW_DAMPING * yaw_rate
                self.drive_swerve(
                    0.0, 0.0, float(np.clip(omega, -omega_limit, omega_limit))
                )
            elif status["state"] != ShooterState.IDLE.value:
                self.drive_swerve(0.0, 0.0, 0.0)
            else:
                # Releasing WASD must decelerate to zero; previously the last
                # wheel velocity command remained latched indefinitely.
                self.drive(0.0, 0.0, 0.0)
        fired = None
        if status["should_feed"]:
            fired = self.fire_setpoint(fuel_view, solution, now_s)
        status["fired_index"] = fired
        status["fired_indices"] = self.last_fired_indices[:]
        status["solution"] = solution
        return status

    def stats(self) -> dict[str, Any]:
        return {
            "robot": "Competition Robot",
            "alliance_lock": self.alliance_lock,
            "magazine": len(self.magazine),
            "intake_transit": len(self.intake_transit),
            "feeder_staged": len(self.feeder_queue),
            "retention_corrections": self.retention_corrections,
            "storage_position": self.storage_position,
            "storage_target": self.storage_target,
            "storage_top_m": self.storage_top,
            "storage_mode": "rectangle/deployed" if self.storage_position >= 0.82 else "square/compact",
            "intake_deployed": self.intake_deployed,
            "mechanism_transitions": self.mechanism_transitions,
            "balls_collected": self.balls_collected,
            "shots_fired": self.shots_fired,
            "volleys_fired": self.volleys_fired,
            "shooter_state": self.state_machine.state.value,
            "auto_aim": {k: v for k, v in self.last_aim_solution.items() if k != "setpoint"},
        }
