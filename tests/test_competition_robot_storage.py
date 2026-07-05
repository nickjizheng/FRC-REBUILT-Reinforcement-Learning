from __future__ import annotations

import math

import numpy as np
import pytest

from xrc_rebuilt.competition_robot import (
    BLUE_TRENCH_ALLIANCE_EDGE_Y_M,
    BLUE_TRENCH_CLEAR_X_MAX_M,
    BLUE_TRENCH_CLEAR_X_MIN_M,
    BLUE_TRENCH_NEUTRAL_EDGE_Y_M,
    BLUE_TRENCH_START_TRANSLATION,
    BLUE_TRENCH_START_YAW_DEG,
    BUMPER_BOTTOM_M,
    BUMPER_BLUE_RGBA,
    BUMPER_CENTER_X_M,
    BUMPER_DEPTH_M,
    BUMPER_OUTER_HALF_M,
    BUMPER_TOP_M,
    CAMERA_ABLATION_NAMES,
    CAMERA_BASELINE_NAMES,
    CAMERA_HOUSING_COLLIDERS,
    CAMERA_RATE_HZ,
    CAMERA_RESOLUTION,
    CAMERA_RIG,
    DRIVER_ACCEL_LIMIT_MPS2,
    DRIVER_CONTROL_DT_S,
    INTAKE_CENTER_LOCAL,
    INTAKE_PATH_LOCAL,
    INTAKE_SIMULTANEOUS_LANES,
    CAD_VERTICAL_LOWER_M,
    FERRY_MAX_EXIT_SPEED_MPS,
    FERRY_AIM_TOLERANCE_DEG,
    FERRY_FIRE_MAX_SPEED_MPS,
    FERRY_FIRE_MAX_YAW_RATE_DPS,
    FERRY_MIN_EXIT_SPEED_MPS,
    FERRY_PITCH_DEG,
    FRONT_SIDE_POST_COLLIDERS,
    FEEDER_READY_LOCAL,
    FEEDER_TUNNEL_COLLIDERS,
    HOPPER_ROLLER_COLLIDERS,
    HOPPER_PRESSURE_CAPACITY,
    MUZZLE_LOCAL,
    ROBOT_ROOT_PATH,
    SHOOTER_CAGE_COLLIDERS,
    SHOOTER_LANES,
    SHOOTER_LANE_SPACING_M,
    SHOOTER_LON_RAND_M,
    SHOOTER_LON_STEP_M,
    SHOOTER_ROW_HALF_WIDTH_M,
    STORAGE_EXTENDED_POSITION,
    STORAGE_EXTENDED_TOP_M,
    STORAGE_LOWERED_POSITION,
    STORAGE_LOWERED_TOP_M,
    UPPER_SIDE_BOARD_COLLIDERS,
    XRC_PRELOAD_COUNT,
    CompetitionRobotController,
    camera_orientation_wxyz,
    _four_corner_soft_net_collision_blocks,
    _four_corner_soft_net_segments,
    _hopper_slots,
    _point_on_polyline,
    _storage_curve_segments,
    storage_top_m,
)


class FakeFuel:
    def __init__(self, positions: np.ndarray):
        self.positions = np.asarray(positions, dtype=np.float32).copy()
        self.count = len(self.positions)
        self.linear = np.zeros_like(self.positions)
        self.angular = np.zeros_like(self.positions)

    def get_world_poses(self):
        orientations = np.tile(np.array([1, 0, 0, 0], np.float32), (self.count, 1))
        return self.positions.copy(), orientations

    def set_world_poses(self, positions, indices):
        self.positions[np.asarray(indices, dtype=int)] = np.asarray(positions)

    def set_linear_velocities(self, values, indices):
        self.linear[np.asarray(indices, dtype=int)] = np.asarray(values)

    def set_angular_velocities(self, values, indices):
        self.angular[np.asarray(indices, dtype=int)] = np.asarray(values)


def stationary_controller() -> CompetitionRobotController:
    controller = CompetitionRobotController()
    controller.chassis_pose = lambda: (
        np.array([0.0, 0.0, 0.0], np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], np.float32),
    )
    controller.chassis_velocity = lambda: (np.zeros(3, np.float32), 0.0)
    return controller


def test_hopper_uses_only_xrc_starting_preloads_not_a_software_capacity():
    slots = _hopper_slots()
    assert XRC_PRELOAD_COUNT == 8
    assert len(slots) == XRC_PRELOAD_COUNT
    distances = np.linalg.norm(slots[:, None, :] - slots[None, :, :], axis=2)
    distances += np.eye(len(slots), dtype=np.float32) * 10.0
    assert float(distances.min()) >= 0.150 - 1e-5


def test_2026_continuous_bumper_is_uniform_and_external():
    assert BUMPER_BOTTOM_M == pytest.approx(0.065)
    assert BUMPER_TOP_M == pytest.approx(5.75 * 0.0254)
    assert BUMPER_DEPTH_M == pytest.approx(0.062)
    # The STEP metal chassis is +/-0.357 m laterally; the pad lies outside it.
    assert BUMPER_OUTER_HALF_M - BUMPER_DEPTH_M > 0.357
    assert BUMPER_BLUE_RGBA[2] > BUMPER_BLUE_RGBA[0] * 10


def test_compact_start_pose_fits_beneath_blue_trench_and_faces_out():
    x, y, z = BLUE_TRENCH_START_TRANSLATION
    assert BLUE_TRENCH_START_YAW_DEG == pytest.approx(90.0)
    # +90 degrees maps the intake/nose (local +X) toward world +Y / neutral.
    nose_y = y + BUMPER_CENTER_X_M + BUMPER_OUTER_HALF_M
    rear_y = y + BUMPER_CENTER_X_M - BUMPER_OUTER_HALF_M
    assert nose_y <= BLUE_TRENCH_NEUTRAL_EDGE_Y_M
    # The reference setup keeps the complete nose behind the overhead beam
    # instead of aligning to the lower trench base.
    assert nose_y >= BLUE_TRENCH_NEUTRAL_EDGE_Y_M - 0.55
    assert rear_y < BLUE_TRENCH_ALLIANCE_EDGE_Y_M
    assert x - BUMPER_OUTER_HALF_M > BLUE_TRENCH_CLEAR_X_MIN_M
    assert x + BUMPER_OUTER_HALF_M < BLUE_TRENCH_CLEAR_X_MAX_M
    left_clearance = x - BUMPER_OUTER_HALF_M - BLUE_TRENCH_CLEAR_X_MIN_M
    right_clearance = BLUE_TRENCH_CLEAR_X_MAX_M - (x + BUMPER_OUTER_HALF_M)
    assert left_clearance > 0.18
    assert right_clearance > 0.18
    assert z + STORAGE_LOWERED_TOP_M < 0.591


def test_onboard_camera_rig_is_trench_safe_and_bumper_protected():
    assert CAMERA_RESOLUTION == (640, 360)
    assert CAMERA_RATE_HZ == 10
    assert CAMERA_BASELINE_NAMES == ("intake", "shooter", "navigation")
    assert CAMERA_ABLATION_NAMES == ("intake", "shooter", "navigation")
    assert len(CAMERA_HOUSING_COLLIDERS) == 3
    for name, spec in CAMERA_RIG.items():
        center = np.asarray(spec["housing_center"], dtype=np.float64)
        half = np.asarray(spec["housing_half_extents"], dtype=np.float64)
        lower = center - half
        upper = center + half
        assert str(spec["parent_path"]).startswith(ROBOT_ROOT_PATH), name
        # Moving camera housings are specified in their link-local frames.
        assert np.all(half > 0.0), name
        assert np.all(upper > lower), name
        # Existing compact CAD, not a camera, remains the height constraint.
        assert float(spec["compact_top_m"]) < 0.533, name

    # The fixed navigation housing remains inside the bumper footprint.
    navigation = CAMERA_RIG["navigation"]
    center = np.asarray(navigation["housing_center"], dtype=np.float64)
    half = np.asarray(navigation["housing_half_extents"], dtype=np.float64)
    lower, upper = center - half, center + half
    assert upper[0] <= BUMPER_CENTER_X_M + BUMPER_OUTER_HALF_M
    assert lower[0] >= BUMPER_CENTER_X_M - BUMPER_OUTER_HALF_M
    assert upper[1] <= BUMPER_OUTER_HALF_M
    assert lower[1] >= -BUMPER_OUTER_HALF_M


def test_camera_quaternions_point_usd_minus_z_along_requested_direction():
    for spec in CAMERA_RIG.values():
        quat = camera_orientation_wxyz(spec["direction"])
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0)
        w, x, y, z = [float(value) for value in quat]
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        requested = np.asarray(spec["direction"], dtype=np.float64)
        requested /= np.linalg.norm(requested)
        actual = rotation @ np.asarray([0.0, 0.0, -1.0])
        assert actual == pytest.approx(requested, abs=1e-6)


def test_weidai_square_rectangle_motion_couples_net_and_intake():
    controller = stationary_controller()
    assert controller.intake_deployed


def test_storage_mode_sets_intake_once_but_manual_override_remains():
    controller = stationary_controller()
    controller.set_storage_extended(False)
    assert not controller.intake_on
    controller.set_storage_extended(True)
    assert controller.intake_on
    controller.intake_on = False
    controller.set_storage_extended(True)
    assert not controller.intake_on
    controller.set_storage_extended(False)
    assert not controller.intake_on
    assert storage_top_m(STORAGE_EXTENDED_POSITION) == pytest.approx(STORAGE_EXTENDED_TOP_M)
    controller.set_storage_extended(False)
    for _ in range(90):
        controller.step_mechanisms(1 / 30)
    assert controller.storage_position == pytest.approx(STORAGE_LOWERED_POSITION)
    assert controller.storage_top == pytest.approx(STORAGE_LOWERED_TOP_M)
    assert controller.storage_top < 22.25 * 0.0254
    assert not controller.intake_deployed
    controller.set_storage_extended(True)
    for _ in range(90):
        controller.step_mechanisms(1 / 30)
    assert controller.storage_position == pytest.approx(STORAGE_EXTENDED_POSITION)
    assert controller.intake_deployed


def test_compact_program_retracts_intake_before_lowering_containers():
    controller = stationary_controller()
    controller.set_storage_extended(False)
    controller.step_mechanisms(1 / 30)
    assert controller.mechanism_phase == "INTAKE_RETRACTING"
    assert controller.intake_extension < 1.0
    assert controller.container_extension == pytest.approx(1.0)
    while controller.intake_extension > 0.0:
        previous_container = controller.container_extension
        controller.step_mechanisms(1 / 30)
        assert controller.container_extension == pytest.approx(previous_container)
    controller.step_mechanisms(1 / 30)
    assert controller.mechanism_phase == "CONTAINER_CLOSING"
    assert controller.container_extension < 1.0
    for _ in range(60):
        controller.step_mechanisms(1 / 30)
    assert controller.mechanism_phase == "COMPACT"
    assert controller.intake_extension == pytest.approx(0.0)
    assert controller.container_extension == pytest.approx(0.0)
    assert controller.storage_top < 22.25 * 0.0254

    controller.set_storage_extended(True)
    controller.step_mechanisms(1 / 30)
    assert controller.mechanism_phase == "CONTAINER_OPENING"
    assert controller.intake_extension == pytest.approx(0.0)


def test_reference_four_bar_raises_without_uniformly_scaling_the_frame():
    compact_side, _, compact_links, compact_t = _storage_curve_segments(
        STORAGE_LOWERED_POSITION
    )
    deployed_side, _, deployed_links, deployed_t = _storage_curve_segments(
        STORAGE_EXTENDED_POSITION
    )
    assert compact_t == pytest.approx(0.0)
    assert deployed_t == pytest.approx(1.0)
    # Same curve topology, but actual endpoints articulate independently.
    assert len(compact_side) == len(deployed_side)
    assert len(compact_links) == len(deployed_links) == 8
    compact_upper_z = compact_links[0][1][2]
    deployed_upper_z = deployed_links[0][1][2]
    assert deployed_upper_z - compact_upper_z > 0.30
    # Paired upper pivots remain close in height like a parallelogram carriage.
    assert abs(deployed_links[0][1][2] - deployed_links[1][1][2]) < 0.05


def test_reference_soft_net_is_tied_at_four_corners_and_sags_between_them():
    anchors = np.asarray(
        [
            [0.1892, -0.317, 0.727], [0.1892, 0.317, 0.727],
            [0.50, -0.35, 0.528], [0.50, 0.35, 0.528],
        ],
        dtype=np.float32,
    )
    segments, returned_anchors = _four_corner_soft_net_segments(anchors)
    assert returned_anchors == pytest.approx(anchors)
    points = np.asarray([point for segment in segments for point in segment])
    for anchor in anchors:
        assert np.min(np.linalg.norm(points - anchor, axis=1)) < 1e-6
    # At the middle, the straight corner-to-corner plane is z=0.6275 m.  The
    # rope net must hang substantially below it, and even a free edge must sag.
    center = points[np.argmin(np.linalg.norm(points[:, :2] - [0.3446, 0.0], axis=1))]
    rear_edge = points[np.argmin(np.linalg.norm(points[:, :2] - [0.1892, 0.0], axis=1))]
    assert center[2] < 0.50
    assert rear_edge[2] < 0.70


def test_reference_net_has_a_watertight_solid_collision_skin():
    anchors = np.asarray(
        [
            [0.1892, -0.317, 0.727], [0.1892, 0.317, 0.727],
            [0.50, -0.35, 0.528], [0.50, 0.35, 0.528],
        ],
        dtype=np.float32,
    )
    blocks = _four_corner_soft_net_collision_blocks(anchors, resolution=6)
    assert len(blocks) == 36
    assert all(block.shape == (12, 3, 3) for block in blocks)
    # Every tile is a closed prism, not an infinitely thin visual triangle.
    for block in blocks:
        vertices = np.unique(np.round(block.reshape(-1, 3), 6), axis=0)
        assert len(vertices) == 8
        assert np.ptp(vertices[:, 2]) >= 0.018 - 1e-6


def test_polyline_transport_does_not_jump_to_destination():
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], np.float32)
    point, done = _point_on_polyline(path, 0.25)
    assert not done
    assert point == pytest.approx([0.25, 0.0, 0.0])
    point, done = _point_on_polyline(path, 2.5)
    assert done
    assert point == pytest.approx([1.0, 1.0, 0.0])


def test_intake_visibly_conveys_one_ball_into_storage():
    controller = stationary_controller()
    controller.intake_on = True
    fuel = FakeFuel(np.array([INTAKE_CENTER_LOCAL], np.float32))
    assert controller.step_intake(fuel, set(), dt_s=1 / 30) == 0
    assert list(controller.intake_transit) == [0]
    original = fuel.positions[0].copy()
    controller.step_intake(fuel, set(), dt_s=1 / 30)
    fuel.positions += fuel.linear * (1 / 30)
    assert np.linalg.norm(fuel.positions[0] - original) > 0.02
    for _ in range(30):
        controller.step_intake(fuel, set(), dt_s=1 / 30)
        fuel.positions += fuel.linear * (1 / 30)
    assert controller.magazine == [0]
    assert not controller.intake_transit


def test_horizontal_intake_accepts_three_balls_across_full_roller_at_once():
    controller = stationary_controller()
    controller.intake_on = True
    positions = np.array(
        [
            INTAKE_CENTER_LOCAL + [0.00, -0.25, 0.00],
            INTAKE_CENTER_LOCAL + [0.02, 0.00, 0.00],
            INTAKE_CENTER_LOCAL + [-0.02, 0.25, 0.00],
            INTAKE_CENTER_LOCAL + [0.04, 0.10, 0.00],
        ],
        np.float32,
    )
    fuel = FakeFuel(positions)
    controller.step_intake(fuel, set(), dt_s=1 / 30)
    assert INTAKE_SIMULTANEOUS_LANES == 3
    assert len(controller.intake_transit) == 3
    assert len(controller.intake_lane_y) == 3
    assert min(controller.intake_lane_y.values()) < -0.20
    assert max(controller.intake_lane_y.values()) > 0.20


def test_three_balls_stage_in_rigid_feeder_without_firing():
    controller = stationary_controller()
    fuel = FakeFuel(np.array(_hopper_slots()[:5], np.float32))
    controller.magazine = list(range(5))
    controller.captured_indices = set(controller.magazine)
    controller.pen_reserved = set(controller.magazine)
    for _ in range(35):
        controller._step_feeder(fuel, 1 / 30)
    assert set(controller.feeder_queue) == set(range(5))
    assert fuel.positions[controller.feeder_queue] == pytest.approx(
        FEEDER_READY_LOCAL[:5], abs=0.02
    )
    assert controller.shots_fired == 0


def test_feeder_holds_eight_balls_as_two_rows_of_four_inside_turret():
    controller = stationary_controller()
    fuel = FakeFuel(np.array(_hopper_slots(), np.float32))
    controller.magazine = list(range(8))
    controller.captured_indices = set(controller.magazine)
    controller.pen_reserved = set(controller.magazine)
    for _ in range(45):
        controller._step_feeder(fuel, 1 / 30)
    assert set(controller.feeder_queue) == set(range(8))
    assert fuel.positions[controller.feeder_queue] == pytest.approx(
        FEEDER_READY_LOCAL, abs=0.02
    )
    assert np.unique(np.round(FEEDER_READY_LOCAL[:, 2], 3)) == pytest.approx(
        [0.215, 0.371]
    )
    for height in (0.215, 0.371):
        row = FEEDER_READY_LOCAL[np.isclose(FEEDER_READY_LOCAL[:, 2], height)]
        assert len(row) == 4
    assert HOPPER_PRESSURE_CAPACITY > len(controller.feeder_queue)


def test_full_hopper_stalls_intake_instead_of_pressurising_body():
    controller = stationary_controller()
    controller.intake_on = True
    positions = np.tile(INTAKE_CENTER_LOCAL, (HOPPER_PRESSURE_CAPACITY + 1, 1))
    fuel = FakeFuel(positions)
    controller.captured_indices = set(range(HOPPER_PRESSURE_CAPACITY))
    controller.pen_reserved = set(controller.captured_indices)
    controller.step_intake(fuel, set(), dt_s=1 / 30)
    assert not controller.intake_transit


def test_blocked_intake_release_waits_at_nip():
    controller = stationary_controller()
    release = np.array([0.42, 0.0, 0.32], np.float32)
    fuel = FakeFuel(np.array([release, release + [0.05, 0.0, 0.0]], np.float32))
    final_waypoint = len(INTAKE_PATH_LOCAL) - 1
    controller.intake_transit[0] = final_waypoint
    controller.intake_lane_y[0] = 0.0
    controller._sync_intake_transit(fuel, 1 / 30)
    assert controller.intake_transit[0] == final_waypoint
    assert 0 not in controller.captured_indices


def test_mechanism_collision_proxies_are_complete_and_trench_safe():
    assert len(HOPPER_ROLLER_COLLIDERS) == 5
    assert len(UPPER_SIDE_BOARD_COLLIDERS) == 2
    assert len(FRONT_SIDE_POST_COLLIDERS) == 2
    assert len(FEEDER_TUNNEL_COLLIDERS) == 5
    assert len(SHOOTER_CAGE_COLLIDERS) == 3
    compact_side_top = max(
        center[2] + half[2] - CAD_VERTICAL_LOWER_M
        for half, center in UPPER_SIDE_BOARD_COLLIDERS
    )
    front_post_top = max(
        center[2] + half[2] for half, center in FRONT_SIDE_POST_COLLIDERS
    )
    assert compact_side_top < 22.25 * 0.0254
    assert front_post_top < 22.25 * 0.0254


def test_robot_large_roller_releases_at_most_three_abreast():
    controller = stationary_controller()
    fuel = FakeFuel(np.zeros((8, 3), np.float32))
    controller.magazine = list(range(8))
    controller.pen_reserved = set(controller.magazine)
    solution = {
        "setpoint": controller.control.solve_shot(
            robot_xy=np.array([1.52, -5.55]),
            robot_velocity_xy=np.zeros(2),
            hub_xy=np.array([-0.0199, -3.6874]),
        )
    }
    first = controller.fire_setpoint(fuel, solution, now_s=10.0)
    assert first == 0
    count = len(controller.last_fired_indices)
    assert 1 <= count <= SHOOTER_LANES
    assert controller.last_fired_indices == list(range(count))
    assert controller.magazine == list(range(count, 8))
    assert controller.shots_fired == count
    assert controller.volleys_fired == 1
    # Full-width drum release: spawns stay inside the row, never behind the
    # muzzle gate, and can never interpenetrate (FUEL diameter 0.152 m).
    spawned = fuel.positions[:count].astype(np.float64)
    velocity = controller.control.launch_velocity_world(
        solution["setpoint"], np.zeros(2)
    )
    direction = velocity / np.linalg.norm(velocity)
    muzzle_center = MUZZLE_LOCAL.astype(np.float64) + direction * 0.14
    lateral = np.array([0.0, 1.0, 0.0])
    lon_axis = direction - float(direction @ lateral) * lateral
    lon_axis = lon_axis / np.linalg.norm(lon_axis)
    axes = np.stack([lateral, lon_axis], axis=1)
    coords, *_ = np.linalg.lstsq(axes, (spawned - muzzle_center).T, rcond=None)
    lats, lons = coords[0], coords[1]
    assert np.all(np.abs(lats) <= SHOOTER_ROW_HALF_WIDTH_M + 1e-5)
    assert np.all(lons >= -1e-5)
    assert np.all(lons <= (count - 1) * SHOOTER_LON_STEP_M + SHOOTER_LON_RAND_M + 1e-5)
    for i in range(count):
        for j in range(i + 1, count):
            assert float(np.linalg.norm(spawned[i] - spawned[j])) >= 0.152


def test_robot_large_roller_regroups_later_volleys():
    controller = stationary_controller()
    fuel = FakeFuel(FEEDER_READY_LOCAL.copy())
    controller.magazine = list(range(8))
    controller.captured_indices = set(controller.magazine)
    controller.pen_reserved = set(controller.magazine)
    solution = {
        "setpoint": controller.control.solve_shot(
            robot_xy=np.array([1.52, -5.55]),
            robot_velocity_xy=np.zeros(2),
            hub_xy=np.array([-0.0199, -3.6874]),
        )
    }
    volley_sizes = []
    for attempt in range(8):
        if not controller.magazine:
            break
        for _ in range(30):
            controller._step_feeder(fuel, 1 / 30)
        controller.fire_setpoint(fuel, solution, now_s=10.0 + 0.3 * attempt)
        volley_sizes.append(len(controller.last_fired_indices))
    assert sum(volley_sizes) == 8
    assert all(1 <= size <= 3 for size in volley_sizes)
    assert 3 in volley_sizes
    assert len(set(volley_sizes)) > 1


def _posed_controller(x: float, y: float) -> CompetitionRobotController:
    controller = stationary_controller()
    controller.chassis_pose = lambda: (
        np.array([x, y, 0.0], np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], np.float32),
    )
    controller.chassis_yaw = lambda: 0.0
    return controller


def test_ferry_refused_inside_own_court():
    controller = _posed_controller(2.6, -4.5)  # blue robot deep in its own court
    solution = controller.solve_ferry("blue")
    assert not solution["valid"]
    assert "own court" in solution["blocked_reason"]


def test_ferry_blocked_over_the_hub_from_mid_field():
    controller = _posed_controller(0.0, 5.0)  # blue robot centered in red zone
    solution = controller.solve_ferry("blue")
    assert not solution["valid"]
    assert "HUB" in solution["blocked_reason"]


def test_ferry_side_lane_returns_fuel_to_own_half():
    controller = _posed_controller(2.6, 0.5)  # neutral zone, side corridor
    solution = controller.solve_ferry("blue")
    assert solution["valid"]
    setpoint = solution["setpoint"]
    assert setpoint.pitch_deg == pytest.approx(FERRY_PITCH_DEG)
    assert (
        FERRY_MIN_EXIT_SPEED_MPS
        <= setpoint.exit_speed_mps
        <= FERRY_MAX_EXIT_SPEED_MPS
    )
    _, target_y = solution["target_xy"]
    assert target_y < -2.7  # lands in the blue half
    # analytic landing check: the solved speed drops the lob exactly from the
    # muzzle height to the floor over the solved range
    theta = math.radians(FERRY_PITCH_DEG)
    horizontal_range = solution["distance_m"]
    drop = horizontal_range * math.tan(theta) - 9.81 * horizontal_range**2 / (
        2.0 * setpoint.exit_speed_mps**2 * math.cos(theta) ** 2
    )
    assert drop == pytest.approx(0.076 - 0.60, abs=1e-6)


def test_ferry_uses_fast_two_degree_alignment_gate():
    controller = _posed_controller(2.6, 0.5)
    desired = controller.solve_ferry("blue")["desired_yaw_rad"]
    controller.chassis_yaw = lambda: desired + math.radians(1.0)
    solution = controller.solve_ferry("blue")
    assert FERRY_AIM_TOLERANCE_DEG == pytest.approx(2.0)
    assert solution["aligned"]

    # The runtime FSM receives the same relaxed moving-shot gates.
    controller.update(
        FakeFuel(np.zeros((1, 3), np.float32)),
        now_s=1.0,
        alliance="blue",
        allow_drive=False,
        fire_mode="ferry",
    )
    assert controller.state_machine.yaw_tolerance_deg == pytest.approx(2.0)
    assert controller.state_machine.max_speed_mps == pytest.approx(
        FERRY_FIRE_MAX_SPEED_MPS
    )
    assert controller.state_machine.max_yaw_rate_dps == pytest.approx(
        FERRY_FIRE_MAX_YAW_RATE_DPS
    )


def test_ferry_reaches_back_from_the_opponent_corner():
    controller = _posed_controller(2.6, 6.5)  # deep red corner, blue alliance
    solution = controller.solve_ferry("blue")
    assert solution["valid"]
    assert solution["speed_mps"] <= FERRY_MAX_EXIT_SPEED_MPS + 1e-6
    _, target_y = solution["target_xy"]
    assert target_y < -2.0  # the big parabola carries FUEL back to blue


def test_blue_alliance_lock_rejects_every_red_target_request():
    controller = CompetitionRobotController(alliance_lock="blue")
    controller.chassis_pose = lambda: (
        np.array([0.0, 0.0, 0.0], np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], np.float32),
    )
    controller.chassis_velocity = lambda: (np.zeros(3, np.float32), 0.0)
    controller.chassis_yaw = lambda: 0.0
    assert controller.solve_auto_aim("red")["alliance"] == "blue"
    assert controller.solve_ferry("red")["alliance"] == "blue"
    with pytest.raises(ValueError):
        CompetitionRobotController(alliance_lock="green")


def test_keyboard_drive_slews_and_release_commands_zero():
    controller = stationary_controller()
    commands = []
    controller.chassis_yaw = lambda: 0.0
    controller.drive_swerve = lambda vx, vy, omega: commands.append((vx, vy, omega))
    controller.drive(1.0, 1.0, 0.0)
    assert commands[-1][0] == pytest.approx(DRIVER_ACCEL_LIMIT_MPS2 * DRIVER_CONTROL_DT_S)
    assert 0.0 < commands[-1][2] < 0.3
    controller.drive(0.0, 0.0, 0.0)
    assert commands[-1] == pytest.approx((0.0, 0.0, 0.0))


def test_rigid_storage_gate_needs_no_scripted_inward_velocity():
    controller = stationary_controller()
    fuel = FakeFuel(np.array([[0.44, 0.0, 0.40]], np.float32))
    controller.captured_indices = {0}
    before = fuel.positions.copy()
    controller.sync_magazine(fuel)
    assert controller.magazine == [0]
    assert fuel.positions == pytest.approx(before)
    assert fuel.linear[0] == pytest.approx([0.0, 0.0, 0.0])
    assert controller.retention_corrections == 0
