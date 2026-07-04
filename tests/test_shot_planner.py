"""Phase D: direct-shot solver, calibration versioning, and global planner.

Pure-Python coverage of the analytic layer. The PhysX shot calibration and the
drive-follow that confirms a planned path is drivable are deferred GPU steps.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from xrc_rebuilt.field_map import OccupancyGrid, candidate_firing_positions, plan_path
from xrc_rebuilt.shot_planner import (
    CALIBRATED_RANGE,
    BlockedReason,
    CalibrationKey,
    DriveAndShootPlan,
    FieldState,
    RobotState,
    ShotCalibration,
    ShotPlan,
    current_calibration_key,
    feasibility_map,
    plan_global_score,
    pursuit_command,
    solve_direct_shot,
)

# Isaac-validated accepted blue pose and its red mirror.
BLUE_POSE = (1.52, -5.55)
RED_POSE = (-1.52, 5.55)


@pytest.fixture(scope="module")
def field_state():
    fs = FieldState()
    fs.occupancy()  # build once for the module
    return fs


# ----------------------------- direct-shot solver -------------------------- #
def test_accepted_blue_pose_is_a_valid_shotplan(field_state):
    plan = solve_direct_shot(RobotState(*BLUE_POSE), "blue", field_state)
    assert isinstance(plan, ShotPlan)
    assert plan.valid and plan.hub == "blue"
    assert CALIBRATED_RANGE[0] <= plan.range_m <= CALIBRATED_RANGE[1]
    assert 0.0 <= plan.aim <= 1.0
    assert abs(plan.vertical_error_m) <= 0.025
    assert plan.calibrated is False  # no PhysX calibration loaded


def test_shotplan_has_all_required_fields_and_is_json_safe(field_state):
    plan = solve_direct_shot(RobotState(*BLUE_POSE), "blue", field_state)
    for f in ("hub", "aim", "speed_mps", "pitch_deg", "muzzle_pose", "exit_direction",
              "desired_yaw_rad", "flight_time_s", "range_m", "vertical_error_m",
              "clearance_margin_m", "uncertainty_m", "calibrated", "valid", "reason"):
        assert hasattr(plan, f)
    json.dumps(asdict(plan))  # must serialise


def test_central_side_is_blocked_unsafe_side(field_state):
    result = solve_direct_shot(RobotState(1.52, -2.5), "blue", field_state)
    assert isinstance(result, BlockedReason)
    assert result.reason == "unsafe_side"


def test_too_close_and_too_far_are_range_blocked(field_state):
    close = solve_direct_shot(RobotState(0.0, -4.3), "blue", field_state)
    far = solve_direct_shot(RobotState(0.0, -8.5), "blue", field_state)
    assert isinstance(close, BlockedReason) and close.reason == "under_range"
    assert isinstance(far, BlockedReason) and far.reason == "over_range"


def test_red_blue_mirror_consistency(field_state):
    blue = solve_direct_shot(RobotState(*BLUE_POSE), "blue", field_state)
    red = solve_direct_shot(RobotState(*RED_POSE), "red", field_state)
    assert isinstance(blue, ShotPlan) and isinstance(red, ShotPlan)
    assert abs(blue.range_m - red.range_m) < 0.02
    assert abs(blue.aim - red.aim) < 0.02


def test_unknown_hub_raises(field_state):
    with pytest.raises(ValueError):
        solve_direct_shot(RobotState(*BLUE_POSE), "green", field_state)


# ----------------------------- calibration --------------------------------- #
def test_calibration_key_is_stable():
    assert asdict(current_calibration_key()) == asdict(current_calibration_key())


def test_calibration_roundtrip_and_match(tmp_path):
    cal = ShotCalibration(key=current_calibration_key(), aim_offset_by_hub={"blue": 0.02})
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = ShotCalibration.load(path)
    assert loaded.matches(current_calibration_key())
    # a different scene hash must NOT match (never use a foreign LUT)
    bad = ShotCalibration(key=CalibrationKey(**{**asdict(current_calibration_key()), "field_hash": "deadbeef"}))
    assert not bad.matches(current_calibration_key())


def test_matching_calibration_is_applied(field_state):
    base = solve_direct_shot(RobotState(*BLUE_POSE), "blue", field_state)
    cal = ShotCalibration(
        key=current_calibration_key(),
        aim_offset_by_hub={"blue": 0.03},
        uncertainty_by_hub={"blue": 0.012},
    )
    tuned = solve_direct_shot(RobotState(*BLUE_POSE), "blue", field_state, cal)
    assert isinstance(tuned, ShotPlan)
    assert tuned.calibrated is True
    assert tuned.uncertainty_m == 0.012
    assert abs(tuned.aim - min(1.0, base.aim + 0.03)) < 1e-6


# ----------------------------- global planner ------------------------------ #
def test_direct_from_accepted_pose_needs_no_drive(field_state):
    plan = plan_global_score(RobotState(*BLUE_POSE), "blue", field_state)
    assert isinstance(plan, DriveAndShootPlan)
    assert plan.reason == "direct" and plan.valid
    assert len(plan.path) == 1
    assert plan.shot is not None and plan.shot.valid


def test_drive_to_shoot_path_is_collision_free(field_state):
    plan = plan_global_score(RobotState(-3.0, 0.0), "blue", field_state)
    assert plan.valid and plan.reason == "drive_to_shoot"
    assert len(plan.path) > 1
    # every waypoint is in free space (never routed through a structure)
    for x, y in plan.path:
        assert field_state.occupancy().is_free(x, y)
    # the firing pose has a valid direct shot
    assert plan.shot is not None and plan.shot.valid
    assert 0 <= plan.braking_from_index < len(plan.path)


# ----------------------------- field map ----------------------------------- #
def test_occupancy_center_free_walls_blocked(field_state):
    grid = field_state.occupancy()
    assert grid.is_free(0.0, 0.0)
    assert not grid.is_free(4.3, 0.0)  # +x perimeter wall
    assert 0.0 < grid.occupied_fraction() < 0.9


def test_plan_path_refuses_occupied_goal(field_state):
    grid = field_state.occupancy()
    assert plan_path(grid, (0.0, 0.0), (4.3, 0.0)) is None


import math


def test_pursuit_drives_forward_when_aligned():
    cmd = pursuit_command((0.0, 0.0, 0.0), [(0.0, 0.0), (2.0, 0.0)], 0.0)
    assert cmd["phase"] == "driving"
    assert cmd["forward"] > 0.0
    assert abs(cmd["turn"]) < 0.2


def test_pursuit_turns_in_place_when_goal_behind():
    cmd = pursuit_command((0.0, 0.0, 0.0), [(0.0, 0.0), (-2.0, 0.0)], 0.0)
    assert cmd["phase"] == "turning"
    assert cmd["forward"] == 0.0
    assert abs(cmd["turn"]) > 0.0


def test_pursuit_orients_then_arrives_at_goal():
    at_goal_wrong_yaw = pursuit_command((2.0, 0.0, 0.0), [(2.0, 0.0)], math.pi / 2)
    assert at_goal_wrong_yaw["phase"] == "orienting"
    at_goal_right_yaw = pursuit_command((2.0, 0.0, math.pi / 2), [(2.0, 0.0)], math.pi / 2)
    assert at_goal_right_yaw["phase"] == "arrived"
    assert at_goal_right_yaw["forward"] == 0.0 and at_goal_right_yaw["turn"] == 0.0


def test_feasibility_map_small_region_labels_both_hubs(field_state):
    result = feasibility_map(field_state, region=(1.0, 2.0, -6.0, -5.0), pos_step=0.5)
    assert result["cells"]
    for cell in result["cells"]:
        assert cell["blue"] in {"direct", "unsafe_side", "under_range", "over_range", "no_trajectory"}
        assert "red" in cell
