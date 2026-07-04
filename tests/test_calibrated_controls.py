import math

import numpy as np

from xrc_rebuilt.calibrated_controls import CalibratedRobotControl, wrap_angle


def test_known_cali4_points():
    control = CalibratedRobotControl()
    speed, pitch, motor, clamped = control.stationary_calibration(1.4)
    assert not clamped
    assert speed == 6.1978
    assert pitch == 77.0
    assert motor == 44.0
    speed, pitch, motor, _ = control.stationary_calibration(5.7)
    assert speed == 8.3662
    assert pitch == 56.31
    assert motor == 60.0


def test_table_interpolation_and_clamping():
    control = CalibratedRobotControl()
    speed, pitch, motor, clamped = control.stationary_calibration(2.345)
    assert abs(speed - 6.96121) < 1e-5
    assert abs(pitch - 73.13) < 1e-5
    assert abs(motor - 53.00253848) < 1e-4
    assert not clamped
    assert control.stationary_calibration(0.5)[3]
    assert control.stationary_calibration(6.2)[3]


def test_robot_shooter_faces_backward_from_chassis():
    control = CalibratedRobotControl()
    shot = control.solve_shot(np.array([0.0, 0.0]), np.zeros(2), np.array([2.0, 0.0]))
    assert abs(wrap_angle(shot.shot_yaw_rad)) < 1e-9
    assert abs(abs(shot.chassis_yaw_rad) - math.pi) < 1e-9


def test_motion_compensation_cancels_chassis_velocity():
    control = CalibratedRobotControl()
    chassis = np.array([0.3, -0.5])
    shot = control.solve_shot(
        np.array([0.0, 0.0]), chassis, np.array([3.0, 0.0]), compensate_motion=True
    )
    world = control.launch_velocity_world(shot, chassis)
    # The requested world horizontal vector remains directed at the HUB.
    assert abs(float(world[1])) < 1e-5
    assert float(world[0]) > 0.0
