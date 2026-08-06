from __future__ import annotations

import argparse

import pytest

from frc_rebuilt.isaac_scene import (
    _parse_gui_camera_views,
    _parse_gui_intake_substeps,
    _parse_gui_render_hz,
    _sleep_until_realtime,
    _step_gui_intake,
)


class _IntakeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, set[int], float]] = []

    def step_intake(self, fuel_view, hub_pending, dt_s: float) -> int:
        self.calls.append((fuel_view, hub_pending, dt_s))
        return len(self.calls)


def test_gui_intake_substeps_repeat_the_unchanged_controller_path():
    controller = _IntakeRecorder()
    fuel_view = object()
    pending = {3, 7}

    completed = _step_gui_intake(
        controller,
        fuel_view,
        pending,
        dt_s=1 / 30,
        substeps=2,
    )

    assert completed == 3
    assert controller.calls == [
        (fuel_view, pending, 1 / 30),
        (fuel_view, pending, 1 / 30),
    ]


@pytest.mark.parametrize("value", ["0", "4", "-1"])
def test_gui_intake_substeps_reject_unsafe_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_gui_intake_substeps(value)


def test_gui_intake_substeps_accept_training_parity_and_recommended_boost():
    assert _parse_gui_intake_substeps("1") == 1
    assert _parse_gui_intake_substeps("2") == 2


@pytest.mark.parametrize("value", ["-1", "4"])
def test_gui_camera_view_count_rejects_out_of_range_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_gui_camera_views(value)


def test_gui_camera_view_count_accepts_smooth_and_complete_presets():
    assert _parse_gui_camera_views("1") == 1
    assert _parse_gui_camera_views("3") == 3


def test_gui_render_rate_accepts_supported_values():
    assert _parse_gui_render_hz("30") == 30
    assert _parse_gui_render_hz("60") == 60
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_gui_render_hz("24")


def test_realtime_pacer_uses_an_absolute_deadline():
    clock = [9.990]
    sleeps: list[float] = []

    def now() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += max(seconds, 0.00025)

    _sleep_until_realtime(10.0, now_fn=now, sleep_fn=sleep)

    assert clock[0] >= 10.0
    assert sleeps[0] == pytest.approx(0.00925)
