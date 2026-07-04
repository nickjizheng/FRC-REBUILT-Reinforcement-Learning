"""Phase F: match-aware autonomous target selection + active-HUB gating."""
from __future__ import annotations

import pytest

from xrc_rebuilt.robot_model import ShooterInputs, ShooterState, ShooterStateMachine
from xrc_rebuilt.shot_planner import (
    FieldState,
    MatchState,
    RobotState,
    select_target_hub,
)

BLUE_POSE = (1.52, -5.55)
RED_POSE = (-1.52, 5.55)


@pytest.fixture(scope="module")
def field_state():
    fs = FieldState()
    fs.occupancy()
    return fs


# ------------------------------- match state ------------------------------- #
def test_both_hubs_active_in_auto():
    assert set(MatchState(10.0).active_hubs()) == {"red", "blue"}


def test_one_hub_inactive_during_shift():
    # SHIFT 1 (30-55 s); first inactive = blue -> blue HUB off in shift 1
    ms = MatchState(40.0, first_inactive="blue")
    assert ms.active_hubs() == ["red"]
    assert ms.hub_active("blue") is False
    assert ms.hub_active("red") is True


def test_shift_alternates_by_shift_index():
    # blue inactive in shifts 1 & 3, red inactive in shifts 2 & 4
    assert MatchState(40.0, first_inactive="blue").active_hubs() == ["red"]   # shift 1
    assert MatchState(65.0, first_inactive="blue").active_hubs() == ["blue"]  # shift 2
    assert MatchState(90.0, first_inactive="blue").active_hubs() == ["red"]   # shift 3


def test_both_active_in_endgame_none_post_match():
    assert set(MatchState(140.0).active_hubs()) == {"red", "blue"}  # endgame
    assert MatchState(165.0).active_hubs() == []                    # post match


def test_shift_without_determination_defaults_both_active():
    assert set(MatchState(40.0, first_inactive=None).active_hubs()) == {"red", "blue"}


# ---------------------------- target selection ----------------------------- #
def test_selects_blue_by_direct_shot_in_auto(field_state):
    hub, reason = select_target_hub(RobotState(*BLUE_POSE), MatchState(10.0), field_state)
    assert hub == "blue" and reason == "direct"


def test_selects_red_from_red_side(field_state):
    hub, reason = select_target_hub(RobotState(*RED_POSE), MatchState(10.0), field_state)
    assert hub == "red" and reason == "direct"


def test_keeps_valid_preferred_hub(field_state):
    # from a central-ish spot both may screen; preference is honoured if valid
    hub, _ = select_target_hub(
        RobotState(*BLUE_POSE), MatchState(10.0), field_state, preferred="blue"
    )
    assert hub == "blue"


def test_no_active_hub_post_match(field_state):
    hub, reason = select_target_hub(RobotState(*BLUE_POSE), MatchState(165.0), field_state)
    assert hub is None and reason == "no active HUB"


def test_does_not_select_deactivated_blue(field_state):
    # blue HUB is off in shift 1; selection must not return blue
    hub, _ = select_target_hub(
        RobotState(*BLUE_POSE), MatchState(40.0, first_inactive="blue"), field_state
    )
    assert hub != "blue"


# ------------------- continuous fire across a phase boundary ---------------- #
def test_continuous_fire_stops_when_selected_hub_deactivates():
    sm = ShooterStateMachine()
    sm.set_continuous(True)

    def inputs(active: bool) -> ShooterInputs:
        return ShooterInputs(
            magazine_count=8, hub_active=active, shot_valid=True,
            yaw_error_deg=0.0, chassis_speed_mps=0.0, yaw_rate_dps=0.0,
            muzzle_clear=True,
        )

    fed_active = any(
        sm.tick(inputs(True), i * 0.1)["should_feed"] for i in range(5)
    )
    status = sm.tick(inputs(False), 0.6)  # HUB deactivates
    assert fed_active is True
    assert status["should_feed"] is False
    assert status["state"] == ShooterState.BLOCKED.value
    assert status["blocked_reason"] == "target HUB inactive"
