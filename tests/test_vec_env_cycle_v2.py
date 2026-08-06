"""CPU-only tests for the opt-in Stage C v2 environment helpers."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from frc_rebuilt.rl.cycle_v2 import (
    CyclePhase,
    FieldRegion,
    Milestone,
    MilestoneEvent,
)
from frc_rebuilt.rl.vec_env import (
    EnvSlot,
    VecCompetitionEnv,
    VecEnvCfg,
    _effective_fire_mode,
)


class _Fuel:
    def __init__(self, count: int = 200):
        self.count = count
        self.pose_calls = []
        self.linear_calls = []
        self.angular_calls = []

    def set_world_poses(self, **kwargs):
        self.pose_calls.append(kwargs)

    def set_linear_velocities(self, values, indices=None):
        self.linear_calls.append((np.asarray(values), np.asarray(indices)))

    def set_angular_velocities(self, values, indices=None):
        self.angular_calls.append((np.asarray(values), np.asarray(indices)))


def _env(**overrides):
    cfg = VecEnvCfg(stagec_v2=True, **overrides)
    env = object.__new__(VecCompetitionEnv)
    env.cfg = cfg
    env.rng = np.random.default_rng(14)
    env._fuel_home = np.zeros((200, 3), np.float32)
    env._fuel_home[:, 2] = 0.076
    return env


def test_large_full_cycle_target_does_not_inflate_return_micro_skill_reserve():
    env = object.__new__(VecCompetitionEnv)
    env.cfg = VecEnvCfg(
        stagec_v2=True,
        num_envs=1,
        cycle_v2_target_load=45,
        cycle_v2_reserve_count=18,
    )

    env._validate_cycle_v2_cfg()

    env.cfg.cycle_v2_reserve_count = 7
    with pytest.raises(ValueError, match="physical return-skill preload"):
        env._validate_cycle_v2_cfg()


def test_complete_cycle_curriculum_requires_target_and_ramp_gates():
    env = _env(
        cycle_v2_postdump_complete_cycle=True,
        cycle_v2_postdump_require_target_load=True,
        cycle_v2_require_ramp_out=False,
    )
    with pytest.raises(ValueError, match="target-load and ramp-out"):
        env._validate_cycle_v2_cfg()


def _slot(fuel):
    return EnvSlot(
        index=0,
        origin=np.zeros(3, np.float32),
        controller=SimpleNamespace(magazine=[]),
        router=SimpleNamespace(pending=set()),
        articulation=None,
        fuel=fuel,
    )


def test_v2_config_is_opt_in_and_rejects_mixed_contracts():
    legacy = object.__new__(VecCompetitionEnv)
    legacy.cfg = VecEnvCfg(stagec_v2=False, cycle_v2_reset_modes=("nonsense",))
    legacy._validate_cycle_v2_cfg()  # legacy behavior does not inspect v2 fields

    env = _env(cycle_v2_reset_modes=("full", "collect"), num_envs=2)
    env._validate_cycle_v2_cfg()
    env.cfg.cycle_v2_reset_modes = ("full", "bad")
    with pytest.raises(ValueError, match="unknown"):
        env._validate_cycle_v2_cfg()

    mixed = _env(neutral_refill_count=12, neutral_refill_prob=0.5)
    with pytest.raises(ValueError, match="cannot be mixed"):
        mixed._validate_cycle_v2_cfg()

    bad_fraction = _env(cycle_v2_score_fraction=0.0)
    with pytest.raises(ValueError, match="score_fraction"):
        bad_fraction._validate_cycle_v2_cfg()

    bad_outer_geometry = _env(
        cycle_v2_outer_rail_exit_x=2.9,
        cycle_v2_outer_rail_enter_x=2.8,
    )
    with pytest.raises(ValueError, match="outer-rail geometry"):
        bad_outer_geometry._validate_cycle_v2_cfg()

    bad_intake = _env(cycle_v2_intake_substeps=4)
    with pytest.raises(ValueError, match="intake_substeps"):
        bad_intake._validate_cycle_v2_cfg()

    bad_ramp_width = _env(cycle_v2_ramp_out_half_width=0.0)
    with pytest.raises(ValueError, match="ramp_out_half_width"):
        bad_ramp_width._validate_cycle_v2_cfg()

    bad_preferred_load = _env(
        cycle_v2_target_load=15,
        cycle_v2_preferred_repeat_load=15,
    )
    with pytest.raises(ValueError, match="preferred_repeat_load"):
        bad_preferred_load._validate_cycle_v2_cfg()

    missing_v9_fallback = _env(
        cycle_v2_target_load=15,
        cycle_v2_preferred_repeat_load=20,
        cycle_v2_collect_until_preferred=True,
    )
    with pytest.raises(ValueError, match="collection-stall"):
        missing_v9_fallback._validate_cycle_v2_cfg()

    # V8 used the preferred load only to scale a soft reward. Replaying that
    # historical contract must not require or invent V9 transition fallbacks.
    historical_v8 = _env(
        cycle_v2_target_load=15,
        cycle_v2_preferred_repeat_load=30,
        cycle_v2_repeat_load_return_bonus=8.0,
        cycle_v2_repeat_load_score_bonus=12.0,
    )
    historical_v8._validate_cycle_v2_cfg()

    # V10 keeps the 15-ball state transition intact: the preferred load is
    # shaping only, with intake forced during RETURN rather than by extending
    # COLLECT. No V9 stall/clock fallback is needed.
    v10_return_intake = _env(
        cycle_v2_target_load=15,
        cycle_v2_preferred_repeat_load=20,
        cycle_v2_collect_until_preferred=False,
        cycle_v2_collect_stall_steps=0,
        cycle_v2_return_time_guard=0.0,
        cycle_v2_intake_during_return=True,
    )
    v10_return_intake._validate_cycle_v2_cfg()

    mixed_transition_modes = _env(
        cycle_v2_target_load=15,
        cycle_v2_preferred_repeat_load=20,
        cycle_v2_collect_until_preferred=True,
        cycle_v2_collect_stall_steps=10,
        cycle_v2_intake_during_return=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        mixed_transition_modes._validate_cycle_v2_cfg()


def test_active_score_dump_ignores_later_ferry_proposal():
    assert _effective_fire_mode(
        "ferry", dumping=True, dump_mode="score"
    ) == "score"
    assert _effective_fire_mode(
        "ferry", dumping=False, dump_mode=None
    ) == "ferry"


def test_shoot_delay_penalty_has_two_second_grace_and_stops_on_dump_start():
    env = _env(
        cycle_v2_shoot_grace_steps=20,
        cycle_v2_shoot_penalty_per_step=0.05,
        cycle_v2_shoot_penalty_cap=0.10,
    )
    slot = _slot(_Fuel())
    slot.controller.magazine = list(range(15))
    slot.cycle_v2_penalty_spent = {"leave": 0.0, "return": 0.0, "score": 0.0}
    slot.dumping = False
    slot.dump_started_this_step = False

    def score_step(elapsed):
        return SimpleNamespace(
            phase=CyclePhase.SCORE,
            region=FieldRegion.HOME,
            qualified_load=15,
            phase_elapsed_steps=elapsed,
        )

    assert env._cycle_v2_delay_penalty(slot, score_step(20)) == 0.0
    assert env._cycle_v2_delay_penalty(slot, score_step(21)) == pytest.approx(-0.05)
    assert env._cycle_v2_delay_penalty(slot, score_step(22)) == pytest.approx(-0.05)
    assert env._cycle_v2_delay_penalty(slot, score_step(23)) == 0.0

    slot.cycle_v2_penalty_spent["score"] = 0.0
    slot.dump_started_this_step = True
    assert env._cycle_v2_delay_penalty(slot, score_step(100)) == 0.0


def test_leave_and_return_delay_cost_only_the_wrong_region_state():
    env = _env(
        cycle_v2_leave_grace_steps=0,
        cycle_v2_return_grace_steps=0,
    )
    slot = _slot(_Fuel())
    slot.cycle_v2_penalty_spent = {"leave": 0.0, "return": 0.0, "score": 0.0}
    slot.dumping = False
    slot.dump_started_this_step = False

    leave = SimpleNamespace(
        phase=CyclePhase.LEAVE,
        region=FieldRegion.HOME,
        qualified_load=0,
        phase_elapsed_steps=1,
    )
    assert env._cycle_v2_delay_penalty(slot, leave) < 0.0
    leave.region = FieldRegion.AWAY
    assert env._cycle_v2_delay_penalty(slot, leave) == 0.0

    slot.controller.magazine = list(range(15))
    returning = SimpleNamespace(
        phase=CyclePhase.RETURN,
        region=FieldRegion.AWAY,
        qualified_load=15,
        phase_elapsed_steps=1,
    )
    assert env._cycle_v2_delay_penalty(slot, returning) < 0.0
    returning.region = FieldRegion.HOME
    assert env._cycle_v2_delay_penalty(slot, returning) == 0.0


def test_return_parks_one_preload_batch_and_releases_it_once():
    env = _env(cycle_v2_reserve_count=18, cycle_v2_reserve_batches=3)
    fuel = _Fuel()
    slot = _slot(fuel)
    slot.cycle_v2_mode = "return"
    slot.cycle_v2_ramp_side = -1.0
    env.slots = [slot]
    original = np.zeros((fuel.count, 3), np.float32)

    parked = env._configure_cycle_v2_reserve(slot, original)
    expected = set(range(182, 200))
    assert slot.cycle_v2_reserved_ids == expected
    assert len(slot.cycle_v2_reserved_batches) == 1
    assert all(len(batch) == 18 for batch in slot.cycle_v2_reserved_batches)
    assert np.all(parked[182:, 0] > 9.0)
    assert np.all(parked[182:, 2] == -2.0)

    first = env.release_cycle_v2_reserve(0)
    second = env.release_cycle_v2_reserve(0)
    assert len(first) == 18
    assert second == ()
    assert not set(first).intersection(slot.cycle_v2_reserved_ids)
    assert len(slot.cycle_v2_reserved_batches) == 0
    np.testing.assert_array_equal(
        fuel.pose_calls[0]["positions"], env._fuel_home[list(first)]
    )

    env._pin_cycle_v2_reserve(slot)
    assert slot.cycle_v2_reserved_ids == set()


@pytest.mark.parametrize("mode", ["full", "postdump", "collect"])
def test_native_modes_keep_entire_field_intact(mode):
    env = _env(cycle_v2_reserve_count=18, cycle_v2_reserve_batches=3)
    fuel = _Fuel()
    slot = _slot(fuel)
    slot.cycle_v2_mode = mode
    original = np.arange(fuel.count * 3, dtype=np.float32).reshape(fuel.count, 3)

    configured = env._configure_cycle_v2_reserve(slot, original)

    assert configured is original
    assert slot.cycle_v2_reserved_ids == set()
    assert slot.cycle_v2_reserved_batches == []


def test_bridge_postdump_can_mimic_a_depleted_first_trip_field():
    env = _env(
        cycle_v2_postdump_depleted_count=60,
        cycle_v2_postdump_depleted_prob=1.0,
    )
    fuel = _Fuel()
    slot = _slot(fuel)
    slot.cycle_v2_mode = "postdump"
    original = np.zeros((fuel.count, 3), np.float32)

    configured = env._configure_cycle_v2_reserve(slot, original)

    assert configured is not original
    assert slot.cycle_v2_postdump_depleted_count == 60
    assert len(slot.cycle_v2_reserved_ids) == 60
    assert slot.cycle_v2_reserved_batches == []
    parked = np.asarray(sorted(slot.cycle_v2_reserved_ids), dtype=np.int32)
    assert np.all(configured[parked, 2] == -2.0)


def test_full_field_configuration_does_not_advance_reset_rng():
    env = _env(cycle_v2_reserve_count=18, cycle_v2_reserve_batches=1)
    slot = _slot(_Fuel())
    slot.cycle_v2_mode = "full"
    expected_next = np.random.default_rng(14).random()

    env._configure_cycle_v2_reserve(
        slot, np.zeros((slot.fuel.count, 3), np.float32)
    )

    assert env.rng.random() == expected_next


def test_positive_progress_only_pays_new_best_distance():
    env = _env(cycle_v2_progress_per_m=5.0, cycle_v2_progress_step_cap=10.0)
    slot = _slot(_Fuel())
    slot.cycle_v2_ramp_side = 1.0

    # First observation establishes the baseline. Moving closer earns; moving
    # away and revisiting the old best do not, so oscillation cannot farm it.
    assert env._cycle_v2_progress_reward(slot, CyclePhase.LEAVE, (1.55, -5.0)) == 0.0
    closer = env._cycle_v2_progress_reward(slot, CyclePhase.LEAVE, (1.55, -4.0))
    assert closer == pytest.approx(5.0)
    assert env._cycle_v2_progress_reward(slot, CyclePhase.LEAVE, (1.55, -4.5)) == 0.0
    assert env._cycle_v2_progress_reward(slot, CyclePhase.LEAVE, (1.55, -4.0)) == 0.0
    assert env._cycle_v2_progress_reward(slot, CyclePhase.LEAVE, (1.55, -3.8)) == pytest.approx(1.0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ramp_out_progress_requires_inward_motion_before_forward_motion(sign):
    env = _env(
        cycle_v2_require_ramp_out=True,
        cycle_v2_ramp_out_half_width=0.90,
        cycle_v2_progress_per_m=5.0,
        cycle_v2_progress_step_cap=10.0,
    )
    slot = _slot(_Fuel())
    slot.cycle_v2_ramp_side = sign

    assert env._cycle_v2_progress_reward(
        slot, CyclePhase.LEAVE, (sign * 3.50, -5.0)
    ) == 0.0
    # Merely driving north beside the rail is no longer rewarded.
    assert env._cycle_v2_progress_reward(
        slot, CyclePhase.LEAVE, (sign * 3.50, -4.0)
    ) == 0.0
    inward = env._cycle_v2_progress_reward(
        slot, CyclePhase.LEAVE, (sign * 2.40, -4.0)
    )
    assert inward > 0.0
    forward = env._cycle_v2_progress_reward(
        slot, CyclePhase.LEAVE, (sign * 2.40, -3.8)
    )
    assert forward > 0.0


def _route_stats():
    return {
        "v2_ramp_out_reward": 0.0,
        "ramp_out_attempts": 0,
        "ramp_out_successes": 0,
        "off_ramp_outs": 0,
        "cycle2_ramp_out_attempts": 0,
        "cycle2_ramp_out_successes": 0,
        "cycle2_off_ramp_outs": 0,
        "cycle3plus_ramp_out_attempts": 0,
        "cycle3plus_ramp_out_successes": 0,
        "cycle3plus_off_ramp_outs": 0,
        "ramp_out_abs_x_sum": 0.0,
        "ramp_return_attempts": 0,
        "ramp_return_successes": 0,
        "off_ramp_returns": 0,
        "ramp_return_abs_x_sum": 0.0,
    }


def test_ramp_out_milestone_rewards_ramp_and_rejects_postdump_trench():
    env = _env(
        cycle_v2_require_ramp_out=True,
        cycle_v2_ramp_out_half_width=0.90,
        cycle_v2_ramp_out_bonus=24.0,
        cycle_v2_off_ramp_exit_penalty=20.0,
    )
    event = MilestoneEvent(Milestone.LEFT_HOME, cycle_index=2, elapsed_steps=10)

    ramp_slot = _slot(_Fuel())
    ramp_slot.cycle_v2_mode = "postdump"
    ramp_slot.cycle_v2_stats = _route_stats()
    ramp_reward = env._cycle_v2_milestone_reward(ramp_slot, (event,), 2.40)
    assert ramp_reward > 24.0
    assert ramp_slot.cycle_v2_stats["ramp_out_successes"] == 1
    assert ramp_slot.cycle_v2_stats["v2_ramp_out_reward"] == 24.0
    assert env._cycle_v2_skill_succeeded(ramp_slot, (event,), 2.40)
    assert not env._cycle_v2_skill_failed(ramp_slot, (event,), 2.40)

    left_ramp_slot = _slot(_Fuel())
    left_ramp_slot.cycle_v2_ramp_side = 1.0
    left_ramp_slot.cycle_v2_stats = _route_stats()
    env._cycle_v2_milestone_reward(left_ramp_slot, (event,), -2.40)
    assert left_ramp_slot.cycle_v2_ramp_side == -1.0

    trench_slot = _slot(_Fuel())
    trench_slot.cycle_v2_mode = "postdump"
    trench_slot.cycle_v2_stats = _route_stats()
    trench_reward = env._cycle_v2_milestone_reward(trench_slot, (event,), 2.60)
    assert trench_reward < 0.0
    assert trench_slot.cycle_v2_stats["off_ramp_outs"] == 1
    assert trench_slot.cycle_v2_stats["v2_ramp_out_reward"] == -20.0
    assert not env._cycle_v2_skill_succeeded(trench_slot, (event,), 2.60)
    assert env._cycle_v2_skill_failed(trench_slot, (event,), 2.60)


def test_v5_postdump_must_clear_ramp_and_collect_target_load():
    env = _env(
        cycle_v2_require_ramp_out=True,
        cycle_v2_postdump_require_target_load=True,
    )
    slot = _slot(_Fuel())
    slot.cycle_v2_mode = "postdump"
    slot.cycle_v2_stats = _route_stats()
    left = MilestoneEvent(Milestone.LEFT_HOME, cycle_index=2, elapsed_steps=12)
    loaded = MilestoneEvent(Milestone.TARGET_LOAD, cycle_index=2, elapsed_steps=74)

    # A clean crossing is now only the first half of the drill.
    env._cycle_v2_milestone_reward(slot, (left,), 1.55)
    assert slot.cycle_v2_stats["ramp_out_successes"] == 1
    assert not env._cycle_v2_skill_succeeded(slot, (left,), 1.55)
    assert env._cycle_v2_skill_succeeded(slot, (loaded,), 0.80)

    no_ramp = _slot(_Fuel())
    no_ramp.cycle_v2_mode = "postdump"
    no_ramp.cycle_v2_stats = _route_stats()
    assert not env._cycle_v2_skill_succeeded(no_ramp, (loaded,), 0.80)


def test_v6_postdump_continues_through_return_and_second_score():
    env = _env(
        cycle_v2_require_ramp_out=True,
        cycle_v2_postdump_require_target_load=True,
        cycle_v2_postdump_complete_cycle=True,
    )
    slot = _slot(_Fuel())
    slot.cycle_v2_mode = "postdump"
    slot.cycle_v2_stats = _route_stats()
    left = MilestoneEvent(Milestone.LEFT_HOME, cycle_index=2, elapsed_steps=10)
    loaded = MilestoneEvent(Milestone.TARGET_LOAD, cycle_index=2, elapsed_steps=80)
    returned = MilestoneEvent(Milestone.RETURNED_HOME, cycle_index=2, elapsed_steps=130)
    scored = MilestoneEvent(Milestone.CYCLE_SCORED, cycle_index=2, elapsed_steps=160)

    env._cycle_v2_milestone_reward(slot, (left,), 1.55)
    assert not env._cycle_v2_skill_succeeded(slot, (loaded,), 0.0)
    assert not env._cycle_v2_skill_succeeded(slot, (returned,), 1.0)
    assert env._cycle_v2_skill_succeeded(slot, (scored,), 1.0)

    no_ramp = _slot(_Fuel())
    no_ramp.cycle_v2_mode = "postdump"
    no_ramp.cycle_v2_stats = _route_stats()
    assert not env._cycle_v2_skill_succeeded(no_ramp, (scored,), 1.0)


def test_v8_soft_load_bonus_is_paid_on_return_and_conversion():
    env = _env(
        cycle_v2_target_load=15,
        cycle_v2_preferred_repeat_load=30,
        cycle_v2_repeat_load_return_bonus=8.0,
        cycle_v2_repeat_load_score_bonus=12.0,
    )
    slot = _slot(_Fuel())
    slot.cycle_v2_stats = _route_stats()
    returned = MilestoneEvent(
        Milestone.RETURNED_HOME, cycle_index=2, elapsed_steps=100
    )
    scored = MilestoneEvent(
        Milestone.CYCLE_SCORED, cycle_index=2, elapsed_steps=130
    )

    env._cycle_v2_milestone_reward(
        slot, (returned,), 1.55, qualified_load=25
    )
    assert slot.cycle_v2_return_loads == {2: 25}
    assert slot.cycle_v2_stats["repeat_return_load_sum"] == 25
    assert slot.cycle_v2_stats["v2_load_efficiency_reward"] == pytest.approx(
        8.0 * 10.0 / 15.0
    )

    env._cycle_v2_milestone_reward(
        slot, (scored,), 1.55, qualified_load=0
    )
    assert slot.cycle_v2_return_loads == {}
    assert slot.cycle_v2_stats["repeat_scored_load_sum"] == 25
    assert slot.cycle_v2_stats["v2_load_efficiency_reward"] == pytest.approx(
        (8.0 + 12.0) * 10.0 / 15.0
    )


def test_outer_rail_penalty_protects_first_score_and_clean_ramp_route():
    env = _env(
        cycle_v2_outer_rail_grace_steps=2,
        cycle_v2_outer_rail_penalty_per_step=0.04,
        cycle_v2_outer_rail_penalty_cap=1.0,
    )
    slot = _slot(_Fuel())

    first = SimpleNamespace(phase=CyclePhase.FIRST_CYCLE, cycle_index=1)
    score = SimpleNamespace(phase=CyclePhase.SCORE, cycle_index=2)
    leave = SimpleNamespace(phase=CyclePhase.LEAVE, cycle_index=2)

    assert env._cycle_v2_outer_rail_penalty(slot, first, 3.55) == 0.0
    assert env._cycle_v2_outer_rail_penalty(slot, score, 3.55) == 0.0
    for _ in range(20):
        assert env._cycle_v2_outer_rail_penalty(slot, leave, 1.55) == 0.0
    assert slot.cycle_v2_outer_rail_streak == 0


@pytest.mark.parametrize("x", [3.60, -3.60])
def test_outer_rail_penalty_is_symmetric_grace_based_and_capped(x):
    env = _env(
        cycle_v2_outer_rail_grace_steps=2,
        cycle_v2_outer_rail_penalty_per_step=0.04,
        cycle_v2_outer_rail_penalty_cap=0.06,
    )
    slot = _slot(_Fuel())
    leave = SimpleNamespace(phase=CyclePhase.LEAVE, cycle_index=2)

    assert env._cycle_v2_outer_rail_penalty(slot, leave, x) == 0.0
    assert env._cycle_v2_outer_rail_penalty(slot, leave, x) == 0.0
    assert env._cycle_v2_outer_rail_penalty(slot, leave, x) == pytest.approx(-0.04)
    assert env._cycle_v2_outer_rail_penalty(slot, leave, x) == pytest.approx(-0.02)
    assert env._cycle_v2_outer_rail_penalty(slot, leave, x) == 0.0
    assert slot.cycle_v2_outer_rail_spent == pytest.approx(0.06)

    # Hysteresis keeps the state active inside the enter threshold, then an
    # actual inner-lane re-entry clears the streak.
    assert env._cycle_v2_outer_rail_penalty(slot, leave, 2.70) == 0.0
    assert slot.cycle_v2_outer_rail_active
    assert env._cycle_v2_outer_rail_penalty(slot, leave, 2.55) == 0.0
    assert not slot.cycle_v2_outer_rail_active
    assert slot.cycle_v2_outer_rail_streak == 0


def test_outer_rail_v2_has_minimum_charge_and_escalates_at_threshold():
    env = _env(
        cycle_v2_outer_rail_enter_x=2.75,
        cycle_v2_outer_rail_grace_steps=1,
        cycle_v2_outer_rail_penalty_per_step=0.12,
        cycle_v2_outer_rail_penalty_cap=10.0,
        cycle_v2_outer_rail_min_scale=0.75,
        cycle_v2_outer_rail_escalation_steps=2,
        cycle_v2_outer_rail_max_multiplier=3.0,
    )
    slot = _slot(_Fuel())
    leave = SimpleNamespace(phase=CyclePhase.LEAVE, cycle_index=2)

    assert env._cycle_v2_outer_rail_penalty(slot, leave, 2.75) == 0.0
    assert env._cycle_v2_outer_rail_penalty(
        slot, leave, 2.75
    ) == pytest.approx(-0.09)
    assert env._cycle_v2_outer_rail_penalty(
        slot, leave, 2.75
    ) == pytest.approx(-0.135)
    assert env._cycle_v2_outer_rail_penalty(
        slot, leave, 2.75
    ) == pytest.approx(-0.18)


def test_outer_rail_telemetry_splits_cycle2_cycle3_and_side():
    env = _env(cycle_v2_outer_rail_penalty_per_step=0.0)
    slot = _slot(_Fuel())
    slot.cycle_v2_stats = {
        "outer_rail_active_steps": 0,
        "outer_rail_steps": 0,
        "outer_rail_positive_steps": 0,
        "outer_rail_negative_steps": 0,
        "outer_rail_max_streak": 0,
        "outer_rail_fraction": 0.0,
        "cycle2_outer_rail_active_steps": 0,
        "cycle2_outer_rail_steps": 0,
        "cycle2_outer_rail_fraction": 0.0,
        "cycle3plus_outer_rail_active_steps": 0,
        "cycle3plus_outer_rail_steps": 0,
        "cycle3plus_outer_rail_fraction": 0.0,
    }
    cycle2 = SimpleNamespace(phase=CyclePhase.COLLECT, cycle_index=2)
    cycle3 = SimpleNamespace(phase=CyclePhase.RETURN, cycle_index=3)

    env._cycle_v2_outer_rail_penalty(slot, cycle2, 3.2)
    env._cycle_v2_outer_rail_penalty(slot, cycle3, -3.2)

    assert slot.cycle_v2_stats["outer_rail_steps"] == 2
    assert slot.cycle_v2_stats["outer_rail_positive_steps"] == 1
    assert slot.cycle_v2_stats["outer_rail_negative_steps"] == 1
    assert slot.cycle_v2_stats["cycle2_outer_rail_steps"] == 1
    assert slot.cycle_v2_stats["cycle3plus_outer_rail_steps"] == 1


def test_outer_rail_fraction_keeps_updating_after_robot_leaves_wall():
    env = _env(cycle_v2_outer_rail_penalty_per_step=0.0)
    slot = _slot(_Fuel())
    slot.cycle_v2_stats = {
        "outer_rail_active_steps": 0,
        "outer_rail_steps": 0,
        "outer_rail_positive_steps": 0,
        "outer_rail_negative_steps": 0,
        "outer_rail_max_streak": 0,
        "outer_rail_fraction": 0.0,
        "cycle2_outer_rail_active_steps": 0,
        "cycle2_outer_rail_steps": 0,
        "cycle2_outer_rail_fraction": 0.0,
        "cycle3plus_outer_rail_active_steps": 0,
        "cycle3plus_outer_rail_steps": 0,
        "cycle3plus_outer_rail_fraction": 0.0,
    }
    leave = SimpleNamespace(phase=CyclePhase.LEAVE, cycle_index=2)

    env._cycle_v2_outer_rail_penalty(slot, leave, 3.2)
    for _ in range(9):
        env._cycle_v2_outer_rail_penalty(slot, leave, 1.55)

    assert slot.cycle_v2_stats["outer_rail_steps"] == 1
    assert slot.cycle_v2_stats["outer_rail_active_steps"] == 10
    assert slot.cycle_v2_stats["outer_rail_fraction"] == pytest.approx(0.1)
    assert slot.cycle_v2_stats["cycle2_outer_rail_fraction"] == pytest.approx(0.1)


def test_ramp_side_refresh_uses_scoring_side_and_centre_deadband():
    env = _env(cycle_v2_ramp_side_deadband_x=0.25)
    slot = _slot(_Fuel())
    slot.cycle_v2_ramp_side = 1.0

    assert env._cycle_v2_select_ramp_side(slot, -1.4) == -1.0
    assert env._cycle_v2_select_ramp_side(slot, 0.1) == -1.0
    assert env._cycle_v2_select_ramp_side(slot, 1.4) == 1.0
