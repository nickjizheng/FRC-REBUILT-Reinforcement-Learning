"""Pure-Python regression tests for the isolated Stage C v2 cycle logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frc_rebuilt.rl.cycle_v2 import (  # noqa: E402
    COLLECT_UNTIL_PREFERRED_REVISIONS,
    RETURN_INTAKE_REVISIONS,
    ROUTE_EFFICIENCY_REVISION,
    SCORE_EFFICIENCY_REVISIONS,
    STAGE_D_REVISIONS,
    SUPPORTED_ROUTE_EFFICIENCY_REVISIONS,
    CyclePhase,
    CycleV2Config,
    CycleV2State,
    FieldRegion,
    Milestone,
    capped_phase_delay_penalty,
    soft_repeat_load_bonus,
    time_decayed_success_bonus,
)

HOME = (3.3, -3.6)
DEAD_BAND = (3.0, -2.8)
AWAY = (0.0, 0.5)


def test_stage_d_revision_is_explicit_and_preserves_legacy_capabilities():
    assert ROUTE_EFFICIENCY_REVISION == "stage_d_v1"
    assert "score_efficiency_v9" in SUPPORTED_ROUTE_EFFICIENCY_REVISIONS
    assert ROUTE_EFFICIENCY_REVISION in SUPPORTED_ROUTE_EFFICIENCY_REVISIONS
    assert "score_efficiency_v9" in SCORE_EFFICIENCY_REVISIONS
    assert ROUTE_EFFICIENCY_REVISION in SCORE_EFFICIENCY_REVISIONS
    assert STAGE_D_REVISIONS == ("stage_d_v1", "stage_d_v2")
    for revision in STAGE_D_REVISIONS:
        assert revision in SUPPORTED_ROUTE_EFFICIENCY_REVISIONS
        assert revision in SCORE_EFFICIENCY_REVISIONS
        assert revision in COLLECT_UNTIL_PREFERRED_REVISIONS
    assert "score_efficiency_v9" in COLLECT_UNTIL_PREFERRED_REVISIONS
    assert RETURN_INTAKE_REVISIONS == ("score_efficiency_v10_return_intake",)
    assert ROUTE_EFFICIENCY_REVISION not in RETURN_INTAKE_REVISIONS


def _milestones(step):
    return [event.name for event in step.milestones]


def _latch(state: CycleV2State, first_ids=range(25)):
    ids = tuple(first_ids)
    state.reset(ids, HOME)
    return state.update(
        [],
        ids,
        HOME,
        score=len(ids),
        score_dump_started=True,
        score_dump_completed=True,
    )


def _reach_score_phase(state: CycleV2State, ids):
    """Leave, gather ``ids`` away, and return home while still loaded."""

    step = state.update([], [], AWAY, score=25)
    assert step.phase is CyclePhase.COLLECT
    step = state.update(ids, [], AWAY, score=25)
    assert step.phase is CyclePhase.RETURN
    step = state.update(ids, [], HOME, score=25)
    assert step.phase is CyclePhase.SCORE
    return step


def test_soft_repeat_load_bonus_prefers_fuller_load_without_a_hard_gate():
    values = [
        soft_repeat_load_bonus(
            qualified_load=load,
            minimum_load=15,
            preferred_load=30,
            max_bonus=12.0,
        )
        for load in (10, 15, 20, 25, 30, 45)
    ]

    assert values == pytest.approx([0.0, 0.0, 4.0, 8.0, 12.0, 12.0])


def test_latch_requires_verified_empty_dump_and_preserves_first_reward():
    state = CycleV2State()
    state.reset([], HOME)

    # Neither an empty chamber nor a high cumulative score can arm the repeat
    # cycle without the verified score-mode dump completion edge.
    step = state.update([], [], HOME, score=40)
    assert not step.latched
    assert step.phase is CyclePhase.FIRST_CYCLE

    first_load = tuple(range(25))
    state.reset(first_load, HOME)
    unverified = state.update([], first_load, HOME, score=25)
    assert not unverified.latched
    assert unverified.phase is CyclePhase.FIRST_CYCLE

    state.reset(first_load, HOME)
    step = state.update(
        [],
        first_load,
        HOME,
        score=25,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert step.score_reward == 25 * 10.0
    assert step.full_score_events == 25
    assert step.unqualified_score_events == 0
    assert step.latched and step.phase is CyclePhase.LEAVE
    assert step.cycle_index == 2
    assert _milestones(step) == [Milestone.LATCHED]


def test_first_empty_dump_latches_after_only_six_landings():
    state = CycleV2State()
    first_load = tuple(range(30))
    state.reset(first_load, HOME)

    step = state.update(
        [],
        first_load[:6],
        HOME,
        score=6,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert step.latched
    assert step.phase is CyclePhase.LEAVE
    assert step.score_reward == 60.0
    assert step.cycles_attempted == 0

    # The other projectiles are still first-volley balls and retain +10 even
    # though navigation has already advanced to LEAVE.
    trailing = state.update([], first_load[6:8], HOME, score=8)
    assert trailing.score_reward == 20.0


def test_hysteresis_and_collect_requires_target_load_not_one_ball():
    state = CycleV2State()
    _latch(state)

    # Dead-band motion retains HOME and emits no phantom leave event.
    step = state.update([], [], DEAD_BAND, score=25)
    assert step.region is FieldRegion.HOME
    assert step.phase is CyclePhase.LEAVE
    assert not step.milestones

    step = state.update([100], [], AWAY, score=25)
    assert step.region is FieldRegion.AWAY
    assert step.phase is CyclePhase.COLLECT
    assert step.qualified_load == 1
    assert step.collect_reward == 1.5
    assert _milestones(step) == [Milestone.LEFT_HOME]

    # Fourteen is still collection; the old one-ball phase switch is gone.
    step = state.update(list(range(100, 114)), [], AWAY, score=25)
    assert step.qualified_load == 14
    assert step.phase is CyclePhase.COLLECT
    assert Milestone.TARGET_LOAD not in _milestones(step)

    step = state.update(list(range(100, 115)), [], AWAY, score=25)
    assert step.qualified_load == 15
    assert step.phase is CyclePhase.RETURN
    assert _milestones(step) == [Milestone.TARGET_LOAD]

    # Returning into the dead band retains AWAY; a full home crossing is needed.
    step = state.update(list(range(100, 115)), [], DEAD_BAND, score=25)
    assert step.region is FieldRegion.AWAY
    assert step.phase is CyclePhase.RETURN


def test_preferred_load_keeps_collecting_past_target_until_fuller_load():
    state = CycleV2State(
        CycleV2Config(
            target_load=15,
            preferred_load=20,
            collect_stall_steps=12,
            return_time_guard=0.20,
        )
    )
    _latch(state)
    state.update([], [], AWAY, score=25)

    step = state.update(list(range(100, 115)), [], AWAY, score=25)
    assert step.qualified_load == 15
    assert step.phase is CyclePhase.COLLECT
    assert _milestones(step) == [Milestone.TARGET_LOAD]

    step = state.update(list(range(100, 120)), [], AWAY, score=25)
    assert step.qualified_load == 20
    assert step.phase is CyclePhase.RETURN
    assert step.collect_exit_reason == "preferred"


def test_preferred_load_stall_and_clock_fallbacks_avoid_deadlock():
    stalled = CycleV2State(
        CycleV2Config(
            target_load=15,
            preferred_load=20,
            collect_stall_steps=3,
        )
    )
    _latch(stalled)
    stalled.update([], [], AWAY, score=25)
    ids = list(range(200, 215))
    assert stalled.update(ids, [], AWAY, score=25).phase is CyclePhase.COLLECT
    assert stalled.update(ids, [], AWAY, score=25).phase is CyclePhase.COLLECT
    assert stalled.update(ids, [], AWAY, score=25).phase is CyclePhase.COLLECT
    step = stalled.update(ids, [], AWAY, score=25)
    assert step.phase is CyclePhase.RETURN
    assert step.collect_exit_reason == "stall"

    clocked = CycleV2State(
        CycleV2Config(
            target_load=15,
            preferred_load=20,
            return_time_guard=0.25,
        )
    )
    _latch(clocked)
    clocked.update([], [], AWAY, score=25)
    step = clocked.update(
        list(range(300, 315)),
        [],
        AWAY,
        score=25,
        time_remaining=0.25,
    )
    assert step.phase is CyclePhase.RETURN
    assert step.collect_exit_reason == "clock"


def test_route_qualified_and_unqualified_scores_have_distinct_credit():
    state = CycleV2State()
    _latch(state)
    qualified = list(range(100, 115))
    _reach_score_phase(state, qualified)

    # 100 was earned through the route; 999 never was.
    step = state.update(
        [],
        [100, 999],
        HOME,
        score=27,
        score_dump_started=True,
        score_dump_completed=True,
    )
    # Only route-qualified repeat-cycle balls earn reward.  The stray event is
    # still counted for diagnostics but cannot teach a scoring shortcut.
    assert step.score_reward == 10.0
    assert step.full_score_events == 1
    assert step.unqualified_score_events == 1
    assert 100 not in step.qualified_ids
    assert step.reward == step.score_reward
    assert step.phase is CyclePhase.LEAVE
    assert step.cycles_attempted == 1
    assert step.cycles_completed == 0
    assert _milestones(step) == [Milestone.CYCLE_DUMPED]


def test_scoring_consumes_qualification_so_repeat_is_not_full():
    state = CycleV2State()
    _latch(state)
    qualified = list(range(200, 215))
    _reach_score_phase(state, qualified)

    first = state.update(
        qualified[1:],
        [200],
        HOME,
        score=26,
        score_dump_started=True,
    )
    assert first.score_reward == 10.0
    assert 200 not in state.qualified_ids

    repeat = state.update([], [200], HOME, score=27)
    assert repeat.score_reward == 0.0
    assert repeat.full_score_events == 0
    assert repeat.unqualified_score_events == 1


def test_trailing_first_dump_events_remain_fully_protected_after_latch():
    state = CycleV2State(CycleV2Config(first_unload_score_floor=25))
    first_load = tuple(range(30))
    state.reset(first_load, HOME)

    # The chamber is already empty, but only the first 25 projectiles have
    # reached the hub. This arms the repeat-cycle latch.
    armed = state.update(
        [],
        first_load[:25],
        HOME,
        score=25,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert armed.latched
    assert armed.score_reward == 250.0

    # Remaining in-flight balls belong to the protected first dump, not home
    # camping. Their first landing remains +10; a duplicate gets no new credit.
    trailing = state.update([], first_load[25:], HOME, score=30)
    assert trailing.score_reward == 50.0
    assert trailing.full_score_events == 5
    duplicate = state.update([], [29], HOME, score=31)
    assert duplicate.score_reward == 0.0
    assert duplicate.unqualified_score_events == 1


def test_home_collection_gets_base_credit_not_route_qualification():
    config = CycleV2Config(base_collect_reward=0.4)
    state = CycleV2State(config)
    _latch(state)

    step = state.update([700], [], HOME, score=25)
    assert step.collect_reward == 0.4
    assert step.qualified_load == 0
    assert 700 not in step.qualified_ids


def test_cycle_milestones_rearm_for_the_next_cycle():
    state = CycleV2State()
    _latch(state)
    cycle2 = list(range(300, 315))
    _reach_score_phase(state, cycle2)
    dumped = state.update(
        [],
        cycle2[:2],
        HOME,
        score=27,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert dumped.cycle_index == 3
    assert dumped.phase is CyclePhase.LEAVE
    assert dumped.cycles_attempted == 1
    assert dumped.cycles_completed == 0
    assert _milestones(dumped) == [Milestone.CYCLE_DUMPED]

    # Quality is allowed to resolve after navigation has already resumed.  A
    # 15-ball start needs ceil(75%) == 12 unique scored IDs.
    quality = state.update([], cycle2[2:12], HOME, score=37)
    assert quality.phase is CyclePhase.LEAVE
    assert quality.cycles_completed == 1
    assert _milestones(quality) == [Milestone.CYCLE_SCORED]
    assert quality.milestones[0].cycle_index == 2

    duplicate = state.update([], cycle2[2:12], HOME, score=47)
    assert duplicate.cycles_completed == 1
    assert Milestone.CYCLE_SCORED not in _milestones(duplicate)

    # The same milestone names emit again, now tagged as cycle 3.
    left = state.update([], [], AWAY, score=47)
    assert _milestones(left) == [Milestone.LEFT_HOME]
    assert left.milestones[0].cycle_index == 3

    cycle3 = list(range(400, 415))
    loaded = state.update(cycle3, [], AWAY, score=47)
    assert _milestones(loaded) == [Milestone.TARGET_LOAD]
    assert loaded.milestones[0].cycle_index == 3

    returned = state.update(cycle3, [], HOME, score=47)
    assert _milestones(returned) == [Milestone.RETURNED_HOME]
    finished = state.update(
        [],
        cycle3[:12],
        HOME,
        score=59,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert _milestones(finished) == [Milestone.CYCLE_DUMPED, Milestone.CYCLE_SCORED]
    assert finished.cycle_index == 4
    assert finished.cycles_attempted == 2
    assert finished.cycles_completed == 2


def test_eight_ball_return_dump_succeeds_at_absolute_floor_six():
    state = CycleV2State()
    load = tuple(range(800, 808))
    state.reset(
        load,
        AWAY,
        phase=CyclePhase.RETURN,
        qualified_ids=load,
    )
    returned = state.update(load, [], HOME, score=0)
    assert returned.phase is CyclePhase.SCORE

    dumped = state.update(
        [],
        load[:6],
        HOME,
        score=6,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert dumped.phase is CyclePhase.LEAVE
    assert dumped.cycles_attempted == 1
    assert dumped.cycles_completed == 1
    assert _milestones(dumped) == [Milestone.CYCLE_DUMPED, Milestone.CYCLE_SCORED]


def test_repeat_dump_quality_threshold_uses_configured_fraction_and_floor():
    state = CycleV2State(
        CycleV2Config(target_load=10, cycle_score_fraction=0.5, cycle_score_floor=3)
    )
    load = tuple(range(900, 910))
    state.reset(load, AWAY, phase=CyclePhase.RETURN, qualified_ids=load)
    state.update(load, [], HOME, score=0)

    four = state.update(
        [],
        load[:4],
        HOME,
        score=4,
        score_dump_started=True,
        score_dump_completed=True,
    )
    assert four.cycles_attempted == 1
    assert four.cycles_completed == 0

    fifth = state.update([], [load[4]], HOME, score=5)
    assert fifth.cycles_completed == 1
    assert _milestones(fifth) == [Milestone.CYCLE_SCORED]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cycle_score_fraction": 0.0},
        {"cycle_score_fraction": 1.01},
        {"cycle_score_floor": 0},
    ],
)
def test_repeat_dump_quality_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CycleV2Config(**kwargs)


def test_phase_elapsed_clock_restarts_on_transition():
    state = CycleV2State()
    _latch(state)
    waiting = state.update([], [], HOME, score=25)
    assert waiting.phase is CyclePhase.LEAVE
    assert waiting.phase_elapsed_steps == 1

    left = state.update([], [], AWAY, score=25)
    assert left.phase is CyclePhase.COLLECT
    assert left.phase_elapsed_steps == 0

    collecting = state.update([], [], AWAY, score=25)
    assert collecting.phase_elapsed_steps == 1


def test_feature_vector_is_phase_one_hot_load_ratio_and_latch():
    state = CycleV2State()
    assert state.feature_vector() == (
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0 / 60.0, 1.0
    )
    _latch(state)

    state.update([], [], AWAY, score=25)
    step = state.update(list(range(10)), [], AWAY, score=25)
    assert step.phase is CyclePhase.COLLECT
    assert step.features[:5] == (0.0, 0.0, 1.0, 0.0, 0.0)
    assert step.features[5] == pytest.approx(10.0 / 60.0)
    assert step.features[6] == pytest.approx(15.0 / 60.0)
    assert step.features[7] == 1.0

    # Ratio is bounded for actor stability if a larger-than-target load exists.
    step = state.update(list(range(30)), [], AWAY, score=25)
    assert step.features[5] == pytest.approx(30.0 / 60.0)


def test_reset_and_done_do_not_leak_state_between_environments():
    state = CycleV2State()
    _latch(state)
    state.update([], [], AWAY, score=25)
    terminal = state.update([50], [], AWAY, score=25, done=True)

    # The caller still gets terminal diagnostics.
    assert terminal.done and terminal.latched
    assert terminal.qualified_ids == frozenset({50})

    # Internal state is clean for the next autoreset episode.
    assert not state.latched
    assert state.phase is CyclePhase.FIRST_CYCLE
    assert state.cycle_index == 1
    assert state.qualified_ids == set()
    assert state.prev_magazine == frozenset()
    assert state.feature_vector() == (
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0 / 60.0, 1.0
    )

    state.reset([8, 9], HOME)
    step = state.update([8, 9], [], HOME, score=0)
    assert step.collected_ids == ()  # preload was seeded, not rewarded
    assert step.collect_reward == 0.0


def test_time_decay_only_pays_nonnegative_success_bonus():
    immediate = time_decayed_success_bonus(6.0, elapsed_steps=0, decay_steps=100)
    later = time_decayed_success_bonus(6.0, elapsed_steps=100, decay_steps=100)
    assert immediate == 6.0
    assert 0.0 < later < immediate
    assert time_decayed_success_bonus(0.0, 999, 1) == 0.0
    with pytest.raises(ValueError):
        time_decayed_success_bonus(-1.0, 0, 100)


def test_capped_phase_delay_penalty_has_grace_and_exact_cap():
    kwargs = {
        "grace_steps": 2,
        "penalty_per_step": 0.4,
        "cap": 1.0,
    }
    assert capped_phase_delay_penalty(elapsed_steps=0, spent=0.0, **kwargs) == 0.0
    assert capped_phase_delay_penalty(elapsed_steps=2, spent=0.0, **kwargs) == 0.0

    first = capped_phase_delay_penalty(elapsed_steps=3, spent=0.0, **kwargs)
    second = capped_phase_delay_penalty(elapsed_steps=4, spent=0.4, **kwargs)
    last = capped_phase_delay_penalty(elapsed_steps=5, spent=0.8, **kwargs)
    capped = capped_phase_delay_penalty(elapsed_steps=6, spent=1.0, **kwargs)
    assert first == pytest.approx(-0.4)
    assert second == pytest.approx(-0.4)
    assert last == pytest.approx(-0.2)
    assert capped == 0.0
    assert first + second + last + capped == pytest.approx(-1.0)

    with pytest.raises(ValueError):
        capped_phase_delay_penalty(
            elapsed_steps=3,
            grace_steps=0,
            penalty_per_step=0.1,
            spent=-0.1,
            cap=1.0,
        )


def test_curriculum_reset_primes_collect_and_return_without_fake_pickups():
    state = CycleV2State()
    state.reset([], AWAY, phase=CyclePhase.COLLECT)
    assert state.latched and state.phase is CyclePhase.COLLECT
    assert state.cycle_index == 2

    loaded = tuple(range(100, 115))
    state.reset(
        loaded,
        AWAY,
        phase=CyclePhase.RETURN,
        qualified_ids=loaded,
    )
    step = state.update(loaded, [], AWAY, score=0, time_remaining=0.4)
    assert step.collected_ids == ()
    assert step.collect_reward == 0.0
    assert step.qualified_load == 15
    assert step.features[5:] == pytest.approx((15 / 60, 15 / 60, 0.4))
