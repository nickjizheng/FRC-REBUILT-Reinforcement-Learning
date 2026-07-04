from pathlib import Path
import random
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from xrc_rebuilt import rules


def test_official_field_and_fuel_constants_are_si_converted():
    assert rules.FIELD_LENGTH_M == pytest.approx(16.54048)
    assert rules.FIELD_WIDTH_M == pytest.approx(8.06958)
    assert rules.FUEL_DIAMETER_M == pytest.approx(0.150)
    assert rules.FUEL_RADIUS_M == pytest.approx(0.075)
    assert rules.FUEL_MASS_MIN_KG == pytest.approx(0.20320938176)
    assert rules.FUEL_MASS_MAX_KG == pytest.approx(0.226796185)
    assert rules.HUB_EXIT_COUNT == 4
    assert rules.BUMP_COUNT == 4
    assert rules.TRENCH_COUNT == 4


@pytest.mark.parametrize(
    ("preloaded", "neutral"),
    [(0, 408), (1, 407), (24, 384), (48, 360)],
)
def test_504_fuel_distribution(preloaded, neutral):
    distribution = rules.fuel_distribution(preloaded)
    assert distribution.total == 504
    assert distribution.depots == 48
    assert distribution.outposts == 48
    assert distribution.robot_preloads == preloaded
    assert distribution.neutral_zone == neutral
    assert (
        distribution.depots
        + distribution.outposts
        + distribution.robot_preloads
        + distribution.neutral_zone
        == 504
    )


@pytest.mark.parametrize("bad", [-1, 49])
def test_fuel_distribution_rejects_illegal_total_preloads(bad):
    with pytest.raises(ValueError):
        rules.fuel_distribution(bad)


def test_match_timeline_boundaries():
    expected = {
        0.0: rules.MatchPhase.AUTO,
        19.999: rules.MatchPhase.AUTO,
        20.0: rules.MatchPhase.TRANSITION,
        29.999: rules.MatchPhase.TRANSITION,
        30.0: rules.MatchPhase.SHIFT_1,
        55.0: rules.MatchPhase.SHIFT_2,
        80.0: rules.MatchPhase.SHIFT_3,
        105.0: rules.MatchPhase.SHIFT_4,
        130.0: rules.MatchPhase.ENDGAME,
        159.999: rules.MatchPhase.ENDGAME,
        160.0: rules.MatchPhase.POST_MATCH_ASSESSMENT,
        162.999: rules.MatchPhase.POST_MATCH_ASSESSMENT,
        163.0: rules.MatchPhase.COMPLETE,
    }
    for elapsed, phase in expected.items():
        assert rules.phase_at(elapsed) is phase

    assert rules.MATCH_DURATION_S == 160
    assert rules.AUTO_DURATION_S == 20
    assert rules.TRANSITION_DURATION_S == 10
    assert rules.SHIFT_DURATION_S == 25
    assert rules.ENDGAME_DURATION_S == 30


def test_arena_timer_resets_for_teleop():
    assert rules.arena_timer_seconds(0) == 20
    assert rules.arena_timer_seconds(19) == 1
    assert rules.arena_timer_seconds(20) == 140
    assert rules.arena_timer_seconds(30) == 130
    assert rules.arena_timer_seconds(160) == 0


def test_higher_auto_scorer_is_first_inactive():
    assert rules.select_first_inactive_alliance(12, 7) is rules.Alliance.RED
    assert rules.select_first_inactive_alliance(3, 9) is rules.Alliance.BLUE


def test_tied_auto_selection_is_seedable_and_does_not_mutate_global_rng():
    results_a = [
        rules.select_first_inactive_alliance(8, 8, seed=seed)
        for seed in range(20)
    ]
    results_b = [
        rules.select_first_inactive_alliance(8, 8, seed=seed)
        for seed in range(20)
    ]
    assert results_a == results_b
    assert set(results_a) == {rules.Alliance.RED, rules.Alliance.BLUE}


@pytest.mark.parametrize(
    ("phase", "red_active", "blue_active"),
    [
        (rules.MatchPhase.AUTO, True, True),
        (rules.MatchPhase.TRANSITION, True, True),
        (rules.MatchPhase.SHIFT_1, False, True),
        (rules.MatchPhase.SHIFT_2, True, False),
        (rules.MatchPhase.SHIFT_3, False, True),
        (rules.MatchPhase.SHIFT_4, True, False),
        (rules.MatchPhase.ENDGAME, True, True),
    ],
)
def test_hubs_alternate_when_red_is_first_inactive(
    phase, red_active, blue_active
):
    assert rules.hub_is_active("red", phase, "red") is red_active
    assert rules.hub_is_active("blue", phase, "red") is blue_active


def test_three_second_deactivation_warning():
    # RED is first inactive, so its first deactivation is at t=30.
    assert not rules.hub_in_deactivation_warning("red", 26.999, "red")
    assert rules.hub_in_deactivation_warning("red", 27.0, "red")
    assert rules.hub_in_deactivation_warning("red", 29.999, "red")
    assert not rules.hub_in_deactivation_warning("red", 30.0, "red")
    # Both active HUBS warn for the final buzzer.
    assert rules.hub_in_deactivation_warning("blue", 157.0, "red")


def test_official_hub_light_sequence_when_red_rests_first():
    state = rules.hub_light_state_at
    lights = rules.HubLightState
    assert state("red", 10.0, None) is lights.ACTIVE
    assert state("blue", 10.0, None) is lights.ACTIVE
    assert state("red", 22.0, "red") is lights.TRANSITION_CHASE
    assert state("blue", 22.0, "red") is lights.ACTIVE
    assert state("red", 27.0, "red") is lights.WARNING
    assert state("red", 30.0, "red") is lights.INACTIVE
    assert state("blue", 52.0, "red") is lights.WARNING
    assert state("blue", 55.0, "red") is lights.INACTIVE
    assert state("red", 140.0, "red") is lights.ACTIVE
    assert state("red", 157.0, "red") is lights.WARNING
    assert state("blue", 157.0, "red") is lights.WARNING
    assert state("red", 160.0, "red") is lights.POST_MATCH_WHITE
    assert state("blue", 162.999, "red") is lights.POST_MATCH_WHITE
    assert state("red", 163.0, "red") is lights.INACTIVE


def test_three_second_scoring_assessment_grace():
    # RED becomes inactive at t=30; processing is credited through t=33.
    assert rules.fuel_score_is_eligible("red", 30.0, "red")
    assert rules.fuel_score_is_eligible("red", 33.0, "red")
    assert not rules.fuel_score_is_eligible("red", 33.0001, "red")
    # BLUE remains active, while RED becomes active again in SHIFT 2.
    assert rules.fuel_score_is_eligible("blue", 40.0, "red")
    assert rules.fuel_score_is_eligible("red", 55.0, "red")
    # Final scoring assessment lasts through 3 seconds after TELEOP.
    assert rules.fuel_score_is_eligible("blue", 163.0, "red")
    assert not rules.fuel_score_is_eligible("blue", 163.0001, "red")


def test_hub_routing_delay_is_bounded_seedable_and_has_target_mean():
    rng = random.Random(2026)
    samples = [rules.sample_hub_routing_delay(rng) for _ in range(50_000)]
    assert min(samples) >= 0.0
    assert max(samples) <= 3.0
    assert sum(samples) / len(samples) == pytest.approx(1.32, abs=0.015)


def test_hub_exit_randomly_uses_all_four_routes():
    rng = random.Random(2026)
    assert {rules.sample_hub_exit(rng) for _ in range(100)} == {0, 1, 2, 3}
