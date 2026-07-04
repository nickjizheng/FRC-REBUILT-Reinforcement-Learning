"""Official 2026 FRC REBUILT match rules used by the simulator.

The constants in this module use SI units for simulation.  Source-unit values
are retained where useful so that the conversion back to the game manual is
unambiguous.  See ``docs/RULES.md`` for citations and modelling notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Sequence


# Unit conversions ---------------------------------------------------------

METERS_PER_INCH = 0.0254
KILOGRAMS_PER_POUND = 0.45359237


# Official field and game-piece constants (2026 Game Manual, sections 5.2-
# 5.11 and 6.3.4).  Dimensions described as approximate in the manual remain
# nominal here; collision geometry should come from the official CAD.

FIELD_LENGTH_IN = 651.2
FIELD_WIDTH_IN = 317.7
FIELD_LENGTH_M = FIELD_LENGTH_IN * METERS_PER_INCH
FIELD_WIDTH_M = FIELD_WIDTH_IN * METERS_PER_INCH

HUB_COUNT = 2
HUB_FOOTPRINT_IN = 47.0
HUB_FOOTPRINT_M = HUB_FOOTPRINT_IN * METERS_PER_INCH
HUB_DISTANCE_FROM_ALLIANCE_WALL_IN = 158.6
HUB_DISTANCE_FROM_ALLIANCE_WALL_M = (
    HUB_DISTANCE_FROM_ALLIANCE_WALL_IN * METERS_PER_INCH
)
HUB_OPENING_IN = 41.7
HUB_OPENING_M = HUB_OPENING_IN * METERS_PER_INCH
HUB_OPENING_FRONT_HEIGHT_IN = 72.0
HUB_OPENING_FRONT_HEIGHT_M = HUB_OPENING_FRONT_HEIGHT_IN * METERS_PER_INCH
HUB_EXIT_COUNT = 4

BUMP_COUNT = 4
BUMP_WIDTH_IN = 73.0
BUMP_DEPTH_IN = 44.4
BUMP_HEIGHT_IN = 6.513
BUMP_RAMP_ANGLE_DEG = 15.0
BUMP_WIDTH_M = BUMP_WIDTH_IN * METERS_PER_INCH
BUMP_DEPTH_M = BUMP_DEPTH_IN * METERS_PER_INCH
BUMP_HEIGHT_M = BUMP_HEIGHT_IN * METERS_PER_INCH

TRENCH_COUNT = 4
TRENCH_WIDTH_IN = 65.65
TRENCH_DEPTH_IN = 47.0
TRENCH_HEIGHT_IN = 40.25
TRENCH_CLEAR_WIDTH_IN = 50.34
TRENCH_CLEAR_HEIGHT_IN = 22.25
TRENCH_WIDTH_M = TRENCH_WIDTH_IN * METERS_PER_INCH
TRENCH_DEPTH_M = TRENCH_DEPTH_IN * METERS_PER_INCH
TRENCH_HEIGHT_M = TRENCH_HEIGHT_IN * METERS_PER_INCH
TRENCH_CLEAR_WIDTH_M = TRENCH_CLEAR_WIDTH_IN * METERS_PER_INCH
TRENCH_CLEAR_HEIGHT_M = TRENCH_CLEAR_HEIGHT_IN * METERS_PER_INCH

DEPOT_COUNT = 2
OUTPOST_COUNT = 2
TOWER_COUNT = 2

APRILTAG_COUNT = 32
APRILTAG_FAMILY = "36h11"
APRILTAG_SIZE_IN = 8.125
APRILTAG_SIZE_M = APRILTAG_SIZE_IN * METERS_PER_INCH

FUEL_DIAMETER_IN = 5.91
FUEL_DIAMETER_M = 0.150  # the manual's metric specification
FUEL_RADIUS_M = FUEL_DIAMETER_M / 2.0
FUEL_MASS_MIN_LB = 0.448
FUEL_MASS_MAX_LB = 0.500
FUEL_MASS_MIN_KG = FUEL_MASS_MIN_LB * KILOGRAMS_PER_POUND
FUEL_MASS_MAX_KG = FUEL_MASS_MAX_LB * KILOGRAMS_PER_POUND
FUEL_MASS_NOMINAL_KG = (FUEL_MASS_MIN_KG + FUEL_MASS_MAX_KG) / 2.0

OFFICIAL_MATCH_FUEL = 504
FUEL_PER_DEPOT = 24
FUEL_PER_OUTPOST = 24
MAX_FUEL_PRELOAD_PER_ROBOT = 8
ROBOTS_PER_MATCH = 6
MAX_TOTAL_PRELOADED_FUEL = MAX_FUEL_PRELOAD_PER_ROBOT * ROBOTS_PER_MATCH
NEUTRAL_FUEL_WITH_MAX_PRELOAD = 360
NEUTRAL_FUEL_WITH_NO_PRELOAD = 408
NEUTRAL_FUEL_COUNT_TOLERANCE = 24
CHAMPIONSHIP_MAX_MATCH_FUEL = 600


@dataclass(frozen=True)
class FuelDistribution:
    """Official pre-match FUEL staging counts."""

    total: int
    depots: int
    outposts: int
    robot_preloads: int
    neutral_zone: int

    def __post_init__(self) -> None:
        if min(
            self.total,
            self.depots,
            self.outposts,
            self.robot_preloads,
            self.neutral_zone,
        ) < 0:
            raise ValueError("FUEL counts cannot be negative")
        staged = self.depots + self.outposts + self.robot_preloads + self.neutral_zone
        if staged != self.total:
            raise ValueError(f"staged FUEL ({staged}) does not equal total ({self.total})")


def fuel_distribution(total_preloaded: int = MAX_TOTAL_PRELOADED_FUEL) -> FuelDistribution:
    """Return the official 504-FUEL staging for a preload count from 0 to 48."""

    if isinstance(total_preloaded, bool) or not isinstance(total_preloaded, int):
        raise TypeError("total_preloaded must be an integer")
    if not 0 <= total_preloaded <= MAX_TOTAL_PRELOADED_FUEL:
        raise ValueError(
            f"total_preloaded must be between 0 and {MAX_TOTAL_PRELOADED_FUEL}"
        )
    depots = DEPOT_COUNT * FUEL_PER_DEPOT
    outposts = OUTPOST_COUNT * FUEL_PER_OUTPOST
    neutral = OFFICIAL_MATCH_FUEL - depots - outposts - total_preloaded
    return FuelDistribution(
        total=OFFICIAL_MATCH_FUEL,
        depots=depots,
        outposts=outposts,
        robot_preloads=total_preloaded,
        neutral_zone=neutral,
    )


NOMINAL_FUEL_DISTRIBUTION = fuel_distribution()


# Match timing -------------------------------------------------------------

AUTO_DURATION_S = 20.0
TELEOP_DURATION_S = 140.0
MATCH_DURATION_S = AUTO_DURATION_S + TELEOP_DURATION_S
TRANSITION_DURATION_S = 10.0
SHIFT_DURATION_S = 25.0
SHIFT_COUNT = 4
ENDGAME_DURATION_S = 30.0
SCORING_ASSESSMENT_GRACE_S = 3.0
HUB_DEACTIVATION_WARNING_S = 3.0
POST_MATCH_ASSESSMENT_END_S = MATCH_DURATION_S + SCORING_ASSESSMENT_GRACE_S


class Alliance(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def opponent(self) -> "Alliance":
        return Alliance.BLUE if self is Alliance.RED else Alliance.RED


class MatchPhase(str, Enum):
    AUTO = "auto"
    TRANSITION = "transition"
    SHIFT_1 = "shift_1"
    SHIFT_2 = "shift_2"
    SHIFT_3 = "shift_3"
    SHIFT_4 = "shift_4"
    ENDGAME = "endgame"
    POST_MATCH_ASSESSMENT = "post_match_assessment"
    COMPLETE = "complete"


class HubLightState(str, Enum):
    """Official 2026 HUB DMX indications used by the field renderer."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    WARNING = "warning"
    TRANSITION_CHASE = "transition_chase"
    POST_MATCH_WHITE = "post_match_white"


@dataclass(frozen=True)
class MatchSegment:
    phase: MatchPhase
    start_s: float
    end_s: float

    def contains(self, elapsed_s: float) -> bool:
        return self.start_s <= elapsed_s < self.end_s


MATCH_SEGMENTS: tuple[MatchSegment, ...] = (
    MatchSegment(MatchPhase.AUTO, 0.0, 20.0),
    MatchSegment(MatchPhase.TRANSITION, 20.0, 30.0),
    MatchSegment(MatchPhase.SHIFT_1, 30.0, 55.0),
    MatchSegment(MatchPhase.SHIFT_2, 55.0, 80.0),
    MatchSegment(MatchPhase.SHIFT_3, 80.0, 105.0),
    MatchSegment(MatchPhase.SHIFT_4, 105.0, 130.0),
    MatchSegment(MatchPhase.ENDGAME, 130.0, 160.0),
)

ALLIANCE_SHIFT_PHASES = (
    MatchPhase.SHIFT_1,
    MatchPhase.SHIFT_2,
    MatchPhase.SHIFT_3,
    MatchPhase.SHIFT_4,
)


def _coerce_alliance(alliance: Alliance | str) -> Alliance:
    try:
        return alliance if isinstance(alliance, Alliance) else Alliance(alliance.lower())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"unknown alliance: {alliance!r}") from exc


def phase_at(elapsed_s: float) -> MatchPhase:
    """Return the half-open match phase at seconds elapsed from AUTO start."""

    if elapsed_s < 0:
        raise ValueError("elapsed_s cannot be negative")
    for segment in MATCH_SEGMENTS:
        if segment.contains(elapsed_s):
            return segment.phase
    if elapsed_s < POST_MATCH_ASSESSMENT_END_S:
        return MatchPhase.POST_MATCH_ASSESSMENT
    return MatchPhase.COMPLETE


def arena_timer_seconds(elapsed_s: float) -> float:
    """Return the arena countdown value (AUTO and TELEOP use separate clocks)."""

    if elapsed_s < 0:
        raise ValueError("elapsed_s cannot be negative")
    if elapsed_s < AUTO_DURATION_S:
        return AUTO_DURATION_S - elapsed_s
    if elapsed_s < MATCH_DURATION_S:
        return MATCH_DURATION_S - elapsed_s
    return 0.0


def select_first_inactive_alliance(
    red_auto_fuel: int,
    blue_auto_fuel: int,
    *,
    seed: int | float | str | bytes | bytearray | None = None,
    rng: random.Random | None = None,
) -> Alliance:
    """Select the SHIFT 1 inactive HUB from AUTO FUEL results.

    The higher AUTO scorer becomes inactive first.  A tie is selected randomly,
    matching FMS; pass ``seed`` (or a dedicated ``rng``) for reproducible runs.
    """

    if red_auto_fuel < 0 or blue_auto_fuel < 0:
        raise ValueError("AUTO FUEL counts cannot be negative")
    if red_auto_fuel > blue_auto_fuel:
        return Alliance.RED
    if blue_auto_fuel > red_auto_fuel:
        return Alliance.BLUE
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    chooser = rng if rng is not None else random.Random(seed)
    return chooser.choice((Alliance.RED, Alliance.BLUE))


def hub_is_active(
    alliance: Alliance | str,
    phase: MatchPhase,
    first_inactive: Alliance | str | None = None,
) -> bool:
    """Return official HUB status, excluding post-period scoring assessment."""

    alliance = _coerce_alliance(alliance)
    if phase in (MatchPhase.AUTO, MatchPhase.TRANSITION, MatchPhase.ENDGAME):
        return True
    if phase in (MatchPhase.POST_MATCH_ASSESSMENT, MatchPhase.COMPLETE):
        return False
    if phase not in ALLIANCE_SHIFT_PHASES:
        raise ValueError(f"unsupported match phase: {phase!r}")
    if first_inactive is None:
        raise ValueError("first_inactive is required during ALLIANCE SHIFTS")
    first_inactive = _coerce_alliance(first_inactive)
    shift_index = ALLIANCE_SHIFT_PHASES.index(phase) + 1
    inactive = first_inactive if shift_index % 2 else first_inactive.opponent
    return alliance is not inactive


def hub_active_at(
    alliance: Alliance | str,
    elapsed_s: float,
    first_inactive: Alliance | str | None = None,
) -> bool:
    """Convenience wrapper around :func:`phase_at` and :func:`hub_is_active`."""

    return hub_is_active(alliance, phase_at(elapsed_s), first_inactive)


def hub_deactivation_times(
    alliance: Alliance | str, first_inactive: Alliance | str
) -> tuple[float, ...]:
    """Return match times when this HUB deactivates, including the final buzzer."""

    alliance = _coerce_alliance(alliance)
    first_inactive = _coerce_alliance(first_inactive)
    shift_deactivations = (30.0, 80.0) if alliance is first_inactive else (55.0, 105.0)
    return (*shift_deactivations, MATCH_DURATION_S)


def hub_in_deactivation_warning(
    alliance: Alliance | str,
    elapsed_s: float,
    first_inactive: Alliance | str,
) -> bool:
    """Whether the HUB should pulse during its official 3-second warning."""

    if elapsed_s < 0 or elapsed_s >= MATCH_DURATION_S:
        return False
    return any(
        deactivation - HUB_DEACTIVATION_WARNING_S <= elapsed_s < deactivation
        for deactivation in hub_deactivation_times(alliance, first_inactive)
    )


def hub_light_state_at(
    alliance: Alliance | str,
    elapsed_s: float,
    first_inactive: Alliance | str | None = None,
) -> HubLightState:
    """Return the official visible HUB-light state at a match timestamp.

    The three-second deactivation pulse takes precedence over the TRANSITION
    white chase during the final three seconds before SHIFT 1.  This keeps the
    imminent scoring-state change unambiguous while preserving the chase for
    the first seven seconds of TRANSITION.
    """

    alliance = _coerce_alliance(alliance)
    phase = phase_at(elapsed_s)
    if phase is MatchPhase.POST_MATCH_ASSESSMENT:
        return HubLightState.POST_MATCH_WHITE
    if phase is MatchPhase.COMPLETE:
        return HubLightState.INACTIVE
    if first_inactive is not None and hub_in_deactivation_warning(
        alliance, elapsed_s, first_inactive
    ):
        return HubLightState.WARNING
    if phase is MatchPhase.TRANSITION and first_inactive is not None:
        if alliance is _coerce_alliance(first_inactive):
            return HubLightState.TRANSITION_CHASE
    if hub_is_active(alliance, phase, first_inactive):
        return HubLightState.ACTIVE
    return HubLightState.INACTIVE


def fuel_score_is_eligible(
    alliance: Alliance | str,
    sensor_elapsed_s: float,
    first_inactive: Alliance | str,
) -> bool:
    """Apply active-HUB status and the 3-second processing grace window.

    ``sensor_elapsed_s`` is when the HUB sensor assesses the FUEL.  This models
    the manual's processing allowance after HUB deactivation and after TELEOP.
    AUTO bookkeeping should separately retain shots assessed through t=23 s as
    AUTO FUEL when they entered the HUB during AUTO.
    """

    if sensor_elapsed_s < 0:
        return False
    if sensor_elapsed_s < MATCH_DURATION_S and hub_active_at(
        alliance, sensor_elapsed_s, first_inactive
    ):
        return True
    return any(
        0.0 <= sensor_elapsed_s - deactivation <= SCORING_ASSESSMENT_GRACE_S
        for deactivation in hub_deactivation_times(alliance, first_inactive)
    )


# HUB routing --------------------------------------------------------------

# The manual specifies random distribution through four exits and a 3-second
# scoring-assessment window, but not the internal travel-time distribution.
# 1.32 s is the xRC-derived calibration target for this simulator.  A triangular
# distribution gives exactly that expectation while keeping every sample <= 3 s.
HUB_ROUTING_DELAY_MEAN_S = 1.32
HUB_ROUTING_DELAY_MIN_S = 0.0
HUB_ROUTING_DELAY_MAX_S = SCORING_ASSESSMENT_GRACE_S
HUB_ROUTING_DELAY_MODE_S = (
    3.0 * HUB_ROUTING_DELAY_MEAN_S
    - HUB_ROUTING_DELAY_MIN_S
    - HUB_ROUTING_DELAY_MAX_S
)


def sample_hub_routing_delay(rng: random.Random | None = None) -> float:
    """Sample a bounded HUB-to-exit delay with a 1.32-second expectation."""

    generator = rng if rng is not None else random
    value = generator.triangular(
        HUB_ROUTING_DELAY_MIN_S,
        HUB_ROUTING_DELAY_MAX_S,
        HUB_ROUTING_DELAY_MODE_S,
    )
    # Defensive clamps make the physics contract explicit across RNG backends.
    return max(HUB_ROUTING_DELAY_MIN_S, min(HUB_ROUTING_DELAY_MAX_S, value))


def sample_hub_exit(rng: random.Random | None = None) -> int:
    """Randomly select one of the HUB's four neutral-zone exits (0 through 3)."""

    generator = rng if rng is not None else random
    return generator.randrange(HUB_EXIT_COUNT)
