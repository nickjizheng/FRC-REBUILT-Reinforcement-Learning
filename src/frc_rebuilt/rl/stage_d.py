"""Stage D official-match helpers for the RL environment.

Stage D uses the full 160-second, 456-ball contract with the accepted starting
pose and exact rules, evaluated on legal score only. This is a thin wiring layer: the
official match model already exists, faithful and tested, in
``frc_rebuilt.rules`` and is already consumed by ``HubRouter`` — the Stage C
RL env merely bypasses it via ``router.sandbox=True`` and by never setting
``router.match_first_inactive``.  This module holds the small pure-Python
pieces that connect ``vec_env`` / the collectors / eval to that model.

Everything here is importable without Isaac; NumPy is only needed by
``pin_prefix_view``.
"""

from __future__ import annotations

import random

from frc_rebuilt.rules import (
    AUTO_DURATION_S,
    MATCH_DURATION_S,
    SCORING_ASSESSMENT_GRACE_S,
    fuel_score_is_eligible,
    hub_active_at,
    hub_in_deactivation_warning,
    select_first_inactive_alliance,
)

# The AUTO result is assessed at 23 s (20 s AUTO + the 3 s scoring grace), the
# same instant the GUI match loop uses (isaac_scene.py match-flow block).  The
# SHIFT 1 decision it produces is needed at t=30 s, so it always lands in time.
AUTO_RESULT_DECISION_S = float(AUTO_DURATION_S + SCORING_ASSESSMENT_GRACE_S)

# The clock scale the frozen FIRST_CYCLE prefix trained on (Stage C episodes).
PREFIX_EPISODE_LEN_S = 90.0

# Legacy proprio channel indices the prefix relies on (see vec_env._observe).
LEGACY_CLOCK_IDX = 7
LEGACY_HUB_ELIGIBLE_IDX = 12

FIRST_INACTIVE_MODES = ("red", "blue", "random", "rules")

# Reward-revision names for the staged bring-up.  D0 deliberately KEEPS
# score_efficiency_v11_rampfree (no reward-contract change; only episode
# length and template move, neither is a route-contract key).  These names
# become active only when cycle_v2.ROUTE_EFFICIENCY_REVISION is flipped at a
# D1/D2 launch — every hardcoded-literal site covered by the revision audit
# must be updated in the same commit (v9-literal lesson).
STAGE_D_REVISIONS = ("stage_d_v1", "stage_d_v2")


def decide_first_inactive(
    mode: str,
    *,
    blue_auto_fuel: int,
    episode_seed: int,
    synthetic_red_auto: tuple[int, int] = (0, 0),
) -> str:
    """Return "red"/"blue": the alliance whose hub rests during SHIFT 1.

    Modes:
      "red" / "blue"  fixed parity (curriculum stages D1a; "red" first-inactive
                      means the BLUE hub is active 0-55 s, matching the Stage C
                      champion's first-dump timing).
      "random"        seeded 50/50 per episode (curriculum stage D1b).
      "rules"         official rule: the higher AUTO scorer rests first, ties
                      broken by a seeded coin (rules.select_first_inactive_alliance).
                      A synthetic red AUTO count stands in for the missing red
                      alliance.  STAGE-D FIX (F2) semantics — an explicit 50/50
                      mixture over ``synthetic_red_auto = (lo, hi)``: half the
                      episodes red scores nothing beyond the floor
                      (``red_auto = lo``, so blue wins the AUTO comparison
                      whenever it out-scores ``lo``), the other half red draws
                      uniformly from ``lo+1..hi`` (red usually wins).  Ties
                      still fall to the seeded coin inside
                      select_first_inactive_alliance, so both parities keep
                      appearing regardless of blue's AUTO output.
    """

    mode = str(mode)
    if mode in ("red", "blue"):
        return mode
    rng = random.Random(int(episode_seed))
    if mode == "random":
        return "red" if rng.random() < 0.5 else "blue"
    if mode != "rules":
        raise ValueError(f"unknown stage_d_first_inactive mode: {mode!r}")
    lo, hi = (int(synthetic_red_auto[0]), int(synthetic_red_auto[1]))
    if lo < 0 or hi < lo:
        raise ValueError(f"invalid synthetic_red_auto range: {(lo, hi)!r}")
    # STAGE-D FIX (F2): a plain uniform draw over lo..hi made "rules" ~98 %
    # red-first-inactive over seeds (a uniform red AUTO count almost always
    # beats blue's small AUTO output).  Explicit 50/50 mixture instead: 50 %
    # red_auto = lo (blue wins the comparison whenever it out-scores lo),
    # 50 % uniform lo+1..hi (red usually wins).  See the docstring.
    if hi > lo:
        red_auto = lo if rng.random() < 0.5 else rng.randint(lo + 1, hi)
    else:
        red_auto = lo
    return select_first_inactive_alliance(
        red_auto, int(blue_auto_fuel), rng=rng
    ).value


def blue_hub_eligible(clock_s: float, first_inactive: str | None) -> bool:
    """Would a FUEL sensed at ``clock_s`` score for blue (incl. the 3 s grace)?

    Mirrors HubRouter._score_eligible for the blue alliance: before the SHIFT 1
    decision (AUTO/TRANSITION) both hubs are active; afterwards the official
    rules.fuel_score_is_eligible applies (active phases plus the 3-second
    scoring-assessment grace after each deactivation).
    """

    t = max(0.0, float(clock_s))
    if first_inactive is None:
        return bool(hub_active_at("blue", min(t, float(MATCH_DURATION_S) - 1e-3)))
    return bool(fuel_score_is_eligible("blue", t, first_inactive))


def blue_hub_obs(clock_s: float, first_inactive: str | None) -> float:
    """Proprio idx-12 encoding of the blue hub state.

    1.0  eligible now (active, or within the 3 s post-deactivation grace);
    0.5  active but inside the official 3 s deactivation warning (dump started
         now will only partially land — the light-pulse cue a human uses);
    0.0  inactive (a ball sensed now scores nothing).
    """

    t = max(0.0, float(clock_s))
    if (
        first_inactive is not None
        and t < float(MATCH_DURATION_S)
        and hub_in_deactivation_warning("blue", t, first_inactive)
    ):
        return 0.5
    return 1.0 if blue_hub_eligible(t, first_inactive) else 0.0


def pin_prefix_view(
    proprio,
    *,
    episode_len_s: float,
    legacy_dim: int = 22,
    prefix_episode_len_s: float = PREFIX_EPISODE_LEN_S,
):
    """Return the frozen prefix's 22-dim view with its Stage-C constants pinned.

    The FIRST_CYCLE prefix trained on (a) a 90 s clock at idx 7 and (b) the
    constant 1.0 "blue hub eligible" at idx 12.  Stage D changes both in the
    stored observation; this helper restores the prefix's training-time view:

      idx 7  = min(clock_s / 90, 1)  (rescaled from clock_s / episode_len_s)
      idx 12 = 1.0

    The suffix and every stored transition keep the TRUE Stage-D values; only
    the tensor handed to ``prefix_agent.act`` is pinned.  Call sites:
    collector_cycle_v2 (prefix act), the GUI policy adapter, and eval scripts.
    """

    import numpy as np

    view = np.array(proprio[:, :legacy_dim], dtype=np.float32, copy=True)
    scale = float(episode_len_s) / float(prefix_episode_len_s)
    view[:, LEGACY_CLOCK_IDX] = np.minimum(view[:, LEGACY_CLOCK_IDX] * scale, 1.0)
    view[:, LEGACY_HUB_ELIGIBLE_IDX] = 1.0
    return view
