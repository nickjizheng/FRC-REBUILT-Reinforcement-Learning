"""Focused tests for the Stage-E production wiring contract:

  1. Anchor sampler EXCLUDES all seven frozen holdout episodes and phase-balances.
  2. PBRS telescopes (Φ=0 at terminal, phase-aware), with no persistent penalty and
     no repeatable crossing bonus.
  3. Drift gate: champion identity passes; drive / shoot / storage regressions stop;
     intake / ferry are reported-not-triggering. Immutable champion SHA-256 guard.
  4. alpha finite and constrained to [1.0, 2.5]; starts at 1.0, rises only after two
     safe held-out windows.

Pure numpy (no Isaac). The anchor-sampler / champion-hash tests use the real frozen
artifacts if present, else skip. Run:
  C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_prefix_takeover_wiring.py -q
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frc_rebuilt.rl import prefix_takeover as pt          # noqa: E402
from frc_rebuilt.rl.spec import CompetitionRLSpec         # noqa: E402

ANCHOR_DIR = ROOT / "runs/phase/stageC_phase_timing_anchor_dev32_anchors"
BOUNDS = {
    "warning": {"drive_l2_p50": 0.35, "shoot_disagree": 0.05, "storage_disagree": 0.08},
    "hard_stop": {"drive_l2_p50": 0.50, "shoot_disagree": 0.08, "storage_disagree": 0.12},
}


# --------------------------- 1. anchor sampler --------------------------- #
@pytest.mark.skipif(not ANCHOR_DIR.exists(), reason="frozen anchor dump not present")
def test_anchor_sampler_excludes_holdout_and_balances():
    s = pt.AnchorSampler(ANCHOR_DIR, pt.FROZEN_HOLDOUT_EPISODES, seed=0)
    # every frozen holdout episode is excluded from the training pool
    assert s.excludes_holdout()
    assert not (set(s.train_episodes) & set(pt.FROZEN_HOLDOUT_EPISODES))
    assert set(s.episode.tolist()).isdisjoint(pt.FROZEN_HOLDOUT_EPISODES)
    # phase-balanced: a batch that is a multiple of #phases is split evenly
    nph = len(s.phases)
    fr, pr, ac = s.sample(nph * 40)
    assert fr.shape[0] == nph * 40 and ac.shape[1] == 7
    counts = Counter(s.phase[s._last_idx].tolist())
    assert max(counts.values()) - min(counts.values()) <= 1, counts


# --------------------------- 2. PBRS telescoping ------------------------- #
def test_pbrs_telescopes_to_minus_phi0_at_terminal():
    shaper = pt.PBRSShaper(gamma=0.99)
    # potentials across a trajectory that includes a curriculum-phase change
    # (values jump when the phase flips at the midpoint) and a TRUE terminal.
    phis = [3.0, 3.5, 4.0, 2.0, 2.5, 1.0]     # last state is terminal -> Phi:=0
    bonus = shaper.trajectory_bonus(phis, terminal=True)
    assert abs(bonus - (-phis[0])) < 1e-9, bonus   # telescopes exactly to -Phi(s0)


def test_pbrs_gamma_must_match_learner():
    # the design note footgun: PBRS gamma must equal the learner's (Stage C = 0.999)
    pt.assert_pbrs_gamma(0.999, 0.999)
    with pytest.raises(ValueError):
        pt.assert_pbrs_gamma(0.997, 0.999)          # the old default vs Stage C
    assert pt.STAGE_C_GAMMA == 0.999
    with pytest.raises(TypeError):
        pt.PBRSShaper()                             # gamma is now REQUIRED (no silent default)


def test_leave_potential_shape_fixed_gateway_and_no_path():
    from frc_rebuilt.field_map import OccupancyGrid
    grid = OccupancyGrid()
    lp = pt.LeavePotential("blue", grid=grid)
    # select g* at a realistic handoff position in the blue court just south of the board
    # (an in-structure coordinate like (2.5,-3.6)/(0,-3.6) correctly yields None -- the
    # robot can't be inside the trench wall or hub centre).
    handoff = (1.5, -3.6)
    g = lp.select_gateway(handoff)
    assert g is not None, "no legal outbound gateway reachable from a normal handoff pos"
    assert g[1] > -pt.LeaveGeom().board_y, "gateway must sit in neutral (past the board)"
    # Φ in [-1, 0]; negative distance potential (closer -> nearer 0)
    phi_h = lp.potential(handoff)
    assert phi_h is not None and -1.0 <= phi_h <= 0.0
    assert lp.potential(g) == 0.0 or abs(lp.potential(g)) < 1e-9   # at g*: d=0 -> Φ=0
    # farther from g* is MORE negative than nearer
    phi_far = lp.potential((2.5, -6.0))
    phi_near = lp.potential((2.5, -3.0))
    assert phi_far is not None and phi_near is not None
    assert phi_far <= phi_h <= phi_near + 1e-9, (phi_far, phi_h, phi_near)
    # g* is FIXED for the segment: querying another position does not reselect it
    assert lp.gstar == g
    # no-path failure returns None (never NaN/inf): an out-of-bounds query
    lp2 = pt.LeavePotential("blue", grid=grid)
    assert lp2.select_gateway((100.0, 100.0)) is None    # nothing reachable -> leave failure
    assert lp2.potential((2.5, -3.6)) is None            # no g* selected -> None


def test_leave_potential_includes_and_selects_physical_trench_lane():
    from frc_rebuilt.field_map import OccupancyGrid

    geom = pt.LeaveGeom()
    assert 3.3582 in geom.lane_x and -3.3582 in geom.lane_x
    assert geom.lane_x == (-3.3582, -1.55, 1.55, 3.3582)
    lp = pt.LeavePotential("blue", geom=geom, grid=OccupancyGrid())
    # The occupancy graph may select the ramp as the shorter fixed PBRS target.
    # A policy is nevertheless successful if it takes the physical trench.
    g = lp.select_gateway((3.1, -7.6))
    assert g is not None
    assert g[0] == pytest.approx(1.55, abs=1e-6)
    assert lp.potential((3.1, -7.6)) is not None
    assert lp.is_legal_crossing((3.609, -2.40))
    assert lp.is_legal_crossing((1.55, -2.40))
    assert lp.crossing_kind((3.609, -2.40)) == "trench"
    assert lp.crossing_kind((1.55, -2.40)) == "ramp"
    assert lp.crossing_kind((-1.55, -2.40)) == "ramp"
    assert not lp.is_legal_crossing((0.0, -2.40))
    assert lp.crossing_kind((0.0, -2.40)) is None
    assert not lp.is_legal_crossing((3.609, -2.60))


def test_leave_gateway_selection_prefers_ramp_unless_trench_is_materially_shorter(monkeypatch):
    import frc_rebuilt.field_map as field_map

    class FreeGrid:
        resolution = 0.1

        @staticmethod
        def is_free(_x, _y):
            return True

    monkeypatch.setattr(field_map, "plan_path", lambda _grid, start, goal: [start, goal])
    geom = pt.LeaveGeom(trench_route_penalty_m=1.25)
    lp = pt.LeavePotential("blue", geom=geom, grid=FreeGrid())
    # The +X trench is geometrically closer here. The preference makes the
    # inner +X ramp the fixed learning target, while the trench remains legal.
    assert lp.select_gateway((3.30, -3.00))[0] == pytest.approx(1.55)
    assert lp.crossing_kind((3.3582, -2.40)) == "trench"


def test_pbrs_no_persistent_penalty_and_no_repeatable_bonus():
    shaper = pt.PBRSShaper(gamma=0.99)
    # A non-terminal there-and-back loop (a -> b -> a) must NOT yield a positive,
    # farmable bonus: it nets to a*(gamma^2 - 1) <= 0.
    a, b = 2.0, 5.0
    loop = shaper.trajectory_bonus([a, b, a], terminal=False)
    assert loop <= 0.0, loop
    # Staying home (constant potential, non-terminal) adds no persistent penalty
    # beyond the single potential-difference term (telescoping), never a per-step cost.
    flat = shaper.trajectory_bonus([1.0, 1.0, 1.0, 1.0], terminal=False)
    assert abs(flat - (0.99 ** 3 * 1.0 - 1.0)) < 1e-9   # = gamma^(T)*phi_T - phi_0 form


# --------------------------- 3. drift gate ------------------------------- #
def _acts(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (n, 7)).astype(np.float32)
    a[:, 3:] = -0.5          # all discrete heads OFF (below 0.25 threshold) by default
    return a


def test_drift_identity_passes():
    spec = CompetitionRLSpec()
    champ = _acts(400)
    r = pt.drift_decision(champ, champ.copy(), spec, BOUNDS)
    assert r["drive_l2_p50"] == 0.0 and not r["hard_stop"] and not r["warning"]
    assert r["intake_disagree"] == 0 and r["ferry_disagree"] == 0


def test_drift_drive_regression_hard_stops():
    spec = CompetitionRLSpec()
    champ = _acts(400)
    cand = champ.copy(); cand[:, :3] += 1.0          # ~sqrt(3) drive shift
    r = pt.drift_decision(champ, cand, spec, BOUNDS)
    assert r["drive_l2_p50"] > 0.5 and r["hard_stop"]


def test_drift_storage_and_shoot_regressions_hard_stop():
    spec = CompetitionRLSpec()
    champ = _acts(400)
    # flip storage (idx4) ON for 20% of states -> storage_disagree 0.20 > 0.12
    cand = champ.copy(); cand[:80, 4] = 1.0
    rs = pt.drift_decision(champ, cand, spec, BOUNDS)
    assert rs["storage_disagree"] >= 0.19 and rs["hard_stop"]
    # flip shoot (idx5) ON for 15% -> shoot_disagree 0.15 > 0.08
    cand2 = champ.copy(); cand2[:60, 5] = 1.0
    rsh = pt.drift_decision(champ, cand2, spec, BOUNDS)
    assert rsh["shoot_disagree"] >= 0.14 and rsh["hard_stop"]


def test_drift_intake_ferry_reported_not_triggering():
    spec = CompetitionRLSpec()
    champ = _acts(400)
    # flip ONLY intake (idx3) for 50% and ferry (idx6) for 50%; drive/shoot/storage same
    cand = champ.copy(); cand[:200, 3] = 1.0; cand[100:300, 6] = 1.0
    r = pt.drift_decision(champ, cand, spec, BOUNDS)
    assert r["intake_disagree"] > 0.1 and r["ferry_disagree"] > 0.1
    assert not r["hard_stop"], "intake/ferry must NOT trigger a hard stop (design note contract)"


# --------------------------- immutable champion -------------------------- #
@pytest.mark.skipif(not (ROOT / "runs/stageC_champion_998753.pt").exists(),
                    reason="champion checkpoint not present")
def test_immutable_champion_guard():
    sha = pt.immutable_champion_ok()
    assert len(sha) == 64
    assert pt.immutable_champion_ok(expected_sha=sha) == sha        # unchanged -> ok
    with pytest.raises(RuntimeError):
        pt.immutable_champion_ok(expected_sha="0" * 64)            # changed -> raises


# --------------------------- 4. alpha schedule -------------------------- #
def test_check_alpha_bounds():
    assert pt.check_alpha(1.0) == 1.0 and pt.check_alpha(2.5) == 2.5
    for bad in (0.99, 2.51, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pt.check_alpha(bad)


def test_alpha_schedule_starts_at_one_rises_after_two_safe_windows():
    sch = pt.AlphaSchedule(start=1.0, end=2.5, increment=0.25, warmup_windows=2)
    assert sch.alpha == 1.0
    assert sch.on_window(success=True, safe_drift=True) == 1.0     # 1 safe window
    assert sch.on_window(success=True, safe_drift=True) == 1.0     # 2 safe windows
    assert sch.on_window(success=True, safe_drift=True) == 1.25    # now it rises
    # a bad window resets the warmup and holds alpha
    a = sch.on_window(success=False, safe_drift=True)
    assert a == 1.25
    assert sch.on_window(success=True, safe_drift=True) == 1.25    # counting restarts
    # never exceeds the ceiling and always validates
    for _ in range(50):
        a = sch.on_window(True, True)
    assert a == 2.5 and pt.check_alpha(a) == 2.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
