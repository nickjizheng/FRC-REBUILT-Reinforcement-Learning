"""Tests for the aggressive multi-cycle reward (AggressiveCycleShaper, user directive
2026-07-13, "bonus + gated penalty" variant). Pure-Python, no Isaac.

Covers: Term A1 pays FRESH scored balls on a full clear (not load, min-load gated);
A1 never fires without a full clear; Term A2 gated abandon penalty (cross out with a big
undumped load, latch-gated); Term B linger penalty (latch-gated, grace, high-water
ratchet vs oscillation, score-freeze); Term C escalating ordered cycle bonus (rejects
home-recycle + shallow dips, escalates, episode cap); champion first cycle earns the
bonus and eats NO penalty (hard constraint); reset on done.
Run: C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_aggressive_cycle.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frc_rebuilt.rl.prefix_takeover import AggressiveCycleShaper, AggressiveCycleCfg  # noqa: E402


def _latch(sh, fresh=30):
    """Drive a productive first unload so A2/B/C arm. Returns after the clear step."""
    sh.update(mag=fresh, score=0, collected=fresh, fresh_score=0, y=-3.6, done=False)
    return sh.update(mag=0, score=fresh, collected=fresh, fresh_score=fresh, y=-3.6, done=False)


def test_atonce_bonus_counts_fresh_not_load():
    # big clean dump: 46 fresh -> +46 (== fresh, capped at 50)
    sh = AggressiveCycleShaper()
    sh.update(46, 0, 46, 0, -3.6, False)
    r = sh.update(0, 46, 46, 46, -3.6, False)
    assert abs(r - 46.0) < 1e-6

    # load 8 but only 5 fresh score (3 ferried/missed) -> pays FRESH (5), not load (8)
    sh2 = AggressiveCycleShaper()
    sh2.update(8, 0, 8, 0, -3.6, False)
    r2 = sh2.update(0, 5, 8, 5, -3.6, False)
    assert abs(r2 - 5.0) < 1e-6

    # fresh below min_load(4) -> no bonus (blocks micro-clear farming)
    sh3 = AggressiveCycleShaper()
    sh3.update(8, 0, 8, 0, -3.6, False)
    r3 = sh3.update(0, 3, 8, 3, -3.6, False)
    assert r3 == 0.0


def test_atonce_only_on_full_clear():
    # magazine never reaches 0 (perpetual dribbler): A1 never fires
    sh = AggressiveCycleShaper()
    rs = [sh.update(8, i, 20, i, -3.6, False) for i in range(1, 15)]  # mag stuck at 8
    assert all(r == 0.0 for r in rs)


def test_atonce_cap():
    sh = AggressiveCycleShaper()
    sh.update(70, 0, 70, 0, -3.6, False)
    r = sh.update(0, 70, 70, 70, -3.6, False)      # 70 fresh, cap 50
    assert abs(r - 50.0) < 1e-6


def test_gated_abandon_penalty():
    # isolate the abandon penalty from the per-leg leave bonus
    C = AggressiveCycleCfg(leave_bonus=0.0)
    # post-latch: carry a big load OUT to neutral -> penalty -abandon_weight*mag; the big load
    # (>=abandon_load) also blocks the leave, so no cycle-start bonus even if leave_bonus>0.
    sh = AggressiveCycleShaper(C)
    _latch(sh)
    sh.update(10, 30, 50, 30, -3.2, False)          # loaded, home (band below)
    r = sh.update(10, 30, 50, 30, -2.4, False)      # crossed out (>= -2.50) holding 10
    assert abs(r - (-0.3 * 10)) < 1e-6

    # empty outbound -> no abandon penalty (and leave_bonus zeroed here -> exactly 0)
    sh2 = AggressiveCycleShaper(C)
    _latch(sh2)
    sh2.update(0, 30, 40, 30, -3.2, False)
    r2 = sh2.update(0, 30, 40, 30, -2.4, False)
    assert r2 == 0.0

    # PRE-latch outbound with a load -> no penalty (gated on the productive-unload latch)
    sh3 = AggressiveCycleShaper(C)
    sh3.update(10, 0, 10, 0, -3.2, False)
    r3 = sh3.update(10, 0, 10, 0, -2.4, False)
    assert r3 == 0.0


def test_linger_grace_and_flat_penalty():
    sh = AggressiveCycleShaper(AggressiveCycleCfg(linger_penalty=0.12, linger_grace_steps=12))
    _latch(sh)                                       # in_dwell starts, counter 0
    rs = [sh.update(0, 30, 40, 30, -3.6, False) for _ in range(20)]
    # counter increments each camping step; penalty only when counter > grace(12)
    penal = [r for r in rs if r < 0]
    assert len(penal) == 20 - 12                     # steps 13..20 penalized
    assert all(abs(r + 0.12) < 1e-6 for r in penal)


def test_linger_highwater_ratchet_blocks_oscillation():
    sh = AggressiveCycleShaper(AggressiveCycleCfg(linger_penalty=0.12, linger_grace_steps=12))
    _latch(sh)
    for _ in range(13):
        sh.update(0, 30, 40, 30, -3.6, False)        # camp -> counter climbs into penalty
    r_nudge = sh.update(0, 30, 40, 30, -3.5, False)  # NEW high -3.5 -> progressing -> reset, no penalty
    assert r_nudge == 0.0
    # re-nudge to the SAME height -3.5 is NOT a new high -> counter climbs again (no free reset)
    r_same = sh.update(0, 30, 40, 30, -3.5, False)
    assert r_same == 0.0                             # counter just 1 now (<= grace) but proves no exploit path:
    # drift back below and camp -> penalty returns after grace (ratchet kept the pressure)
    back = [sh.update(0, 30, 40, 30, -3.6, False) for _ in range(13)]
    assert any(r < 0 for r in back)


def test_linger_frozen_while_scoring():
    sh = AggressiveCycleShaper()
    _latch(sh)
    # trailing first-load landings: score keeps rising while home+empty -> NO linger penalty
    rs = [sh.update(0, 30 + i, 40, 30 + i, -3.6, False) for i in range(1, 20)]
    assert all(r >= 0.0 for r in rs)


def test_multicycle_ordered_bonus_and_escalation():
    # pin Term-C params + zero the per-leg bonuses so we isolate A1+C on the clear step
    sh = AggressiveCycleShaper(AggressiveCycleCfg(
        mc_per_ball=1.5, mc_cap=30.0, leave_bonus=0.0, collect_bonus=0.0, return_bonus=0.0))
    _latch(sh, fresh=30)                              # fresh_at_last_empty = 30
    # additional cycle: out -> deep collect -> return loaded -> full clear
    sh.update(0, 30, 40, 30, -2.4, False)            # outbound (empty) -> stage 1
    assert sh.stage == 1
    sh.update(8, 30, 48, 30, -1.0, False)            # deep neutral (y>-1.5), collected rose -> stage 2
    assert sh.stage == 2
    sh.update(8, 30, 48, 30, -3.2, False)            # inbound loaded -> stage 3
    assert sh.stage == 3
    r1 = sh.update(0, 38, 48, 38, -3.6, False)       # clear, 8 fresh this cycle
    assert sh.cycles_completed == 1
    # A1: fresh_delta 38-30=8 -> +8 ; C: min(30,1.5*8=12)*1.2^0=12 -> +12
    assert abs(r1 - (8.0 + 12.0)) < 1e-6

    # second additional cycle -> escalation 1.2^1
    sh.update(0, 38, 48, 38, -2.4, False)
    sh.update(8, 38, 56, 38, -1.0, False)
    sh.update(8, 38, 56, 38, -3.2, False)
    r2 = sh.update(0, 46, 56, 46, -3.6, False)
    assert sh.cycles_completed == 2
    assert abs(r2 - (8.0 + 12.0 * 1.2)) < 1e-6       # 8 (A1) + 14.4 (C escalated)


def test_leg_progress_bonuses_fire_once():
    # aggressive per-leg journey rewards: +bonus for leave/collect/return, ONCE per episode
    cfg = AggressiveCycleCfg(leave_bonus=15.0, collect_bonus=15.0, return_bonus=15.0,
                             linger_penalty=0.0)     # isolate the leg bonuses
    sh = AggressiveCycleShaper(cfg)
    _latch(sh, fresh=30)
    assert abs(sh.update(0, 30, 40, 30, -2.4, False) - 15.0) < 1e-6   # leave -> +15
    assert abs(sh.update(8, 30, 48, 30, -1.0, False) - 15.0) < 1e-6   # collect -> +15
    assert abs(sh.update(8, 30, 48, 30, -3.2, False) - 15.0) < 1e-6   # return -> +15
    sh.update(0, 38, 48, 38, -3.6, False)            # clear (completes the cycle, re-arms)
    assert sh.update(0, 38, 48, 38, -2.4, False) == 0.0   # 2nd leave -> NO re-award (once/ep)


def test_multicycle_rejects_recycle_and_shallow_dip():
    # home recycle (no field trip): C never pays
    sh = AggressiveCycleShaper()
    _latch(sh)
    sh.update(3, 30, 43, 30, -3.6, False)
    sh.update(0, 33, 43, 33, -3.6, False)
    assert sh.cycles_completed == 0

    # shallow dip: crosses out but only to y=-2.0 (never past neutral_deep_y=-1.5)
    sh2 = AggressiveCycleShaper()
    _latch(sh2)
    sh2.update(0, 30, 40, 30, -2.4, False)           # stage 1
    sh2.update(5, 30, 45, 30, -2.0, False)           # collected rose but NOT deep -> stays stage 1
    assert sh2.stage == 1
    sh2.update(5, 30, 45, 30, -3.2, False)
    sh2.update(0, 35, 45, 35, -3.6, False)
    assert sh2.cycles_completed == 0


def test_multicycle_min_cycle_and_episode_cap():
    # a 1-ball "cycle" pays no C bonus (min_cycle_score=2)
    sh = AggressiveCycleShaper()
    _latch(sh)
    sh.update(0, 30, 40, 30, -2.4, False)
    sh.update(1, 30, 41, 30, -1.0, False)
    sh.update(1, 30, 41, 30, -3.2, False)
    sh.update(0, 31, 41, 31, -3.6, False)            # only 1 fresh this cycle
    assert sh.cycles_completed == 0

    # episode cap: many big cycles cannot exceed mc_episode_cap
    sh2 = AggressiveCycleShaper(AggressiveCycleCfg(mc_episode_cap=20.0))
    _latch(sh2)
    total_c = 0.0
    fs = 30
    for _ in range(4):
        sh2.update(0, fs, 100, fs, -2.4, False)
        sh2.update(10, fs, 100 + 10, fs, -1.0, False)
        sh2.update(10, fs, 100 + 10, fs, -3.2, False)
        r = sh2.update(0, fs + 10, 110, fs + 10, -3.6, False)
        # C part = r - A1 part; A1 = min(50, fresh_delta) ; here fresh_delta = 10 -> A1=10
        total_c += (r - 10.0)
        fs += 10
    assert total_c <= 20.0 + 1e-6


def test_champion_first_cycle_no_penalty():
    """Hard constraint: a champion-style first cycle (collect ~46, stream-dump to empty,
    then leave) earns the full-clear bonus and eats ZERO penalty."""
    sh = AggressiveCycleShaper()
    rs = []
    for m in range(0, 46, 5):                        # collect out in neutral, empty->loaded
        rs.append(sh.update(m, 0, m, 0, 0.5, False))
    rs.append(sh.update(45, 0, 46, 0, -3.2, False))  # cross inbound, loaded
    fresh = 0
    for m in (35, 25, 15, 5, 0):                     # stream-dump, score/fresh rise together
        fresh = 46 - m
        rs.append(sh.update(m, fresh, 46, fresh, -3.5, False))
    for y in (-3.4, -3.0, -2.6, -2.0):               # drive out to leave
        rs.append(sh.update(0, 46, 46, 46, y, False))
    assert max(rs) >= 40.0                            # the A1 full-clear bonus landed
    assert min(rs) >= -1e-9                           # NO penalty anywhere in cycle 1
    assert sum(rs) > 40.0


def test_phase_report_timeline():
    sh = AggressiveCycleShaper()
    # 1st-cycle unload -> latch (records t_unload; still stage 0 = hasn't left)
    sh.update(30, 0, 40, 0, -3.6, False)
    sh.update(0, 30, 40, 30, -3.6, False)
    rep0 = sh.phase_report()
    assert rep0["t_unload"] is not None and rep0["max_stage"] == 0
    # ordered 2nd cycle: leave -> deep collect -> return loaded -> clear/score
    sh.update(0, 30, 40, 30, -2.4, False)          # left -> stage 1
    assert sh.phase_report()["max_stage"] == 1 and sh.phase_report()["t_leave"] is not None
    sh.update(8, 30, 48, 30, -1.0, False)          # collected deep -> 2
    sh.update(8, 30, 48, 30, -3.2, False)          # returned loaded -> 3
    sh.update(0, 38, 48, 38, -3.6, False)          # scored again -> 4
    rep = sh.phase_report()
    assert rep["max_stage"] == 4
    ts = [rep["t_unload"], rep["t_leave"], rep["t_collect"], rep["t_return"], rep["t_score2"]]
    assert all(t is not None for t in ts) and ts == sorted(ts)   # monotone timeline
    assert rep["cycles_completed"] == 1 and rep["n_leaves"] == 1


def test_phase_report_partial_leave_only():
    # left the field but never completed -> max_stage 1, downstream times None
    sh = AggressiveCycleShaper()
    sh.update(30, 0, 40, 0, -3.6, False)
    sh.update(0, 30, 40, 30, -3.6, False)
    sh.update(0, 30, 40, 30, -2.4, False)          # leave only
    rep = sh.phase_report()
    assert rep["max_stage"] == 1 and rep["t_leave"] is not None
    assert rep["t_collect"] is None and rep["t_score2"] is None and rep["cycles_completed"] == 0


def test_reset_on_done():
    sh = AggressiveCycleShaper()
    sh.update(30, 0, 40, 0, -3.6, False)
    sh.update(0, 30, 40, 30, -3.6, True)             # done -> reset
    assert not sh.latched and sh.stage == 0 and sh.cycles_completed == 0


# --------------------------------------------------------------------------- #
# Term D — time-decay ramps + ramp-lane preference (2026-07-14 user directive)
# --------------------------------------------------------------------------- #
# Term D defaults to no-op; every test below explicitly enables the pieces it exercises.
CORE = dict(core_slope=0.005, core_step_cap=0.50, core_grace_steps=8,
            core_freeze_confirm=15, core_freeze_cap=30,
            leave_bonus=0.0, collect_bonus=0.0, return_bonus=0.0, linger_penalty=0.0)


def test_rampA_monotone_and_leave_does_not_stop_it():
    # Ramp A grows monotonically with idle-since-empty, and crossing OUT (leave) does NOT
    # stop it -- only the deep collect (stage 1->2) does.
    sh = AggressiveCycleShaper(AggressiveCycleCfg(**CORE))
    _latch(sh, fresh=30)
    home = [-sh.update(0, 30, 30, 30, -3.6, False) for _ in range(40)]   # idle home, score flat
    assert home[-1] > 0.0                                                # charging by now
    assert all(b >= a - 1e-9 for a, b in zip(home, home[1:]))            # monotone non-decreasing
    before = home[-1]
    sh.update(0, 30, 30, 30, -2.4, False)                               # cross OUT (leave), stage 1
    post = [-sh.update(0, 30, 30, 30, -2.0, False) for _ in range(5)]    # in neutral, not collected
    assert all(p >= before - 1e-9 for p in post)                        # still charging, still growing


def test_rampA_latch_gated_and_champion_cycle1_zero():
    # Cycle 1 (collect -> full dump -> trailing landings -> brief quiet) eats NO ramp penalty:
    # pre-latch is ungated, and freeze-confirm + grace cover the dump tail.
    sh = AggressiveCycleShaper(AggressiveCycleCfg(**CORE))
    rs = [sh.update(mag=k + 1, score=0, collected=k + 1, fresh_score=0, y=-3.6, done=False)
          for k in range(30)]                                            # collecting, pre-latch
    rs.append(sh.update(0, 40, 40, 40, -3.6, False))                    # full dump -> latch + A1
    rs += [sh.update(0, s, 46, s, -3.6, False) for s in range(41, 46)]   # trailing in-flight landings
    rs += [sh.update(0, 46, 46, 46, -3.6, False) for _ in range(10)]     # quiet, still in confirm window
    assert min(rs) >= 0.0


def test_rampA_never_collect_bounded_by_episode_cap():
    sh = AggressiveCycleShaper(AggressiveCycleCfg(**{**CORE, "ramp_episode_cap": 60.0}))
    _latch(sh, fresh=30)
    total = sum(-sh.update(0, 30, 30, 30, -3.6, False) for _ in range(2000))  # idle forever
    assert 60.0 - 1e-6 <= total <= 60.0 + 1e-6                          # saturates at the shared cap


def test_rampsBC_clean_leg_zero_and_budget_cap():
    BC = dict(core_slope=0.0, linger_penalty=0.0, leave_bonus=0.0, collect_bonus=0.0,
              return_bonus=0.0, rampB_slope=0.0125, rampB_step_cap=0.25, rampB_grace=30,
              rampB_budget=10.0, neutral_deep_y=-1.5)
    sh = AggressiveCycleShaper(AggressiveCycleCfg(**BC))
    _latch(sh, fresh=30)
    sh.update(0, 30, 30, 30, -2.4, False)                              # leave -> stage 1
    sh.update(0, 30, 31, 30, -1.0, False)                              # deep collect -> stage 2
    prompt = [-sh.update(0, 30, 31, 30, -1.0, False) for _ in range(25)]  # within grace -> 0
    assert max(prompt) == 0.0
    total_B = sum(-sh.update(0, 30, 31, 30, -1.0, False) for _ in range(4000))  # dawdle in-leg
    assert 10.0 - 1e-6 <= total_B <= 10.0 + 1e-6                        # capped at the per-leg budget


def test_ramp_pref_lane_and_no_strand():
    cfg = AggressiveCycleCfg(ramp_pref=4.0, ramp_center=1.55, ramp_tol=0.9,
                             leave_bonus=5.0, return_bonus=5.0, core_slope=0.0, linger_penalty=0.0)
    sh = AggressiveCycleShaper(cfg)
    assert abs(sh._ramp_score(1.55) - 1.0) < 1e-9                        # 1.0 at the ramp lane
    assert sh._ramp_score(3.3582) == 0.0                                # 0 at the trench lane
    assert 0.0 < sh._ramp_score(2.0) < 1.0
    assert sh._ramp_score(1.55) >= sh._ramp_score(1.9) >= sh._ramp_score(2.3) >= sh._ramp_score(3.0)
    sh1 = AggressiveCycleShaper(cfg); _latch(sh1, 30)
    assert abs(sh1.update(0, 30, 30, 30, -2.4, False, x=1.55) - 9.0) < 1e-6    # base 5 + full pref 4
    sh2 = AggressiveCycleShaper(cfg); _latch(sh2, 30)
    r_trench = sh2.update(0, 30, 30, 30, -2.4, False, x=3.3582)          # trench: base only
    assert abs(r_trench - 5.0) < 1e-6 and r_trench > 0.0                # net-positive => never strands
    assert sh2.update(0, 30, 30, 30, -2.4, False, x=1.55) <= 1e-9        # once-only: no 2nd leave bonus


def test_shallow_collect_does_not_disarm_and_deadline_arm():
    # (a) with neutral_deep_y=0.0 a shallow collect (y=-1.0) does NOT disarm Ramp A; deep does.
    sh = AggressiveCycleShaper(AggressiveCycleCfg(**{**CORE, "neutral_deep_y": 0.0}))
    _latch(sh, 30)
    sh.update(0, 30, 30, 30, -2.4, False)                              # leave -> stage 1
    sh.update(0, 30, 31, 30, -1.0, False)                              # shallow "collect" (y<0.0)
    assert sh.stage == 1 and sh._core_off is False
    sh.update(0, 30, 32, 30, 0.5, False)                               # genuine deep collect
    assert sh.stage == 2 and sh._core_off is True
    # (b) deadline-arm: score>=floor but chamber never empties -> latch after the deadline.
    sh2 = AggressiveCycleShaper(AggressiveCycleCfg(**{**CORE, "arm_deadline_steps": 50}))
    latched_at = None
    for k in range(1, 80):
        sh2.update(mag=5, score=30, collected=30, fresh_score=30, y=-3.6, done=False)
        if sh2.latched and latched_at is None:
            latched_at = k
    assert latched_at == 51                                             # _t_reach_floor=1 -> _t>=51


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
