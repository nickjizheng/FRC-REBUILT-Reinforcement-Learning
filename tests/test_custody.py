"""Pure-Python tests for custody-weighted reward.

No Isaac. Covers: fresh vs recycled score/collect, preload not credited, fire (magazine
removal) not credited, and episode reset clearing the ledger.
Run: C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_custody.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frc_rebuilt.rl.custody import CustodyState, collect_custody, score_custody  # noqa: E402

WS, WC, RS, RC = 10.0, 0.3, 0.2, 0.2   # live Stage-C scales + rho


def test_first_score_full_repeat_discounted():
    s = CustodyState(); s.reset([])
    assert score_custody([1, 2, 3], s, WS, RS) == 30.0        # three fresh balls
    assert s.fresh_score == 3 and s.recycled_score == 0
    assert score_custody([1], s, WS, RS) == WS * RS           # re-score ball 1 -> 2.0
    assert s.recycled_score == 1 and s.fresh_score == 3
    # a mixed batch: 2 (recycled) + 9 (fresh)
    assert abs(score_custody([2, 9], s, WS, RS) - (WS * RS + WS)) < 1e-9
    assert s.fresh_score == 4 and s.recycled_score == 2


def test_collect_fresh_vs_recycled_keyed_on_ever_scored():
    s = CustodyState(); s.reset([])
    score_custody([1], s, WS, RS)                              # ball 1 has been scored
    # magazine gains a fresh field ball (10) and the recycled ball (1)
    r = collect_custody([10, 1], s, WC, RC)
    assert abs(r - (WC + WC * RC)) < 1e-9                      # 0.3 fresh + 0.06 recycled
    assert s.fresh_collect == 1 and s.recycled_collect == 1


def test_preloaded_balls_not_credited_as_collect():
    s = CustodyState(); s.reset([5, 6, 7])                     # preload seeds prev_magazine
    assert collect_custody([5, 6, 7], s, WC, RC) == 0.0        # no new appearance
    assert collect_custody([5, 6, 7, 8], s, WC, RC) == WC      # only 8 is newly collected
    assert s.fresh_collect == 1


def test_fired_ball_removal_is_not_a_collect():
    s = CustodyState(); s.reset([])
    assert collect_custody([1, 2], s, WC, RC) == 2 * WC        # collected 1,2
    assert collect_custody([2], s, WC, RC) == 0.0              # fired 1 -> removal, no credit
    assert collect_custody([2, 3], s, WC, RC) == WC            # collected 3
    assert s.fresh_collect == 3


def test_reset_clears_ledger_and_reseeds_magazine():
    s = CustodyState(); s.reset([])
    score_custody([1, 2], s, WS, RS); collect_custody([1], s, WC, RC)
    assert s.ever_scored and s.fresh_score == 2
    s.reset([9])                                              # new episode, preloaded ball 9
    assert s.ever_scored == set() and s.fresh_score == 0 and s.score_events_seen == 0
    assert score_custody([1], s, WS, RS) == WS                # ball 1 is fresh again this episode
    assert collect_custody([9], s, WC, RC) == 0.0             # preloaded 9 not credited


def test_rho_one_reproduces_raw_reward():
    s = CustodyState(); s.reset([])
    # rho=1.0 must reproduce the original flat per-ball reward exactly
    assert score_custody([1, 1, 1], s, WS, 1.0) == 3 * WS
    assert collect_custody([1, 2, 3], s, WC, 1.0) == 3 * WC


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
