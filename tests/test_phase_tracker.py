"""Regression tests for the Stage C phase-timing metrics.

Pure EpisodeTracker, no Isaac. Pins the promotion-gate metric so it cannot silently
regress: ordered completed_cycle_2, recycling-only score, trailing first-load flight
delay, inbound-without-inventory, collection-before-outbound, and repeated-crossing
hysteresis.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_phase_timing",
    Path(__file__).resolve().parents[1] / "scripts" / "rl" / "eval_phase_timing.py",
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
EpisodeTracker = _m.EpisodeTracker


# step args: x, y, vxy, act_trans, shoot, mag, score, collected, shots, cext
def _run(steps):
    tr = EpisodeTracker("anchor_dev", 0)
    for s in steps:
        tr.step(*s)
    last = steps[-1]
    tr.finalize(last[6], last[7], last[8])
    return tr.record()


def _escape_collect_unload():
    """spawn under trench -> outbound -> collect 8 -> inbound -> fire load1 with a
    flight delay (magazine empties before the balls land and score)."""
    seq = [
        (3.35, -3.885, .5, .8, False, 0, 0, 0, 0, 0.0),   # spawn below band -> silent
        (3.0, -3.1, .8, .8, False, 0, 0, 0, 0, 1.0),
        (2.0, -2.4, .8, .8, False, 0, 0, 0, 0, 1.0),      # OUTBOUND #1
    ]
    seq += [(0.0, 0.5, .8, .8, False, k, 0, k, 0, 1.0) for k in range(1, 9)]  # collect 8
    seq += [
        (1.5, -3.1, .8, .5, False, 8, 0, 8, 0, 1.0),      # INBOUND #1
        (3.35, -3.6, .05, .1, True, 8, 0, 8, 0, 1.0),     # aim + fire request
        (3.35, -3.6, .05, .1, True, 0, 0, 8, 8, 1.0),     # launched: mag0, score0 (in flight)
        (3.35, -3.6, .05, .1, False, 0, 3, 8, 8, 1.0),    # first volley lands: score 3
        (3.35, -3.6, .05, .1, False, 0, 8, 8, 8, 1.0),    # trailing lands: score 8
    ]
    return seq


def test_trailing_first_load_counts_full_score():
    r = _run(_escape_collect_unload())
    assert r["score_at_first_unload"] == 8            # full load, not the mid-flight 3
    assert r["recycling_post_unload_score"] == 0      # trailing balls are NOT recycling
    assert r["completed_cycle_2_t"] is None
    assert r["chamber_empty_after_score_t"] is not None


def test_ordered_completed_cycle_2():
    seq = _escape_collect_unload() + [
        (2.0, -2.4, .8, .8, False, 0, 8, 8, 8, 1.0),      # OUTBOUND #2
        (0.0, 0.5, .8, .8, False, 5, 8, 13, 8, 1.0),      # collect 13>8 in neutral
        (1.5, -3.1, .8, .5, False, 5, 8, 13, 8, 1.0),     # INBOUND #2 with mag 5>0
        (3.35, -3.6, .05, .1, True, 5, 8, 13, 8, 1.0),
        (3.35, -3.6, .05, .1, False, 0, 13, 13, 13, 1.0),  # genuine 2nd-cycle score
    ]
    r = _run(seq)
    assert r["completed_cycle_2_t"] is not None
    assert r["recycling_post_unload_score"] == 0
    assert [c["dir"] for c in r["crossings"]] == ["outbound", "inbound", "outbound", "inbound"]


def test_recycling_only_is_not_a_cycle():
    seq = _escape_collect_unload() + [
        (3.3, -3.6, .3, .5, False, 3, 8, 11, 8, 1.0),     # collect 11>8 but stays home
        (3.3, -3.6, .05, .1, True, 0, 13, 11, 11, 1.0),   # score 8->13 at home (no field trip)
    ]
    r = _run(seq)
    assert r["completed_cycle_2_t"] is None
    assert r["recycling_post_unload_score"] == 5


def test_inbound_without_inventory_is_not_a_cycle():
    seq = _escape_collect_unload() + [
        (2.0, -2.4, .8, .8, False, 0, 8, 8, 8, 1.0),      # OUTBOUND #2 -> stage 1
        (0.0, 0.5, .8, .8, False, 4, 8, 12, 8, 1.0),      # collect 12>8 neutral -> stage 2
        (0.0, 0.3, .05, .1, True, 0, 8, 12, 12, 1.0),     # empties in neutral: mag -> 0
        (1.5, -3.1, .8, .5, False, 0, 8, 12, 12, 1.0),    # INBOUND with mag==0 -> NOT stage 3
        (3.35, -3.6, .05, .1, False, 0, 12, 12, 12, 1.0),  # score at home
    ]
    r = _run(seq)
    assert r["completed_cycle_2_t"] is None            # returned without inventory


def test_collection_before_outbound_does_not_advance_cycle2():
    seq = _escape_collect_unload() + [
        (3.3, -3.6, .3, .5, False, 4, 8, 12, 8, 1.0),     # collect at home, no 2nd outbound
        (3.35, -3.6, .05, .1, False, 0, 16, 12, 12, 1.0),  # score -> recycling, not cycle-2
    ]
    r = _run(seq)
    assert r["completed_cycle_2_t"] is None
    assert r["second_outbound_t"] is None


def test_repeated_band_jitter_logs_no_phantom_crossing():
    seq = [(3.35, -3.885, .5, .8, False, 0, 0, 0, 0, 0.0)]  # spawn below band -> silent
    # oscillate strictly inside the hysteresis dead-zone (-3.05 .. -2.50)
    seq += [(3.0, y, .5, .5, False, 0, 0, 0, 0, 1.0) for y in (-2.9, -2.7, -2.9, -2.7, -2.9)]
    r = _run(seq)
    assert r["crossings"] == []                        # no full crossing -> no phantom cycle
