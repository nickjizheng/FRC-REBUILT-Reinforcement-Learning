"""Tests for the second-cycle phase-PBRS CyclePotential.

Verifies: no shaping before the first unload; each leg's Phi band; monotone progress
through leave->collect->return->unload; continuity at the leave->collect board crossing;
Phi~0 at the hub (cycle complete); board hysteresis (no leg flicker); no-path -> None;
reset clears state. Uses the real OccupancyGrid (built once). No Isaac.
Run: C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_cycle_potential.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frc_rebuilt.field_map import OccupancyGrid          # noqa: E402
from frc_rebuilt.rl.prefix_takeover import CyclePotential, CycleGeom  # noqa: E402

GRID = OccupancyGrid()   # slow to build; share across tests


def _cp():
    return CyclePotential("blue", grid=GRID)


def test_no_shaping_before_first_unload():
    cp = _cp()
    cp.update_leg((1.5, -4.0), 0)
    assert cp.leg is None and cp.potential((1.5, -4.0)) == 0.0


def test_leg_bands_and_gateway_fixed():
    cp = _cp(); cp.note_unload()
    cp.update_leg((1.5, -4.5), 0)                       # empty, deep own court -> leave
    assert cp.leg == "leave" and -1.0 <= cp.potential((1.5, -4.5)) <= -0.75
    assert cp.gstar_out is not None and cp.gstar_out[1] > -CycleGeom().board_y  # gateway in neutral
    g0 = cp.gstar_out
    cp.update_leg((1.4, -4.0), 0)                       # still leave -> gateway must NOT change
    assert cp.gstar_out == g0
    cp.update_leg((0.0, 0.6), 0)                        # empty, neutral -> collect
    assert cp.leg == "collect" and -0.75 <= cp.potential((0.0, 0.6)) <= -0.5
    cp.update_leg((0.0, 0.6), 5)                        # loaded, neutral -> return
    assert cp.leg == "return" and -0.5 <= cp.potential((0.0, 0.6)) <= -0.25
    cp.update_leg((1.5, -3.6), 5)                       # loaded, home at hub -> unload
    assert cp.leg == "unload" and -0.25 <= cp.potential((1.5, -3.6)) <= 0.0


def test_monotone_progress_through_cycle():
    cp = _cp(); cp.note_unload()
    # a plausible progression; Phi must be non-decreasing leg-to-leg (each band is higher)
    seq = [((1.5, -4.5), 0), ((1.5, -2.7), 0),          # leave: far -> near gateway
           ((0.0, 1.4), 0),                              # collect (near collect target)
           ((0.0, 0.0), 5), ((1.5, -2.9), 5),            # return: far -> near board
           ((1.5, -3.6), 5)]                             # unload at hub
    phis = []
    for pos, mag in seq:
        cp.update_leg(pos, mag)
        phi = cp.potential(pos)
        assert phi is not None, f"no-path at {pos} leg={cp.leg}"
        phis.append(phi)
    assert phis == sorted(phis), f"Phi not monotone through the cycle: {[round(p,3) for p in phis]}"
    assert abs(phis[-1]) < 0.06, f"Phi at hub should be ~0, got {phis[-1]:.3f}"


def test_continuity_at_board_crossing():
    # crossing leave->collect at the board: Phi must not jump (both ~ -0.75 at the boundary)
    cp = _cp(); cp.note_unload()
    cp.update_leg((1.55, -2.6), 0); phi_leave = cp.potential((1.55, -2.6))   # home side, near gateway
    cp.update_leg((1.55, -2.45), 0); phi_collect = cp.potential((1.55, -2.45))  # away side
    assert cp.leg == "collect"
    assert abs(phi_leave - phi_collect) < 0.15, f"discontinuous at board: {phi_leave:.3f} vs {phi_collect:.3f}"


def test_board_hysteresis_no_flicker():
    cp = _cp(); cp.note_unload()
    cp.update_leg((1.5, -2.0), 0)                       # clearly away -> collect
    assert cp.leg == "collect"
    cp.update_leg((1.5, -2.7), 0)                       # in the hysteresis band (-3.05..-2.50): keep away
    assert cp.leg == "collect", "leg flickered back inside the hysteresis band"
    cp.update_leg((1.5, -3.2), 0)                       # past band_in -> home -> leave
    assert cp.leg == "leave"


def test_no_path_returns_none_never_nan():
    cp = _cp(); cp.note_unload()
    cp.update_leg((1.5, -4.0), 0)                       # leave leg set
    assert cp.potential((100.0, 100.0)) is None         # off-grid start -> no path, not NaN


def test_reset_clears_state():
    cp = _cp(); cp.note_unload()
    cp.update_leg((1.5, -4.0), 0)
    assert cp.leg == "leave"
    cp.reset()
    cp.update_leg((1.5, -4.0), 0)
    assert cp.leg is None and cp.potential((1.5, -4.0)) == 0.0   # unloaded flag cleared


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
