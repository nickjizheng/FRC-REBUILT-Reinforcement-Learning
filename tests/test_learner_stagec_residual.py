from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl" / "learner_stagec_residual.py"
    spec = importlib.util.spec_from_file_location("learner_stagec_residual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_rows_are_one_phase_and_stop_at_last_success():
    module = _module()
    proprio = np.zeros((8, 30), dtype=np.float32)
    phases = [0, 1, 1, 2, 1, 3, 1, 4]
    for row, phase in enumerate(phases):
        proprio[row, 22 + phase] = 1.0
    episode = {"cycle_success_steps": [4]}
    assert module._selected_rows(proprio, episode, "leave").tolist() == [1, 2, 4]
    assert module._selected_rows(proprio, episode, "collect").tolist() == [3]
    assert module._selected_rows(proprio, episode, "return") == pytest.approx([])


def test_selected_rows_reject_malformed_phase_or_missing_success():
    module = _module()
    proprio = np.zeros((2, 30), dtype=np.float32)
    proprio[:, 23] = 1.0
    with pytest.raises(ValueError, match="success"):
        module._selected_rows(proprio, {}, "leave")
    proprio[0, 24] = 1.0
    with pytest.raises(ValueError, match="one-hot"):
        module._selected_rows(proprio, {"cycle_success_steps": [1]}, "leave")


def test_episode_weight_is_positive_bounded_and_rewards_outcome_gain():
    module = _module()
    baseline = module._episode_weight(
        {
            "candidate_score": 80,
            "control_score": 80,
            "candidate_cycles": 1,
            "control_cycles": 1,
        }
    )
    improved = module._episode_weight(
        {
            "candidate_score": 90,
            "control_score": 80,
            "candidate_cycles": 2,
            "control_cycles": 1,
        }
    )
    huge = module._episode_weight(
        {
            "candidate_score": 200,
            "control_score": 0,
            "candidate_cycles": 8,
            "control_cycles": 0,
        }
    )
    assert baseline == pytest.approx(1.0)
    assert improved > baseline
    assert huge == pytest.approx(4.0)
