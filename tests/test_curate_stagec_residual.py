from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl" / "curate_stagec_residual.py"
    spec = importlib.util.spec_from_file_location("curate_stagec_residual", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(tmp_path: Path, *, action_mode: str, score: int, cycles: int) -> dict:
    return {
        "env_seed": 10,
        "env_index": 0,
        "env_episode_sequence": 0,
        "checkpoint_sha256": "a" * 64,
        "prefix_sha256": "b" * 64,
        "template_sha256": "c" * 64,
        "mode": "full",
        "episode_len_s": 90.0,
        "num_envs": 1,
        "stagec_v2_metadata": {"reward_revision": "score_efficiency_v8"},
        "action_mode": action_mode,
        "noise_phases": ["leave"] if action_mode == "smooth-drive" else [],
        "noise_cap": 0.05 if action_mode == "smooth-drive" else None,
        "scored": score,
        "cycles_completed": cycles,
        "cycle_success_steps": [500] if cycles else [],
        "repeat_scored_load_sum": 17 if cycles else 0,
        "outer_rail_fraction": 0.2,
        "outer_rail_max_streak": 50,
        "capture_path": None,
    }


def _write_jsonl(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_curator_accepts_only_noninferior_strict_improvement(tmp_path):
    module = _module()
    capture = tmp_path / "episode.npz"
    capture.write_bytes(b"capture")
    control = _record(tmp_path, action_mode="deterministic", score=80, cycles=1)
    candidate = _record(tmp_path, action_mode="smooth-drive", score=86, cycles=1)
    candidate["capture_path"] = str(capture)
    control_path = tmp_path / "control.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(control_path, control)
    _write_jsonl(candidate_path, candidate)
    result = module.curate(
        [control_path], [candidate_path], tmp_path / "curated", "leave"
    )
    assert result["status"] == "READY"
    assert result["accepted"] == 1
    assert result["decisions"][0]["improvements"] == ["score"]
    assert (tmp_path / "curated" / "captures" / capture.name).read_bytes() == b"capture"


def test_curator_rejects_score_or_cycle_regression(tmp_path):
    module = _module()
    control = _record(tmp_path, action_mode="deterministic", score=80, cycles=2)
    candidate = _record(tmp_path, action_mode="smooth-drive", score=79, cycles=1)
    control_path = tmp_path / "control.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(control_path, control)
    _write_jsonl(candidate_path, candidate)
    result = module.curate(
        [control_path], [candidate_path], tmp_path / "curated", "leave"
    )
    assert result["status"] == "NO_QUALIFIED_DATA"
    assert result["accepted"] == 0
    assert result["decisions"][0]["reason"] == "score_regression"


def test_curator_rejects_contract_mismatch(tmp_path):
    module = _module()
    control = _record(tmp_path, action_mode="deterministic", score=80, cycles=1)
    candidate = _record(tmp_path, action_mode="smooth-drive", score=85, cycles=1)
    candidate["template_sha256"] = "d" * 64
    control_path = tmp_path / "control.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(control_path, control)
    _write_jsonl(candidate_path, candidate)
    try:
        module.curate(
            [control_path], [candidate_path], tmp_path / "curated", "leave"
        )
    except ValueError as exc:
        assert "template_sha256" in str(exc)
    else:
        raise AssertionError("expected a fail-closed contract mismatch")
