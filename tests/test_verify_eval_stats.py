from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl" / "verify_eval_stats.py"
    spec = importlib.util.spec_from_file_location("verify_eval_stats", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block():
    return {
        "episodes": 2,
        "mean_return": 2.0,
        "mean_collected": 3.0,
        "max_collected": 4,
        "mean_scored": 1.0,
        "max_scored": 2,
        "per_episode": [
            {"return": 1.0, "collected": 2, "scored": 0},
            {"return": 3.0, "collected": 4, "scored": 2},
        ],
    }


def test_recompute_block_includes_count_summaries_and_sample_sd():
    module = _module()
    result = module.recompute_block(_block())

    assert result["episodes"] == 2
    assert result["mean_return"] == pytest.approx(2.0)
    assert result["sd_return"] == pytest.approx(2**0.5)
    assert result["mean_collected"] == pytest.approx(3.0)
    assert result["max_scored"] == 2


def test_check_block_detects_tampering_and_missing_header():
    module = _module()
    block = _block()
    block["mean_collected"] = 99.0
    del block["max_scored"]

    problems = module.check_block("candidate", block)

    assert any("mean_collected" in problem for problem in problems)
    assert any("missing summary header max_scored" in problem for problem in problems)


def test_recompute_block_rejects_malformed_rows():
    module = _module()
    with pytest.raises(ValueError, match="missing: scored"):
        module.recompute_block(
            {"per_episode": [{"return": 1.0, "collected": 2}]}
        )


def test_main_accepts_consistent_file_and_rejects_tampering(tmp_path, monkeypatch):
    module = _module()
    path = tmp_path / "eval.json"
    data = {"candidate": _block()}
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verify_eval_stats.py", str(path)])
    assert module.main() == 0

    data["candidate"]["episodes"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verify_eval_stats.py", str(path)])
    assert module.main() == 1
