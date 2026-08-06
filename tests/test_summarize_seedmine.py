from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rl"
        / "summarize_seedmine.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_seedmine", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(**overrides):
    value = {
        "checkpoint": "/models/v2_000075000.pt",
        "checkpoint_sha256": "7" * 64,
        "checkpoint_v2_updates": 75000,
        "mode": "full",
        "scored": 55,
        "collected": 70,
        "cycles_completed": 0,
        "dump_empty_completions": 1,
        "partial_dumps": 0,
        "milestones": {"latched": 1, "left_home": 1},
        "capture_path": None,
        "capture_tier": None,
    }
    value.update(overrides)
    return value


def test_active_jsonl_tail_is_ignored_until_newline(tmp_path):
    module = _module()
    path = tmp_path / "lane_a.episodes.jsonl"
    first = json.dumps(_record())
    second = json.dumps(_record(scored=71))
    path.write_bytes((first + "\n" + "not-json\n" + second[:20]).encode("utf-8"))
    records, diagnostics = module.read_jsonl_snapshot(path)
    assert len(records) == 1
    assert records[0]["scored"] == 55
    assert diagnostics["incomplete_tail"] is True
    assert diagnostics["parse_errors"] == 1


def test_aggregate_reports_clean_cycles_funnel_scores_and_near_success():
    module = _module()
    cycle = _record(
        scored=71,
        cycles_completed=1,
        dump_empty_completions=2,
        milestones={
            "latched": 1,
            "left_home": 1,
            "target_load": 1,
            "returned_home": 1,
            "cycle_dumped": 1,
            "cycle_scored": 1,
        },
        capture_tier="cycle",
        capture_path="/captures/episode_cycle.npz",
    )
    near = _record(
        scored=62,
        milestones={
            "latched": 1,
            "left_home": 1,
            "target_load": 1,
            "returned_home": 1,
        },
    )
    result = module.aggregate_records([cycle, near])
    assert result["episodes"] == 2
    assert result["cycle_successes"] == 1
    assert result["clean_cycle_successes"] == 1
    assert result["funnel"]["returned_home"]["count"] == 2
    assert result["near_success"]["returned_home_not_scored"] == 1
    assert result["score"]["mean"] == 66.5
    assert result["score"]["max"] == 71
    assert result["captures"]["cycle"] == 1


def test_build_summary_groups_checkpoint_lanes_and_capture_files(tmp_path, monkeypatch):
    module = _module()
    lane_a = tmp_path / "v2_75" / "lane_a.episodes.jsonl"
    lane_b = tmp_path / "v2_75" / "lane_b.episodes.jsonl"
    lane_a.parent.mkdir(parents=True)
    capture = tmp_path / "captures" / "episode_cycle.npz"
    capture.parent.mkdir()
    capture.write_bytes(b"standalone")
    lane_a.write_text(
        json.dumps(
            _record(
                scored=71,
                cycles_completed=1,
                dump_empty_completions=2,
                milestones={"cycle_dumped": 1, "cycle_scored": 1},
                capture_tier="cycle",
                capture_path=str(capture),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    lane_b.write_text(json.dumps(_record(scored=52)) + "\n", encoding="utf-8")
    (lane_a.parent / "lane_a.pid").write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(module, "local_pid_alive", lambda pid: pid == os.getpid())

    summary = module.build_summary(tmp_path)
    assert summary["files"]["matched"] == 2
    assert summary["files"]["records"] == 2
    assert len(summary["checkpoints"]) == 1
    assert len(summary["lanes"]) == 2
    assert summary["checkpoints"][0]["clean_cycle_successes"] == 1
    assert summary["captured_files"]["referenced"] == 1
    assert summary["captured_files"]["discovered"] == 1
    assert summary["captured_files"]["missing_basenames"] == []
    assert summary["processes"][0]["status"] == "running"


def test_remote_pid_is_reported_without_local_probe(tmp_path, monkeypatch):
    module = _module()
    pid_path = tmp_path / "lane.pid"
    pid_path.write_text(
        json.dumps({"pid": 12345, "host": "definitely.remote.invalid"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "local_pid_alive",
        lambda pid: (_ for _ in ()).throw(AssertionError("must not inspect remote PID")),
    )
    process = module.discover_processes(tmp_path)[0]
    assert process["status"] == "remote_unchecked"
    assert process["alive"] is None


def test_local_pid_probe_recognizes_current_process():
    module = _module()
    assert module.local_pid_alive(os.getpid()) is True


def test_atomic_summary_replaces_complete_file_and_leaves_no_temp(tmp_path):
    module = _module()
    path = tmp_path / "combined_summary.json"
    path.write_text("old", encoding="utf-8")
    module.atomic_write_json(path, {"schema": module.SCHEMA, "episodes": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["episodes"] == 2
    assert not list(tmp_path.glob(".*.tmp"))
