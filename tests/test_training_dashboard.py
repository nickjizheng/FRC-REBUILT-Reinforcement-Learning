from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rl"
        / "training_dashboard.py"
    )
    spec = importlib.util.spec_from_file_location("training_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_jsonl_skips_incomplete_tail(tmp_path):
    dashboard = _module()
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"transitions": 10, "recent_return_mean": 1.5}\n'
        '{"transitions": 20, "recent_return_mean": 2.5}\n'
        '{"transitions":',
        encoding="utf-8",
    )
    rows = dashboard.read_jsonl(path)
    assert [row["transitions"] for row in rows] == [10, 20]


def test_cli_value_reads_training_arguments():
    dashboard = _module()
    command = [
        "python",
        "scripts/rl/train_drqv2.py",
        "--num-envs",
        "4",
        "--minutes",
        "90",
    ]
    assert dashboard.cli_value(command, "--num-envs") == "4"
    assert dashboard.cli_value(command, "--minutes") == "90"
    assert dashboard.cli_value(command, "--missing", 7) == 7


def test_stopped_run_without_summary_is_reported(tmp_path):
    dashboard = _module()
    run = tmp_path / "drqv2_interrupted"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        json.dumps(
            {
                "transitions": 7360,
                "updates": 6362,
                "recent_return_mean": 9.69,
                "recent_score_reward": 0.0,
                "recent_collect_reward": 13.65,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "latest.pt").write_bytes(b"checkpoint")
    payload = dashboard.run_payload(run, [])
    assert payload["status"] == "stopped"
    assert payload["latest"]["transitions"] == 7360
    assert payload["checkpoint"]["latest"]
    assert any(alert["level"] == "error" for alert in payload["alerts"])
