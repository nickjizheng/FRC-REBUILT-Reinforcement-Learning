from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime
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


def test_process_discovery_skips_zombies_without_killing_sampler(monkeypatch):
    dashboard = _module()
    import psutil

    class Zombie:
        info = {"pid": 123, "name": "python"}

        def cmdline(self):
            raise psutil.ZombieProcess(123)

    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: [Zombie()])

    assert dashboard.find_training_processes() == []


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_v2_funnel_uses_selected_run_and_separates_skill_resets(tmp_path):
    dashboard = _module()
    run = tmp_path / "selected"
    other = tmp_path / "other"
    run.mkdir()
    other.mkdir()
    rows = [
        {
            "stagec_v2": True,
            "reset_mode": "full",
            "scored": 56,
            "latched": True,
            "milestones": {
                "latched": 1,
                "left_home": 1,
                "target_load": 1,
                "returned_home": 1,
            },
            "dump_empty_completions": 1,
            "partial_dumps": 0,
            "cycles_completed": 0,
        },
        {
            "stagec_v2": True,
            "reset_mode": "full",
            "scored": 50,
            "latched": True,
            "milestones": {"latched": 1},
            "dump_empty_completions": 1,
            "partial_dumps": 2,
            "cycles_completed": 0,
        },
        {"stagec_v2": True, "reset_mode": "postdump", "milestones": {"left_home": 1}},
        {"stagec_v2": True, "reset_mode": "collect", "milestones": {"target_load": 1}},
        {"stagec_v2": True, "reset_mode": "return", "milestones": {"returned_home": 1}},
    ]
    _write_jsonl(run / "cycle_telemetry.jsonl", rows)
    _write_jsonl(
        other / "cycle_telemetry.jsonl",
        [
            {
                "stagec_v2": True,
                "reset_mode": "full",
                "scored": 99,
                "cycles_completed": 1,
                "milestones": {"cycle_scored": 1},
            }
        ],
    )

    funnel = dashboard.compute_cycle_funnel(run)

    assert funnel is not None
    assert funnel["episodes"] == 2
    assert funnel["curriculum_episodes"] == 3
    assert [stage["count"] for stage in funnel["stages"]] == [2, 1, 1, 1, 0]
    assert funnel["score_mean"] == 53.0
    assert funnel["score_max"] == 56
    assert funnel["clean_dump_episodes"] == 1
    assert funnel["partial_dumps"] == 2
    skills = {item["key"]: item for item in funnel["skill_streams"]}
    assert skills["postdump"]["successes"] == 1
    assert skills["collect"]["successes"] == 1
    assert skills["return"]["successes"] == 1
    assert skills["return_fire"]["successes"] == 0


def test_v2_funnel_counts_actual_ramp_out_not_any_leave_milestone(tmp_path):
    dashboard = _module()
    run = tmp_path / "rampout"
    run.mkdir()
    rows = [
        {
            "stagec_v2": True,
            "reset_mode": "full",
            "scored": 60,
            "latched": True,
            "ramp_out_attempts": 1,
            "ramp_out_successes": 1,
            "milestones": {"left_home": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "full",
            "scored": 58,
            "latched": True,
            "ramp_out_attempts": 1,
            "ramp_out_successes": 0,
            "milestones": {"left_home": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "postdump",
            "terminal_reason": "off_ramp_exit",
            "ramp_out_attempts": 1,
            "ramp_out_successes": 0,
            "milestones": {"left_home": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "postdump",
            "terminal_reason": "skill_success",
            "ramp_out_attempts": 1,
            "ramp_out_successes": 1,
            "milestones": {"left_home": 1},
        },
    ]
    _write_jsonl(run / "cycle_telemetry.jsonl", rows)

    funnel = dashboard.compute_cycle_funnel(run)

    assert funnel is not None
    assert funnel["stages"][1] == {
        "key": "left",
        "label": "Ramp out",
        "count": 1,
        "pct": 50.0,
    }
    postdump = next(
        item for item in funnel["skill_streams"] if item["key"] == "postdump"
    )
    assert postdump["label"] == "Ramp out"
    assert postdump["successes"] == 1
    assert funnel["ramp_out_attempts"] == 2
    assert funnel["ramp_out_successes"] == 1


def test_v5_postdump_dashboard_requires_ramp_and_target_load(tmp_path):
    dashboard = _module()
    run = tmp_path / "efficiency_v5"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "stagec_v2": {
                    "target_load": 20,
                    "postdump_require_target_load": True,
                }
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "stagec_v2": True,
            "reset_mode": "full",
            "scored": 61,
            "ramp_out_attempts": 1,
            "ramp_out_successes": 1,
            "milestones": {"latched": 1, "left_home": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "postdump",
            "terminal_reason": "horizon",
            "ramp_out_attempts": 1,
            "ramp_out_successes": 1,
            "milestones": {"left_home": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "postdump",
            "terminal_reason": "skill_success",
            "ramp_out_attempts": 1,
            "ramp_out_successes": 1,
            "milestones": {"left_home": 1, "target_load": 1},
        },
    ]
    _write_jsonl(run / "cycle_telemetry.jsonl", rows)

    funnel = dashboard.compute_cycle_funnel(run)

    assert funnel is not None
    assert funnel["stages"][2]["label"] == "Collect 20+"
    postdump = next(
        item for item in funnel["skill_streams"] if item["key"] == "postdump"
    )
    assert postdump["label"] == "Ramp out + collect 20"
    assert postdump["successes"] == 1
    assert postdump["episodes"] == 2


def test_v6_dashboard_counts_only_complete_repeat_cycle(tmp_path):
    dashboard = _module()
    run = tmp_path / "cycle_bridge_v6"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "stagec_v2": {
                    "target_load": 45,
                    "postdump_require_target_load": True,
                    "postdump_complete_cycle": True,
                }
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "stagec_v2": True,
            "reset_mode": "full",
            "scored": 60,
            "milestones": {"latched": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "postdump",
            "terminal_reason": "horizon",
            "ramp_out_successes": 1,
            "milestones": {"left_home": 1, "target_load": 1},
        },
        {
            "stagec_v2": True,
            "reset_mode": "postdump",
            "terminal_reason": "skill_success",
            "ramp_out_successes": 1,
            "cycles_completed": 1,
            "milestones": {
                "left_home": 1,
                "target_load": 1,
                "returned_home": 1,
                "cycle_scored": 1,
            },
        },
    ]
    _write_jsonl(run / "cycle_telemetry.jsonl", rows)

    funnel = dashboard.compute_cycle_funnel(run)
    postdump = next(
        item for item in funnel["skill_streams"] if item["key"] == "postdump"
    )
    assert postdump["label"] == "Complete repeat cycle (45)"
    assert postdump["successes"] == 1
    assert postdump["episodes"] == 2


def test_v2_payload_normalizes_physical_scores_and_process_metadata(tmp_path):
    dashboard = _module()
    run = tmp_path / "stagec_v2_latest"
    run.mkdir()
    started = time.time() - 120
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "started_at_unix": started,
                "minutes": "600.0",
                "num_collectors": "3",
                "collector_envs": "2",
                "collect_weight": "0.3",
                "stagec_v2": {"schema_version": "stagec_v2.3", "return_skill_preload": 8},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run / "metrics.jsonl",
        [
            {
                "transitions": 32000,
                "updates": 28000,
                "episodes": 100,
                "recent_full_score_mean": 58.5,
                "recent_full_score_max": 62,
                "q_pi": 42.5,
                "by_mode": {"return": {"success_rate": 0.0}},
            }
        ],
    )
    _write_jsonl(
        run / "cycle_telemetry.jsonl",
        [
            {
                "stagec_v2": True,
                "reset_mode": "full",
                "scored": 50,
                "score_reward": 484.0,
                "collect_reward": 18.0,
                "return": 465.0,
                "shots_fired": 64,
                "cycles_completed": 0,
                "partial_dumps": 2,
                "latched": True,
            },
            {
                "stagec_v2": True,
                "reset_mode": "full",
                "scored": 60,
                "score_reward": 600.0,
                "collect_reward": 20.0,
                "return": 580.0,
                "shots_fired": 65,
                "cycles_completed": 0,
                "partial_dumps": 0,
                "latched": True,
            },
            {
                "stagec_v2": True,
                "reset_mode": "collect",
                "scored": 0,
                "score_reward": 0.0,
                "collect_reward": 24.0,
            },
        ],
    )
    process = dashboard.TrainingProcess(
        pid=123,
        command=["python", "learner_cycle_v2.py"],
        create_time=started + 60,
        cpu_percent=50.0,
        rss_bytes=100,
        private_bytes=100,
        num_envs=6,
        target_minutes=600.0,
        replay_capacity=400000,
        output_dir=str(run),
    )

    payload = dashboard.run_payload(run, [process])

    assert payload["running"] is True
    assert payload["status"] == "running"
    assert payload["started_at"] == datetime.fromtimestamp(started).astimezone().isoformat()
    assert payload["config"]["num_envs"] == 6
    assert payload["config"]["preload_display"] == "return 8"
    assert payload["latest"]["recent_scored_balls"] == 55.0
    assert payload["latest"]["recent_scored_max"] == 60
    assert payload["latest"]["q1"] == 42.5
    assert [row["scored"] for row in payload["episode_history"]] == [50, 60]
    assert payload["episode_history"][0]["score_reward"] == 484.0
    assert [
        row["rolling_scored_mean"] for row in payload["episode_history"]
    ] == [50.0, 55.0]


def test_cycle_learner_command_reports_total_collector_envs():
    dashboard = _module()
    command = [
        "python",
        "scripts/rl/learner_cycle_v2.py",
        "--num-collectors",
        "3",
        "--collector-envs",
        "2",
    ]
    assert dashboard.command_script(command) == "learner_cycle_v2.py"
    assert dashboard.command_num_envs(command) == 6


def test_final_checkpoint_is_complete_without_summary(tmp_path):
    dashboard = _module()
    run = tmp_path / "finished_v2"
    run.mkdir()
    _write_jsonl(run / "metrics.jsonl", [{"transitions": 10, "updates": 5}])
    (run / "final.pt").write_bytes(b"checkpoint")

    payload = dashboard.run_payload(run, [])

    assert payload["status"] == "complete"
    assert not any("without final summary" in alert["text"] for alert in payload["alerts"])


def test_read_jsonl_tail_returns_actual_last_records(tmp_path):
    dashboard = _module()
    path = tmp_path / "many.jsonl"
    _write_jsonl(path, [{"index": index} for index in range(dashboard.MAX_HISTORY + 5)])
    assert [row["index"] for row in dashboard.read_jsonl_tail(path, 3)] == [602, 603, 604]


def test_stagec_scoring_chart_uses_the_headline_rolling_average():
    html_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rl"
        / "training_dashboard.html"
    )
    html = html_path.read_text(encoding="utf-8")

    assert 'id="scoreChartUnit"' in html
    assert "eh.map(v=>v.rolling_scored_mean==null?NaN:+v.rolling_scored_mean)" in html
    assert "20-FULL rolling avg" in html
