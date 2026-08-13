"""Live local dashboard for FRC reinforcement-learning runs.

Usage:
    python scripts/rl/training_dashboard.py

The server binds to localhost only.  It discovers ``runs/drqv2_*`` folders,
tails JSONL metrics, checks the real training process, and reports laptop
CPU/RAM/GPU telemetry without adding dependencies to the training process.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import shlex
import statistics
import subprocess
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"
HTML_PATH = Path(__file__).with_name("training_dashboard.html")
MAX_HISTORY = 600
TRAINING_SCRIPTS = {"train_drqv2.py", "learner_cycle_v2.py", "learner_finetune.py"}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    if len(rows) <= MAX_HISTORY:
        return rows
    stride = max(1, len(rows) // (MAX_HISTORY - 1))
    return rows[::stride][-MAX_HISTORY + 1 :] + [rows[-1]]


def read_jsonl_all(path: Path) -> list[dict[str, Any]]:
    """Read every complete JSON object in a JSONL file.

    Training metrics are deliberately downsampled by :func:`read_jsonl` for charting,
    but episode telemetry must not be downsampled before counts and rates are computed.
    """
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    """Return the actual last ``limit`` complete JSONL records."""
    if limit <= 0:
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return list(rows)


def cli_value(command: list[str], flag: str, default: Any = None) -> Any:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def command_script(command: list[str]) -> str | None:
    for part in command:
        name = Path(part).name.lower()
        if name in TRAINING_SCRIPTS:
            return name
    return None


def command_num_envs(command: list[str], script: str | None = None) -> int:
    script = script or command_script(command)
    if script == "learner_cycle_v2.py":
        collectors = int(cli_value(command, "--num-collectors", 1))
        collector_envs = int(cli_value(command, "--collector-envs", 1))
        return collectors * collector_envs
    return int(cli_value(command, "--num-envs", 2))


@dataclass
class TrainingProcess:
    pid: int
    command: list[str]
    create_time: float
    cpu_percent: float
    rss_bytes: int
    private_bytes: int
    num_envs: int
    target_minutes: float
    replay_capacity: int
    output_dir: str


def find_training_processes() -> list[TrainingProcess]:
    try:
        import psutil
    except ImportError:
        return []
    found: list[TrainingProcess] = []
    # Querying cmdline for every Windows process is surprisingly expensive.
    # Filter to Python first, then inspect the handful of plausible workers.
    for process in psutil.process_iter(["pid", "name"]):
        try:
            if "python" not in str(process.info.get("name") or "").lower():
                continue
            command = process.cmdline()
            script = command_script(command)
            if script is None:
                continue
            memory = process.memory_info()
            output = str(cli_value(command, "--out", "runs/drqv2_stageA"))
            found.append(
                TrainingProcess(
                    pid=int(process.info["pid"]),
                    command=[str(part) for part in command],
                    create_time=float(process.create_time()),
                    cpu_percent=float(process.cpu_percent(interval=None)),
                    rss_bytes=int(getattr(memory, "rss", 0)),
                    private_bytes=int(getattr(memory, "private", 0)),
                    num_envs=command_num_envs(command, script),
                    target_minutes=float(cli_value(command, "--minutes", 20.0)),
                    replay_capacity=int(cli_value(command, "--replay-capacity", 60_000)),
                    output_dir=output,
                )
            )
        except (OSError, ValueError, TypeError, psutil.Error):
            continue
    return found


def resolve_output_dir(value: str) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def system_resources() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil

        memory = psutil.virtual_memory()
        result.update(
            cpu_percent=round(float(psutil.cpu_percent(interval=None)), 1),
            ram_percent=round(float(memory.percent), 1),
            ram_used_bytes=int(memory.used),
            ram_total_bytes=int(memory.total),
        )
    except ImportError:
        pass
    try:
        query = (
            "utilization.gpu,utilization.memory,memory.used,memory.total,"
            "temperature.gpu,power.draw"
        )
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows: list[list[float]] = []
        for line in completed.stdout.splitlines():
            try:
                values = [float(part.strip()) for part in line.split(",")]
            except ValueError:
                continue
            if len(values) >= 6:
                rows.append(values)
        if rows:
            count = len(rows)
            result.update(
                gpu_count=count,
                gpu_name=f"{count} GPU{'s' if count != 1 else ''} avg",
                gpu_percent=round(sum(row[0] for row in rows) / count, 1),
                gpu_memory_percent=round(sum(row[1] for row in rows) / count, 1),
                gpu_memory_used_mb=round(sum(row[2] for row in rows), 1),
                gpu_memory_total_mb=round(sum(row[3] for row in rows), 1),
                gpu_temperature_c=max(row[4] for row in rows),
                gpu_power_w=round(sum(row[5] for row in rows), 1),
            )
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return result


def discover_runs() -> list[Path]:
    if not RUNS_ROOT.exists():
        return []
    candidates = [
        path
        for path in RUNS_ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")  # skip _smoke / _cachetest scratch dirs
        and ((path / "metrics.jsonl").exists() or (path / "latest.pt").exists())
    ]
    return sorted(
        candidates,
        key=lambda path: max(
            (child.stat().st_mtime for child in path.iterdir() if child.is_file()),
            default=path.stat().st_mtime,
        ),
        reverse=True,
    )


def current_run_only(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """metrics.jsonl is appended across restarts, so the charts show every past run as a
    cliff and become unreadable. Keep only the CURRENT run: slice from the last point where
    ``updates`` drops to a lower value (a fresh learner restarts its update counter). Capped
    so a single very long run can't overwhelm the chart either."""
    start = 0
    for i in range(1, len(metrics)):
        try:
            if float(metrics[i].get("updates", 0)) < float(metrics[i - 1].get("updates", 0)):
                start = i
        except (TypeError, ValueError):
            continue
    return metrics[start:][-4000:]


def _mode(record: dict[str, Any]) -> str:
    return str(record.get("reset_mode") or record.get("stream_mode") or "")


def _milestone(record: dict[str, Any], key: str) -> bool:
    milestones = record.get("milestones")
    return bool(isinstance(milestones, dict) and milestones.get(key))


def _cycle_scored(record: dict[str, Any]) -> bool:
    return bool(
        int(record.get("cycles_completed", 0) or 0) > 0
        or _milestone(record, "cycle_scored")
    )


def compute_cycle_funnel(
    run: Path,
    window: int | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build the selected run's real full-episode cycle funnel.

    Stage C v2 skill resets start part-way through a cycle, so they are reported as
    separate skill streams and never mixed into end-to-end FULL performance.
    """
    all_records = records if records is not None else read_jsonl_all(run / "cycle_telemetry.jsonl")
    if window is not None:
        all_records = all_records[-window:]
    if not all_records:
        return None

    run_config = dict(read_json(run / "run_config.json") or {})
    stage_meta = run_config.get("stagec_v2")
    if not isinstance(stage_meta, dict):
        stage_meta = {}
    try:
        target_load = int(stage_meta.get("target_load", 15))
    except (TypeError, ValueError):
        target_load = 15
    postdump_requires_target = bool(
        stage_meta.get("postdump_require_target_load", False)
    )
    postdump_completes_cycle = bool(
        stage_meta.get("postdump_complete_cycle", False)
    )

    is_v2 = any(record.get("stagec_v2") or _mode(record) for record in all_records)
    if not is_v2:
        curriculum = sum(1 for record in all_records if record.get("neutral_loaded"))
        full_records = [record for record in all_records if not record.get("neutral_loaded")]
        if not full_records:
            return None

        def reached(stage: int) -> int:
            return sum(
                1 for record in full_records if int(record.get("max_stage", 0) or 0) >= stage
            )

        def median_s(key: str) -> float | None:
            values = [float(record[key]) for record in full_records if record.get(key)]
            return round(statistics.median(values) * 0.1, 0) if values else None

        stages = [
            {"key": "left", "label": "Go out", "count": reached(1)},
            {"key": "collect", "label": "Collect", "count": reached(2)},
            {"key": "return", "label": "Return", "count": reached(3)},
            {"key": "shoot", "label": "Shoot again", "count": reached(4)},
        ]
        n = len(full_records)
        for stage in stages:
            stage["pct"] = round(100.0 * int(stage["count"]) / n, 0)
        scored = [int(record.get("scored", 0) or 0) for record in full_records]
        return {
            "episodes": n,
            "curriculum_episodes": curriculum,
            "stages": stages,
            "cycles_completed": sum(
                int(record.get("cycles_completed", 0) or 0) for record in full_records
            ),
            "score_mean": round(statistics.mean(scored), 1),
            "score_max": max(scored),
            "t_leave_s": median_s("t_leave"),
            "t_collect_s": median_s("t_collect"),
            "t_return_s": median_s("t_return"),
            "t_shoot_s": median_s("t_score2"),
            "skill_streams": [],
        }

    full_records = [record for record in all_records if _mode(record) == "full"]
    if not full_records:
        return None
    curriculum = len(all_records) - len(full_records)
    n = len(full_records)
    first_dump = sum(
        1
        for record in full_records
        if record.get("latched")
        or _milestone(record, "latched")
        or int(record.get("dump_empty_completions", 0) or 0) > 0
    )
    has_ramp_out_telemetry = any(
        "ramp_out_attempts" in record for record in full_records
    )
    ramp_out_episodes = sum(
        int(record.get("ramp_out_successes", 0) or 0) > 0
        for record in full_records
    )
    stages = [
        {"key": "dump", "label": "First dump", "count": first_dump},
        {
            "key": "left",
            "label": "Ramp out" if has_ramp_out_telemetry else "Go out",
            "count": (
                ramp_out_episodes
                if has_ramp_out_telemetry
                else sum(_milestone(record, "left_home") for record in full_records)
            ),
        },
        {
            "key": "collect",
            "label": f"Collect {target_load}+",
            "count": sum(_milestone(record, "target_load") for record in full_records),
        },
        {
            "key": "return",
            "label": "Return home",
            "count": sum(_milestone(record, "returned_home") for record in full_records),
        },
        {
            "key": "shoot",
            "label": "Shoot again",
            "count": sum(_cycle_scored(record) for record in full_records),
        },
    ]
    for stage in stages:
        stage["pct"] = round(100.0 * int(stage["count"]) / n, 0)

    def skill_stat(key: str, label: str, predicate: Any) -> dict[str, Any]:
        selected = [record for record in all_records if _mode(record) == key]
        successes = sum(bool(predicate(record)) for record in selected)
        return {
            "key": key,
            "label": label,
            "episodes": len(selected),
            "successes": successes,
            "pct": round(100.0 * successes / len(selected), 1) if selected else None,
        }

    return_records = [record for record in all_records if _mode(record) == "return"]
    return_fire = sum(_cycle_scored(record) or int(record.get("scored", 0) or 0) > 0 for record in return_records)
    skill_streams = [
        skill_stat(
            "postdump",
            (
                f"Complete repeat cycle ({target_load})"
                if postdump_completes_cycle
                else (
                    f"Ramp out + collect {target_load}"
                    if postdump_requires_target
                    else ("Ramp out" if has_ramp_out_telemetry else "Leave")
                )
            ),
            lambda record: (
                (
                    int(record.get("ramp_out_successes", 0) or 0) > 0
                    and _milestone(record, "target_load")
                    and _milestone(record, "returned_home")
                    and _cycle_scored(record)
                    and record.get("terminal_reason") == "skill_success"
                )
                if postdump_completes_cycle
                else (
                    (
                        int(record.get("ramp_out_successes", 0) or 0) > 0
                        and _milestone(record, "target_load")
                        and record.get("terminal_reason") == "skill_success"
                    )
                    if postdump_requires_target
                    else (
                        int(record.get("ramp_out_successes", 0) or 0) > 0
                        if "ramp_out_attempts" in record
                        else _milestone(record, "left_home")
                    )
                )
            ),
        ),
        skill_stat("collect", "Collect", lambda record: _milestone(record, "target_load")),
        skill_stat("return", "Return home", lambda record: _milestone(record, "returned_home")),
        {
            "key": "return_fire",
            "label": "Return + fire",
            "episodes": len(return_records),
            "successes": return_fire,
            "pct": round(100.0 * return_fire / len(return_records), 1)
            if return_records
            else None,
        },
    ]
    scored = [int(record.get("scored", 0) or 0) for record in full_records]
    clean_dumps = sum(
        int(record.get("dump_empty_completions", 0) or 0) > 0
        and int(record.get("partial_dumps", 0) or 0) == 0
        for record in full_records
    )
    return {
        "episodes": n,
        "curriculum_episodes": curriculum,
        "stages": stages,
        "cycles_completed": sum(
            int(record.get("cycles_completed", 0) or 0) for record in full_records
        ),
        "score_mean": round(statistics.mean(scored), 1),
        "score_max": max(scored),
        "clean_dump_episodes": clean_dumps,
        "partial_dumps": sum(int(record.get("partial_dumps", 0) or 0) for record in full_records),
        "ramp_out_attempts": sum(
            int(record.get("ramp_out_attempts", 0) or 0)
            for record in full_records
        ),
        "ramp_out_successes": sum(
            int(record.get("ramp_out_successes", 0) or 0)
            for record in full_records
        ),
        "skill_streams": skill_streams,
        "t_leave_s": None,
        "t_collect_s": None,
        "t_return_s": None,
        "t_shoot_s": None,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.mean(values) if values else None


def match_time_profile(
    full_records: list[dict[str, Any]],
    window: int = 120,
    bin_s: float = 5.0,
    span_s: float = 160.0,
) -> dict[str, Any]:
    """Balls SHOT and balls FERRIED HOME per 5 s bin of match time.

    This is the shape that matters for Stage D: legal scoring only happens while
    our HUB is active, and ferrying is only useful during a blackout, so the
    per-bin profile shows directly whether the policy is using each window.
    ``ferry_land`` events carry the true landed count; older episodes only have
    ferry presses, so their episode total is spread across those presses.
    """
    rows = full_records[-window:]
    n = len(rows)
    bins = max(1, int(span_s / bin_s))
    scored = [0.0] * bins
    ferried = [0.0] * bins
    parity: str | None = None

    def index_of(t: float) -> int:
        return min(bins - 1, max(0, int(float(t) / bin_s)))

    for record in rows:
        parity = record.get("stage_d_first_inactive") or parity
        timeline = record.get("timeline") or []
        presses: list[float] = []
        landed = False
        for event in timeline:
            kind = event.get("ev")
            try:
                when = float(event.get("t", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if kind == "score" and event.get("elig", True):
                scored[index_of(when)] += int(event.get("q", 1) or 1)
            elif kind == "ferry_land":
                ferried[index_of(when)] += int(event.get("n", 1) or 1)
                landed = True
            elif kind == "ferry":
                presses.append(when)
        if not landed and presses:
            total = float(record.get("ferried_balls", 0) or 0)
            if total > 0.0:
                share = total / len(presses)
                for when in presses:
                    ferried[index_of(when)] += share

    dark: list[list[int]] = []
    if parity == "blue":
        dark = [[30, 55], [80, 105]]
    elif parity == "red":
        dark = [[55, 80], [105, 130]]
    return {
        "bin_s": bin_s,
        "n": n,
        "parity": parity,
        "edges": [int(i * bin_s) for i in range(bins)],
        "scored": [round(v / max(n, 1), 2) for v in scored],
        "ferried": [round(v / max(n, 1), 2) for v in ferried],
        "dark": dark,
        "total_scored": round(sum(scored) / max(n, 1), 1),
        "total_ferried": round(sum(ferried) / max(n, 1), 1),
    }


def section_lanes(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-section results for SECTION training (opener / bank / live).

    The old panels all aggregate reset_mode=='full', so during section training
    they go blank. Each lane is judged on a different metric, and the live lane
    is split by its start clock because its windows are NOT the same length:
    SHIFT 2 55-80 and SHIFT 4 105-130 are 25 s, the ENDGAME 130-160 is 30 s.
    """

    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if not n:
            return {}
        num = lambda k: sum(float(r.get(k) or 0) for r in rows) / n
        scored = [int(r.get("scored") or 0) for r in rows]
        return {
            "n": n,
            "scored": round(sum(scored) / n, 1),
            "scored_max": max(scored),
            "dumps": round(num("dump_attempts"), 2),
            "ferried": round(num("ferried_balls"), 1),
            "chamber": round(num("chamber_load"), 1),
            "stockpile": round(num("own_court_stockpile"), 1),
            "conversions": round(num("owncourt_ledger_scored"), 1),
            "cycles": round(num("cycles_completed"), 2),
            "dumped_eps": sum(1 for r in rows if (r.get("dump_attempts") or 0) > 0),
            "conv_eps": sum(1 for r in rows if (r.get("owncourt_ledger_scored") or 0) > 0),
        }

    lanes: dict[str, Any] = {}
    for lane in ("opener", "bank", "live"):
        rows = [r for r in telemetry if _mode(r) == lane]
        if not rows:
            continue
        entry = agg(rows[-400:])
        # split by the window each episode actually drilled
        windows: dict[str, Any] = {}
        for row in rows[-400:]:
            t0 = row.get("lane_t0")
            if t0 is None or float(t0) < 0:
                key = "unknown"
            else:
                key = "%g" % round(float(t0))
            windows.setdefault(key, []).append(row)
        entry["windows"] = {
            k: agg(v) | {"span_s": round(float(v[0].get("lane_end") or 0) - float(k), 1)
                         if k != "unknown" else None}
            for k, v in sorted(windows.items())
        }
        lanes[lane] = entry
    return lanes


def score_distribution(
    full_records: list[dict[str, Any]], window: int = 200, bin_width: int = 10
) -> dict[str, Any]:
    """Histogram of FULL-episode legal scores, split deterministic vs explore.

    Stage-D collectors run a mix of deterministic ("mean") and exploration
    suffix action modes; the deterministic distribution is the honest read and
    the exploration one shows where behavior is heading.  Records without a
    ``suffix_action_mode`` (pre-Stage-D runs) all land in the ``det`` series so
    the chart stays useful on old runs.
    """
    edges = list(range(0, 121, bin_width))  # last bin is open-ended (120+)

    def hist(values: list[int]) -> list[int]:
        counts = [0] * len(edges)
        for value in values:
            counts[min(max(value, 0) // bin_width, len(edges) - 1)] += 1
        return counts

    det = [
        int(record.get("scored", 0) or 0)
        for record in full_records
        if record.get("suffix_action_mode", "mean") == "mean"
    ][-window:]
    explore = [
        int(record.get("scored", 0) or 0)
        for record in full_records
        if record.get("suffix_action_mode") == "explore"
    ][-window:]
    return {
        "edges": edges,
        "det": hist(det),
        "explore": hist(explore),
        "det_n": len(det),
        "explore_n": len(explore),
        "det_mean": round(statistics.mean(det), 1) if det else None,
        "explore_mean": round(statistics.mean(explore), 1) if explore else None,
    }


def normalize_v2_latest(
    latest: dict[str, Any], full_records: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = dict(latest)
    recent = full_records[-20:]
    telemetry_mean = _mean(recent, "scored")
    telemetry_max = max(
        (int(record.get("scored", 0) or 0) for record in recent),
        default=None,
    )
    # Collector telemetry lands as soon as a FULL episode ends, while learner
    # metrics are emitted roughly once a minute.  Prefer telemetry so the
    # headline and chart advance together instead of freezing on an older
    # learner-side aggregate.
    normalized["recent_scored_balls"] = (
        telemetry_mean
        if telemetry_mean is not None
        else normalized.get("recent_full_score_mean")
    )
    normalized["recent_scored_max"] = (
        telemetry_max
        if telemetry_max is not None
        else normalized.get("recent_full_score_max")
    )
    normalized["full_episodes"] = len(full_records)
    normalized.setdefault("recent_score_reward", _mean(recent, "score_reward"))
    normalized.setdefault("recent_collect_reward", _mean(recent, "collect_reward"))
    normalized.setdefault("recent_return_mean", _mean(recent, "return"))
    normalized.setdefault(
        "recent_return_max",
        max((float(record.get("return", 0.0) or 0.0) for record in recent), default=None),
    )
    if normalized.get("q1") is None and normalized.get("q_pi") is not None:
        normalized["q1"] = normalized["q_pi"]
    return normalized


def full_episode_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "scored",
        "score_reward",
        "collect_reward",
        "return",
        "shots_fired",
        "cycles_completed",
        "partial_dumps",
        "policy_train_steps",
        "ferried_balls",
        "owncourt_scored",
    )
    history: list[dict[str, Any]] = []
    recent_scores: deque[float] = deque(maxlen=20)
    for record in records:
        try:
            recent_scores.append(float(record.get("scored", 0) or 0))
        except (TypeError, ValueError):
            recent_scores.append(0.0)
        row = {key: record.get(key) for key in keys}
        row["rolling_scored_mean"] = statistics.mean(recent_scores)
        history.append(row)
    return history[-MAX_HISTORY:]


# --- drive-speed curriculum (added 2026-07-31) ------------------------------
MAX_WHEEL_SPEED_MPS = 4.59
DRIVER_SPEED_RATE = 0.7
DAMPED_SCALE = 0.70
SPEED_ENV_FILE = Path("/root/blue2_env.sh")


def _speed_env():
    out = {}
    try:
        for line in SPEED_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("export "):
                continue
            body = line[len("export "):]
            if "=" not in body:
                continue
            key, _, value = body.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def speed_payload(records):
    """Suffix drive speed + ramp progress at the newest checkpoint step."""
    env = _speed_env()
    steps = 0.0
    for record in reversed(records):
        value = record.get("policy_train_steps")
        if value:
            try:
                steps = float(value)
                break
            except (TypeError, ValueError):
                continue
    fixed = env.get("FRC_POLICY_SPEED_SCALE")
    ramp = env.get("FRC_POLICY_SPEED_RAMP")
    mode = "damped"
    progress = 0.0
    scale = DAMPED_SCALE
    if fixed:
        try:
            scale = float(fixed)
            mode = "fixed"
            progress = 1.0 if scale >= 1.0 else 0.0
        except ValueError:
            pass
    elif ramp and "," in ramp:
        try:
            a, b = (float(x) for x in ramp.split(",", 1))
            if b > a:
                progress = max(0.0, min(1.0, (steps - a) / (b - a)))
                scale = DAMPED_SCALE + progress * (1.0 - DAMPED_SCALE)
                mode = "ramp"
        except ValueError:
            pass
    elif env.get("FRC_POLICY_FULL_SPEED") == "1":
        scale = 1.0
        mode = "full"
        progress = 1.0
    top = MAX_WHEEL_SPEED_MPS * DRIVER_SPEED_RATE
    return {
        "mode": mode,
        "scale": round(scale, 4),
        "suffix_mps": round(top * scale, 3),
        "prefix_mps": round(top * DAMPED_SCALE, 3),
        "cap_mps": round(top, 3),
        "ramp_pct": round(100.0 * progress, 1),
        "pct_of_cap": round(100.0 * scale, 1),
        "train_steps": int(steps),
    }


def run_payload(run: Path, processes: list[TrainingProcess]) -> dict[str, Any]:
    metrics_path = run / "metrics.jsonl"
    metrics = current_run_only(read_jsonl(metrics_path))
    latest = dict(metrics[-1]) if metrics else {}
    summary = read_json(run / "summary.json")
    config = dict(read_json(run / "run_config.json") or {})
    launcher = dict(read_json(run / "launcher_manifest.json") or {})
    stagec_v2 = config.get("stagec_v2")
    is_v2 = isinstance(stagec_v2, dict)
    telemetry = read_jsonl_all(run / "cycle_telemetry.jsonl")
    speed_info = speed_payload(telemetry)
    full_records = [record for record in telemetry if _mode(record) == "full"]
    if is_v2:
        try:
            config.setdefault(
                "num_envs",
                int(config.get("num_collectors", 1)) * int(config.get("collector_envs", 1)),
            )
        except (TypeError, ValueError):
            pass
        if launcher.get("full_episode_s") is not None:
            config.setdefault(
                "episode_len_display",
                f"full {launcher['full_episode_s']} / postdump "
                f"{launcher.get('postdump_episode_s', '?')} s",
            )
        else:
            config.setdefault("episode_len_display", "120 / 75 s")
        preload = stagec_v2.get("return_skill_preload")
        if preload is not None:
            config.setdefault("preload_display", f"return {preload}")
        if config.get("collect_weight") is not None:
            config.setdefault("collect_weight_start", config["collect_weight"])
        latest = normalize_v2_latest(latest, full_records)
    funnel = compute_cycle_funnel(run, records=telemetry)
    run_resolved = run.resolve()
    matches = [
        process
        for process in processes
        if resolve_output_dir(process.output_dir) == run_resolved
    ]
    running = bool(matches)
    configured_start = config.get("started_at_unix")
    started_at = float(
        configured_start
        or (min(process.create_time for process in matches) if matches else run.stat().st_ctime)
    )
    updated_at = (
        metrics_path.stat().st_mtime if metrics_path.exists() else run.stat().st_mtime
    )
    target_minutes = float(
        config.get("minutes")
        or (matches[0].target_minutes if matches else 0.0)
        or 0.0
    )
    elapsed_s = max(
        0.0,
        (time.time() if running else updated_at) - started_at,
    )
    transitions = int(latest.get("transitions") or summary and summary.get("transitions") or 0)
    overall_tps = (
        float(latest.get("transitions_per_s"))
        if latest.get("transitions_per_s") is not None
        else transitions / elapsed_s
        if elapsed_s > 0
        else 0.0
    )
    latest.setdefault("transitions_per_s", round(overall_tps, 3))
    expected_transitions = overall_tps * target_minutes * 60.0
    progress = (
        min(1.0, elapsed_s / (target_minutes * 60.0))
        if target_minutes > 0
        else None
    )
    eta_s = max(0.0, target_minutes * 60.0 - elapsed_s) if running and target_minutes else None
    age_s = max(0.0, time.time() - updated_at)
    final_exists = (run / "final.pt").exists()
    complete = summary is not None or final_exists
    alerts: list[dict[str, str]] = []
    if running and age_s > 150:
        alerts.append({"level": "error", "text": "Metrics are stale; the process may be hung."})
    if not running and not complete and metrics:
        alerts.append(
            {
                "level": "error",
                "text": "Run stopped without final summary; latest checkpoint may still be usable.",
            }
        )
    if is_v2 and latest:
        recent_score = latest.get("recent_scored_balls")
        if recent_score is not None:
            alerts.append(
                {
                    "level": "success",
                    "text": (
                        f"First-dump performance is intact: {float(recent_score):.1f} "
                        f"balls recent FULL average, max {latest.get('recent_scored_max', '—')}."
                    ),
                }
            )
        skill_streams = {
            item["key"]: item for item in (funnel or {}).get("skill_streams", [])
        }
        home = skill_streams.get("return")
        fire = skill_streams.get("return_fire")
        if fire and not fire.get("successes"):
            home_text = (
                f"{home['successes']}/{home['episodes']} return-skill episodes reached home"
                if home and home.get("episodes")
                else "no return-skill episode has reached home"
            )
            alerts.append(
                {
                    "level": "info",
                    "text": (
                        f"Second-cycle bottleneck: {home_text}, but 0/{fire['episodes']} "
                        "returned and fired successfully."
                    ),
                }
            )
        restarts = int(latest.get("collector_restart_boundaries", 0) or 0)
        rejected = int(latest.get("rejected_transitions", 0) or 0)
        if restarts or rejected:
            level = "error" if rejected else "info"
            alerts.append(
                {
                    "level": level,
                    "text": (
                        f"Watchdog recoveries: {restarts}; rejected replay transitions: "
                        f"{rejected}."
                    ),
                }
            )
    else:
        if latest and float(latest.get("recent_score_reward") or 0.0) == 0.0:
            alerts.append(
                {
                    "level": "info",
                    "text": "No scoring reward yet; current behavior is collection-focused.",
                }
            )
        if latest and float(latest.get("recent_collect_reward") or 0.0) > 0.0:
            alerts.append(
                {
                    "level": "success",
                    "text": "Collection reward is active and being measured separately.",
                }
            )
    return {
        "name": run.name,
        "path": str(run),
        "running": running,
        "status": "running" if running else "complete" if complete else "stopped",
        "started_at": datetime.fromtimestamp(started_at).astimezone().isoformat(),
        "updated_at": datetime.fromtimestamp(updated_at).astimezone().isoformat(),
        "metrics_age_s": round(age_s, 1),
        "elapsed_s": round(elapsed_s, 1),
        "target_minutes": target_minutes or None,
        "progress": progress,
        "eta_s": round(eta_s, 1) if eta_s is not None else None,
        "overall_transitions_per_s": round(overall_tps, 3),
        "expected_transitions": round(expected_transitions) if target_minutes else None,
        "latest": latest,
        "history": metrics,
        "episode_history": full_episode_history(full_records),
        "score_distribution": score_distribution(full_records),
        "match_profile": match_time_profile(full_records),
        "section_lanes": section_lanes(telemetry),
        "cycle_funnel": funnel,
        "telemetry_episodes": len(telemetry),
        "drive_speed": speed_info,
        "summary": summary,
        "config": config,
        "processes": [asdict(process) for process in matches],
        "checkpoint": {
            "latest": (run / "latest.pt").exists(),
            "final": final_exists,
            "latest_updated_at": (
                datetime.fromtimestamp((run / "latest.pt").stat().st_mtime)
                .astimezone()
                .isoformat()
                if (run / "latest.pt").exists()
                else None
            ),
        },
        "alerts": alerts,
    }


class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.resource_history: list[dict[str, Any]] = []
        self.telemetry_at = 0.0
        self.cached_processes: list[TrainingProcess] = []
        self.cached_resources: dict[str, Any] = {}
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        threading.Thread(
            target=self._sample_forever,
            name="dashboard-telemetry",
            daemon=True,
        ).start()

    def _sample_forever(self) -> None:
        while True:
            processes = find_training_processes()
            resources = system_resources()
            now = time.time()
            with self.lock:
                self.cached_processes = processes
                self.cached_resources = resources
                self.telemetry_at = now
                self.resource_history.append({"time": now, **resources})
                self.resource_history = self.resource_history[-300:]
            time.sleep(3.0)

    def payload(self, selected: str | None) -> dict[str, Any]:
        runs = discover_runs()
        run = next((path for path in runs if path.name == selected), None)
        if run is None and runs:
            run = runs[0]
        with self.lock:
            resource_history = list(self.resource_history)
            processes = list(self.cached_processes)
            resources = dict(self.cached_resources)
        return {
            "generated_at": datetime.now().astimezone().isoformat(),
            "runs": [
                {
                    "name": path.name,
                    "updated_at": datetime.fromtimestamp(
                        max(
                            (
                                child.stat().st_mtime
                                for child in path.iterdir()
                                if child.is_file()
                            ),
                            default=path.stat().st_mtime,
                        )
                    )
                    .astimezone()
                    .isoformat(),
                }
                for path in runs
            ],
            "selected": run_payload(run, processes) if run else None,
            "resources": resources,
            "resource_history": resource_history,
        }


STATE = DashboardState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path != "/api/status":
            super().log_message(fmt, *args)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            try:
                self.send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self.send_bytes(str(exc).encode(), "text/plain", HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/status":
            selected = parse_qs(parsed.query).get("run", [None])[0]
            body = json.dumps(STATE.payload(selected), ensure_ascii=False).encode()
            self.send_bytes(body, "application/json; charset=utf-8")
            return
        if parsed.path == "/trace":
            run = parse_qs(parsed.query).get("run", ["a"])[0]
            if run not in ("a", "b", "c"):
                run = "a"
            try:
                self.send_bytes((RUNS_ROOT / f"live_trace_{run}.html").read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self.send_bytes(
                    b"<!doctype html><meta http-equiv=refresh content=15>"
                    b"<body style='font:14px/1.6 ui-monospace,monospace;background:#0b0f14;"
                    b"color:#8b98a5;padding:44px'>Trace for run <b>" + run.encode() + b"</b> not "
                    b"generated yet - the eval loop is booting Isaac + running the first "
                    b"deterministic episode (~5-8 min per run). This page auto-refreshes.</body>",
                    "text/html; charset=utf-8",
                )
            return
        self.send_bytes(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="override the runs directory (e.g. /root/autodl-tmp/runs on the "
        "training box); defaults to <repo>/runs",
    )
    args = parser.parse_args()
    if args.runs_root is not None:
        global RUNS_ROOT
        RUNS_ROOT = Path(args.runs_root).resolve()
    STATE.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"TRAINING_DASHBOARD {url}", flush=True)
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
