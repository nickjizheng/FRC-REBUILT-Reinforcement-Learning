"""Live local dashboard for xRC reinforcement-learning runs.

Usage:
    python scripts/rl/training_dashboard.py

The server binds to localhost only.  It discovers ``runs/drqv2_*`` folders,
tails JSONL metrics, checks the real training process, and reports laptop
CPU/RAM/GPU telemetry without adding dependencies to the training process.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
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


def cli_value(command: list[str], flag: str, default: Any = None) -> Any:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default


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
            if not any(Path(part).name.lower() == "train_drqv2.py" for part in command):
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
                    num_envs=int(cli_value(command, "--num-envs", 2)),
                    target_minutes=float(cli_value(command, "--minutes", 20.0)),
                    replay_capacity=int(cli_value(command, "--replay-capacity", 60_000)),
                    output_dir=output,
                )
            )
        except (OSError, ValueError, TypeError):
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
        values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        if len(values) >= 6:
            result.update(
                gpu_percent=float(values[0]),
                gpu_memory_percent=float(values[1]),
                gpu_memory_used_mb=float(values[2]),
                gpu_memory_total_mb=float(values[3]),
                gpu_temperature_c=float(values[4]),
                gpu_power_w=float(values[5]),
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
        and path.name.startswith("drqv2_")
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


def run_payload(run: Path, processes: list[TrainingProcess]) -> dict[str, Any]:
    metrics_path = run / "metrics.jsonl"
    metrics = read_jsonl(metrics_path)
    latest = metrics[-1] if metrics else {}
    summary = read_json(run / "summary.json")
    config = read_json(run / "run_config.json") or {}
    run_resolved = run.resolve()
    matches = [
        process
        for process in processes
        if resolve_output_dir(process.output_dir) == run_resolved
    ]
    running = bool(matches)
    started_at = (
        min(process.create_time for process in matches)
        if matches
        else float(config.get("started_at_unix") or run.stat().st_ctime)
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
    expected_transitions = overall_tps * target_minutes * 60.0
    progress = (
        min(1.0, elapsed_s / (target_minutes * 60.0))
        if target_minutes > 0
        else None
    )
    eta_s = max(0.0, target_minutes * 60.0 - elapsed_s) if running and target_minutes else None
    age_s = max(0.0, time.time() - updated_at)
    alerts: list[dict[str, str]] = []
    if running and age_s > 150:
        alerts.append({"level": "error", "text": "Metrics are stale; the process may be hung."})
    if not running and summary is None and metrics:
        alerts.append(
            {
                "level": "error",
                "text": "Run stopped without final summary; latest checkpoint may still be usable.",
            }
        )
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
        "status": "running" if running else "complete" if summary else "stopped",
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
        "summary": summary,
        "config": config,
        "processes": [asdict(process) for process in matches],
        "checkpoint": {
            "latest": (run / "latest.pt").exists(),
            "final": (run / "final.pt").exists(),
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
        self.send_bytes(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
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
