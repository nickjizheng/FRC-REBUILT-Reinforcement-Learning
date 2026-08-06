"""Safely summarize partial Stage C seed-mining results.

The reader takes a snapshot of every ``*.episodes.jsonl`` beneath a run
directory.  A writer may be appending at the same time: only newline-terminated
records are considered, so a half-written trailing JSON object is ignored until
the next invocation.  This script never opens checkpoints, replay transport, or
simulator state.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import socket
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "stagec_seedmine_combined_v1"
FUNNEL_MILESTONES = (
    "latched",
    "left_home",
    "target_load",
    "returned_home",
    "cycle_dumped",
    "cycle_scored",
)


def read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read complete JSONL records without treating an active tail as corrupt."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        return [], {
            "path": str(path),
            "bytes": 0,
            "records": 0,
            "incomplete_tail": False,
            "parse_errors": 0,
            "read_error": repr(exc),
        }
    incomplete_tail = bool(data and not data.endswith(b"\n"))
    lines = data.split(b"\n")
    if incomplete_tail:
        lines = lines[:-1]
    records: list[dict[str, Any]] = []
    parse_errors = 0
    for raw in lines:
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parse_errors += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            parse_errors += 1
    return records, {
        "path": str(path),
        "bytes": len(data),
        "records": len(records),
        "incomplete_tail": incomplete_tail,
        "parse_errors": parse_errors,
        "read_error": None,
    }


def _milestone(record: dict[str, Any], name: str) -> bool:
    return int((record.get("milestones") or {}).get(name, 0) or 0) > 0


def _cycle_success(record: dict[str, Any]) -> bool:
    return bool(
        int(record.get("cycles_completed", 0) or 0) >= 1
        or _milestone(record, "cycle_scored")
    )


def _clean_cycle_success(record: dict[str, Any]) -> bool:
    if not _cycle_success(record):
        return False
    required_dumps = 2 if str(record.get("mode", record.get("reset_mode", "full"))) == "full" else 1
    return bool(
        int(record.get("partial_dumps", 0) or 0) == 0
        and int(record.get("dump_empty_completions", 0) or 0) >= required_dumps
        and (_milestone(record, "cycle_dumped") or _milestone(record, "cycle_scored"))
    )


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    count = len(rows)
    scores = [int(row.get("scored", 0) or 0) for row in rows]
    cycles = sum(_cycle_success(row) for row in rows)
    clean_cycles = sum(_clean_cycle_success(row) for row in rows)
    funnel_counts = {
        name: sum(_milestone(row, name) for row in rows) for name in FUNNEL_MILESTONES
    }
    # Older/equivalent telemetry can carry cycles_completed without duplicating
    # cycle_scored in milestones.  Do not undercount the final funnel stage.
    funnel_counts["cycle_scored"] = max(funnel_counts["cycle_scored"], cycles)
    funnel = {
        name: {
            "count": int(value),
            "rate": round(float(value) / count, 6) if count else 0.0,
        }
        for name, value in funnel_counts.items()
    }
    returned_not_scored = sum(
        _milestone(row, "returned_home") and not _cycle_success(row) for row in rows
    )
    loaded_not_returned = sum(
        _milestone(row, "target_load") and not _milestone(row, "returned_home")
        for row in rows
    )
    left_not_loaded = sum(
        _milestone(row, "left_home") and not _milestone(row, "target_load")
        for row in rows
    )
    capture_paths = sorted(
        {str(row["capture_path"]) for row in rows if row.get("capture_path")}
    )
    return {
        "episodes": count,
        "cycle_successes": int(cycles),
        "cycle_rate": round(float(cycles) / count, 6) if count else 0.0,
        "clean_cycle_successes": int(clean_cycles),
        "clean_cycle_rate": round(float(clean_cycles) / count, 6) if count else 0.0,
        "funnel": funnel,
        "near_success": {
            "returned_home_not_scored": int(returned_not_scored),
            "target_load_not_returned": int(loaded_not_returned),
            "left_home_not_loaded": int(left_not_loaded),
        },
        "score": {
            "mean": round(float(statistics.fmean(scores)), 3) if scores else 0.0,
            "median": round(float(statistics.median(scores)), 3) if scores else 0.0,
            "p90": round(_percentile(scores, 0.9), 3),
            "min": min(scores, default=0),
            "max": max(scores, default=0),
            "total": sum(scores),
            "episodes_65_plus": sum(score >= 65 for score in scores),
            "episodes_100_plus": sum(score >= 100 for score in scores),
        },
        "captures": {
            "records": len(capture_paths),
            "cycle": sum(row.get("capture_tier") == "cycle" for row in rows),
            "returned_home": sum(
                row.get("capture_tier") == "returned_home" for row in rows
            ),
            "paths": capture_paths,
        },
    }


def _lane_id(run_dir: Path, episode_file: Path, records: list[dict[str, Any]]) -> str:
    for record in records:
        if record.get("lane"):
            return str(record["lane"])
    relative = episode_file.relative_to(run_dir).as_posix()
    suffix = ".episodes.jsonl"
    return relative[: -len(suffix)] if relative.endswith(suffix) else relative


def _checkpoint_identity(record: dict[str, Any]) -> tuple[str, str, str | None]:
    sha = str(record.get("checkpoint_sha256") or "").lower()
    path = str(record.get("checkpoint") or "unknown")
    key = sha if sha else path
    label = Path(path).name if path != "unknown" else "unknown"
    return key, label, sha or None


def parse_pid_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {"pid": None, "host": None, "parse_error": repr(exc)}
    if not text:
        return {"pid": None, "host": None, "parse_error": "empty PID file"}
    host = None
    pid: int | None = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        raw_pid = value.get("pid")
        host = value.get("host", value.get("hostname"))
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            pid = None
    else:
        candidate = text
        if "=" in candidate:
            candidate = candidate.rsplit("=", 1)[-1].strip()
        try:
            pid = int(candidate)
        except ValueError:
            pid = None
    if pid is None or pid <= 0:
        return {"pid": None, "host": host, "parse_error": "invalid PID"}
    return {"pid": pid, "host": str(host) if host else None, "parse_error": None}


def _host_is_local(host: str | None) -> bool:
    if not host:
        return True
    candidate = host.lower().rstrip(".")
    local = {
        "localhost",
        "127.0.0.1",
        "::1",
        socket.gethostname().lower().rstrip("."),
        socket.getfqdn().lower().rstrip("."),
    }
    short = candidate.split(".", 1)[0]
    return candidate in local or short in {value.split(".", 1)[0] for value in local}


def local_pid_alive(pid: int) -> bool:
    """Portable existence check that never sends a terminating Windows signal."""

    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        open_process = kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        open_process.restype = ctypes.c_void_p
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        get_exit_code.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        handle = open_process(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            close_handle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def discover_processes(run_dir: Path) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.pid")):
        parsed = parse_pid_file(path)
        host = parsed["host"]
        if parsed["pid"] is None:
            status = "invalid"
            alive = None
        elif host and not _host_is_local(host):
            status = "remote_unchecked"
            alive = None
        else:
            alive = local_pid_alive(int(parsed["pid"]))
            status = "running" if alive else "stopped"
        processes.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "name": path.stem,
                **parsed,
                "host_assumed_local": host is None,
                "alive": alive,
                "status": status,
            }
        )
    return processes


def build_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    episode_files = sorted(run_dir.rglob("*.episodes.jsonl"))
    file_diagnostics: list[dict[str, Any]] = []
    lane_records: dict[str, list[dict[str, Any]]] = {}
    lane_files: dict[str, str] = {}
    all_records: list[dict[str, Any]] = []
    for path in episode_files:
        records, diagnostics = read_jsonl_snapshot(path)
        diagnostics["path"] = path.relative_to(run_dir).as_posix()
        file_diagnostics.append(diagnostics)
        lane = _lane_id(run_dir, path, records)
        lane_records.setdefault(lane, []).extend(records)
        lane_files[lane] = path.relative_to(run_dir).as_posix()
        all_records.extend(records)

    checkpoint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoint_info: dict[str, tuple[str, str | None, str | None, int | None]] = {}
    lane_checkpoint: dict[str, str] = {}
    lanes: list[dict[str, Any]] = []
    processes = discover_processes(run_dir)
    for lane, records in sorted(lane_records.items()):
        identities = [_checkpoint_identity(record) for record in records]
        if identities:
            key, label, sha = identities[0]
            path = str(records[0].get("checkpoint") or "unknown")
            steps = records[0].get("checkpoint_v2_updates")
            steps = int(steps) if steps is not None else None
        else:
            key, label, sha, path, steps = f"empty:{lane}", "unknown", None, None, None
        lane_checkpoint[lane] = key
        checkpoint_groups[key].extend(records)
        checkpoint_info[key] = (label, sha, path, steps)
        process_paths = [
            process["path"]
            for process in processes
            if Path(process["path"]).parent.as_posix() == Path(lane_files[lane]).parent.as_posix()
        ]
        lanes.append(
            {
                "lane": lane,
                "episode_file": lane_files[lane],
                "checkpoint_key": key,
                "checkpoint_label": label,
                "checkpoint_sha256": sha,
                "process_pid_files": process_paths,
                **aggregate_records(records),
            }
        )

    checkpoints = []
    for key, records in sorted(checkpoint_groups.items()):
        label, sha, path, steps = checkpoint_info[key]
        checkpoints.append(
            {
                "checkpoint_key": key,
                "checkpoint_label": label,
                "checkpoint": path,
                "checkpoint_sha256": sha,
                "checkpoint_v2_updates": steps,
                "lanes": sorted(lane for lane, lane_key in lane_checkpoint.items() if lane_key == key),
                **aggregate_records(records),
            }
        )

    referenced = {
        Path(path).name
        for path in aggregate_records(all_records)["captures"]["paths"]
    }
    discovered_paths = sorted(run_dir.rglob("episode_*.npz"))
    discovered = {path.name for path in discovered_paths}
    warnings = []
    if not episode_files:
        warnings.append("no *.episodes.jsonl files found")
    if any(item["incomplete_tail"] for item in file_diagnostics):
        warnings.append("one or more active JSONL tails were ignored")
    if any(item["parse_errors"] for item in file_diagnostics):
        warnings.append("one or more complete JSONL lines could not be parsed")
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "clean_cycle_definition": (
            "qualified cycle_scored/cycles_completed, zero partial dumps, and at least "
            "two clean dumps for FULL or one for RETURN"
        ),
        "files": {
            "matched": len(episode_files),
            "records": len(all_records),
            "incomplete_tails": sum(item["incomplete_tail"] for item in file_diagnostics),
            "parse_errors": sum(item["parse_errors"] for item in file_diagnostics),
            "diagnostics": file_diagnostics,
        },
        "overall": aggregate_records(all_records),
        "checkpoints": checkpoints,
        "lanes": lanes,
        "captured_files": {
            "referenced": len(referenced),
            "discovered": len(discovered),
            "missing_basenames": sorted(referenced - discovered),
            "orphaned_basenames": sorted(discovered - referenced),
            "paths": [path.relative_to(run_dir).as_posix() for path in discovered_paths],
        },
        "processes": processes,
        "warnings": warnings,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def render_text(summary: dict[str, Any]) -> str:
    lines = [
        f"Seed mining: {summary['files']['records']} episodes across "
        f"{len(summary['lanes'])} lanes / {len(summary['checkpoints'])} checkpoints"
    ]
    for checkpoint in summary["checkpoints"]:
        lines.append(
            f"CHECKPOINT {checkpoint['checkpoint_label']} "
            f"eps={checkpoint['episodes']} cycles={checkpoint['cycle_successes']} "
            f"clean={checkpoint['clean_cycle_successes']} "
            f"score_mean={checkpoint['score']['mean']:.1f} "
            f"score_max={checkpoint['score']['max']}"
        )
        for lane in summary["lanes"]:
            if lane["checkpoint_key"] != checkpoint["checkpoint_key"]:
                continue
            funnel = lane["funnel"]
            lines.append(
                f"  {lane['lane']}: eps={lane['episodes']} clean={lane['clean_cycle_successes']} "
                f"load={funnel['target_load']['count']} home={funnel['returned_home']['count']} "
                f"score2={funnel['cycle_scored']['count']} max={lane['score']['max']}"
            )
    for process in summary["processes"]:
        lines.append(
            f"PROCESS {process['path']} pid={process['pid']} status={process['status']}"
        )
    if summary["files"]["incomplete_tails"]:
        lines.append(
            f"ACTIVE_TAILS ignored={summary['files']['incomplete_tails']} "
            "(normal while writers are running)"
        )
    for warning in summary["warnings"]:
        lines.append(f"WARNING {warning}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="print combined JSON")
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically write combined_summary.json beneath run_dir",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override summary output path; requires --write",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")
    if args.out is not None and not args.write:
        parser.error("--out requires --write")
    summary = build_summary(args.run_dir)
    if args.write:
        output = args.out or (args.run_dir / "combined_summary.json")
        atomic_write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else render_text(summary))


if __name__ == "__main__":
    main()
