from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rl"
        / "select_fullspeed_retention_teachers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "select_fullspeed_retention_teachers", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _timeline(
    opener: int = 50,
    live1: int = 25,
    live2: int = 10,
    endgame: int = 10,
) -> list[dict[str, object]]:
    return [
        {"ev": "score", "t": 10.0, "q": opener},
        {"ev": "score", "t": 60.0, "q": live1},
        {"ev": "score", "t": 110.0, "q": live2},
        {"ev": "score", "t": 140.0, "q": endgame},
    ]


def _write_harvest(
    tmp_path: Path,
    source_hash: str,
    specifications: list[dict[str, object]],
    *,
    name: str = "harvest",
) -> Path:
    harvest = tmp_path / name
    harvest.mkdir()
    rows = []
    for index, specification in enumerate(specifications):
        capture = harvest / f"capture_{index:02d}.npz"
        timeline = specification.get("timeline", _timeline())
        score = sum(
            int(event.get("q", 0) or 0) + int(event.get("u", 0) or 0)
            for event in timeline
            if event.get("ev") == "score"
        )
        stage_d_contract = specification.get(
            "stage_d_contract",
            {
                "stage_d": True,
                "first_inactive": "blue",
                "ferry": False,
                "return_when_live": False,
                "owncourt_loop": False,
                "policy_speed_scale": 1.0,
                "prefix_rescue_s": 35.0,
            },
        )
        row = {
            "capture_path": str(capture),
            "checkpoint_sha256": specification.get("checkpoint_sha256", source_hash),
            "action_mode": specification.get("action_mode", "deterministic"),
            "mode": specification.get("mode", "full"),
            "episode_len_s": specification.get("episode_len_s", 160.0),
            "terminal_reason": specification.get("terminal_reason", "horizon"),
            "env_seed": specification.get("seed", index),
            "env_index": specification.get("env_index", index % 2),
            "scored": specification.get("scored", score),
            "collected": specification.get("collected", score + 30),
            "cycles_completed": specification.get("cycles", 3),
            "repeat_scored_load_max": specification.get(
                "repeat_scored_load_max", 0
            ),
            "ferried_balls": specification.get("ferried_balls", 0),
            "owncourt_score_entries": specification.get("owncourt_score_entries", 0),
            "owncourt_shots": specification.get("owncourt_shots", 0),
            "owncourt_scored": specification.get("owncourt_scored", 0),
            "stage_d_contract": stage_d_contract,
            "timeline": timeline,
        }
        episode = {key: value for key, value in row.items() if key != "capture_path"}
        episode.update(specification.get("episode_overrides", {}))
        metadata = {
            "schema": "stagec_training_episode_v1",
            "episode": episode,
        }
        np.savez_compressed(
            capture,
            metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
        )
        row.update(specification.get("row_overrides", {}))
        rows.append(row)
    (harvest / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return harvest


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    source: Path,
    harvest: Path | list[Path],
    output: Path,
    report: Path,
    extra: list[str] | None = None,
) -> None:
    harvests = [harvest] if isinstance(harvest, Path) else harvest
    harvest_args = [
        argument
        for path in harvests
        for argument in ("--harvest-dir", str(path))
    ]
    argv = [
        "select_fullspeed_retention_teachers.py",
        "--source-checkpoint",
        str(source),
        *harvest_args,
        "--output-dir",
        str(output),
        "--report",
        str(report),
        *(extra or []),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def test_phase_scores_use_legal_windows_and_grace_boundaries():
    module = _module()
    timeline = [
        {"ev": "score", "t": 0.0, "q": 1},
        {"ev": "score", "t": 32.999, "q": 2},
        {"ev": "score", "t": 33.0, "q": 50},
        {"ev": "score", "t": 55.0, "q": 3},
        {"ev": "score", "t": 82.999, "u": 4},
        {"ev": "score", "t": 83.0, "q": 50},
        {"ev": "score", "t": 105.0, "q": 5},
        {"ev": "score", "t": 129.999, "q": 6},
        {"ev": "score", "t": 130.0, "q": 7},
        {"ev": "score", "t": 160.0, "q": 8},
        {"ev": "score", "t": 160.001, "q": 50},
        {"ev": "collect", "t": 10.0, "q": 50},
    ]
    assert module.phase_scores(timeline) == {
        "opener": 3,
        "live1": 7,
        "live2": 11,
        "endgame": 15,
    }


def test_main_selects_phase_complete_seed_and_env_diverse_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": 1, "env_index": 0, "scored": 200, "cycles": 4},
            {"seed": 1, "env_index": 0, "scored": 199, "cycles": 4},
            {"seed": 1, "env_index": 1, "scored": 198, "cycles": 4},
            {"seed": 2, "env_index": 0, "scored": 180, "cycles": 5},
            {"seed": 3, "env_index": 0, "scored": 179, "cycles": 4},
            {"seed": 4, "env_index": 1, "scored": 178, "cycles": 4},
            {"seed": 5, "env_index": 1, "scored": 177, "cycles": 3},
            # These must never rescue the bank: one is unhealthy, one lacks
            # endgame score, and one has the wrong checkpoint provenance.
            {"seed": 6, "terminal_reason": "unhealthy", "scored": 999},
            {"seed": 7, "timeline": _timeline(endgame=9), "scored": 999},
            {"seed": 8, "checkpoint_sha256": "f" * 64, "scored": 999},
        ],
    )
    output = tmp_path / "teachers"
    report_path = tmp_path / "report.json"
    _run_main(
        monkeypatch,
        module,
        source=source,
        harvest=harvest,
        output=output,
        report=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["selected"]
    seed_counts = Counter(row["seed"] for row in selected)
    assert report["selected_count"] == 5
    assert report["qualifying_candidate_count"] == 7
    assert report["selected_unique_seed_count"] >= 4
    assert max(seed_counts.values()) <= 2
    assert {row["env_index"] for row in selected} == {0, 1}
    assert all(
        row["phase_scores"]
        == {"opener": 50, "live1": 25, "live2": 10, "endgame": 10}
        for row in selected
    )
    assert len(list(output.glob("teacher_*.npz"))) == 5


def test_main_combines_harvests_and_applies_all_hard_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    phase_complete = _timeline(opener=55, live1=40, live2=15, endgame=15)
    common = {
        "timeline": phase_complete,
        "collected": 200,
        "cycles": 4,
        "repeat_scored_load_max": 35,
    }
    wave1 = _write_harvest(
        tmp_path,
        source_hash,
        [
            {**common, "seed": 101, "env_index": 0, "scored": 190},
            {**common, "seed": 102, "env_index": 0, "scored": 184},
            # Each near miss isolates one of the new global hard gates.
            {**common, "seed": 103, "env_index": 1, "scored": 159},
            {
                **common,
                "seed": 104,
                "env_index": 1,
                "scored": 180,
                "collected": 179,
            },
        ],
        name="wave1",
    )
    wave2 = _write_harvest(
        tmp_path,
        source_hash,
        [
            {**common, "seed": 201, "env_index": 0, "scored": 186},
            {**common, "seed": 202, "env_index": 0, "scored": 184},
            {**common, "seed": 203, "env_index": 1, "scored": 168},
            {
                **common,
                "seed": 204,
                "env_index": 1,
                "scored": 180,
                "cycles": 3,
            },
            {
                **common,
                "seed": 205,
                "env_index": 1,
                "scored": 180,
                "repeat_scored_load_max": 31,
            },
        ],
        name="wave2",
    )
    output = tmp_path / "teachers"
    report_path = tmp_path / "report.json"

    _run_main(
        monkeypatch,
        module,
        source=source,
        harvest=[wave1, wave2],
        output=output,
        report=report_path,
        extra=[
            "--score-slots",
            "5",
            "--min-score-cycles",
            "4",
            "--min-score",
            "160",
            "--min-collected",
            "180",
            "--min-cycles",
            "4",
            "--min-repeat-scored-load-max",
            "32",
            "--min-opener-score",
            "55",
            "--min-live1-score",
            "40",
            "--min-live2-score",
            "15",
            "--min-endgame-score",
            "15",
        ],
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["selected"]
    assert report["harvest_dirs"] == [str(wave1.resolve()), str(wave2.resolve())]
    assert report["candidate_minimums"] == {
        "score": 160,
        "collected": 180,
        "cycles": 4,
        "repeat_scored_load_max": 32,
    }
    assert report["candidate_count"] == 9
    assert report["qualifying_candidate_count"] == 5
    assert {row["seed"] for row in selected} == {101, 102, 201, 202, 203}
    assert {row["env_index"] for row in selected} == {0, 1}
    assert all(row["repeat_scored_load_max"] >= 32 for row in selected)
    assert len(list(output.glob("teacher_*.npz"))) == 5


def test_main_fails_closed_when_phase_complete_clean_count_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": 1},
            {"seed": 2},
            {"seed": 3},
            {"seed": 4},
            {"seed": 5, "timeline": _timeline(endgame=0), "scored": 500},
            {"seed": 6, "terminal_reason": "unhealthy", "scored": 500},
        ],
    )
    output = tmp_path / "teachers"
    with pytest.raises(SystemExit, match="4 qualifying clean exact-source"):
        _run_main(
            monkeypatch,
            module,
            source=source,
            harvest=harvest,
            output=output,
            report=tmp_path / "report.json",
        )
    assert not output.exists()


def test_main_fails_closed_when_seed_diversity_is_impossible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": 1, "env_index": 0},
            {"seed": 1, "env_index": 1},
            {"seed": 2, "env_index": 0},
            {"seed": 2, "env_index": 1},
            {"seed": 3, "env_index": 0},
            {"seed": 3, "env_index": 1},
        ],
    )
    output = tmp_path / "teachers"
    with pytest.raises(SystemExit, match="cannot satisfy teacher-bank diversity"):
        _run_main(
            monkeypatch,
            module,
            source=source,
            harvest=harvest,
            output=output,
            report=tmp_path / "report.json",
        )
    assert not output.exists()


def test_embedded_capture_metadata_cannot_be_overridden_by_jsonl_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": 1},
            {"seed": 2},
            {"seed": 3},
            {"seed": 4},
            # JSONL still claims a clean exact-source deterministic full match,
            # but the immutable capture says it terminated unhealthy.
            {"seed": 5, "episode_overrides": {"terminal_reason": "unhealthy"}},
            # JSONL has a complete four-window timeline, while the capture's
            # embedded endgame evidence is below the hard phase minimum.
            {"seed": 6, "episode_overrides": {"timeline": _timeline(endgame=0)}},
        ],
    )
    output = tmp_path / "teachers"
    with pytest.raises(SystemExit, match="4 qualifying clean exact-source"):
        _run_main(
            monkeypatch,
            module,
            source=source,
            harvest=harvest,
            output=output,
            report=tmp_path / "report.json",
        )
    assert not output.exists()


def test_no_ferry_owncourt_contract_checks_jsonl_and_embedded_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    ferry_timeline = [*_timeline(), {"ev": "ferry", "t": 40.0}]
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": 1, "env_index": 0},
            {"seed": 2, "env_index": 1},
            {"seed": 3, "env_index": 0},
            {"seed": 4, "env_index": 1},
            {"seed": 5, "env_index": 0},
            # Only the embedded record discloses this ferry.
            {"seed": 6, "episode_overrides": {"ferried_balls": 1}},
            # Only JSONL discloses this own-court usage.
            {"seed": 7, "owncourt_shots": 1, "episode_overrides": {"owncourt_shots": 0}},
            # A timeline event is forbidden even when every counter is zero.
            {"seed": 8, "episode_overrides": {"timeline": ferry_timeline}},
            {"seed": 9, "episode_overrides": {"timeline": [*_timeline(), {"ev": "oc_entry", "t": 90.0}]}},
        ],
    )
    output = tmp_path / "teachers"
    report_path = tmp_path / "report.json"
    _run_main(
        monkeypatch,
        module,
        source=source,
        harvest=harvest,
        output=output,
        report=report_path,
        extra=["--require-no-ferry-owncourt"],
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["require_no_ferry_owncourt"] is True
    assert report["expected_stage_d_contract"] == {
        "stage_d": True,
        "first_inactive": "blue",
        "ferry": False,
        "return_when_live": False,
        "owncourt_loop": False,
        "policy_speed_scale": 1.0,
        "prefix_rescue_s": 35.0,
    }
    assert report["auxiliary_behavior_contract"] == {
        "enabled": True,
        "counter_fields": [
            "ferried_balls",
            "owncourt_score_entries",
            "owncourt_shots",
            "owncourt_scored",
        ],
        "forbidden_timeline_events": ["ferry", "oc_entry"],
        "accepted_candidate_usage_count": 0,
        "selected_usage_count": 0,
    }
    assert report["rejection_counts"]["capture_auxiliary_behavior"] == 3
    assert report["rejection_counts"]["jsonl_auxiliary_behavior"] == 1
    assert all(not row["auxiliary_usage"]["used"] for row in report["selected"])


def test_no_aux_contract_allows_frozen_prefix_ferry_press(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"frozen-source")
    output = tmp_path / "selected"
    report_path = tmp_path / "report.json"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    specs = []
    for seed in range(10, 15):
        timeline = [{"ev": "ferry", "t": 3.5}, *_timeline()]
        specs.append({"seed": seed, "timeline": timeline})
    harvest = _write_harvest(tmp_path, source_hash, specs)

    _run_main(
        monkeypatch,
        module,
        source=source,
        harvest=harvest,
        output=output,
        report=report_path,
        extra=["--require-no-ferry-owncourt"],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["selected_count"] == 5
    assert all(not row["auxiliary_usage"]["used"] for row in report["selected"])


def test_no_aux_contract_requires_declared_disabled_mechanics_in_both_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    valid_contract = {
        "stage_d": True,
        "first_inactive": "blue",
        "ferry": False,
        "return_when_live": False,
        "owncourt_loop": False,
        "policy_speed_scale": 1.0,
        "prefix_rescue_s": 35.0,
    }
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": 1, "env_index": 0},
            {"seed": 2, "env_index": 1},
            {"seed": 3, "env_index": 0},
            {"seed": 4, "env_index": 1},
            {"seed": 5, "env_index": 0},
            {"seed": 6, "row_overrides": {"stage_d_contract": None}},
            {
                "seed": 7,
                "episode_overrides": {
                    "stage_d_contract": {**valid_contract, "return_when_live": True}
                },
            },
            {
                "seed": 8,
                "episode_overrides": {
                    "stage_d_contract": {**valid_contract, "policy_speed_scale": 0.7}
                },
            },
            {
                "seed": 9,
                "row_overrides": {
                    "stage_d_contract": {**valid_contract, "prefix_rescue_s": 0.0}
                },
            },
        ],
    )
    report_path = tmp_path / "report.json"
    _run_main(
        monkeypatch,
        module,
        source=source,
        harvest=harvest,
        output=tmp_path / "teachers",
        report=report_path,
        extra=["--require-no-ferry-owncourt"],
    )

    rejection_counts = json.loads(report_path.read_text())["rejection_counts"]
    assert rejection_counts["jsonl_missing_stage_d_contract"] == 1
    assert rejection_counts["capture_stage_d_contract_return_when_live"] == 1
    assert rejection_counts["capture_stage_d_contract_policy_speed_scale"] == 1
    assert rejection_counts["jsonl_stage_d_contract_prefix_rescue_s"] == 1


def test_explicit_prefix_rescue_contract_overrides_no_aux_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = {
        "stage_d": True,
        "first_inactive": "blue",
        "ferry": False,
        "return_when_live": False,
        "owncourt_loop": False,
        "policy_speed_scale": 1.0,
        "prefix_rescue_s": 42.5,
    }
    harvest = _write_harvest(
        tmp_path,
        source_hash,
        [
            {"seed": seed, "env_index": seed % 2, "stage_d_contract": contract}
            for seed in range(1, 6)
        ],
    )
    report_path = tmp_path / "report.json"
    _run_main(
        monkeypatch,
        module,
        source=source,
        harvest=harvest,
        output=tmp_path / "teachers",
        report=report_path,
        extra=[
            "--require-no-ferry-owncourt",
            "--expected-prefix-rescue-s",
            "42.5",
        ],
    )

    report = json.loads(report_path.read_text())
    assert report["selected_count"] == 5
    assert report["expected_stage_d_contract"]["prefix_rescue_s"] == 42.5
