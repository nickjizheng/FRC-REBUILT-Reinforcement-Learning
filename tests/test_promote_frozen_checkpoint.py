from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rl"
        / "promote_frozen_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("promote_frozen_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path: Path, name: str = "frozen_001.pt") -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(b"opaque checkpoint payload\x00\x01")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _row(sha256: str, **overrides):
    row = {
        "checkpoint": "/root/preserved/frozen_001.pt",
        "checkpoint_sha256": sha256,
        "action_mode": "deterministic",
        "mode": "full",
        "episode_len_s": 160.0,
        "terminal_reason": "horizon",
        "scored": 100,
        "cycles_completed": 2,
        "repeat_scored_load_count": 2,
        "repeat_scored_load_sum": 80,
        "repeat_scored_load_max": 45,
        "timeline": [
            {"t": 10.0, "ev": "score", "q": 60, "u": 0},
            {"t": 140.0, "ev": "score", "q": 35, "u": 5},
        ],
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_promotes_exact_bytes_and_writes_provenance_sidecar(tmp_path):
    module = _module()
    source, source_sha = _checkpoint(tmp_path)
    evaluation = tmp_path / "isolated_det_eval.jsonl"
    # The valid zero must remain in the denominator: (100 + 0) / 2 = 50.
    _write_jsonl(
        evaluation,
        [
            _row(source_sha),
            _row(
                source_sha,
                scored=0,
                cycles_completed=0,
                repeat_scored_load_count=0,
                repeat_scored_load_sum=0,
                repeat_scored_load_max=0,
                timeline=[],
            ),
        ],
    )
    output = tmp_path / "promoted" / "champion.pt"
    sidecar = tmp_path / "promoted" / "champion.promotion.json"
    gates = module.Gates(
        min_episodes=2,
        min_mean_score=50,
        min_max_score=100,
        min_cycle2_rate=0.5,
        min_cycle3_rate=0.5,
        min_endgame_mean=20,
        min_endgame_rate=0.5,
        min_repeat_load_mean=40,
    )

    report = module.promote(source, evaluation, output, sidecar, gates)

    assert output.read_bytes() == source.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == source_sha
    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert persisted == report
    assert persisted["schema"] == module.SCHEMA
    assert persisted["checkpoint"]["source_sha256"] == source_sha
    assert persisted["checkpoint"]["promoted_sha256"] == source_sha
    assert persisted["source_evaluation"]["path"] == str(evaluation.resolve())
    assert persisted["source_evaluation"]["sha256"] == hashlib.sha256(
        evaluation.read_bytes()
    ).hexdigest()
    metrics = persisted["metrics"]
    assert metrics["eligible_episodes"] == 2
    assert metrics["score_mean"] == 50.0
    assert metrics["score_max"] == 100
    assert metrics["cycle2_rate"] == 0.5
    assert metrics["cycle3_rate"] == 0.5
    assert metrics["endgame_score_mean"] == 20.0
    assert metrics["endgame_score_rate"] == 0.5
    assert metrics["repeat_load_mean"] == 40.0
    assert not list(output.parent.glob(".*.tmp"))


def test_corrupt_rows_are_excluded_but_zero_horizon_is_not_score_filtered(tmp_path):
    module = _module()
    source, source_sha = _checkpoint(tmp_path)
    rows = [
        _row(
            source_sha,
            scored=0,
            cycles_completed=0,
            repeat_scored_load_count=0,
            repeat_scored_load_sum=0,
            repeat_scored_load_max=0,
            timeline=[],
        ),
        _row(source_sha, terminal_reason="unhealthy", scored=999),
        _row(source_sha, infra_invalid=True, scored=999),
        _row(source_sha, restart_boundary=True, scored=999),
        _row(
            source_sha,
            collector_generation_start=3,
            collector_generation_end=4,
            scored=999,
        ),
    ]

    module.validate_eval_contract(rows, source_sha)
    metrics = module.compute_metrics(rows)

    assert metrics["input_rows"] == 5
    assert metrics["eligible_episodes"] == 1
    assert metrics["rejected_rows"] == 4
    assert metrics["score_mean"] == 0.0
    assert metrics["score_max"] == 0
    assert metrics["cycle2_rate"] == 0.0
    assert {reason for item in metrics["rejections"] for reason in item["reasons"]} >= {
        "unhealthy",
        "infra_invalid",
        "restart_crossing",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"checkpoint_sha256": "0" * 64}, "checkpoint SHA"),
        ({"action_mode": "policy-noise"}, "action_mode"),
        ({"mode": "return"}, "mode"),
        ({"episode_len_s": 159.0}, "episode_len_s"),
        ({"checkpoint": "/tmp/latest.pt"}, "mutable latest.pt"),
    ],
)
def test_every_row_must_match_frozen_deterministic_full_contract(
    tmp_path, override, message
):
    module = _module()
    _, source_sha = _checkpoint(tmp_path)
    rows = [_row(source_sha), _row(source_sha, **override)]

    with pytest.raises(module.PromotionError, match=message):
        module.validate_eval_contract(rows, source_sha)


def test_failed_gate_creates_no_output(tmp_path):
    module = _module()
    source, source_sha = _checkpoint(tmp_path)
    evaluation = tmp_path / "eval.jsonl"
    _write_jsonl(evaluation, [_row(source_sha, scored=99)])
    output = tmp_path / "promoted.pt"
    sidecar = tmp_path / "promoted.json"

    with pytest.raises(module.PromotionError, match="promotion gates failed"):
        module.promote(
            source,
            evaluation,
            output,
            sidecar,
            module.Gates(min_mean_score=100),
        )

    assert not output.exists()
    assert not sidecar.exists()


def test_latest_checkpoint_is_rejected_before_hash_or_read(tmp_path, monkeypatch):
    module = _module()
    latest, _ = _checkpoint(tmp_path, "latest.pt")
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text("this must not be read", encoding="utf-8")
    touched = False

    def forbidden_hash(_path):
        nonlocal touched
        touched = True
        raise AssertionError("latest.pt was inspected")

    monkeypatch.setattr(module, "sha256_file", forbidden_hash)
    with pytest.raises(module.PromotionError, match="must not be latest.pt"):
        module.promote(
            latest,
            evaluation,
            tmp_path / "promoted.pt",
            tmp_path / "promoted.json",
            module.Gates(),
        )
    assert touched is False


def test_incomplete_jsonl_tail_is_not_silently_accepted(tmp_path):
    module = _module()
    path = tmp_path / "eval.jsonl"
    path.write_text('{"partial": true}', encoding="utf-8")

    with pytest.raises(module.PromotionError, match="incomplete trailing record"):
        module.read_jsonl(path)


def test_existing_outputs_are_never_overwritten(tmp_path):
    module = _module()
    source, source_sha = _checkpoint(tmp_path)
    evaluation = tmp_path / "eval.jsonl"
    _write_jsonl(evaluation, [_row(source_sha)])
    output = tmp_path / "promoted.pt"
    sidecar = tmp_path / "promoted.json"
    output.write_bytes(b"keep me")

    with pytest.raises(module.PromotionError, match="outputs already exist"):
        module.promote(source, evaluation, output, sidecar, module.Gates())

    assert output.read_bytes() == b"keep me"
    assert not sidecar.exists()
