"""Focused tests for the evaluation-only checkpoint interpolation tool."""
from __future__ import annotations

import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "derive_interpolated_checkpoint",
        ROOT / "tools" / "derive_interpolated_checkpoint.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_counter_derivation_rejects_fractional_values():
    module = _module()
    assert module._derive_counter(3_195_000, 3_200_000, module.Fraction(1, 5), "v2") == 3_196_000
    assert module._derive_counter(3_200_000, 3_205_000, module.Fraction(2, 5), "v2") == 3_202_000
    with pytest.raises(ValueError, match="does not land on an integer"):
        module._derive_counter(0, 1, module.Fraction(1, 2), "counter")


def test_sha_parser_is_fail_closed():
    module = _module()
    assert module.expected_sha256("A" * 64, "test") == "a" * 64
    with pytest.raises(ValueError, match="complete SHA-256"):
        module.expected_sha256("abc", "test")


def _state(torch, value: float):
    state = OrderedDict(
        (
            ("weight", torch.tensor([[value, value + 1.0]], dtype=torch.float32)),
            ("counter", torch.tensor(7, dtype=torch.int64)),
        )
    )
    state._metadata = OrderedDict((("", {"version": 1}),))
    return state


def _payload(torch, model_value: float, *, v2_updates: int, train_steps: int):
    optimizer = {
        "state": {
            0: {
                "step": torch.tensor(float(v2_updates), dtype=torch.float32),
                "exp_avg": torch.tensor([model_value], dtype=torch.float32),
                "exp_avg_sq": torch.tensor([model_value + 1.0], dtype=torch.float32),
            }
        },
        "param_groups": [
            {
                "lr": 1e-5,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "params": [0],
            }
        ],
    }
    return {
        "encoder": _state(torch, model_value),
        "actor": _state(torch, model_value + 10.0),
        "critic": _state(torch, model_value + 20.0),
        "critic_target": _state(torch, model_value + 30.0),
        "encoder_opt": optimizer,
        "actor_opt": optimizer,
        "critic_opt": optimizer,
        "train_steps": train_steps,
        "skipped_updates": 1 if model_value == 0.0 else 2,
        "explore_offset": 1_163_753,
        "v2_updates": v2_updates,
        "elite_updates": 10 if model_value == 0.0 else 20,
        "actor_updates": 100 if model_value == 0.0 else 200,
        "stagec_v2": {
            "schema": "test_static_contract",
            "prefix_sha256": "f" * 64,
            "policy_speed_scale": 1.0,
        },
    }


def test_end_to_end_interpolates_models_but_preserves_optimizer_and_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    module = _module()
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    output = tmp_path / "derived.pt"
    provenance = tmp_path / "derived.provenance.json"
    torch.save(
        _payload(torch, 0.0, v2_updates=3_195_000, train_steps=4_358_753),
        left,
    )
    torch.save(
        _payload(torch, 10.0, v2_updates=3_200_000, train_steps=4_363_753),
        right,
    )

    args = module.parse_args(
        [
            "--left",
            str(left),
            "--expected-left-sha256",
            module.sha256_file(left),
            "--right",
            str(right),
            "--expected-right-sha256",
            module.sha256_file(right),
            "--target-step",
            "3196000",
            "--output",
            str(output),
            "--provenance-out",
            str(provenance),
        ]
    )
    report = module.derive(args)
    assert report["interpolation"]["alpha_fraction"] == "1/5"
    assert report["interpolation"]["derived_counters"] == {
        "v2_updates": 3_196_000,
        "train_steps": 4_359_753,
    }
    assert output.is_file() and provenance.is_file()

    derived = torch.load(output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
        derived["encoder"]["weight"],
        torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert int(derived["encoder"]["counter"]) == 7
    # Optimizer tensors and unpredictable scalar counters come from the left
    # metadata anchor; they are never blended into a resumable state.
    assert float(derived["encoder_opt"]["state"][0]["exp_avg"][0]) == 0.0
    assert derived["skipped_updates"] == 1
    assert derived["elite_updates"] == 10
    assert derived["actor_updates"] == 100
    assert derived["stagec_v2"]["schema"] == "test_static_contract"
    assert derived["derived_interpolation"]["evaluation_only"] is True


def test_static_contract_difference_is_rejected_before_output(tmp_path):
    torch = pytest.importorskip("torch")
    module = _module()
    left_payload = _payload(
        torch, 0.0, v2_updates=3_195_000, train_steps=4_358_753
    )
    right_payload = _payload(
        torch, 10.0, v2_updates=3_200_000, train_steps=4_363_753
    )
    right_payload["stagec_v2"]["policy_speed_scale"] = 0.99
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    torch.save(left_payload, left)
    torch.save(right_payload, right)
    args = module.parse_args(
        [
            "--left",
            str(left),
            "--expected-left-sha256",
            module.sha256_file(left),
            "--right",
            str(right),
            "--expected-right-sha256",
            module.sha256_file(right),
            "--target-step",
            "3196000",
            "--output",
            str(tmp_path / "must-not-exist.pt"),
        ]
    )
    with pytest.raises(ValueError, match="static value differs"):
        module.derive(args)
    assert not (tmp_path / "must-not-exist.pt").exists()
