"""Derive an evaluation-only checkpoint between aligned training snapshots.

The tool is intentionally conservative.  It verifies the immutable endpoint
hashes, requires identical checkpoint/module topology and static metadata, and
interpolates only floating tensors in explicitly selected model state_dicts.
Optimizer state and all non-floating values are copied verbatim from one
declared metadata anchor.  A small set of scalar progress counters can be
derived exactly from the checkpoint-step fraction.

This is a model-soup approximation, not a reconstruction of the exact missing
training snapshot.  Outputs are marked evaluation-only and must never be used
to resume training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA = "frc_aligned_checkpoint_interpolation_v1"
DEFAULT_TENSOR_ROOTS = ("encoder", "actor", "critic", "critic_target")
DEFAULT_OPTIMIZER_ROOTS = ("encoder_opt", "actor_opt", "critic_opt")
DEFAULT_DYNAMIC_SCALARS = (
    "actor_updates",
    "elite_updates",
    "skipped_updates",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be one complete SHA-256")
    return normalized


def require_file_sha(path: Path, expected: str, label: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    wanted = expected_sha256(expected, f"expected {label} SHA-256")
    if actual != wanted:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {wanted}")
    return actual


def _load_checkpoint(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older Isaac/PyTorch exposes no weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _path(parent: str, child: Any) -> str:
    token = str(child)
    return token if not parent else f"{parent}.{token}"


def _same_container(left: Any, right: Any, path: str) -> None:
    if type(left) is not type(right):
        raise ValueError(
            f"{path or '<root>'} type differs: "
            f"{type(left).__name__} != {type(right).__name__}"
        )


def _tensor_contract(torch: Any, left: Any, right: Any, path: str) -> None:
    if not torch.is_tensor(right):
        raise ValueError(f"{path} is a tensor only in the left checkpoint")
    if left.shape != right.shape:
        raise ValueError(f"{path} shape differs: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.dtype != right.dtype:
        raise ValueError(f"{path} dtype differs: {left.dtype} != {right.dtype}")
    if left.layout != right.layout:
        raise ValueError(f"{path} layout differs: {left.layout} != {right.layout}")
    if left.device.type == "meta" or right.device.type == "meta":
        raise ValueError(f"{path} uses unsupported meta tensors")
    if bool(getattr(left, "is_quantized", False)) or bool(
        getattr(right, "is_quantized", False)
    ):
        raise ValueError(f"{path} uses unsupported quantized tensors")


def _finite_tensor(torch: Any, value: Any, path: str) -> None:
    if value.is_floating_point() or value.is_complex():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{path} contains NaN or infinity")


def _mapping_metadata(value: Any) -> Any:
    """Return state_dict's attached version metadata, if present."""

    return getattr(value, "_metadata", None)


def _assert_static_equal(torch: Any, left: Any, right: Any, path: str) -> None:
    """Require exact equality for static metadata and non-selected payloads."""

    if torch.is_tensor(left):
        _tensor_contract(torch, left, right, path)
        _finite_tensor(torch, left, path)
        _finite_tensor(torch, right, path)
        if not bool(torch.equal(left, right)):
            raise ValueError(f"static tensor differs at {path}")
        return
    if torch.is_tensor(right):
        raise ValueError(f"{path} is a tensor only in the right checkpoint")
    _same_container(left, right, path)
    if isinstance(left, Mapping):
        left_keys = list(left.keys())
        right_keys = list(right.keys())
        if left_keys != right_keys:
            raise ValueError(f"{path or '<root>'} mapping keys/order differ")
        if _mapping_metadata(left) != _mapping_metadata(right):
            raise ValueError(f"{path or '<root>'} attached state_dict metadata differs")
        for key in left_keys:
            _assert_static_equal(torch, left[key], right[key], _path(path, key))
        return
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"{path} length differs: {len(left)} != {len(right)}")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _assert_static_equal(
                torch, left_value, right_value, _path(path, index)
            )
        return
    if isinstance(left, (str, bytes, int, float, bool, type(None))):
        if isinstance(left, float):
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"static float is non-finite at {path}")
        if left != right:
            raise ValueError(f"static value differs at {path}: {left!r} != {right!r}")
        return
    raise ValueError(f"unsupported static value type at {path}: {type(left).__name__}")


def _validate_preserved_tree(
    torch: Any,
    left: Any,
    right: Any,
    path: str,
    stats: dict[str, Any],
) -> None:
    """Validate topology while allowing tensor values to differ.

    This is used for optimizer state.  Tensor moments and floating step
    counters are expected to differ between sequential checkpoints, but their
    shapes/dtypes must remain compatible and all scalar hyperparameters must be
    identical.  The entire subtree is copied from the declared anchor.
    """

    if torch.is_tensor(left):
        _tensor_contract(torch, left, right, path)
        _finite_tensor(torch, left, path)
        _finite_tensor(torch, right, path)
        stats["preserved_tensor_count"] += 1
        stats["preserved_tensor_elements"] += int(left.numel())
        if not bool(torch.equal(left, right)):
            stats["preserved_changed_tensor_count"] += 1
        return
    if torch.is_tensor(right):
        raise ValueError(f"{path} is a tensor only in the right checkpoint")
    _same_container(left, right, path)
    if isinstance(left, Mapping):
        left_keys = list(left.keys())
        right_keys = list(right.keys())
        if left_keys != right_keys:
            raise ValueError(f"{path} mapping keys/order differ")
        if _mapping_metadata(left) != _mapping_metadata(right):
            raise ValueError(f"{path} attached state_dict metadata differs")
        for key in left_keys:
            _validate_preserved_tree(
                torch, left[key], right[key], _path(path, key), stats
            )
        return
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"{path} length differs: {len(left)} != {len(right)}")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _validate_preserved_tree(
                torch, left_value, right_value, _path(path, index), stats
            )
        return
    if isinstance(left, (str, bytes, int, float, bool, type(None))):
        if isinstance(left, float):
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"optimizer scalar is non-finite at {path}")
        if left != right:
            raise ValueError(
                f"preserved subtree scalar differs at {path}: {left!r} != {right!r}"
            )
        return
    raise ValueError(f"unsupported preserved value type at {path}: {type(left).__name__}")


def _interpolate_model_tree(
    torch: Any,
    left: Any,
    right: Any,
    anchor: Any,
    alpha: Fraction,
    path: str,
    stats: dict[str, Any],
) -> Any:
    if torch.is_tensor(left):
        _tensor_contract(torch, left, right, path)
        _finite_tensor(torch, left, path)
        _finite_tensor(torch, right, path)
        if not left.is_floating_point():
            if left.is_complex():
                raise ValueError(f"{path} uses unsupported complex model tensors")
            if not bool(torch.equal(left, right)):
                raise ValueError(f"non-floating model tensor differs at {path}")
            stats["preserved_model_buffer_count"] += 1
            return anchor.detach().clone()
        if left.layout != torch.strided:
            raise ValueError(f"{path} uses unsupported non-strided model tensors")
        numerator = float(alpha.numerator)
        denominator = float(alpha.denominator)
        # Float64 arithmetic gives one deterministic, nearest-representable
        # result when cast back to the checkpoint dtype.
        left_work = left.detach().to(device="cpu", dtype=torch.float64)
        right_work = right.detach().to(device="cpu", dtype=torch.float64)
        result = (left_work + (right_work - left_work) * (numerator / denominator)).to(
            dtype=left.dtype
        )
        _finite_tensor(torch, result, path)
        delta = right_work - left_work
        stats["interpolated_tensor_count"] += 1
        stats["interpolated_tensor_elements"] += int(left.numel())
        if left.numel():
            stats["maximum_endpoint_abs_delta"] = max(
                stats["maximum_endpoint_abs_delta"],
                float(delta.abs().max().item()),
            )
            stats["endpoint_delta_l2_squared"] += float(
                torch.sum(delta * delta).item()
            )
            stats["left_l2_squared"] += float(
                torch.sum(left_work * left_work).item()
            )
        return result
    if torch.is_tensor(right):
        raise ValueError(f"{path} is a tensor only in the right checkpoint")
    _same_container(left, right, path)
    if isinstance(left, Mapping):
        left_keys = list(left.keys())
        right_keys = list(right.keys())
        if left_keys != right_keys:
            raise ValueError(f"{path} mapping keys/order differ")
        if _mapping_metadata(left) != _mapping_metadata(right):
            raise ValueError(f"{path} attached state_dict metadata differs")
        result = copy.deepcopy(anchor)
        for key in left_keys:
            result[key] = _interpolate_model_tree(
                torch,
                left[key],
                right[key],
                anchor[key],
                alpha,
                _path(path, key),
                stats,
            )
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(f"{path} length differs: {len(left)} != {len(right)}")
        return [
            _interpolate_model_tree(
                torch, left_value, right_value, anchor[index], alpha, _path(path, index), stats
            )
            for index, (left_value, right_value) in enumerate(zip(left, right))
        ]
    if isinstance(left, tuple):
        if len(left) != len(right):
            raise ValueError(f"{path} length differs: {len(left)} != {len(right)}")
        return type(anchor)(
            _interpolate_model_tree(
                torch, left_value, right_value, anchor[index], alpha, _path(path, index), stats
            )
            for index, (left_value, right_value) in enumerate(zip(left, right))
        )
    _assert_static_equal(torch, left, right, path)
    return copy.deepcopy(anchor)


def _int_counter(payload: Mapping[str, Any], key: str, label: str) -> int:
    if key not in payload:
        raise ValueError(f"{label} checkpoint is missing integer counter {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} checkpoint counter {key!r} is not a Python int")
    return int(value)


def _derive_counter(left: int, right: int, alpha: Fraction, key: str) -> int:
    value = Fraction(left, 1) + alpha * (right - left)
    if value.denominator != 1:
        raise ValueError(
            f"counter {key!r} does not land on an integer at alpha "
            f"{alpha.numerator}/{alpha.denominator}: {value}"
        )
    return int(value)


def _same_payload(torch: Any, expected: Any, actual: Any, path: str = "") -> None:
    """Bitwise round-trip validation after torch.save/torch.load."""

    if torch.is_tensor(expected):
        _tensor_contract(torch, expected, actual, path)
        if not bool(torch.equal(expected, actual)):
            raise RuntimeError(f"serialized tensor changed at {path}")
        return
    if torch.is_tensor(actual):
        raise RuntimeError(f"serialized type changed at {path}")
    _same_container(expected, actual, path)
    if isinstance(expected, Mapping):
        if list(expected.keys()) != list(actual.keys()):
            raise RuntimeError(f"serialized mapping keys changed at {path or '<root>'}")
        if _mapping_metadata(expected) != _mapping_metadata(actual):
            raise RuntimeError(f"serialized state_dict metadata changed at {path or '<root>'}")
        for key in expected:
            _same_payload(torch, expected[key], actual[key], _path(path, key))
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError(f"serialized sequence length changed at {path}")
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            _same_payload(
                torch, expected_value, actual_value, _path(path, index)
            )
        return
    if expected != actual:
        raise RuntimeError(f"serialized value changed at {path}")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> Path:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temp
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def derive(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on host runtime.
        raise RuntimeError("PyTorch is required to derive a checkpoint") from exc

    left = args.left.resolve()
    right = args.right.resolve()
    if left == right:
        raise ValueError("left and right checkpoints must be different files")
    left_sha = require_file_sha(left, args.expected_left_sha256, "left checkpoint")
    right_sha = require_file_sha(right, args.expected_right_sha256, "right checkpoint")
    left_payload = _load_checkpoint(torch, left)
    right_payload = _load_checkpoint(torch, right)
    if list(left_payload.keys()) != list(right_payload.keys()):
        raise ValueError("checkpoint top-level keys/order differ")

    step_key = str(args.step_key)
    left_step = _int_counter(left_payload, step_key, "left")
    right_step = _int_counter(right_payload, step_key, "right")
    target_step = int(args.target_step)
    if not left_step < target_step < right_step:
        raise ValueError(
            f"target {step_key} must be strictly bracketed: "
            f"{left_step} < {target_step} < {right_step}"
        )
    alpha = Fraction(target_step - left_step, right_step - left_step)

    tensor_roots = tuple(dict.fromkeys(args.tensor_root or DEFAULT_TENSOR_ROOTS))
    optimizer_roots = tuple(
        root for root in DEFAULT_OPTIMIZER_ROOTS if root in left_payload
    )
    derive_counters = tuple(
        dict.fromkeys(args.derive_counter or (step_key, "train_steps"))
    )
    dynamic_scalars = set(DEFAULT_DYNAMIC_SCALARS)
    dynamic_scalars.update(args.allow_scalar_difference or ())
    dynamic_scalars.update(derive_counters)

    for root in tensor_roots:
        if root not in left_payload:
            raise ValueError(f"checkpoint is missing selected tensor root {root!r}")
        if not isinstance(left_payload[root], Mapping):
            raise ValueError(f"selected tensor root {root!r} is not a state_dict mapping")

    anchor_payload = left_payload if args.metadata_anchor == "left" else right_payload
    output_payload = copy.deepcopy(anchor_payload)
    stats: dict[str, Any] = {
        "interpolated_tensor_count": 0,
        "interpolated_tensor_elements": 0,
        "preserved_model_buffer_count": 0,
        "preserved_tensor_count": 0,
        "preserved_tensor_elements": 0,
        "preserved_changed_tensor_count": 0,
        "maximum_endpoint_abs_delta": 0.0,
        "endpoint_delta_l2_squared": 0.0,
        "left_l2_squared": 0.0,
    }
    preserved_scalar_differences: dict[str, dict[str, Any]] = {}

    for key in left_payload:
        path = str(key)
        if key in tensor_roots:
            output_payload[key] = _interpolate_model_tree(
                torch,
                left_payload[key],
                right_payload[key],
                anchor_payload[key],
                alpha,
                path,
                stats,
            )
        elif key in optimizer_roots:
            _validate_preserved_tree(
                torch, left_payload[key], right_payload[key], path, stats
            )
        elif key in dynamic_scalars:
            left_value = left_payload[key]
            right_value = right_payload[key]
            if type(left_value) is not type(right_value):
                raise ValueError(f"dynamic scalar type differs at {key!r}")
            if not isinstance(left_value, (str, bytes, int, float, bool, type(None))):
                raise ValueError(f"allowed dynamic value {key!r} is not a scalar")
            if left_value != right_value:
                preserved_scalar_differences[key] = {
                    "left": left_value,
                    "right": right_value,
                    "selected": anchor_payload[key],
                }
        else:
            _assert_static_equal(
                torch, left_payload[key], right_payload[key], path
            )

    derived_counter_values: dict[str, int] = {}
    for key in derive_counters:
        left_value = _int_counter(left_payload, key, "left")
        right_value = _int_counter(right_payload, key, "right")
        derived = _derive_counter(left_value, right_value, alpha, key)
        output_payload[key] = derived
        derived_counter_values[key] = derived
    if derived_counter_values[step_key] != target_step:
        raise AssertionError("derived checkpoint step differs from requested target")

    script_sha = sha256_file(Path(__file__).resolve())
    embedded = {
        "schema": SCHEMA,
        "evaluation_only": True,
        "left_sha256": left_sha,
        "right_sha256": right_sha,
        "step_key": step_key,
        "left_step": left_step,
        "right_step": right_step,
        "target_step": target_step,
        "alpha_fraction": f"{alpha.numerator}/{alpha.denominator}",
        "metadata_anchor": args.metadata_anchor,
        "tensor_roots": list(tensor_roots),
        "script_sha256": script_sha,
    }
    if "derived_interpolation" in output_payload:
        raise ValueError("checkpoint already contains a derived_interpolation record")
    output_payload["derived_interpolation"] = embedded

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evaluation_only": True,
        "inputs": {
            "left": {"path": str(left), "sha256": left_sha, "bytes": left.stat().st_size},
            "right": {
                "path": str(right),
                "sha256": right_sha,
                "bytes": right.stat().st_size,
            },
        },
        "interpolation": {
            "step_key": step_key,
            "left_step": left_step,
            "right_step": right_step,
            "target_step": target_step,
            "alpha_fraction": f"{alpha.numerator}/{alpha.denominator}",
            "alpha_decimal": float(alpha),
            "tensor_roots": list(tensor_roots),
            "optimizer_roots_preserved": list(optimizer_roots),
            "metadata_anchor": args.metadata_anchor,
            "derived_counters": derived_counter_values,
            "preserved_scalar_differences": preserved_scalar_differences,
        },
        "tensor_audit": {
            **stats,
            "endpoint_delta_l2": math.sqrt(stats["endpoint_delta_l2_squared"]),
            "left_l2": math.sqrt(stats["left_l2_squared"]),
            "relative_endpoint_delta_l2": (
                math.sqrt(stats["endpoint_delta_l2_squared"])
                / math.sqrt(stats["left_l2_squared"])
                if stats["left_l2_squared"] > 0.0
                else None
            ),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
    }

    if args.inspect_only:
        return report
    if args.output is None:
        raise ValueError("--output is required unless --inspect-only is used")
    output = args.output.resolve()
    provenance = (
        args.provenance_out.resolve()
        if args.provenance_out is not None
        else output.with_suffix(output.suffix + ".provenance.json")
    )
    if output in (left, right) or provenance in (left, right, output):
        raise ValueError("output/provenance paths must be distinct from both endpoints")
    if output.exists() or provenance.exists():
        raise FileExistsError("refusing to overwrite output or provenance")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    output_temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    provenance_temp: Path | None = None
    try:
        with output_temp.open("xb") as handle:
            torch.save(output_payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = _load_checkpoint(torch, output_temp)
        _same_payload(torch, output_payload, reloaded)
        output_sha = sha256_file(output_temp)
        report["output"] = {
            "path": str(output),
            "sha256": output_sha,
            "bytes": output_temp.stat().st_size,
            "provenance": str(provenance),
        }
        provenance_temp = _atomic_write_json(provenance, report)
        os.replace(output_temp, output)
        os.replace(provenance_temp, provenance)
    finally:
        output_temp.unlink(missing_ok=True)
        if provenance_temp is not None:
            provenance_temp.unlink(missing_ok=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--expected-left-sha256", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--expected-right-sha256", required=True)
    parser.add_argument(
        "--step-key",
        default="v2_updates",
        help="top-level integer progress key used to compute the exact factor",
    )
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument(
        "--metadata-anchor",
        choices=("left", "right"),
        default="left",
        help="endpoint copied for optimizer state and all preserved values",
    )
    parser.add_argument(
        "--tensor-root",
        action="append",
        help="model state_dict root to interpolate (repeatable; defaults to all four modules)",
    )
    parser.add_argument(
        "--derive-counter",
        action="append",
        help="top-level Python-int counter to derive exactly (defaults to step key and train_steps)",
    )
    parser.add_argument(
        "--allow-scalar-difference",
        action="append",
        help="extra top-level scalar allowed to differ and copied from the metadata anchor",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provenance-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        report = derive(parse_args(argv))
    except Exception as exc:
        print(f"CHECKPOINT_INTERPOLATION_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("CHECKPOINT_INTERPOLATION_OK " + json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
