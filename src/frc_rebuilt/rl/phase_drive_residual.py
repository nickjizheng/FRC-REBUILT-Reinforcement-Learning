"""Fail-closed, phase-specific drive residual for the immutable Stage-C V4 policy.

The residual is deliberately a separate module rather than an extension of
``DrQV2Agent.actor``.  This preserves every encoder/actor tensor and state-dict
key from the V4 parent.  A composite policy:

* evaluates the frozen base encoder and actor under ``torch.no_grad``;
* feeds the frozen actor's policy-relevant trunk representation to three small
  residual heads (LEAVE, COLLECT, RETURN);
* changes only forward/strafe/turn, with a hard +/-0.05 bound; and
* leaves FIRST and SCORE actions bit-identical to the base policy.

Residual checkpoints opt in through a distinct ``action_policy`` string.
Existing Stage-C collectors/evaluators therefore reject them until those tools
explicitly wire this module, instead of silently ignoring the residual.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from frc_rebuilt.rl.policy_v2 import (
    ACTION_DIM,
    ACTION_POLICY,
    LEGACY_PROPRIO_DIM,
    PolicyPhase,
    V2_PROPRIO_DIM,
    validate_composite_metadata,
)


RESIDUAL_SCHEMA_VERSION = "stagec_phase_drive_residual_v1"
RESIDUAL_POLICY_REVISION = "exact_v4_phase_drive_residual_v1"
RESIDUAL_ACTION_POLICY = (
    "frozen_prefix_exact_first_v2_plus_phase_drive_residual_v1"
)
RESIDUAL_CHECKPOINT_KEY = "stagec_phase_drive_residual"
RESIDUAL_METADATA_KEY = "phase_drive_residual"
BASE_REWARD_REVISION = "outer_rail_v4_ramp_out"
RESIDUAL_ARCHITECTURE = "three_actor_trunk_mlp_heads_v1"
RESIDUAL_OUTPUT = "tanh_times_cap"
RESIDUAL_CAP = 0.05
RESIDUAL_HIDDEN_DIM = 128
RESIDUAL_ACTION_INDICES = (0, 1, 2)
RESIDUAL_PHASES = (
    PolicyPhase.LEAVE,
    PolicyPhase.COLLECT,
    PolicyPhase.RETURN,
)
_PHASE_OFFSET = LEGACY_PROPRIO_DIM
_PHASE_WIDTH = len(PolicyPhase)


def _require_plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _normalise_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    normalised = value.lower()
    if len(normalised) != 64 or any(
        character not in "0123456789abcdef" for character in normalised
    ):
        raise ValueError(f"{name} is not a valid SHA-256")
    return normalised


def _all_tensors_finite(value: object) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


@dataclass(frozen=True)
class PhaseDriveResidualContract:
    """Strict portable description of the V4 residual policy."""

    base_checkpoint_sha256: str
    trunk_dim: int
    hidden_dim: int = RESIDUAL_HIDDEN_DIM
    proprio_dim: int = V2_PROPRIO_DIM
    cap: float = RESIDUAL_CAP

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": RESIDUAL_SCHEMA_VERSION,
            "revision": RESIDUAL_POLICY_REVISION,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_reward_revision": BASE_REWARD_REVISION,
            "base_action_policy": ACTION_POLICY,
            "action_policy": RESIDUAL_ACTION_POLICY,
            "base_encoder_frozen": True,
            "base_actor_frozen": True,
            "architecture": RESIDUAL_ARCHITECTURE,
            "output": RESIDUAL_OUTPUT,
            "trunk_dim": int(self.trunk_dim),
            "hidden_dim": int(self.hidden_dim),
            "proprio_dim": int(self.proprio_dim),
            "phase_offset": _PHASE_OFFSET,
            "phases": [phase.name.lower() for phase in RESIDUAL_PHASES],
            "action_indices": list(RESIDUAL_ACTION_INDICES),
            "cap": float(self.cap),
        }


_CONTRACT_KEYS = frozenset(
    PhaseDriveResidualContract("0" * 64, 1).metadata()
)


def build_phase_drive_residual_contract(
    *,
    base_checkpoint_sha256: str,
    trunk_dim: int,
) -> PhaseDriveResidualContract:
    """Build the only contract supported by residual revision v1."""

    sha256 = _normalise_sha256(
        base_checkpoint_sha256, "base_checkpoint_sha256"
    )
    trunk_dim = _require_plain_int(trunk_dim, "trunk_dim")
    if trunk_dim <= 0:
        raise ValueError("trunk_dim must be positive")
    return PhaseDriveResidualContract(
        base_checkpoint_sha256=sha256,
        trunk_dim=trunk_dim,
    )


def validate_phase_drive_residual_metadata(
    metadata: object,
    *,
    expected_base_sha256: str | None = None,
    expected_trunk_dim: int | None = None,
) -> PhaseDriveResidualContract:
    """Validate every semantic field; unknown/missing fields fail closed."""

    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint is missing phase-drive residual metadata")
    if set(metadata) != _CONTRACT_KEYS:
        missing = sorted(_CONTRACT_KEYS - set(metadata))
        unexpected = sorted(set(metadata) - _CONTRACT_KEYS)
        raise ValueError(
            "phase-drive residual metadata keys differ; "
            f"missing={missing}, unexpected={unexpected}"
        )

    base_sha256 = _normalise_sha256(
        metadata["base_checkpoint_sha256"], "base_checkpoint_sha256"
    )
    trunk_dim = _require_plain_int(metadata["trunk_dim"], "trunk_dim")
    hidden_dim = _require_plain_int(metadata["hidden_dim"], "hidden_dim")
    proprio_dim = _require_plain_int(metadata["proprio_dim"], "proprio_dim")
    phase_offset = _require_plain_int(metadata["phase_offset"], "phase_offset")
    try:
        cap = float(metadata["cap"])
    except (TypeError, ValueError) as exc:
        raise ValueError("residual cap must be numeric") from exc

    expected = build_phase_drive_residual_contract(
        base_checkpoint_sha256=base_sha256,
        trunk_dim=trunk_dim,
    ).metadata()
    for key in ("base_encoder_frozen", "base_actor_frozen"):
        if type(metadata[key]) is not bool:  # bool, not truthy 0/1
            raise ValueError(f"phase-drive residual {key} must be boolean")
    for key, wanted in expected.items():
        actual = metadata[key]
        if actual != wanted:
            raise ValueError(
                f"phase-drive residual metadata mismatch for {key}: "
                f"{actual!r} != {wanted!r}"
            )
    # Keep these explicit so an accidental dataclass/default edit cannot widen
    # revision v1 without changing its validator and tests.
    if hidden_dim != RESIDUAL_HIDDEN_DIM:
        raise ValueError("phase-drive residual hidden_dim is not revision v1")
    if proprio_dim != V2_PROPRIO_DIM:
        raise ValueError("phase-drive residual proprio_dim is not Stage C v2")
    if phase_offset != _PHASE_OFFSET:
        raise ValueError("phase-drive residual phase offset changed")
    if cap != RESIDUAL_CAP:
        raise ValueError("phase-drive residual cap is not the v1 hard cap")

    if expected_base_sha256 is not None:
        expected_sha256 = _normalise_sha256(
            expected_base_sha256, "expected_base_sha256"
        )
        if base_sha256 != expected_sha256:
            raise ValueError(
                "phase-drive residual parent SHA-256 mismatch: "
                f"{base_sha256} != {expected_sha256}"
            )
    if expected_trunk_dim is not None:
        wanted_dim = _require_plain_int(
            expected_trunk_dim, "expected_trunk_dim"
        )
        if trunk_dim != wanted_dim:
            raise ValueError(
                f"phase-drive residual trunk_dim {trunk_dim} != {wanted_dim}"
            )
    return PhaseDriveResidualContract(
        base_checkpoint_sha256=base_sha256,
        trunk_dim=trunk_dim,
        hidden_dim=hidden_dim,
        proprio_dim=proprio_dim,
        cap=cap,
    )


def _strict_phase_indices(proprio: torch.Tensor) -> torch.Tensor:
    if proprio.ndim != 2 or int(proprio.shape[1]) != V2_PROPRIO_DIM:
        raise ValueError(
            f"Stage C residual proprio must have shape (N, {V2_PROPRIO_DIM})"
        )
    if not proprio.is_floating_point() or not bool(torch.isfinite(proprio).all()):
        raise ValueError("Stage C residual proprio must be finite floating point")
    phase = proprio[:, _PHASE_OFFSET : _PHASE_OFFSET + _PHASE_WIDTH]
    zero = torch.zeros((), dtype=phase.dtype, device=phase.device)
    one = torch.ones((), dtype=phase.dtype, device=phase.device)
    near_zero = torch.isclose(phase, zero, rtol=0.0, atol=1e-6)
    near_one = torch.isclose(phase, one, rtol=0.0, atol=1e-6)
    if not bool((near_zero | near_one).all()) or not bool(
        near_one.sum(dim=1).eq(1).all()
    ):
        raise ValueError(
            "Stage C residual phase features must be a strict one-hot"
        )
    return torch.argmax(phase, dim=1)


class PhaseDriveResidual(nn.Module):
    """Three independent, zero-output MLP heads over the frozen actor trunk."""

    def __init__(
        self,
        trunk_dim: int,
        *,
        hidden_dim: int = RESIDUAL_HIDDEN_DIM,
        cap: float = RESIDUAL_CAP,
    ) -> None:
        super().__init__()
        trunk_dim = _require_plain_int(trunk_dim, "trunk_dim")
        hidden_dim = _require_plain_int(hidden_dim, "hidden_dim")
        if trunk_dim <= 0 or hidden_dim <= 0:
            raise ValueError("residual dimensions must be positive")
        if float(cap) != RESIDUAL_CAP:
            raise ValueError(
                f"revision v1 residual cap must be exactly {RESIDUAL_CAP}"
            )
        self.trunk_dim = trunk_dim
        self.hidden_dim = hidden_dim
        self.cap = float(cap)
        self.heads = nn.ModuleDict(
            {
                phase.name.lower(): nn.Sequential(
                    nn.Linear(trunk_dim, hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(hidden_dim, len(RESIDUAL_ACTION_INDICES)),
                )
                for phase in RESIDUAL_PHASES
            }
        )
        # Only the output layer must start at zero.  The hidden layer retains a
        # useful random basis, while every initial residual remains exactly zero.
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self, actor_trunk: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        if actor_trunk.ndim != 2 or int(actor_trunk.shape[1]) != self.trunk_dim:
            raise ValueError(
                "actor trunk batch has the wrong shape for phase-drive residual"
            )
        if int(actor_trunk.shape[0]) != int(proprio.shape[0]):
            raise ValueError("actor trunk and proprio batch sizes differ")
        if actor_trunk.device != proprio.device:
            raise ValueError("actor trunk and proprio must share a device")
        if not actor_trunk.is_floating_point() or not bool(
            torch.isfinite(actor_trunk).all()
        ):
            raise ValueError("actor trunk values must be finite floating point")
        phases = _strict_phase_indices(proprio)
        residual = actor_trunk.new_zeros(
            (int(actor_trunk.shape[0]), len(RESIDUAL_ACTION_INDICES))
        )
        for phase in RESIDUAL_PHASES:
            delta = torch.tanh(
                self.heads[phase.name.lower()](actor_trunk)
            ) * self.cap
            residual = residual + delta * phases.eq(int(phase)).unsqueeze(1)
        return residual


def apply_phase_drive_residual(
    base_actions: torch.Tensor,
    residual: torch.Tensor,
    proprio: torch.Tensor,
    *,
    cap: float = RESIDUAL_CAP,
) -> torch.Tensor:
    """Apply a bounded drive residual while preserving protected actions."""

    if (
        base_actions.ndim != 2
        or int(base_actions.shape[1]) != ACTION_DIM
        or residual.ndim != 2
        or int(residual.shape[1]) != len(RESIDUAL_ACTION_INDICES)
    ):
        raise ValueError("base actions or drive residual have the wrong shape")
    if int(base_actions.shape[0]) != int(residual.shape[0]):
        raise ValueError("base action and residual batch sizes differ")
    if base_actions.device != residual.device or base_actions.dtype != residual.dtype:
        raise ValueError("base actions and residual must share device and dtype")
    if base_actions.device != proprio.device:
        raise ValueError("base actions and proprio must share a device")
    if not bool(torch.isfinite(base_actions).all()) or not bool(
        torch.isfinite(residual).all()
    ):
        raise ValueError("base actions and residual must be finite")
    if float(cap) != RESIDUAL_CAP:
        raise ValueError(
            f"revision v1 residual cap must be exactly {RESIDUAL_CAP}"
        )

    phases = _strict_phase_indices(proprio)
    eligible = torch.zeros_like(phases, dtype=torch.bool)
    for phase in RESIDUAL_PHASES:
        eligible |= phases.eq(int(phase))
    tolerance = 8.0 * torch.finfo(residual.dtype).eps
    if bool((residual.abs() > float(cap) + tolerance).any()):
        raise ValueError("drive residual exceeds its hard cap")
    if bool(residual[~eligible].ne(0.0).any()):
        raise ValueError("drive residual is non-zero in FIRST or SCORE")

    composed = base_actions.clone()
    if bool(eligible.any()):
        base_drive = base_actions[eligible, : len(RESIDUAL_ACTION_INDICES)]
        delta = residual[eligible]
        adjusted = torch.clamp(base_drive + delta, -1.0, 1.0)
        # ``base + 0`` is value-exact for normal finite actor outputs while
        # retaining the derivative needed to move a zero-initialized head.
        # Selecting ``base`` directly here would make the initial residual
        # untrainable because torch.where would cut its gradient at delta == 0.
        adjusted = torch.where(delta.eq(0.0), base_drive + delta, adjusted)
        composed[eligible, : len(RESIDUAL_ACTION_INDICES)] = adjusted
    return composed


def freeze_base_encoder_actor(encoder: nn.Module, actor: nn.Module) -> None:
    """Freeze the exact base policy without changing any state-dict tensor."""

    for module in (encoder, actor):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def _assert_base_frozen(encoder: nn.Module, actor: nn.Module) -> None:
    if any(
        parameter.requires_grad
        for module in (encoder, actor)
        for parameter in module.parameters()
    ):
        raise RuntimeError("base encoder and actor must remain frozen")


class FrozenBasePhaseDrivePolicy:
    """Composite inference/training seam with gradients only to the residual."""

    def __init__(
        self,
        encoder: nn.Module,
        actor: nn.Module,
        residual: PhaseDriveResidual | None,
    ) -> None:
        if not hasattr(actor, "trunk"):
            raise TypeError("base actor must expose its frozen .trunk module")
        self.encoder = encoder
        self.actor = actor
        self.residual = residual
        freeze_base_encoder_actor(self.encoder, self.actor)

    def mean_actions(
        self, frames: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        _assert_base_frozen(self.encoder, self.actor)
        with torch.no_grad():
            features = self.encoder(frames)
            base_actions = self.actor(features, proprio)
            actor_trunk = self.actor.trunk(
                torch.cat((features, proprio), dim=-1)
            )
        if self.residual is None:
            return base_actions
        if int(actor_trunk.shape[1]) != self.residual.trunk_dim:
            raise ValueError(
                "base actor trunk width does not match residual checkpoint"
            )
        delta = self.residual(actor_trunk.detach(), proprio.detach())
        return apply_phase_drive_residual(base_actions, delta, proprio)


@dataclass(frozen=True)
class LoadedPhaseDriveResidual:
    module: PhaseDriveResidual
    contract: PhaseDriveResidualContract
    updates: int
    optimizer_state: Mapping[str, Any] | None


_ENTRY_KEYS = frozenset(
    {"schema_version", "state_dict", "optimizer_state", "updates"}
)


def _validate_base_stagec_metadata(
    metadata: object, *, residual_enabled: bool
) -> tuple[dict[str, object], dict[str, object] | None]:
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint is missing Stage C v2 metadata")
    stagec = dict(metadata)
    residual_meta = stagec.pop(RESIDUAL_METADATA_KEY, None)
    expected_action_policy = (
        RESIDUAL_ACTION_POLICY if residual_enabled else ACTION_POLICY
    )
    if stagec.get("action_policy") != expected_action_policy:
        raise ValueError(
            "Stage C action policy does not match residual checkpoint presence"
        )
    # Reconstruct the exact parent policy metadata for the existing validator.
    stagec["action_policy"] = ACTION_POLICY
    prefix_sha256 = stagec.get("prefix_sha256")
    validate_composite_metadata(stagec, str(prefix_sha256))
    if stagec.get("reward_revision") != BASE_REWARD_REVISION:
        raise ValueError(
            "phase-drive residual requires exact V4 reward mechanics"
        )
    if int(stagec.get("proprio_dim", -1)) != V2_PROPRIO_DIM:
        raise ValueError("phase-drive residual requires 30-wide Stage C proprio")
    if residual_enabled and not isinstance(residual_meta, Mapping):
        raise ValueError("checkpoint is missing nested residual metadata")
    if not residual_enabled and residual_meta is not None:
        raise ValueError("legacy checkpoint contains orphan residual metadata")
    return stagec, dict(residual_meta) if residual_meta is not None else None


def build_phase_drive_residual_checkpoint(
    base_payload: Mapping[str, Any],
    residual: PhaseDriveResidual,
    *,
    base_checkpoint_sha256: str,
    updates: int = 0,
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a residual snapshot without modifying any base policy tensor."""

    if not isinstance(base_payload, Mapping):
        raise TypeError("base checkpoint payload must be a mapping")
    if RESIDUAL_CHECKPOINT_KEY in base_payload:
        raise ValueError("base checkpoint already contains a drive residual")
    stagec, _ = _validate_base_stagec_metadata(
        base_payload.get("stagec_v2"), residual_enabled=False
    )
    for name in ("encoder", "actor"):
        state = base_payload.get(name)
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise ValueError(f"base checkpoint is missing tensor {name} state")
        if not _all_tensors_finite(state):
            raise ValueError(f"base checkpoint contains non-finite {name} state")
    if not _all_tensors_finite(residual.state_dict()):
        raise ValueError("residual state contains non-finite tensors")
    if optimizer_state is not None and (
        not isinstance(optimizer_state, Mapping)
        or not _all_tensors_finite(optimizer_state)
    ):
        raise ValueError("residual optimizer state is invalid or non-finite")
    updates = _require_plain_int(updates, "updates")
    if updates < 0:
        raise ValueError("residual updates cannot be negative")

    contract = build_phase_drive_residual_contract(
        base_checkpoint_sha256=base_checkpoint_sha256,
        trunk_dim=residual.trunk_dim,
    )
    if residual.hidden_dim != contract.hidden_dim or residual.cap != contract.cap:
        raise ValueError("residual module does not match revision v1")

    payload = dict(base_payload)
    residual_stagec = dict(stagec)
    residual_stagec["action_policy"] = RESIDUAL_ACTION_POLICY
    residual_stagec[RESIDUAL_METADATA_KEY] = contract.metadata()
    payload["stagec_v2"] = residual_stagec
    payload[RESIDUAL_CHECKPOINT_KEY] = {
        "schema_version": RESIDUAL_SCHEMA_VERSION,
        "state_dict": {
            key: value.detach().clone()
            for key, value in residual.state_dict().items()
        },
        "optimizer_state": (
            copy.deepcopy(dict(optimizer_state))
            if optimizer_state is not None
            else None
        ),
        "updates": updates,
    }
    return payload


def load_optional_phase_drive_residual(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
    expected_base_sha256: str | None = None,
    expected_trunk_dim: int | None = None,
) -> LoadedPhaseDriveResidual | None:
    """Load a residual, return ``None`` for a genuine V4 payload, reject hybrids."""

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    has_entry = RESIDUAL_CHECKPOINT_KEY in payload
    stagec_metadata = payload.get("stagec_v2")
    if not isinstance(stagec_metadata, Mapping):
        raise ValueError("checkpoint is missing Stage C v2 metadata")
    has_metadata = RESIDUAL_METADATA_KEY in stagec_metadata
    has_action_policy = (
        stagec_metadata.get("action_policy") == RESIDUAL_ACTION_POLICY
    )
    residual_flags = (has_entry, has_metadata, has_action_policy)
    if not any(residual_flags):
        _validate_base_stagec_metadata(stagec_metadata, residual_enabled=False)
        return None
    if not all(residual_flags):
        raise ValueError(
            "partial phase-drive residual checkpoint fails closed: "
            f"entry={has_entry}, metadata={has_metadata}, "
            f"action_policy={has_action_policy}"
        )

    _, nested_metadata = _validate_base_stagec_metadata(
        stagec_metadata, residual_enabled=True
    )
    contract = validate_phase_drive_residual_metadata(
        nested_metadata,
        expected_base_sha256=expected_base_sha256,
        expected_trunk_dim=expected_trunk_dim,
    )
    entry = payload[RESIDUAL_CHECKPOINT_KEY]
    if not isinstance(entry, Mapping) or set(entry) != _ENTRY_KEYS:
        raise ValueError("phase-drive residual checkpoint entry is malformed")
    if entry.get("schema_version") != RESIDUAL_SCHEMA_VERSION:
        raise ValueError("phase-drive residual checkpoint schema mismatch")
    updates = _require_plain_int(entry.get("updates"), "updates")
    if updates < 0:
        raise ValueError("residual updates cannot be negative")
    optimizer_state = entry.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise ValueError("residual optimizer state must be a mapping or null")
    if not _all_tensors_finite(optimizer_state):
        raise ValueError("residual optimizer state contains non-finite tensors")

    module = PhaseDriveResidual(
        contract.trunk_dim,
        hidden_dim=contract.hidden_dim,
        cap=contract.cap,
    ).to(device)
    state = entry.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("phase-drive residual state_dict is missing")
    target = module.state_dict()
    if set(state) != set(target):
        raise ValueError("phase-drive residual state_dict keys differ")
    for key, wanted in target.items():
        value = state[key]
        if not torch.is_tensor(value) or value.shape != wanted.shape:
            raise ValueError(
                f"phase-drive residual tensor {key!r} has the wrong shape"
            )
    if not _all_tensors_finite(state):
        raise ValueError("phase-drive residual state contains non-finite tensors")
    module.load_state_dict(state, strict=True)
    return LoadedPhaseDriveResidual(
        module=module,
        contract=contract,
        updates=updates,
        optimizer_state=optimizer_state,
    )


__all__ = [
    "BASE_REWARD_REVISION",
    "FrozenBasePhaseDrivePolicy",
    "LoadedPhaseDriveResidual",
    "PhaseDriveResidual",
    "PhaseDriveResidualContract",
    "RESIDUAL_ACTION_INDICES",
    "RESIDUAL_ACTION_POLICY",
    "RESIDUAL_CAP",
    "RESIDUAL_CHECKPOINT_KEY",
    "RESIDUAL_HIDDEN_DIM",
    "RESIDUAL_METADATA_KEY",
    "RESIDUAL_PHASES",
    "RESIDUAL_POLICY_REVISION",
    "RESIDUAL_SCHEMA_VERSION",
    "apply_phase_drive_residual",
    "build_phase_drive_residual_checkpoint",
    "build_phase_drive_residual_contract",
    "freeze_base_encoder_actor",
    "load_optional_phase_drive_residual",
    "validate_phase_drive_residual_metadata",
]
