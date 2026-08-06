"""Safe warm-start of a phase-augmented Stage-C agent.

The legacy DrQ-v2 actor consumes ``[visual_features, proprio]`` while its
asymmetric critic consumes ``[visual_features, proprio, privileged]``.  Stage
C v2 appends phase features to proprio.  A normal ``load_state_dict`` cannot
load the wider first layers, and padding the end of the critic would put the
new inputs *after* privileged state rather than between proprio and privileged
state.

This module performs that one structural migration and deliberately starts
all optimizers fresh.  It is kept separate from :mod:`drqv2` so legacy loading
and checkpoints remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import torch


_ACTOR_INPUT_WEIGHT = "trunk.0.weight"
_CRITIC_INPUT_WEIGHT = "trunk.0.weight"
_MODULE_NAMES = ("encoder", "actor", "critic", "critic_target")
_OPTIMIZER_NAMES = ("encoder_opt", "actor_opt", "critic_opt")


@dataclass(frozen=True)
class V2CheckpointLoadReport:
    """Dimensions and schedule metadata applied by a v2 warm start."""

    checkpoint: str
    feature_dim: int
    legacy_proprio_dim: int
    new_proprio_dim: int
    appended_proprio_dim: int
    privileged_dim: int
    train_steps: int
    explore_offset: int
    skipped_updates: int
    optimizer_state_restored: bool = False


def _load_payload(
    checkpoint: str | PathLike[str] | Mapping[str, Any], map_location: Any
) -> tuple[Mapping[str, Any], str]:
    if isinstance(checkpoint, Mapping):
        return checkpoint, "<mapping>"

    path = Path(checkpoint)
    # ``weights_only`` became a torch.load argument before it became the
    # default.  Prefer the restricted loader, but retain compatibility with
    # the older PyTorch bundled on some Isaac machines.
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - only exercised by old server torch
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint {path} is not a mapping")
    return payload, str(path)


def _state(payload: Mapping[str, Any], name: str) -> Mapping[str, torch.Tensor]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is missing a {name!r} state_dict")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items()):
        raise ValueError(f"checkpoint {name!r} is not a tensor state_dict")
    return value


def _check_keys_and_compatible_shapes(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    name: str,
    *,
    except_key: str | None = None,
) -> None:
    source_keys = set(source)
    target_keys = set(target)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise ValueError(
            f"{name} state_dict keys differ; missing={missing}, unexpected={unexpected}"
        )
    for key in source_keys:
        if key == except_key:
            continue
        if source[key].shape != target[key].shape:
            raise ValueError(
                f"{name}.{key} shape changed from {tuple(source[key].shape)} "
                f"to {tuple(target[key].shape)}"
            )


def _expanded_actor_state(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    legacy_proprio_dim: int,
    new_proprio_dim: int,
) -> tuple[dict[str, torch.Tensor], int]:
    _check_keys_and_compatible_shapes(
        source, target, "actor", except_key=_ACTOR_INPUT_WEIGHT
    )
    old_weight = source[_ACTOR_INPUT_WEIGHT]
    new_weight_template = target[_ACTOR_INPUT_WEIGHT]
    if old_weight.ndim != 2 or new_weight_template.ndim != 2:
        raise ValueError("actor.trunk.0.weight must be a matrix")
    if old_weight.shape[0] != new_weight_template.shape[0]:
        raise ValueError("actor hidden dimension changed during v2 migration")

    old_feature_dim = old_weight.shape[1] - legacy_proprio_dim
    new_feature_dim = new_weight_template.shape[1] - new_proprio_dim
    if old_feature_dim < 0 or new_feature_dim < 0:
        raise ValueError("actor input is narrower than its declared proprio vector")
    if old_feature_dim != new_feature_dim:
        raise ValueError(
            f"visual feature dimension changed from {old_feature_dim} to "
            f"{new_feature_dim}"
        )
    if new_proprio_dim <= legacy_proprio_dim:
        raise ValueError(
            "v2 proprio must append at least one feature to the legacy proprio vector"
        )

    expanded = old_weight.new_zeros(new_weight_template.shape)
    legacy_end = old_feature_dim + legacy_proprio_dim
    expanded[:, :legacy_end].copy_(old_weight)
    result = dict(source)
    result[_ACTOR_INPUT_WEIGHT] = expanded
    return result, old_feature_dim


def _expanded_critic_state(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    name: str,
    *,
    feature_dim: int,
    legacy_proprio_dim: int,
    new_proprio_dim: int,
    legacy_privileged_dim: int,
    new_privileged_dim: int,
) -> dict[str, torch.Tensor]:
    _check_keys_and_compatible_shapes(
        source, target, name, except_key=_CRITIC_INPUT_WEIGHT
    )
    old_weight = source[_CRITIC_INPUT_WEIGHT]
    new_weight_template = target[_CRITIC_INPUT_WEIGHT]
    if old_weight.ndim != 2 or new_weight_template.ndim != 2:
        raise ValueError(f"{name}.trunk.0.weight must be a matrix")
    if old_weight.shape[0] != new_weight_template.shape[0]:
        raise ValueError(f"{name} hidden dimension changed during v2 migration")
    if new_privileged_dim != legacy_privileged_dim:
        raise ValueError(
            f"{name} privileged dimension changed from {legacy_privileged_dim} "
            f"to {new_privileged_dim}; only appended proprio is supported"
        )

    expected_old = feature_dim + legacy_proprio_dim + legacy_privileged_dim
    expected_new = feature_dim + new_proprio_dim + new_privileged_dim
    if old_weight.shape[1] != expected_old:
        raise ValueError(
            f"{name} legacy input has {old_weight.shape[1]} columns; expected "
            f"{expected_old} (feature={feature_dim}, proprio={legacy_proprio_dim}, "
            f"privileged={legacy_privileged_dim})"
        )
    if new_weight_template.shape[1] != expected_new:
        raise ValueError(
            f"{name} v2 input has {new_weight_template.shape[1]} columns; "
            f"expected {expected_new}"
        )

    old_privileged_start = feature_dim + legacy_proprio_dim
    new_privileged_start = feature_dim + new_proprio_dim
    expanded = old_weight.new_zeros(new_weight_template.shape)
    expanded[:, :old_privileged_start].copy_(old_weight[:, :old_privileged_start])
    expanded[:, new_privileged_start:].copy_(old_weight[:, old_privileged_start:])
    result = dict(source)
    result[_CRITIC_INPUT_WEIGHT] = expanded
    return result


def load_legacy_checkpoint_into_v2(
    agent: Any,
    checkpoint: str | PathLike[str] | Mapping[str, Any],
    *,
    legacy_proprio_dim: int = 22,
    legacy_privileged_dim: int = 26,
) -> V2CheckpointLoadReport:
    """Warm-start a wider-proprio DrQV2 agent from a legacy checkpoint.

    The destination ``agent`` must already be constructed with the v2 proprio
    dimension.  All module layouts other than the first actor/critic input
    layers must be identical.  Validation and tensor construction happen
    before any destination module is mutated.

    Legacy optimizer states are never read.  Existing destination optimizer
    state is cleared as well, guaranteeing a fresh optimizer even if the
    caller reuses an agent object.
    """

    if not hasattr(agent, "cfg"):
        raise TypeError("agent must expose its DrQ config as .cfg")
    new_proprio_dim = int(agent.cfg.proprio_dim)
    new_privileged_dim = int(agent.cfg.privileged_dim)
    if legacy_proprio_dim < 0 or legacy_privileged_dim < 0:
        raise ValueError("legacy dimensions must be non-negative")

    payload, checkpoint_label = _load_payload(checkpoint, agent.device)
    source_states = {name: _state(payload, name) for name in _MODULE_NAMES}
    target_states = {
        name: getattr(agent, name).state_dict() for name in _MODULE_NAMES
    }

    # Prepare and validate every state_dict before loading any of them.  This
    # gives callers an all-or-nothing result for dimensional mistakes.
    _check_keys_and_compatible_shapes(
        source_states["encoder"], target_states["encoder"], "encoder"
    )
    actor_state, feature_dim = _expanded_actor_state(
        source_states["actor"],
        target_states["actor"],
        legacy_proprio_dim=legacy_proprio_dim,
        new_proprio_dim=new_proprio_dim,
    )
    configured_feature_dim = int(getattr(agent, "feat_dim", feature_dim))
    if configured_feature_dim != feature_dim:
        raise ValueError(
            f"agent.feat_dim={configured_feature_dim}, but actor layout implies "
            f"{feature_dim}"
        )
    critic_state = _expanded_critic_state(
        source_states["critic"],
        target_states["critic"],
        "critic",
        feature_dim=feature_dim,
        legacy_proprio_dim=legacy_proprio_dim,
        new_proprio_dim=new_proprio_dim,
        legacy_privileged_dim=legacy_privileged_dim,
        new_privileged_dim=new_privileged_dim,
    )
    critic_target_state = _expanded_critic_state(
        source_states["critic_target"],
        target_states["critic_target"],
        "critic_target",
        feature_dim=feature_dim,
        legacy_proprio_dim=legacy_proprio_dim,
        new_proprio_dim=new_proprio_dim,
        legacy_privileged_dim=legacy_privileged_dim,
        new_privileged_dim=new_privileged_dim,
    )

    agent.encoder.load_state_dict(source_states["encoder"], strict=True)
    agent.actor.load_state_dict(actor_state, strict=True)
    agent.critic.load_state_dict(critic_state, strict=True)
    agent.critic_target.load_state_dict(critic_target_state, strict=True)

    for name in _OPTIMIZER_NAMES:
        optimizer = getattr(agent, name, None)
        if optimizer is not None:
            optimizer.state.clear()

    agent.train_steps = int(payload.get("train_steps", 0))
    agent.explore_offset = int(payload.get("explore_offset", 0))
    agent.skipped_updates = int(payload.get("skipped_updates", 0))

    return V2CheckpointLoadReport(
        checkpoint=checkpoint_label,
        feature_dim=feature_dim,
        legacy_proprio_dim=legacy_proprio_dim,
        new_proprio_dim=new_proprio_dim,
        appended_proprio_dim=new_proprio_dim - legacy_proprio_dim,
        privileged_dim=new_privileged_dim,
        train_steps=agent.train_steps,
        explore_offset=agent.explore_offset,
        skipped_updates=agent.skipped_updates,
    )


__all__ = ["V2CheckpointLoadReport", "load_legacy_checkpoint_into_v2"]
