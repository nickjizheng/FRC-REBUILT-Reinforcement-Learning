"""Pure NumPy Stage C v2.3 composite and executed-action policy.

The Stage C observation appends eight values to the legacy 22-value proprio
vector.  The first five appended values are a one-hot cycle phase.  This module
uses that explicit phase to:

* retain the frozen legacy policy for the protected first cycle;
* hand control to the trainable 30-input policy after the first unload; and
* enforce the small set of action invariants shared by collection, replay,
  deterministic evaluation, and GUI playback.

There are intentionally no Torch or Isaac imports here.  Keeping the policy
layer as a batch of NumPy transformations makes its exact executed actions easy
to test and, critically, lets the collector store the action the simulator
actually received rather than the candidate network's pre-mask suggestion.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


SCHEMA_VERSION = "stagec_v2.3"
ACTION_POLICY = "frozen_prefix_exact_first_v2"
FIELD_STRATEGY = "native_field_return_preload_v1"
RETURN_SKILL_PRELOAD = 8
LEGACY_PROPRIO_DIM = 22
V2_PROPRIO_DIM = 30

ACTION_FORWARD = 0
ACTION_STRAFE = 1
ACTION_TURN = 2
ACTION_INTAKE = 3
ACTION_STORAGE = 4
ACTION_SHOOT = 5
ACTION_FERRY = 6
ACTION_DIM = 7


class PolicyPhase(IntEnum):
    """Indices of the five phase bits appended to the legacy proprio."""

    FIRST = 0
    LEAVE = 1
    COLLECT = 2
    RETURN = 3
    SCORE = 4


def validate_composite_metadata(metadata: object, prefix_sha256: str) -> dict:
    """Validate the portable v2.3 checkpoint side of the composite contract."""

    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing Stage C v2 metadata")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "proprio_dim": V2_PROPRIO_DIM,
        "legacy_proprio_dim": LEGACY_PROPRIO_DIM,
        "prefix_sha256": str(prefix_sha256).lower(),
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "return_skill_preload": RETURN_SKILL_PRELOAD,
    }
    for key, wanted in expected.items():
        actual = metadata.get(key)
        if key == "prefix_sha256" and isinstance(actual, str):
            actual = actual.lower()
        if actual != wanted:
            raise ValueError(
                f"Stage C v2 metadata mismatch for {key}: {actual!r} != {wanted!r}"
            )
    return metadata


def _as_proprio_batch(proprio: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(proprio, dtype=np.float32)
    squeezed = values.ndim == 1
    if squeezed:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != V2_PROPRIO_DIM:
        raise ValueError(
            f"Stage C v2 proprio must be shaped (N, {V2_PROPRIO_DIM}), "
            f"got {values.shape}"
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError("Stage C v2 proprio contains non-finite values")
    return values, squeezed


def _as_action_batch(actions: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    values = np.asarray(actions, dtype=np.float32)
    squeezed = values.ndim == 1
    if squeezed:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"{name} actions must be shaped (N, {ACTION_DIM}), got {values.shape}")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} actions contain non-finite values")
    return values, squeezed


def phase_from_proprio(proprio: np.ndarray) -> np.ndarray | PolicyPhase:
    """Decode the explicit phase from a 30-value Stage C proprio vector.

    The environment emits a strict one-hot, but accepting the largest finite
    phase value is robust to normal floating-point serialization.  Ties and a
    non-positive phase vector are rejected so a malformed observation cannot
    silently hand control to the wrong network.
    """

    values, squeezed = _as_proprio_batch(proprio)
    phase_values = values[:, LEGACY_PROPRIO_DIM : LEGACY_PROPRIO_DIM + 5]
    maxima = phase_values.max(axis=1)
    winners = phase_values == maxima[:, None]
    if bool((maxima <= 0.0).any()) or bool((winners.sum(axis=1) != 1).any()):
        raise ValueError("Stage C v2 phase features must identify one unambiguous phase")
    decoded = np.argmax(phase_values, axis=1).astype(np.int8)
    if squeezed:
        return PolicyPhase(int(decoded[0]))
    return decoded


def compose_phase_actions(
    prefix_actions: np.ndarray,
    candidate_actions: np.ndarray,
    proprio: np.ndarray,
) -> np.ndarray:
    """Choose frozen-prefix actions in FIRST and candidate actions thereafter."""

    prefix, prefix_squeezed = _as_action_batch(prefix_actions, name="prefix")
    candidate, candidate_squeezed = _as_action_batch(candidate_actions, name="candidate")
    state, state_squeezed = _as_proprio_batch(proprio)
    if prefix.shape != candidate.shape or prefix.shape[0] != state.shape[0]:
        raise ValueError(
            "prefix, candidate, and proprio batches must contain the same number of rows"
        )
    if not (prefix_squeezed == candidate_squeezed == state_squeezed):
        raise ValueError("prefix, candidate, and proprio must all be batched or all be vectors")
    phases = np.asarray(phase_from_proprio(state), dtype=np.int8)
    use_prefix = phases == int(PolicyPhase.FIRST)
    composed = np.where(use_prefix[:, None], prefix, candidate).astype(np.float32, copy=False)
    return composed[0] if state_squeezed else composed


def apply_executed_action_policy(
    actions: np.ndarray,
    proprio: np.ndarray,
    *,
    intake_during_return: bool = False,
    stage_d_ferry: bool = False,
) -> np.ndarray:
    """Return the exact action batch allowed to reach the simulator.

    Invariants:

    * FIRST retains every frozen-prefix output exactly;
    * ferry is disabled only after FIRST;
    * post-first storage remains extended;
    * intake is on during COLLECT and, when explicitly requested, RETURN;
      otherwise it remains off in LEAVE/RETURN/SCORE for V8/V9 compatibility;
    * shoot is masked in LEAVE/COLLECT/RETURN and remains policy-controlled in
      SCORE.

    Discrete actions use +/-1 rather than a threshold-adjacent zero, making the
    contract explicit and insensitive to the decoder threshold.
    """

    proposed, action_squeezed = _as_action_batch(actions, name="composed")
    state, state_squeezed = _as_proprio_batch(proprio)
    if proposed.shape[0] != state.shape[0]:
        raise ValueError("action and proprio batches must contain the same number of rows")
    if action_squeezed != state_squeezed:
        raise ValueError("actions and proprio must both be batched or both be vectors")

    phases = np.asarray(phase_from_proprio(state), dtype=np.int8)
    post_first = phases != int(PolicyPhase.FIRST)
    executed = proposed.copy()
    # GATE B (STAGE-D1B): ferry is normally forced off for every post-FIRST
    # step.  Under stage_d_ferry keep it policy-controlled in the suffix and
    # force it off ONLY in SCORE, mirroring vec_env's ferry_phase gate so the
    # stored (replay) action equals what the simulator executed.
    if stage_d_ferry:
        ferry_off = post_first & (phases == int(PolicyPhase.SCORE))
    else:
        ferry_off = post_first
    executed[ferry_off, ACTION_FERRY] = -1.0
    executed[post_first, ACTION_STORAGE] = 1.0

    intake_on = phases == int(PolicyPhase.COLLECT)
    if intake_during_return:
        intake_on |= phases == int(PolicyPhase.RETURN)
    executed[post_first, ACTION_INTAKE] = -1.0
    executed[intake_on, ACTION_INTAKE] = 1.0

    not_score = post_first & (phases != int(PolicyPhase.SCORE))
    executed[not_score, ACTION_SHOOT] = -1.0
    return executed[0] if state_squeezed else executed


# Concise aliases for call sites and downstream experiments.
compose_actions = compose_phase_actions
executed_action_policy = apply_executed_action_policy


__all__ = [
    "ACTION_POLICY",
    "FIELD_STRATEGY",
    "LEGACY_PROPRIO_DIM",
    "PolicyPhase",
    "RETURN_SKILL_PRELOAD",
    "SCHEMA_VERSION",
    "V2_PROPRIO_DIM",
    "apply_executed_action_policy",
    "compose_actions",
    "compose_phase_actions",
    "executed_action_policy",
    "phase_from_proprio",
    "validate_composite_metadata",
]
