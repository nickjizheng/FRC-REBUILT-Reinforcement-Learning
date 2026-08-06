"""Pure NumPy tests for the Stage C v2.2 composite policy."""
from __future__ import annotations

import numpy as np
import pytest

from frc_rebuilt.rl.policy_v2 import (
    ACTION_POLICY,
    FIELD_STRATEGY,
    SCHEMA_VERSION,
    PolicyPhase,
    RETURN_SKILL_PRELOAD,
    apply_executed_action_policy,
    compose_phase_actions,
    phase_from_proprio,
    validate_composite_metadata,
)


def _proprio(*phases: PolicyPhase) -> np.ndarray:
    values = np.zeros((len(phases), 30), np.float32)
    for row, phase in enumerate(phases):
        values[row, 22 + int(phase)] = 1.0
    return values


def test_contract_version_and_policy_name():
    assert SCHEMA_VERSION == "stagec_v2.3"
    assert ACTION_POLICY == "frozen_prefix_exact_first_v2"
    assert FIELD_STRATEGY == "native_field_return_preload_v1"
    assert RETURN_SKILL_PRELOAD == 8


def test_composite_metadata_binds_exact_prefix():
    sha = "a" * 64
    meta = {
        "schema_version": SCHEMA_VERSION,
        "proprio_dim": 30,
        "legacy_proprio_dim": 22,
        "prefix_sha256": sha.upper(),
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "return_skill_preload": RETURN_SKILL_PRELOAD,
    }
    assert validate_composite_metadata(meta, sha) is meta
    with pytest.raises(ValueError, match="prefix_sha256"):
        validate_composite_metadata(meta, "b" * 64)

    old_contract = dict(meta, schema_version="stagec_v2.2")
    with pytest.raises(ValueError, match="schema_version"):
        validate_composite_metadata(old_contract, sha)


def test_phase_from_30_proprio_batch_and_vector():
    proprio = _proprio(*list(PolicyPhase))
    assert phase_from_proprio(proprio).tolist() == [0, 1, 2, 3, 4]
    assert phase_from_proprio(proprio[2]) is PolicyPhase.COLLECT


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1, 30), np.float32),
        np.r_[np.zeros(22), [1, 1, 0, 0, 0], np.zeros(3)].astype(np.float32)[None],
        np.zeros((1, 29), np.float32),
    ],
)
def test_phase_rejects_missing_ambiguous_or_wrong_width(bad):
    with pytest.raises(ValueError):
        phase_from_proprio(bad)


def test_composite_handoff_uses_prefix_only_in_first():
    proprio = _proprio(*list(PolicyPhase))
    prefix = np.full((5, 7), -0.25, np.float32)
    candidate = np.full((5, 7), 0.75, np.float32)
    got = compose_phase_actions(prefix, candidate, proprio)
    np.testing.assert_array_equal(got[0], prefix[0])
    np.testing.assert_array_equal(got[1:], candidate[1:])


def test_executed_action_mask_all_phases():
    proprio = _proprio(*list(PolicyPhase))
    proposed = np.array(
        [[0.1, 0.2, 0.3, 0.4, -0.4, 0.6, 0.8]] * 5,
        dtype=np.float32,
    )
    got = apply_executed_action_policy(proposed, proprio)

    # Drive is untouched in every phase.  FIRST is the exact prefix, including
    # ferry; ferry is disabled only after the handoff.
    np.testing.assert_array_equal(got[:, :3], proposed[:, :3])
    assert got[0, 6] == proposed[0, 6]
    np.testing.assert_array_equal(got[1:, 6], -np.ones(4, np.float32))

    # FIRST is byte-for-byte the prefix output.
    np.testing.assert_array_equal(got[0], proposed[0])

    # Post-first storage is extended and only COLLECT has intake enabled.
    np.testing.assert_array_equal(got[1:, 4], np.ones(4, np.float32))
    np.testing.assert_array_equal(got[1:, 3], [-1.0, 1.0, -1.0, -1.0])

    # LEAVE/COLLECT/RETURN cannot shoot; SCORE retains the proposal.
    np.testing.assert_array_equal(got[1:4, 5], -np.ones(3, np.float32))
    assert got[4, 5] == proposed[4, 5]


def test_return_intake_is_explicit_and_changes_only_return_intake():
    proprio = _proprio(*list(PolicyPhase))
    proposed = np.array(
        [[0.1, 0.2, 0.3, 0.4, -0.4, 0.6, 0.8]] * 5,
        dtype=np.float32,
    )
    legacy = apply_executed_action_policy(proposed, proprio)
    enabled = apply_executed_action_policy(
        proposed,
        proprio,
        intake_during_return=True,
    )

    expected = legacy.copy()
    expected[int(PolicyPhase.RETURN), 3] = 1.0
    np.testing.assert_array_equal(enabled, expected)
    assert legacy[int(PolicyPhase.RETURN), 3] == -1.0
    assert enabled[int(PolicyPhase.RETURN), 3] == 1.0


def test_vector_handoff_and_mask_preserve_vector_shape():
    proprio = _proprio(PolicyPhase.SCORE)[0]
    prefix = np.arange(7, dtype=np.float32)
    candidate = -prefix
    composed = compose_phase_actions(prefix, candidate, proprio)
    executed = apply_executed_action_policy(composed, proprio)
    assert composed.shape == executed.shape == (7,)
    np.testing.assert_array_equal(executed[:3], candidate[:3])
    assert executed[4] == 1.0
    assert executed[5] == candidate[5]
    assert executed[6] == -1.0


def test_helpers_do_not_mutate_network_outputs():
    proprio = _proprio(PolicyPhase.COLLECT)
    prefix = np.zeros((1, 7), np.float32)
    candidate = np.full((1, 7), 0.5, np.float32)
    prefix_before, candidate_before = prefix.copy(), candidate.copy()
    composed = compose_phase_actions(prefix, candidate, proprio)
    _ = apply_executed_action_policy(composed, proprio)
    np.testing.assert_array_equal(prefix, prefix_before)
    np.testing.assert_array_equal(candidate, candidate_before)
