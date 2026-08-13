"""CPU-only tests for Stage C v2.3 learner policy and resume helpers."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.rl.learner_cycle_v2 import (
    ACTION_POLICY,
    ELITE_ARCHIVE_POOL_WEIGHTS,
    ELITE_POOL_WEIGHTS,
    ELITE_POOLS,
    FIELD_STRATEGY,
    FIRST_PHASE_INDEX,
    SCHEMA_VERSION,
    SEEDMINE_CAPTURE_SCHEMA,
    SEEDMINE_EVAL_SCHEMA,
    V2_PROPRIO_DIM,
    EliteScoreBehaviorPool,
    _add_seedmine_episode_to_replay,
    _archive_elite_episode,
    _elite_behavior_eligible,
    _elite_capture_groups,
    _elite_classification,
    _elite_replay_rings_for_update,
    _elite_tier,
    _exact_pool_quotas,
    _load_elite_archive,
    _load_elite_archive_record,
    _load_seedmine_archive,
    _load_seedmine_archive_record,
    _prune_elite_archives,
    _prepare_elite_episode,
    _sample_elite,
    _schedule_origin_updates,
    _sha256_file,
    _initial_explore_offset,
    _suffix_actor_mask,
    _allow_actor_q_center_fraction_mismatch,
    _allow_target_load_mismatch,
    _allow_suffix_alpha_mismatch,
    _summarize_actor_interval_metrics,
    _validate_resume_metadata,
    _validate_route_efficiency_resume,
    _validate_seedmine_source_metadata,
    _validate_seedmine_options,
)
from frc_rebuilt.rl.replay import ReplayRing
from frc_rebuilt.rl.replay_v2 import CapturedEpisode


def test_suffix_actor_mask_excludes_first_cycle_rows():
    proprio = np.zeros((6, V2_PROPRIO_DIM), np.float32)
    proprio[:, FIRST_PHASE_INDEX] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]

    mask = _suffix_actor_mask(proprio)

    assert mask.dtype == np.bool_
    assert mask.tolist() == [False, True, False, True, True, False]


def test_elite_capture_groups_follow_the_configured_curriculum():
    groups = ("full", "full", "return", "return")
    assert _elite_capture_groups(groups, enabled=True) == ("full", "return")
    assert _elite_capture_groups(groups, enabled=False) == ()


def test_suffix_actor_mask_rejects_wrong_schema_width():
    with pytest.raises(ValueError, match=r"shape \(N, 30\)"):
        _suffix_actor_mask(np.zeros((2, V2_PROPRIO_DIM - 1), np.float32))


def test_new_branch_resets_schedules_without_resetting_lifetime_updates():
    metadata = {"schedule_origin_updates": 100_000}
    assert _schedule_origin_updates(350_000, metadata, reset=False) == 100_000
    assert _schedule_origin_updates(350_000, metadata, reset=True) == 350_000


def test_initial_explore_offset_rewarms_resumed_agent_to_requested_stddev():
    offset = _initial_explore_offset(
        1_500_000,
        stddev_start=1.0,
        stddev_end=0.18,
        stddev_steps=150_000,
        initial_stddev=0.50,
    )
    elapsed = 1_500_000 - offset
    mix = min(1.0, elapsed / 150_000)
    stddev = 1.0 + (0.18 - 1.0) * mix
    assert stddev == pytest.approx(0.50, abs=1e-5)


def test_prefix_sha256_hashes_exact_file_bytes(tmp_path):
    checkpoint = tmp_path / "prefix.pt"
    payload = b"immutable-prefix-checkpoint\x00\x01"
    checkpoint.write_bytes(payload)
    assert _sha256_file(checkpoint) == hashlib.sha256(payload).hexdigest()


def test_v23_resume_metadata_pins_prefix_and_action_policy():
    expected = {
        "schema_version": SCHEMA_VERSION,
        "prefix_sha256": "a" * 64,
        "action_policy": ACTION_POLICY,
        "suffix_alpha": 1.0,
        "encoder_frozen": True,
    }
    _validate_resume_metadata(dict(expected), expected)

    wrong_prefix = dict(expected, prefix_sha256="b" * 64)
    with pytest.raises(ValueError, match="prefix_sha256"):
        _validate_resume_metadata(wrong_prefix, expected)

    wrong_policy = dict(expected, action_policy="mutable_prefix")
    with pytest.raises(ValueError, match="action_policy"):
        _validate_resume_metadata(wrong_policy, expected)


def test_actor_interval_summary_keeps_actor_steps_and_ignores_nonfinite_values():
    summary = _summarize_actor_interval_metrics(
        [
            {
                "actor_rows": 4.0,
                "actor_applied": 1.0,
                "actor_loss": 2.0,
                "q_pi": 10.0,
                "q_pi_noisy": 8.0,
            },
            {
                "actor_rows": 0.0,
                "actor_applied": 0.0,
                "actor_loss": 999.0,
                "q_pi": 999.0,
            },
            {
                "actor_rows": 8.0,
                "actor_applied": 0.0,
                "actor_loss": float("nan"),
                "q_pi": 14.0,
                "q_pi_noisy": 12.0,
                "q_pi_center": 16.0,
            },
        ]
    )

    assert summary["actor_interval_updates"] == 2
    assert summary["actor_interval_applied"] == 1
    assert summary["actor_rows_mean"] == pytest.approx(6.0)
    assert summary["actor_loss_mean"] == pytest.approx(2.0)
    assert summary["q_pi_mean"] == pytest.approx(12.0)
    assert summary["q_pi_noisy_mean"] == pytest.approx(10.0)
    assert summary["q_pi_center_mean"] == pytest.approx(16.0)
    assert _summarize_actor_interval_metrics(
        [{"actor_rows": 0.0, "actor_loss": 3.0}]
    ) == {
        "actor_interval_updates": 0,
        "actor_interval_applied": 0,
    }


def test_actor_q_center_resume_migration_is_explicit_and_legacy_zero_is_safe():
    assert _allow_actor_q_center_fraction_mismatch(
        {}, 0.0, explicitly_allowed=False
    )
    assert not _allow_actor_q_center_fraction_mismatch(
        {"actor_q_center_fraction": 0.5},
        0.5,
        explicitly_allowed=False,
    )
    with pytest.raises(ValueError, match="actor_q_center_fraction"):
        _allow_actor_q_center_fraction_mismatch(
            {}, 0.5, explicitly_allowed=False
        )
    assert _allow_actor_q_center_fraction_mismatch(
        {}, 0.5, explicitly_allowed=True
    )
    with pytest.raises(ValueError, match="invalid"):
        _allow_actor_q_center_fraction_mismatch(
            {"actor_q_center_fraction": float("nan")},
            0.5,
            explicitly_allowed=True,
        )


def test_seedmine_teacher_validation_ignores_actor_q_training_fraction_only():
    source = {
        "schema_version": SCHEMA_VERSION,
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
    }
    expected = dict(source, actor_q_center_fraction=0.5)

    _validate_seedmine_source_metadata(source, expected)
    with pytest.raises(ValueError, match="action_policy"):
        _validate_seedmine_source_metadata(
            dict(source, action_policy="wrong"), expected
        )


def test_route_efficiency_revision_requires_explicit_one_time_migration():
    expected = {
        "reward_revision": "outer_rail_v1",
        "refresh_ramp_side_on_dump": True,
        "outer_rail_penalty_per_step": 0.04,
    }

    assert _validate_route_efficiency_resume(
        {}, expected, allow_legacy_missing=True
    )
    with pytest.raises(ValueError, match="predates"):
        _validate_route_efficiency_resume(
            {}, expected, allow_legacy_missing=False
        )
    with pytest.raises(ValueError, match="partial"):
        _validate_route_efficiency_resume(
            {"reward_revision": "outer_rail_v1"},
            expected,
            allow_legacy_missing=True,
        )
    assert not _validate_route_efficiency_resume(
        dict(expected), expected, allow_legacy_missing=False
    )


def test_route_efficiency_v1_to_v2_requires_explicit_revision_migration():
    old = {
        "reward_revision": "outer_rail_v1",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.85,
        "outer_rail_exit_x": 2.55,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 20,
        "outer_rail_penalty_per_step": 0.04,
        "outer_rail_penalty_cap": 8.0,
    }
    new = dict(
        old,
        reward_revision="outer_rail_v2",
        outer_rail_min_scale=0.75,
        outer_rail_escalation_steps=20,
        outer_rail_max_multiplier=3.0,
        intake_substeps=2,
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        _validate_route_efficiency_resume(
            old,
            new,
            allow_legacy_missing=False,
        )
    assert _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )


def test_route_efficiency_v2_to_v3_requires_explicit_revision_migration():
    old = {
        "reward_revision": "outer_rail_v2",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.75,
        "outer_rail_exit_x": 2.45,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 5,
        "outer_rail_penalty_per_step": 0.12,
        "outer_rail_penalty_cap": 25.0,
        "outer_rail_min_scale": 0.75,
        "outer_rail_escalation_steps": 20,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
    }
    new = dict(
        old,
        reward_revision="outer_rail_v3",
        outer_rail_enter_x=2.55,
        outer_rail_exit_x=2.20,
        outer_rail_grace_steps=2,
        outer_rail_penalty_per_step=0.30,
        outer_rail_penalty_cap=150.0,
        outer_rail_min_scale=1.0,
        outer_rail_escalation_steps=10,
        outer_rail_max_multiplier=5.0,
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        _validate_route_efficiency_resume(
            old,
            new,
            allow_legacy_missing=False,
        )
    assert _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )


def test_route_efficiency_v3_to_ramp_out_v4_requires_explicit_migration():
    old = {
        "reward_revision": "outer_rail_v3",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.55,
        "outer_rail_exit_x": 2.20,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 10,
        "outer_rail_penalty_per_step": 0.22,
        "outer_rail_penalty_cap": 110.0,
        "outer_rail_min_scale": 0.35,
        "outer_rail_escalation_steps": 120,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
    }
    new = dict(
        old,
        reward_revision="outer_rail_v4_ramp_out",
        require_ramp_out=True,
        ramp_out_half_width=0.90,
        ramp_out_bonus=24.0,
        off_ramp_exit_penalty=20.0,
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        _validate_route_efficiency_resume(
            old, new, allow_legacy_missing=False
        )
    assert _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )


def test_ramp_out_v4_to_cycle_efficiency_v5_and_target_load_are_explicit():
    old = {
        "reward_revision": "outer_rail_v4_ramp_out",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.55,
        "outer_rail_exit_x": 2.20,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 10,
        "outer_rail_penalty_per_step": 0.22,
        "outer_rail_penalty_cap": 110.0,
        "outer_rail_min_scale": 0.35,
        "outer_rail_escalation_steps": 120,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
        "require_ramp_out": True,
        "ramp_out_half_width": 0.90,
        "ramp_out_bonus": 24.0,
        "off_ramp_exit_penalty": 20.0,
        "target_load": 15,
    }
    new = dict(
        old,
        reward_revision="cycle_efficiency_v5",
        postdump_require_target_load=True,
        target_load=20,
    )

    migrated = _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )
    assert migrated
    assert _allow_target_load_mismatch(
        old,
        20,
        route_revision_migrated=migrated,
        explicitly_allowed=True,
    )
    with pytest.raises(ValueError, match="target_load"):
        _allow_target_load_mismatch(
            old,
            20,
            route_revision_migrated=migrated,
            explicitly_allowed=False,
        )
    with pytest.raises(ValueError, match="active supported"):
        _allow_target_load_mismatch(
            old,
            15,
            route_revision_migrated=False,
            explicitly_allowed=True,
        )


def test_cycle_efficiency_v5_to_cycle_bridge_v6_requires_complete_parent():
    old = {
        "reward_revision": "cycle_efficiency_v5",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.55,
        "outer_rail_exit_x": 2.20,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 10,
        "outer_rail_penalty_per_step": 0.22,
        "outer_rail_penalty_cap": 110.0,
        "outer_rail_min_scale": 0.35,
        "outer_rail_escalation_steps": 120,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
        "require_ramp_out": True,
        "ramp_out_half_width": 0.90,
        "ramp_out_bonus": 24.0,
        "off_ramp_exit_penalty": 20.0,
        "postdump_require_target_load": True,
    }
    new = dict(
        old,
        reward_revision="cycle_bridge_v6",
        postdump_complete_cycle=True,
    )

    assert _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )

    incomplete = dict(old)
    incomplete.pop("postdump_require_target_load")
    with pytest.raises(ValueError, match="incomplete"):
        _validate_route_efficiency_resume(
            incomplete,
            new,
            allow_legacy_missing=False,
            allow_revision_migration=True,
        )
    changed_route = dict(new, ramp_out_bonus=99.0)
    with pytest.raises(ValueError, match="ramp_out_bonus"):
        _validate_route_efficiency_resume(
            old,
            changed_route,
            allow_legacy_missing=False,
            allow_revision_migration=True,
        )


def test_ramp_out_v4_can_branch_directly_to_soft_score_efficiency_v8():
    old = {
        "reward_revision": "outer_rail_v4_ramp_out",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.55,
        "outer_rail_exit_x": 2.20,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 10,
        "outer_rail_penalty_per_step": 0.22,
        "outer_rail_penalty_cap": 110.0,
        "outer_rail_min_scale": 0.35,
        "outer_rail_escalation_steps": 120,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
        "require_ramp_out": True,
        "ramp_out_half_width": 0.90,
        "ramp_out_bonus": 24.0,
        "off_ramp_exit_penalty": 20.0,
    }
    new = dict(
        old,
        reward_revision="score_efficiency_v8",
        postdump_require_target_load=False,
        postdump_complete_cycle=False,
        postdump_depleted_count=0,
        postdump_depleted_prob=0.0,
        preferred_repeat_load=30,
        repeat_load_return_bonus=8.0,
        repeat_load_score_bonus=12.0,
    )

    assert _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )
    assert _allow_suffix_alpha_mismatch(
        {"suffix_alpha": 1.0},
        0.15,
        route_revision_migrated=True,
        explicitly_allowed=True,
    )
    with pytest.raises(ValueError, match="suffix_alpha"):
        _allow_suffix_alpha_mismatch(
            {"suffix_alpha": 1.0},
            0.15,
            route_revision_migrated=True,
            explicitly_allowed=False,
        )
    changed_parent = dict(new, outer_rail_penalty_per_step=0.30)
    with pytest.raises(ValueError, match="outer_rail_penalty_per_step"):
        _validate_route_efficiency_resume(
            old,
            changed_parent,
            allow_legacy_missing=False,
            allow_revision_migration=True,
        )


def test_score_efficiency_v8_to_v9_only_adds_transition_fallbacks():
    old = {
        "reward_revision": "score_efficiency_v8",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.55,
        "outer_rail_exit_x": 2.20,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 10,
        "outer_rail_penalty_per_step": 0.22,
        "outer_rail_penalty_cap": 110.0,
        "outer_rail_min_scale": 0.35,
        "outer_rail_escalation_steps": 120,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
        "require_ramp_out": True,
        "ramp_out_half_width": 0.90,
        "ramp_out_bonus": 24.0,
        "off_ramp_exit_penalty": 20.0,
        "postdump_require_target_load": False,
        "postdump_complete_cycle": False,
        "postdump_depleted_count": 0,
        "postdump_depleted_prob": 0.0,
        "preferred_repeat_load": 30,
        "repeat_load_return_bonus": 8.0,
        "repeat_load_score_bonus": 12.0,
    }
    new = dict(
        old,
        reward_revision="score_efficiency_v9",
        collect_stall_steps=12,
        return_time_guard=0.20,
    )

    assert _validate_route_efficiency_resume(
        old,
        new,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )
    with pytest.raises(ValueError, match="outer_rail_penalty_per_step"):
        _validate_route_efficiency_resume(
            old,
            dict(new, outer_rail_penalty_per_step=0.30),
            allow_legacy_missing=False,
            allow_revision_migration=True,
        )


def test_v10_return_intake_supports_exact_v4_and_v9_migrations():
    v4 = {
        "reward_revision": "outer_rail_v4_ramp_out",
        "refresh_ramp_side_on_dump": True,
        "ramp_side_deadband_x": 0.25,
        "outer_rail_enter_x": 2.55,
        "outer_rail_exit_x": 2.20,
        "outer_rail_max_x": 3.60,
        "outer_rail_grace_steps": 10,
        "outer_rail_penalty_per_step": 0.22,
        "outer_rail_penalty_cap": 110.0,
        "outer_rail_min_scale": 0.35,
        "outer_rail_escalation_steps": 120,
        "outer_rail_max_multiplier": 3.0,
        "intake_substeps": 2,
        "require_ramp_out": True,
        "ramp_out_half_width": 0.90,
        "ramp_out_bonus": 24.0,
        "off_ramp_exit_penalty": 20.0,
    }
    v10 = dict(
        v4,
        reward_revision="score_efficiency_v10_return_intake",
        postdump_require_target_load=False,
        postdump_complete_cycle=False,
        postdump_depleted_count=0,
        postdump_depleted_prob=0.0,
        preferred_repeat_load=20,
        repeat_load_return_bonus=8.0,
        repeat_load_score_bonus=12.0,
        collect_stall_steps=0,
        return_time_guard=0.0,
        intake_during_return=True,
    )
    assert _validate_route_efficiency_resume(
        v4,
        v10,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )

    v9 = dict(
        v10,
        reward_revision="score_efficiency_v9",
        collect_stall_steps=30,
        return_time_guard=0.25,
    )
    v9.pop("intake_during_return")
    assert _validate_route_efficiency_resume(
        v9,
        v10,
        allow_legacy_missing=False,
        allow_revision_migration=True,
    )

    changed_v4 = dict(v4, ramp_out_bonus=99.0)
    with pytest.raises(ValueError, match="ramp_out_bonus"):
        _validate_route_efficiency_resume(
            changed_v4,
            v10,
            allow_legacy_missing=False,
            allow_revision_migration=True,
        )


def test_schema_and_action_policy_are_v23_exact_prefix_contract():
    assert SCHEMA_VERSION == "stagec_v2.3"
    assert ACTION_POLICY == "frozen_prefix_exact_first_v2"


def _captured_elite(
    *,
    cycles: int = 1,
    group: str = "full",
    terminal_reason: str = "skill_success",
    ramp_out_successes: int | None = None,
    milestones: dict[str, int] | None = None,
) -> CapturedEpisode:
    steps = 5
    proprio = np.zeros((steps, V2_PROPRIO_DIM), np.float32)
    proprio[:2, FIRST_PHASE_INDEX] = 1.0
    proprio[2:, FIRST_PHASE_INDEX + 1] = 1.0
    done = np.zeros(steps, bool)
    done[-1] = True
    if ramp_out_successes is None:
        ramp_out_successes = max(0, cycles)
    if milestones is None:
        milestones = {
            "target_load": int(cycles >= 1),
            "returned_home": int(cycles >= 1),
            "cycle_scored": int(cycles),
        }
    stats = {
        "reset_mode": group,
        "terminal_reason": terminal_reason,
        "cycles_completed": cycles,
        "ramp_out_attempts": max(0, cycles),
        "ramp_out_successes": ramp_out_successes,
        "milestones": milestones,
        "collector": 0,
        "env_index": 1,
        "episode_seq": 9,
        "policy_train_steps": 123,
    }
    return CapturedEpisode(
        stream_index=1,
        group=group,
        arrays={
            "obs": np.zeros((steps, 1, 2, 2), np.uint8),
            "proprio": proprio,
            "privileged": np.zeros((steps, 1), np.float32),
            "action": np.arange(steps, dtype=np.float32)[:, None],
            "reward": np.arange(steps, dtype=np.float32),
            "done": done,
        },
        stats=stats,
    )


def test_elite_episode_keeps_only_trainable_suffix_and_prefers_cycle_tier():
    episode = _captured_elite(cycles=1)

    assert _elite_tier(episode.stats) == "cycle"
    tier, arrays = _prepare_elite_episode(episode)

    assert tier == "cycle"
    assert arrays["reward"].tolist() == [2.0, 3.0, 4.0]
    assert arrays["action"][:, 0].tolist() == [2.0, 3.0, 4.0]
    assert arrays["done"].tolist() == [False, False, True]


def test_multi_cycle_elite_has_higher_tier_than_cycle2_only():
    assert _elite_tier(_captured_elite(cycles=2).stats) == "multi_cycle"
    tier, arrays = _prepare_elite_episode(_captured_elite(cycles=2))
    assert tier == "multi_cycle"
    assert len(arrays["reward"]) == 3


def test_full_route_elite_requires_one_ramp_out_per_completed_cycle():
    wall_cycle = {
        "reset_mode": "full",
        "cycles_completed": 1,
        "ramp_out_attempts": 1,
        "ramp_out_successes": 0,
        "milestones": {"cycle_scored": 1, "returned_home": 1},
    }
    clean_cycle = dict(wall_cycle, ramp_out_successes=1)
    mixed_multi = dict(
        wall_cycle,
        cycles_completed=2,
        ramp_out_attempts=2,
        ramp_out_successes=1,
        milestones={"cycle_scored": 2, "returned_home": 2},
    )

    assert _elite_tier(wall_cycle) is None
    assert _elite_tier(clean_cycle) == "cycle"
    assert _elite_tier(mixed_multi) == "cycle"


def test_elite_classification_keeps_outcome_and_source_independent():
    assert _elite_classification(_captured_elite(cycles=2)) == (
        "multi_cycle",
        "full_multi",
    )
    assert _elite_classification(_captured_elite(cycles=1)) == (
        "cycle",
        "full_cycle",
    )
    assert _elite_classification(
        _captured_elite(
            cycles=0,
            milestones={"returned_home": 1},
        )
    ) == ("return", "full_return")
    assert _elite_classification(_captured_elite(cycles=1, group="return")) == (
        "cycle",
        "return",
    )
    assert _elite_classification(
        _captured_elite(
            cycles=0,
            group="return",
            milestones={"returned_home": 1},
        )
    ) == ("return", "return")


def test_postdump_elite_requires_the_complete_success_chain():
    qualified = _captured_elite(cycles=1, group="postdump")
    assert _elite_classification(qualified) == ("cycle", "postdump_cycle")
    assert _prepare_elite_episode(qualified) is not None

    variants = []
    for update in (
        {"terminal_reason": "horizon"},
        {"cycles_completed": 0},
        {"ramp_out_successes": 0},
    ):
        stats = dict(qualified.stats)
        stats.update(update)
        variants.append(
            CapturedEpisode(
                qualified.stream_index,
                qualified.group,
                qualified.arrays,
                stats,
            )
        )
    for missing in ("target_load", "returned_home", "cycle_scored"):
        stats = dict(qualified.stats)
        stats["milestones"] = dict(stats["milestones"], **{missing: 0})
        variants.append(
            CapturedEpisode(
                qualified.stream_index,
                qualified.group,
                qualified.arrays,
                stats,
            )
        )
    assert all(_elite_classification(episode) is None for episode in variants)


def test_elite_capture_rejects_group_reset_mode_mismatch():
    episode = _captured_elite()
    stats = dict(episode.stats, reset_mode="return")
    mismatch = CapturedEpisode(
        episode.stream_index, episode.group, episode.arrays, stats
    )

    with pytest.raises(ValueError, match="group/reset mode mismatch"):
        _prepare_elite_episode(mismatch)


def test_only_integrated_success_sources_feed_behavior_anchor():
    assert _elite_behavior_eligible("full_multi")
    assert _elite_behavior_eligible("full_cycle")
    assert _elite_behavior_eligible("postdump_cycle")
    assert not _elite_behavior_eligible("full_return")
    assert not _elite_behavior_eligible("return")


def test_elite_archive_round_trip_and_contract_rejection(tmp_path):
    contract = {
        "schema_version": SCHEMA_VERSION,
        "prefix_sha256": "a" * 64,
        "proprio_dim": V2_PROPRIO_DIM,
    }
    archived = _archive_elite_episode(tmp_path, _captured_elite(), contract)
    assert archived is not None
    path, tier, pool, expected_arrays = archived

    loaded_tier, loaded_arrays = _load_elite_archive(path, contract)
    record_tier, record_pool, _, group = _load_elite_archive_record(path, contract)

    assert loaded_tier == tier == "cycle"
    assert record_tier == tier
    assert record_pool == pool == "full_cycle"
    assert group == "full"
    assert "full_cycle" in path.name
    for key in expected_arrays:
        np.testing.assert_array_equal(loaded_arrays[key], expected_arrays[key])
    with pytest.raises(ValueError, match="prefix_sha256"):
        _load_elite_archive(path, dict(contract, prefix_sha256="b" * 64))

    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    metadata = json.loads(bytes(payload["metadata"]).decode("utf-8"))
    metadata["pool"] = "return"
    payload["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="pool disagrees"):
        _load_elite_archive(path, contract)


def test_elite_archive_pruning_is_per_pool_without_backfill(tmp_path):
    contract = {"schema_version": SCHEMA_VERSION}
    paths: dict[str, list] = {"full_cycle": [], "return": []}
    for index in range(4):
        archived = _archive_elite_episode(
            tmp_path, _captured_elite(cycles=1), contract
        )
        assert archived is not None
        paths["full_cycle"].append(archived[0])
    for index in range(5):
        archived = _archive_elite_episode(
            tmp_path, _captured_elite(cycles=1, group="return"), contract
        )
        assert archived is not None
        paths["return"].append(archived[0])
    legacy = tmp_path / "elite_legacy_cycle.npz"
    legacy.write_bytes(b"legacy is not a v7 archive")

    removed = _prune_elite_archives(tmp_path, 12)

    # 12 slots map to 3/3/1/4/1; unused pools do not donate their quota.
    assert _exact_pool_quotas(12, ELITE_ARCHIVE_POOL_WEIGHTS) == {
        "full_multi": 3,
        "full_cycle": 3,
        "full_return": 1,
        "postdump_cycle": 4,
        "return": 1,
    }
    assert _exact_pool_quotas(96, ELITE_ARCHIVE_POOL_WEIGHTS) == {
        "full_multi": 24,
        "full_cycle": 24,
        "full_return": 8,
        "postdump_cycle": 32,
        "return": 8,
    }
    assert len(removed) == 5
    assert len(list(tmp_path.glob("elite_full_cycle_*.npz"))) == 3
    assert len(list(tmp_path.glob("elite_return_*.npz"))) == 1
    assert legacy.exists()
    with pytest.raises(ValueError, match="positive"):
        _prune_elite_archives(tmp_path, 0)


def _elite_test_ring(seed: int, reward: float) -> ReplayRing:
    kwargs = dict(
        capacity=32,
        obs_shape=(1, 2, 2),
        proprio_dim=V2_PROPRIO_DIM,
        privileged_dim=1,
        action_dim=1,
        n_step=1,
        gamma=0.9,
    )
    ring = ReplayRing(seed=seed, **kwargs)
    for step in range(8):
        ring.add(
            np.zeros((1, 2, 2), np.uint8),
            np.zeros(V2_PROPRIO_DIM, np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, np.float32),
            reward,
            step == 7,
        )
    return ring


def test_elite_capacity_and_sampler_use_fixed_source_quotas():
    assert tuple(ELITE_POOL_WEIGHTS) == ELITE_POOLS
    assert _exact_pool_quotas(50_000, ELITE_POOL_WEIGHTS) == {
        "full_multi": 15_000,
        "full_cycle": 10_000,
        "full_return": 5_000,
        "postdump_cycle": 15_000,
        "return": 5_000,
    }
    rewards = {
        "full_multi": 50.0,
        "full_cycle": 40.0,
        "full_return": 30.0,
        "postdump_cycle": 20.0,
        "return": 10.0,
    }
    rings = {
        pool: _elite_test_ring(index + 1, rewards[pool])
        for index, pool in enumerate(ELITE_POOLS)
    }

    batch = _sample_elite(rings, 20, np.random.default_rng(3))

    assert batch is not None
    assert len(batch.reward) == 20
    assert np.count_nonzero(batch.reward == 50.0) == 6
    assert np.count_nonzero(batch.reward == 40.0) == 4
    assert np.count_nonzero(batch.reward == 30.0) == 2
    assert np.count_nonzero(batch.reward == 20.0) == 6
    assert np.count_nonzero(batch.reward == 10.0) == 2


def test_elite_sampler_does_not_renormalize_an_available_return_pool():
    rings = {"return": _elite_test_ring(1, 10.0)}

    batch = _sample_elite(rings, 20, np.random.default_rng(4))

    assert batch is not None
    assert len(batch.reward) == 2
    assert bool((batch.reward == 10.0).all())


def test_elite_behavior_pool_keeps_full_suffix_and_guarantees_fire_examples():
    steps = 12
    arrays = {
        "obs": np.arange(steps, dtype=np.uint8)[:, None, None, None],
        "proprio": np.zeros((steps, V2_PROPRIO_DIM), np.float32),
        "action": np.full((steps, 7), -1.0, np.float32),
    }
    arrays["proprio"][:3, FIRST_PHASE_INDEX + 3] = 1.0
    arrays["proprio"][3:10, FIRST_PHASE_INDEX + 4] = 1.0
    arrays["proprio"][10:, FIRST_PHASE_INDEX + 1] = 1.0
    arrays["action"][[5, 9], 5] = 1.0
    pool = EliteScoreBehaviorPool(score_capacity=20, trigger_capacity=4, seed=7)

    pool.add(arrays)
    batch = pool.sample(8, trigger_fraction=0.25)

    assert pool.score_rows == 8
    assert pool.trigger_rows == 2
    assert batch is not None
    assert bool((batch.proprio[:, FIRST_PHASE_INDEX] == 0.0).all())
    assert np.count_nonzero(batch.action[:, 5] > 0.0) == 2


def test_elite_behavior_pool_is_bounded_and_can_sample_trigger_only():
    steps = 6
    arrays = {
        "obs": np.arange(steps, dtype=np.uint8)[:, None, None, None],
        "proprio": np.zeros((steps, V2_PROPRIO_DIM), np.float32),
        "action": np.full((steps, 7), -1.0, np.float32),
    }
    arrays["proprio"][:5, FIRST_PHASE_INDEX + 4] = 1.0
    arrays["proprio"][5, FIRST_PHASE_INDEX + 1] = 1.0
    arrays["action"][:, 5] = 1.0
    pool = EliteScoreBehaviorPool(score_capacity=3, trigger_capacity=2, seed=8)

    pool.add(arrays)
    batch = pool.sample(5, trigger_fraction=0.25)

    assert pool.score_rows == 0
    assert pool.trigger_rows == 2
    assert batch is not None
    assert len(batch.proprio) == 5
    assert bool((batch.action[:, 5] > 0.0).all())


def _behavior_custody_episode(source: int) -> dict[str, np.ndarray]:
    steps = 6
    arrays = {
        "obs": np.asarray(
            [source * 10 + row for row in range(steps)], dtype=np.uint8
        )[:, None, None, None],
        "proprio": np.zeros((steps, V2_PROPRIO_DIM), np.float32),
        "action": np.full((steps, 7), -1.0, np.float32),
    }
    arrays["proprio"][:5, FIRST_PHASE_INDEX + 4] = 1.0
    arrays["proprio"][5, FIRST_PHASE_INDEX + 1] = 1.0
    return arrays


def test_elite_behavior_pool_reservoir_retains_multiple_file_additions():
    pool = EliteScoreBehaviorPool(score_capacity=6, trigger_capacity=2, seed=31)

    for source in range(4):
        pool.add(_behavior_custody_episode(source))

    retained = pool._score.obs[:, 0, 0, 0].astype(int)  # noqa: SLF001
    retained_sources = set((retained // 10).tolist())
    assert pool.score_rows == 6
    assert len(retained_sources) >= 3
    assert retained_sources != {3}


def test_elite_behavior_pool_reservoir_is_seed_deterministic():
    pools = [
        EliteScoreBehaviorPool(score_capacity=6, trigger_capacity=2, seed=32)
        for _ in range(2)
    ]
    for pool in pools:
        for source in range(5):
            pool.add(_behavior_custody_episode(source))

    assert np.array_equal(pools[0]._score.obs, pools[1]._score.obs)  # noqa: SLF001


def test_elite_behavior_pool_rejects_unfinished_tail_only_episode():
    steps = 6
    arrays = {
        "obs": np.arange(steps, dtype=np.uint8)[:, None, None, None],
        "proprio": np.zeros((steps, V2_PROPRIO_DIM), np.float32),
        "action": np.full((steps, 7), -1.0, np.float32),
    }
    arrays["proprio"][:, FIRST_PHASE_INDEX + 2] = 1.0
    pool = EliteScoreBehaviorPool(score_capacity=8, trigger_capacity=2, seed=9)

    with pytest.raises(ValueError, match="SCORE-to-LEAVE"):
        pool.add(arrays)


def _write_seedmine_archive(
    path,
    *,
    source_sha: str,
    prefix_sha: str,
    tier: str = "cycle",
    early_done: bool = False,
    action_mode: str = "deterministic",
    episode_len_s: float = 90.0,
    success_step: int = 4,
):
    steps = 5
    proprio = np.zeros((steps, V2_PROPRIO_DIM), np.float32)
    proprio[:2, FIRST_PHASE_INDEX] = 1.0
    if tier == "cycle":
        proprio[2, FIRST_PHASE_INDEX + 4] = 1.0
        proprio[3:, FIRST_PHASE_INDEX + 1] = 1.0
    else:
        proprio[2:, FIRST_PHASE_INDEX + 1] = 1.0
    arrays = {
        "obs": np.zeros((steps, 1, 2, 2), np.uint8),
        "proprio": proprio,
        "privileged": np.zeros((steps, 2), np.float32),
        "action": np.zeros((steps, 7), np.float32),
        "reward": np.arange(steps, dtype=np.float32),
        "done": np.asarray([False, early_done, False, False, True], bool),
    }
    milestones = (
        {"cycle_scored": 1, "returned_home": 1}
        if tier == "cycle"
        else {"returned_home": 1}
    )
    episode = {
        "schema": SEEDMINE_EVAL_SCHEMA,
        "checkpoint_sha256": source_sha,
        "prefix_sha256": prefix_sha,
        "stagec_v2_metadata": {
            "schema_version": SCHEMA_VERSION,
            "prefix_sha256": prefix_sha,
            "action_policy": ACTION_POLICY,
            "field_strategy": FIELD_STRATEGY,
            "proprio_dim": V2_PROPRIO_DIM,
        },
        "mode": "full",
        "action_mode": action_mode,
        "episode_len_s": episode_len_s,
        "episode_steps": steps,
        "cycle_success_steps": [success_step] if tier == "cycle" else [],
        "env_index": 0,
        "num_envs": 2,
        "cycles_completed": int(tier == "cycle"),
        "milestones": milestones,
        "capture_tier": "cycle" if tier == "cycle" else "returned_home",
    }
    metadata = {
        "schema": SEEDMINE_CAPTURE_SCHEMA,
        "capture_tier": episode["capture_tier"],
        "field_keys": list(arrays),
        "length": steps,
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "episode": episode,
    }
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            metadata=np.frombuffer(
                json.dumps(metadata, sort_keys=True).encode("utf-8"),
                dtype=np.uint8,
            ),
        )


def _seedmine_contract(prefix_sha: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "prefix_sha256": prefix_sha,
        "action_policy": ACTION_POLICY,
        "field_strategy": FIELD_STRATEGY,
        "proprio_dim": V2_PROPRIO_DIM,
        "privileged_dim": 2,
        "action_dim": 7,
        "obs_shape": [1, 2, 2],
        "full_episode_s": 90.0,
    }


def test_seedmine_behavior_only_requires_complete_provenance_pair(tmp_path):
    with pytest.raises(ValueError, match="seedmine-behavior-only requires"):
        _validate_seedmine_options(None, None, behavior_only=True)
    with pytest.raises(ValueError, match="must be used together"):
        _validate_seedmine_options(
            tmp_path / "episodes", None, behavior_only=False
        )

    _validate_seedmine_options(
        tmp_path / "episodes",
        tmp_path / "source.pt",
        behavior_only=True,
    )


def test_seedmine_behavior_only_never_enters_critic_replay(tmp_path):
    archive = tmp_path / "episode_cycle.npz"
    _write_seedmine_archive(
        archive, source_sha="b" * 64, prefix_sha="a" * 64
    )
    _, pool, arrays = _load_seedmine_archive_record(
        archive, _seedmine_contract("a" * 64), "b" * 64
    )
    ring = ReplayRing(
        20,
        obs_shape=(1, 2, 2),
        proprio_dim=V2_PROPRIO_DIM,
        privileged_dim=2,
        action_dim=7,
        n_step=1,
        seed=3,
    )
    rings = {pool: ring}

    added = _add_seedmine_episode_to_replay(
        rings, pool, arrays, behavior_only=True
    )
    assert added is False
    assert len(ring) == 0

    added = _add_seedmine_episode_to_replay(
        rings, pool, arrays, behavior_only=False
    )
    assert added is True
    assert len(ring) == len(arrays["reward"])


def test_critic_warmup_selects_only_seedmine_elite_replay():
    live = {"full_cycle": _elite_test_ring(1, 10.0)}
    seedmine = {"full_cycle": _elite_test_ring(2, 20.0)}

    assert (
        _elite_replay_rings_for_update(
            critic_only=True,
            elite_replays=live,
            seedmine_warmup_replays=seedmine,
        )
        is seedmine
    )
    assert (
        _elite_replay_rings_for_update(
            critic_only=False,
            elite_replays=live,
            seedmine_warmup_replays=seedmine,
        )
        is live
    )


def test_seedmine_bridge_validates_provenance_and_imports_suffix(tmp_path):
    source_sha = "b" * 64
    prefix_sha = "a" * 64
    archive = tmp_path / "episode_cycle.npz"
    _write_seedmine_archive(
        archive, source_sha=source_sha, prefix_sha=prefix_sha
    )

    tier, arrays = _load_seedmine_archive(
        archive, _seedmine_contract(prefix_sha), source_sha
    )
    record_tier, pool, record_arrays = _load_seedmine_archive_record(
        archive, _seedmine_contract(prefix_sha), source_sha
    )

    assert tier == "cycle"
    assert record_tier == tier
    assert pool == "full_cycle"
    assert arrays["reward"].tolist() == [2.0, 3.0, 4.0]
    assert record_arrays["reward"].tolist() == arrays["reward"].tolist()
    assert arrays["done"].tolist() == [False, False, True]
    with pytest.raises(ValueError, match="different candidate checkpoint"):
        _load_seedmine_archive(
            archive, _seedmine_contract(prefix_sha), "c" * 64
        )
    with pytest.raises(ValueError, match="different frozen prefix"):
        _load_seedmine_archive(
            archive, _seedmine_contract("d" * 64), source_sha
        )


def test_seedmine_bridge_rejects_nonfinal_terminal(tmp_path):
    archive = tmp_path / "episode_bad_done.npz"
    _write_seedmine_archive(
        archive,
        source_sha="b" * 64,
        prefix_sha="a" * 64,
        early_done=True,
    )

    with pytest.raises(ValueError, match="terminate exactly once"):
        _load_seedmine_archive(
            archive, _seedmine_contract("a" * 64), "b" * 64
        )


def test_seedmine_bridge_rejects_noisy_actor_custody(tmp_path):
    archive = tmp_path / "episode_noisy.npz"
    _write_seedmine_archive(
        archive,
        source_sha="b" * 64,
        prefix_sha="a" * 64,
        action_mode="policy-noise",
    )

    with pytest.raises(ValueError, match="deterministic action_mode"):
        _load_seedmine_archive(
            archive, _seedmine_contract("a" * 64), "b" * 64
        )


def test_seedmine_bridge_rejects_horizon_mismatch_and_truncates_tail(tmp_path):
    wrong_horizon = tmp_path / "episode_120s.npz"
    _write_seedmine_archive(
        wrong_horizon,
        source_sha="b" * 64,
        prefix_sha="a" * 64,
        episode_len_s=120.0,
    )
    with pytest.raises(ValueError, match="horizon mismatch"):
        _load_seedmine_archive(
            wrong_horizon, _seedmine_contract("a" * 64), "b" * 64
        )

    tailed = tmp_path / "episode_tail.npz"
    _write_seedmine_archive(
        tailed,
        source_sha="b" * 64,
        prefix_sha="a" * 64,
        success_step=3,
    )
    _, arrays = _load_seedmine_archive(
        tailed, _seedmine_contract("a" * 64), "b" * 64
    )
    assert arrays["reward"].tolist() == [2.0, 3.0]
    assert arrays["done"].tolist() == [False, True]


def test_seedmine_bridge_maps_returned_home_to_lower_elite_tier(tmp_path):
    archive = tmp_path / "episode_returned_home.npz"
    _write_seedmine_archive(
        archive,
        source_sha="b" * 64,
        prefix_sha="a" * 64,
        tier="return",
    )

    tier, arrays = _load_seedmine_archive(
        archive, _seedmine_contract("a" * 64), "b" * 64
    )
    record_tier, pool, _ = _load_seedmine_archive_record(
        archive, _seedmine_contract("a" * 64), "b" * 64
    )

    assert tier == "return"
    assert record_tier == tier
    assert pool == "full_return"
    assert len(arrays["reward"]) == 3
