from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import json

import numpy as np
import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "rl"
        / "eval_stagec_seedmine.py"
    )
    spec = importlib.util.spec_from_file_location("eval_stagec_seedmine", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "candidate.pt"),
        "--prefix-checkpoint",
        str(tmp_path / "prefix.pt"),
        "--out",
        str(tmp_path / "episodes.jsonl"),
    ]


def test_parser_caps_physics_environments_at_two(tmp_path):
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(_required_args(tmp_path) + ["--num-envs", "3"])


def test_parser_separates_environment_and_action_seeds(tmp_path):
    module = _module()
    args = module.parse_args(
        _required_args(tmp_path)
        + ["--env-seed", "7000", "--action-seed", "9000", "--mode", "return"]
    )
    assert args.env_seed == 7000
    assert args.action_seed == 9000
    assert module.resolved_episode_len_s(args) == 75.0


def test_fixed_gaussian_requires_positive_noise(tmp_path):
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(
            _required_args(tmp_path) + ["--action-mode", "fixed-gaussian"]
        )
    args = module.parse_args(
        _required_args(tmp_path)
        + ["--action-mode", "fixed-gaussian", "--noise-std", "0.3"]
    )
    assert args.noise_std == pytest.approx(0.3)


def test_smooth_drive_defaults_correlation_and_rejects_invalid_values(tmp_path):
    module = _module()
    args = module.parse_args(
        _required_args(tmp_path)
        + ["--action-mode", "smooth-drive", "--noise-std", "0.05"]
    )
    assert args.noise_correlation == pytest.approx(0.95)
    assert args.noise_cap == pytest.approx(0.05)
    assert args.noise_phases == ("leave",)
    with pytest.raises(SystemExit):
        module.parse_args(
            _required_args(tmp_path)
            + [
                "--action-mode",
                "smooth-drive",
                "--noise-std",
                "0.05",
                "--noise-correlation",
                "1.0",
            ]
        )


def test_parser_refuses_output_aliasing_checkpoint(tmp_path):
    module = _module()
    checkpoint = tmp_path / "candidate.pt"
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--prefix-checkpoint",
                str(tmp_path / "prefix.pt"),
                "--out",
                str(checkpoint),
            ]
        )


def test_returned_home_capture_is_opt_in_but_cycle_capture_is_not():
    module = _module()
    returned = {"cycles_completed": 0, "milestones": {"returned_home": 1}}
    cycle = {"cycles_completed": 1, "milestones": {"cycle_scored": 1}}
    assert module.capture_tier(returned) is None
    assert module.capture_tier(returned, include_returned_home=True) == "returned_home"
    assert module.capture_tier(cycle) == "cycle"


def test_returned_home_capture_flag_requires_capture_dir(tmp_path):
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(_required_args(tmp_path) + ["--capture-returned-home"])


def test_all_episode_trajectories_require_trajectory_output(tmp_path):
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(
            _required_args(tmp_path) + ["--trajectory-all-episodes"]
        )
    args = module.parse_args(
        _required_args(tmp_path)
        + [
            "--trajectory-out",
            str(tmp_path / "trajectories.jsonl"),
            "--trajectory-all-episodes",
        ]
    )
    assert args.trajectory_all_episodes


def _capture_arrays(module):
    buffer = module.new_capture_buffer()
    for step in range(3):
        module.append_capture_transition(
            buffer,
            obs=np.full((9, 2, 2), step, dtype=np.uint8),
            proprio=np.full(30, step, dtype=np.float32),
            privileged=np.full(26, step, dtype=np.float32),
            action=np.full(7, step / 10, dtype=np.float32),
            reward=float(step),
            done=step == 2,
        )
    return module.stack_capture_buffer(buffer)


def test_capture_schema_loads_directly_into_replay_ring():
    module = _module()
    arrays = _capture_arrays(module)
    assert tuple(arrays) == module.TRAINING_FIELD_KEYS
    assert arrays["obs"].shape == (3, 9, 2, 2)
    assert arrays["obs"].dtype == np.uint8
    assert arrays["proprio"].dtype == np.float32
    assert arrays["privileged"].dtype == np.float32
    assert arrays["action"].dtype == np.float32
    assert arrays["reward"].dtype == np.float32
    assert arrays["done"].dtype == np.bool_
    assert arrays["done"].tolist() == [False, False, True]

    from frc_rebuilt.rl.replay import ReplayRing

    ring = ReplayRing(
        capacity=8,
        obs_shape=(9, 2, 2),
        proprio_dim=30,
        privileged_dim=26,
        action_dim=7,
    )
    for step in range(3):
        ring.add(*(arrays[key][step] for key in module.TRAINING_FIELD_KEYS))
    assert len(ring) == 3
    assert ring.done[:3].tolist() == [False, False, True]


def test_atomic_capture_has_metadata_unique_names_and_no_temp_file(tmp_path):
    module = _module()
    arrays = _capture_arrays(module)
    record = {
        "episode_index": 4,
        "env_index": 1,
        "env_seed": 750000,
        "action_seed": 750000,
        "checkpoint_sha256": "a" * 64,
        "scored": 71,
        "cycles_completed": 1,
        "milestones": {"cycle_scored": 1},
    }
    basename = module.capture_basename(record, "cycle")
    metadata = module.build_capture_metadata(record, arrays, "cycle")
    first = module.atomic_save_capture(tmp_path, arrays, metadata, basename)
    second = module.atomic_save_capture(tmp_path, arrays, metadata, basename)
    assert first.name == basename
    assert second.stem.endswith("_001")
    assert first != second
    assert not list(tmp_path.glob(".*.tmp"))

    with np.load(first, allow_pickle=False) as saved:
        assert set(saved.files) == {*module.TRAINING_FIELD_KEYS, "metadata"}
        decoded = json.loads(bytes(saved["metadata"]).decode("utf-8"))
        assert decoded["schema"] == module.CAPTURE_SCHEMA
        assert decoded["capture_tier"] == "cycle"
        assert decoded["length"] == 3
        assert decoded["field_keys"] == list(module.TRAINING_FIELD_KEYS)
        np.testing.assert_array_equal(saved["done"], arrays["done"])


def test_jsonable_converts_numpy_and_nonfinite_values():
    module = _module()
    converted = module._jsonable(
        {"count": np.int64(3), "route": np.asarray([1.0, 2.0]), "bad": float("nan")}
    )
    assert converted == {"count": 3, "route": [1.0, 2.0], "bad": None}


def test_fixed_gaussian_actions_repeat_from_action_seed():
    module = _module()

    class MeanAgent:
        def act(self, frames, proprio, explore):
            assert not explore
            return np.zeros((2, 7), dtype=np.float32)

    kwargs = {
        "agent": MeanAgent(),
        "frames": np.zeros((2, 9, 2, 2), dtype=np.uint8),
        "proprio": np.zeros((2, 30), dtype=np.float32),
        "action_mode": "fixed-gaussian",
        "noise_std": 0.3,
    }
    first = module._candidate_actions(
        **kwargs, action_rng=np.random.default_rng(750000)
    )
    second = module._candidate_actions(
        **kwargs, action_rng=np.random.default_rng(750000)
    )
    different = module._candidate_actions(
        **kwargs, action_rng=np.random.default_rng(750001)
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_smooth_drive_only_changes_suffix_travel_and_resets_per_environment():
    module = _module()

    class MeanAgent:
        def act(self, frames, proprio, explore):
            assert not explore
            return np.zeros((5, 7), dtype=np.float32)

    proprio = np.zeros((5, 30), dtype=np.float32)
    for row, phase in enumerate(range(5)):
        proprio[row, 22 + phase] = 1.0
    rng = np.random.default_rng(42)
    noise = module.SmoothDriveNoise(5, 0.05, 0.05, 0.95, {1, 2, 3}, rng)
    actions = module._candidate_actions(
        agent=MeanAgent(),
        frames=np.zeros((5, 9, 2, 2), dtype=np.uint8),
        proprio=proprio,
        action_mode="smooth-drive",
        noise_std=0.05,
        action_rng=rng,
        smooth_noise=noise,
    )
    np.testing.assert_array_equal(actions[0], np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(actions[4], np.zeros(7, dtype=np.float32))
    assert np.any(actions[1:4, :3] != 0.0)
    np.testing.assert_array_equal(actions[:, 3:], np.zeros((5, 4), dtype=np.float32))
    assert np.any(noise.state[2] != 0.0)
    noise.reset([2])
    np.testing.assert_array_equal(noise.state[2], np.zeros(3, dtype=np.float32))
    assert np.any(noise.state[1] != 0.0)


def test_smooth_drive_is_phase_selected_capped_and_resets_on_phase_entry():
    module = _module()
    rng = np.random.default_rng(7)
    noise = module.SmoothDriveNoise(1, 0.05, 0.05, 0.0, {2}, rng)
    proprio = np.zeros((1, 30), dtype=np.float32)
    proprio[0, 23] = 1.0  # LEAVE is not selected.
    np.testing.assert_array_equal(noise.sample(proprio), np.zeros((1, 3)))
    proprio[0, 23] = 0.0
    proprio[0, 24] = 1.0  # COLLECT is selected.
    first = noise.sample(proprio)
    assert np.any(first != 0.0)
    assert np.max(np.abs(first)) <= 0.05
    noise.state[:] = 0.05
    proprio[0, 24] = 0.0
    proprio[0, 25] = 1.0  # RETURN entry clears COLLECT history.
    np.testing.assert_array_equal(noise.sample(proprio), np.zeros((1, 3)))


def test_summary_uses_qualified_cycle_milestone_and_dump_counts():
    module = _module()
    records = [
        {
            "scored": 71,
            "collected": 79,
            "cycles_completed": 1,
            "dump_attempts": 2,
            "dump_empty_completions": 2,
            "partial_dumps": 0,
            "milestones": {"returned_home": 1, "cycle_scored": 1},
        },
        {
            "scored": 55,
            "collected": 70,
            "cycles_completed": 0,
            "dump_attempts": 1,
            "dump_empty_completions": 1,
            "partial_dumps": 0,
            "milestones": {"left_home": 1},
        },
    ]
    summary = module.summarize(records)
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["mean_scored"] == pytest.approx(63.0)
    assert summary["cycles_completed"] == 1
    assert summary["clean_dumps"] == 3
    assert summary["milestone_episodes"]["cycle_scored"] == 1


def test_output_refuses_to_overwrite_by_default(tmp_path):
    module = _module()
    output = tmp_path / "episodes.jsonl"
    output.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._prepare_output(output, overwrite=False)
    assert output.read_text(encoding="utf-8") == "preserve me"
