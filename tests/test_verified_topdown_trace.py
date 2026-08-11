from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_visual_state_sync_renders_without_advancing_physics():
    module = _load("render_verified_trace_topdown")

    class Sim:
        current_time = 12.5

        def __init__(self):
            self.kinematic_calls = 0
            self.physics_sim_view = self

        def update_articulations_kinematic(self):
            self.kinematic_calls += 1

    sim = Sim()
    published = []
    errors = module.synchronize_visual_state(
        sim,
        12.5,
        context="unit test",
        publish_to_fabric=lambda: published.append(True) or {"readback": 0.0},
    )
    assert sim.kinematic_calls == 1
    assert published == [True]
    assert errors == {"readback": 0.0}


def test_visual_state_sync_fails_closed_if_time_advances():
    module = _load("render_verified_trace_topdown")

    class Sim:
        current_time = 7.0
        physics_sim_view = None

        def __init__(self):
            self.physics_sim_view = self

        def update_articulations_kinematic(self):
            pass

        def publish(self):
            self.current_time += 1 / 60
            return {"readback": 0.0}

    sim = Sim()
    with pytest.raises(RuntimeError, match="advanced simulation time"):
        module.synchronize_visual_state(
            sim, 7.0, context="unit test", publish_to_fabric=sim.publish
        )


def _fixture(module, *, score: int = 202):
    steps, fuel_count, joints = 4, 3, 2
    module.STEPS = steps
    module.FUEL_COUNT = fuel_count
    arrays = {
        "robot_position": np.zeros((steps, 3), np.float32),
        "robot_orientation_wxyz": np.tile(
            np.asarray([1, 0, 0, 0], np.float32), (steps, 1)
        ),
        "robot_joint_position": np.zeros((steps, joints), np.float32),
        "robot_joint_velocity": np.zeros((steps, joints), np.float32),
        "fuel_position": np.zeros((steps, fuel_count, 3), np.float32),
        "fuel_orientation_wxyz": np.tile(
            np.asarray([1, 0, 0, 0], np.float32), (steps, fuel_count, 1)
        ),
        "mechanism": np.zeros((steps, 3), np.float32),
        "clock_s": np.asarray([0.0, 0.1, 0.2, 0.3], np.float32),
        "score": np.asarray([0, 2, 2, score], np.int32),
        "collected": np.asarray([0, 2, 2, 4], np.int32),
        "magazine": np.asarray([0, 2, 0, 2], np.int16),
        "phase": np.asarray(["OPENER", "LIVE1", "LIVE2", "ENDGAME"], dtype="<U24"),
        "hub_active": np.asarray([True, False, True, True], bool),
        "cycles": np.asarray([0, 0, 1, 2], np.int16),
        "action": np.zeros((steps, 2, 7), np.float32),
        "proprio": np.zeros((steps, 30), np.float32),
        "privileged": np.zeros((steps, 26), np.float32),
        "reward": np.zeros(steps, np.float32),
        "done": np.asarray([False, False, False, True], bool),
    }
    metadata = {
        "schema": module.TRACE_SCHEMA,
        "steps": steps,
        "fuel_count": fuel_count,
        "joint_count": joints,
        "joint_names": ["left", "right"],
        "env_index": 1,
        "episode_len_s": 160.0,
        "checkpoint_sha256": "a" * 64,
        "prefix_checkpoint_sha256": "b" * 64,
        "bundle_sha256": "c" * 64,
        "publication_min_score": module.MIN_LIVE_SCORE,
        "live_terminal_score": score,
        "contract": {
            "stage_d": True,
            "first_inactive": "blue",
            "ferry": False,
            "return_when_live": False,
            "owncourt_loop": False,
            "policy_speed_scale": 1.0,
            "prefix_rescue_s": 35.0,
        },
    }
    return metadata, arrays


def test_atomic_trace_roundtrip_enforces_sha_and_live_score(tmp_path):
    module = _load("verified_topdown_trace")
    metadata, arrays = _fixture(module)
    path = tmp_path / "trace.npz"
    digest = module.atomic_save_trace(path, metadata, arrays)
    trace = module.load_verified_trace(
        path,
        expected_trace_sha256=digest,
        expected_checkpoint_sha256="a" * 64,
    )
    assert trace.metadata["live_terminal_score"] == 202
    assert module.frame_telemetry(trace, 0)["score"] == 0
    assert module.frame_telemetry(trace, 3)["score"] == 202
    assert module.frame_telemetry(trace, 1)["hub"] == "INACTIVE"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.load_verified_trace(
            path,
            expected_trace_sha256="d" * 64,
            expected_checkpoint_sha256="a" * 64,
        )


def test_trace_fails_closed_below_200_or_with_missing_full_state(tmp_path):
    module = _load("verified_topdown_trace")
    metadata, arrays = _fixture(module, score=199)
    with pytest.raises(ValueError, match="below publication gate"):
        module.atomic_save_trace(tmp_path / "low.npz", metadata, arrays)
    metadata, arrays = _fixture(module)
    del arrays["fuel_orientation_wxyz"]
    with pytest.raises(ValueError, match="trace fields mismatch"):
        module.atomic_save_trace(tmp_path / "incomplete.npz", metadata, arrays)


def test_trace_terminal_metadata_must_match_monotonic_live_score(tmp_path):
    module = _load("verified_topdown_trace")
    metadata, arrays = _fixture(module)
    metadata["live_terminal_score"] = 203
    with pytest.raises(ValueError, match="metadata differs"):
        module.atomic_save_trace(tmp_path / "metadata-mismatch.npz", metadata, arrays)

    metadata, arrays = _fixture(module)
    arrays["score"] = np.asarray([0, 202, 201, 202], np.int32)
    with pytest.raises(ValueError, match="score must be monotonic"):
        module.atomic_save_trace(tmp_path / "score-regression.npz", metadata, arrays)


def test_offline_restore_sets_every_visual_surface_without_step_calls():
    trace_module = _load("verified_topdown_trace")
    render_module = _load("render_verified_trace_topdown")
    metadata, arrays = _fixture(trace_module)

    class Articulation:
        def __init__(self):
            self.calls = []

        def set_world_pose(self, position, orientation):
            self.calls.append(("pose", np.array(position), np.array(orientation)))

        def set_joint_positions(self, value):
            self.calls.append(("joint_position", np.array(value)))

        def set_joint_velocities(self, value):
            self.calls.append(("joint_velocity", np.array(value)))

    class Fuel:
        def __init__(self):
            self.calls = []

        def set_world_poses(self, **kwargs):
            self.calls.append(kwargs)

    class Controller:
        def __init__(self):
            self.visual_calls = 0

        def _apply_mechanism_visuals(self):
            self.visual_calls += 1

    class Slot:
        articulation = Articulation()
        fuel = Fuel()
        controller = Controller()

    class Trace:
        pass

    trace = Trace()
    trace.arrays = arrays
    render_module.restore_visual_state(trace, 2, Slot)
    assert [call[0] for call in Slot.articulation.calls] == [
        "pose",
        "joint_position",
        "joint_velocity",
    ]
    assert Slot.fuel.calls[0]["positions"].shape == (3, 3)
    assert Slot.fuel.calls[0]["orientations"].shape == (3, 4)
    assert Slot.controller.visual_calls == 1


def test_default_oblique_camera_is_slanted_and_keeps_full_field_in_frame():
    module = _load("render_verified_trace_topdown")
    origin = np.asarray([1.0, 2.0, 0.0], np.float64)
    camera = module.oblique_camera_layout(
        origin,
        height_m=module.DEFAULT_CAMERA_HEIGHT_M,
        tilt_deg=module.DEFAULT_CAMERA_TILT_DEG,
        azimuth_deg=module.DEFAULT_CAMERA_AZIMUTH_DEG,
    )
    assert camera["eye"][2] == pytest.approx(14.0)
    assert camera["eye"][0] == pytest.approx(origin[0])
    assert camera["eye"][1] < origin[1]
    assert camera["horizontal_offset_m"] == pytest.approx(
        14.0 * np.tan(np.deg2rad(18.0))
    )
    framing = module.verify_full_field_framing(
        camera,
        focal_length_mm=module.DEFAULT_CAMERA_FOCAL_LENGTH_MM,
        field_margin_m=module.DEFAULT_FIELD_FRAME_MARGIN_M,
    )
    assert framing["validated"] is True
    assert 0.0 < framing["maximum_frame_utilization"] < 1.0


def test_oblique_camera_rejects_a_cropped_field_configuration():
    module = _load("render_verified_trace_topdown")
    camera = module.oblique_camera_layout(
        np.zeros(3), height_m=14.0, tilt_deg=18.0, azimuth_deg=-90.0
    )
    with pytest.raises(ValueError, match="crops the regulation field"):
        module.verify_full_field_framing(
            camera,
            focal_length_mm=40.0,
            field_margin_m=0.5,
        )
    with pytest.raises(ValueError, match="between 0 and 40"):
        module.oblique_camera_layout(
            np.zeros(3), height_m=14.0, tilt_deg=41.0, azimuth_deg=-90.0
        )


def test_renderer_cli_defaults_to_the_validated_oblique_view(tmp_path):
    module = _load("render_verified_trace_topdown")
    placeholder = tmp_path / "x"
    args = module.parse_args(
        [
            "--trace", str(placeholder),
            "--expected-trace-sha256", "0" * 64,
            "--bundle", str(placeholder) + "1",
            "--expected-bundle-sha256", "1" * 64,
            "--checkpoint", str(placeholder) + "2",
            "--expected-checkpoint-sha256", "2" * 64,
            "--prefix-checkpoint", str(placeholder) + "3",
            "--expected-prefix-checkpoint-sha256", "3" * 64,
            "--code-root", str(tmp_path),
            "--code-archive", str(placeholder) + "4",
            "--expected-code-archive-sha256", "4" * 64,
            "--template", str(placeholder) + "5",
            "--expected-template-sha256", "5" * 64,
            "--output", str(tmp_path / "video.avi"),
        ]
    )
    assert args.camera_height_m == pytest.approx(14.0)
    assert args.camera_tilt_deg == pytest.approx(18.0)
    assert args.camera_azimuth_deg == pytest.approx(-90.0)
    assert args.camera_focal_length_mm == pytest.approx(14.0)


def test_trace_renderer_accepts_exact_saved_gui_camera_and_hd_layout(tmp_path):
    module = _load("render_verified_trace_topdown")
    camera_state = tmp_path / "gui_camera_pose.json"
    camera_state.write_text(
        json.dumps(
            {
                "schema": "frc-rebuilt-gui-camera-v1",
                "fuel_template_count": 456,
                "projection": "perspective",
                "eye_xyz": [6.0, -13.0, 10.0],
                "target_xyz": [-1.0, -4.5, -0.4],
                "up_xyz": [-0.42, 0.52, 0.74],
                "focal_length_mm": 15.8,
                "horizontal_aperture_mm": 20.955,
                "vertical_aperture_mm": 11.7871875,
                "horizontal_aperture_offset_mm": 0.0,
                "vertical_aperture_offset_mm": 0.0,
                "exposure": 0.0,
                "clipping_range": [0.01, 10000000.0],
                "viewport_resolution": [1920, 1080],
            }
        ),
        encoding="utf-8",
    )
    placeholder = tmp_path / "x"
    args = module.parse_args(
        [
            "--trace", str(placeholder),
            "--expected-trace-sha256", "0" * 64,
            "--bundle", str(placeholder) + "1",
            "--expected-bundle-sha256", "1" * 64,
            "--checkpoint", str(placeholder) + "2",
            "--expected-checkpoint-sha256", "2" * 64,
            "--prefix-checkpoint", str(placeholder) + "3",
            "--expected-prefix-checkpoint-sha256", "3" * 64,
            "--code-root", str(tmp_path),
            "--code-archive", str(placeholder) + "4",
            "--expected-code-archive-sha256", "4" * 64,
            "--template", str(placeholder) + "5",
            "--expected-template-sha256", "5" * 64,
            "--camera-state-json", str(camera_state),
            "--camera-focal-length-mm", "1000",
            "--output", str(tmp_path / "video.avi"),
        ]
    )
    assert args.camera_state_json == camera_state
    assert module.CAMERA_SIZE == (1920, 1080)
    assert module.SIDEBAR_WIDTH == 0


def test_trace_renderer_reads_three_policy_cameras_as_chw_panels():
    module = _load("render_verified_trace_topdown")

    class Camera:
        def __init__(self, value):
            self.value = value

        def get_rgba(self):
            y, x = np.indices((360, 640))
            return np.stack(
                ((x + self.value) % 256, (y + self.value) % 256, x ^ y), axis=-1
            ).astype(np.uint8)

    class Env:
        camera_names = ("intake", "shooter", "navigation")
        cameras = {
            (1, "intake"): Camera(1),
            (1, "shooter"): Camera(2),
            (1, "navigation"): Camera(3),
        }

    panels = module.policy_camera_frames(Env(), 1)
    assert panels.shape == (3, 3, 90, 160)
    assert panels.dtype == np.uint8
    assert panels.flags.c_contiguous
    assert panels[0, 0, 0, 1] == 5
    assert panels[0, 1, 1, 0] == 5


def test_seed_search_renderer_keeps_source_custody_and_uses_fresh_run_seeds():
    module = _load("render_verified_trace_topdown")
    metadata = {
        "env_seed": 2000001,
        "action_seed": 2000007,
        "env_index": 0,
        "classification": {"seed_search": True},
        "source_capture": {
            "target_env_seed": 1003201,
            "target_action_seed": 1003201,
            "target_env_index": 1,
        },
    }
    assert module.source_bundle_identity(metadata) == {
        "env_seed": 1003201,
        "action_seed": 1003201,
        "env_index": 1,
    }
    source_episode = {"env_seed": 1003201, "action_seed": 1003201, "sentinel": 9}
    assert module.run_episode_contract(source_episode, metadata) == {
        "env_seed": 2000001,
        "action_seed": 2000007,
        "sentinel": 9,
    }
    assert source_episode["env_seed"] == 1003201


def test_seed_search_renderer_fails_closed_without_source_capture_keys():
    module = _load("render_verified_trace_topdown")
    with pytest.raises(ValueError, match="source-capture"):
        module.source_bundle_identity(
            {
                "env_seed": 2,
                "action_seed": 2,
                "env_index": 0,
                "classification": {"seed_search": True},
            }
        )


def test_recorder_defaults_to_closed_loop_policy_mode(tmp_path):
    module = _load("record_verified_policy_trace")
    placeholder = tmp_path / "x"
    argv = [
        "--bundle", str(placeholder),
        "--expected-bundle-sha256", "a" * 64,
        "--companion-bundle", str(placeholder) + "2",
        "--expected-companion-bundle-sha256", "b" * 64,
        "--checkpoint", str(placeholder) + "3",
        "--expected-checkpoint-sha256", "c" * 64,
        "--prefix-checkpoint", str(placeholder) + "4",
        "--expected-prefix-checkpoint-sha256", "d" * 64,
        "--expected-env-seed", "1",
        "--expected-action-seed", "1",
        "--env-index", "1",
        "--code-root", str(tmp_path),
        "--code-archive", str(placeholder) + "5",
        "--expected-code-archive-sha256", "e" * 64,
        "--template", str(placeholder) + "6",
        "--expected-template-sha256", "f" * 64,
        "--trace-out", str(tmp_path / "trace.npz"),
    ]
    args = module.parse_args(argv)
    assert args.action_source == "closed-loop"
    assert args.race_both_envs is False
    assert args.advance_full_horizons == 0
    assert args.advance_selected_env_horizons == 0
    assert module._seed_mode(args) == {
        "search": False,
        "source_env_seed": 1,
        "source_action_seed": 1,
        "run_env_seed": 1,
        "run_action_seed": 1,
    }


def test_recorder_seed_search_requires_fresh_paired_seeds_and_closed_loop(tmp_path):
    module = _load("record_verified_policy_trace")
    placeholder = tmp_path / "x"
    argv = [
        "--bundle", str(placeholder),
        "--expected-bundle-sha256", "a" * 64,
        "--companion-bundle", str(placeholder) + "2",
        "--expected-companion-bundle-sha256", "b" * 64,
        "--checkpoint", str(placeholder) + "3",
        "--expected-checkpoint-sha256", "c" * 64,
        "--prefix-checkpoint", str(placeholder) + "4",
        "--expected-prefix-checkpoint-sha256", "d" * 64,
        "--expected-env-seed", "1003201",
        "--expected-action-seed", "1003201",
        "--env-index", "0",
        "--code-root", str(tmp_path),
        "--code-archive", str(placeholder) + "5",
        "--expected-code-archive-sha256", "e" * 64,
        "--template", str(placeholder) + "6",
        "--expected-template-sha256", "f" * 64,
        "--trace-out", str(tmp_path / "trace.npz"),
    ]
    with pytest.raises(SystemExit, match="must be supplied together"):
        module.parse_args(argv + ["--run-env-seed", "2000000"])
    with pytest.raises(SystemExit, match="fresh env/action seed pair"):
        module.parse_args(
            argv
            + [
                "--run-env-seed", "1003201",
                "--run-action-seed", "1003201",
            ]
        )
    with pytest.raises(SystemExit, match="uint32 range"):
        module.parse_args(
            argv
            + [
                "--run-env-seed", "-1",
                "--run-action-seed", "2000000",
            ]
        )
    with pytest.raises(SystemExit, match="requires --action-source closed-loop"):
        module.parse_args(
            argv
            + [
                "--run-env-seed", "2000000",
                "--run-action-seed", "2000000",
                "--action-source", "archived",
            ]
        )

    args = module.parse_args(
        argv
        + [
            "--run-env-seed", "2000000",
            "--run-action-seed", "2000007",
        ]
    )
    assert args.env_index == 0
    assert module._seed_mode(args) == {
        "search": True,
        "source_env_seed": 1003201,
        "source_action_seed": 1003201,
        "run_env_seed": 2000000,
        "run_action_seed": 2000007,
    }


def _race_arrays(score: int, *, final_done: bool = True):
    return {
        "score": np.asarray([0, score], np.int32),
        "done": np.asarray([False, final_done], bool),
    }


def test_recorder_dual_env_race_selects_one_complete_higher_scoring_trace():
    module = _load("record_verified_policy_trace")
    env0 = _race_arrays(202)
    env1 = _race_arrays(211)
    winner, arrays, terminal, outcomes = module._select_race_winner(
        {0: env0, 1: env1},
        {
            0: {"terminal_reason": "horizon", "scored": 202},
            1: {"terminal_reason": "horizon", "scored": 211},
        },
        {0: None, 1: None},
        preferred_env_index=0,
    )
    assert winner == 1
    assert arrays is env1
    assert int(arrays["score"][-1]) == terminal["scored"] == 211
    assert outcomes["0"]["eligible"] is True
    assert outcomes["1"]["eligible"] is True


def test_recorder_dual_env_race_rejects_unhealthy_or_partial_slot():
    module = _load("record_verified_policy_trace")
    env0 = _race_arrays(208)
    env1 = _race_arrays(202)
    winner, arrays, _, outcomes = module._select_race_winner(
        {0: env0, 1: env1},
        {
            0: {"terminal_reason": "horizon", "scored": 208},
            1: {"terminal_reason": "horizon", "scored": 202},
        },
        {0: 917, 1: None},
        preferred_env_index=0,
    )
    assert winner == 1
    assert arrays is env1
    assert outcomes["0"]["eligible"] is False
    assert "early_termination" in outcomes["0"]["rejection_reasons"]

    with pytest.raises(RuntimeError, match=r"no healthy 200\+ horizon"):
        module._select_race_winner(
            {0: _race_arrays(199), 1: _race_arrays(220, final_done=False)},
            {
                0: {"terminal_reason": "horizon", "scored": 199},
                1: {"terminal_reason": "horizon", "scored": 220},
            },
            {0: None, 1: None},
            preferred_env_index=1,
        )


def test_recorder_dual_env_race_is_opt_in_closed_loop_only(tmp_path):
    module = _load("record_verified_policy_trace")
    placeholder = tmp_path / "x"
    argv = [
        "--bundle", str(placeholder),
        "--expected-bundle-sha256", "a" * 64,
        "--companion-bundle", str(placeholder) + "2",
        "--expected-companion-bundle-sha256", "b" * 64,
        "--checkpoint", str(placeholder) + "3",
        "--expected-checkpoint-sha256", "c" * 64,
        "--prefix-checkpoint", str(placeholder) + "4",
        "--expected-prefix-checkpoint-sha256", "d" * 64,
        "--expected-env-seed", "1003201",
        "--expected-action-seed", "1003201",
        "--env-index", "1",
        "--code-root", str(tmp_path),
        "--code-archive", str(placeholder) + "5",
        "--expected-code-archive-sha256", "e" * 64,
        "--template", str(placeholder) + "6",
        "--expected-template-sha256", "f" * 64,
        "--trace-out", str(tmp_path / "trace.npz"),
        "--race-both-envs",
    ]
    args = module.parse_args(argv)
    assert args.race_both_envs is True
    assert args.env_index == 1
    assert args.advance_full_horizons == 0
    with pytest.raises(SystemExit, match="dual-environment race requires"):
        module.parse_args(argv + ["--action-source", "archived"])


def test_recorder_full_horizon_advance_is_bounded_race_only(tmp_path):
    module = _load("record_verified_policy_trace")
    placeholder = tmp_path / "x"
    base = [
        "--bundle", str(placeholder),
        "--expected-bundle-sha256", "a" * 64,
        "--companion-bundle", str(placeholder) + "2",
        "--expected-companion-bundle-sha256", "b" * 64,
        "--checkpoint", str(placeholder) + "3",
        "--expected-checkpoint-sha256", "c" * 64,
        "--prefix-checkpoint", str(placeholder) + "4",
        "--expected-prefix-checkpoint-sha256", "d" * 64,
        "--expected-env-seed", "1012101",
        "--expected-action-seed", "1012101",
        "--env-index", "1",
        "--code-root", str(tmp_path),
        "--code-archive", str(placeholder) + "5",
        "--expected-code-archive-sha256", "e" * 64,
        "--template", str(placeholder) + "6",
        "--expected-template-sha256", "f" * 64,
        "--trace-out", str(tmp_path / "trace.npz"),
    ]
    with pytest.raises(SystemExit, match="requires --race-both-envs"):
        module.parse_args(base + ["--advance-full-horizons", "1"])
    with pytest.raises(SystemExit):
        module.parse_args(
            base
            + [
                "--race-both-envs",
                "--advance-full-horizons", "2",
            ]
        )
    args = module.parse_args(
        base
        + [
            "--race-both-envs",
            "--advance-full-horizons", "1",
        ]
    )
    assert args.race_both_envs is True
    assert args.advance_full_horizons == 1
    assert args.advance_selected_env_horizons == 0
    search_args = module.parse_args(
        base
        + [
            "--race-both-envs",
            "--advance-full-horizons", "1",
            "--run-env-seed", "2020001",
            "--run-action-seed", "2020007",
        ]
    )
    assert search_args.advance_full_horizons == 1
    assert module._seed_mode(search_args)["search"] is True


def test_recorder_selected_env_advance_is_async_closed_loop_only(tmp_path):
    module = _load("record_verified_policy_trace")
    placeholder = tmp_path / "x"
    base = [
        "--bundle", str(placeholder),
        "--expected-bundle-sha256", "a" * 64,
        "--companion-bundle", str(placeholder) + "2",
        "--expected-companion-bundle-sha256", "b" * 64,
        "--checkpoint", str(placeholder) + "3",
        "--expected-checkpoint-sha256", "c" * 64,
        "--prefix-checkpoint", str(placeholder) + "4",
        "--expected-prefix-checkpoint-sha256", "d" * 64,
        "--expected-env-seed", "1003201",
        "--expected-action-seed", "1003201",
        "--env-index", "1",
        "--code-root", str(tmp_path),
        "--code-archive", str(placeholder) + "5",
        "--expected-code-archive-sha256", "e" * 64,
        "--template", str(placeholder) + "6",
        "--expected-template-sha256", "f" * 64,
        "--trace-out", str(tmp_path / "selected-trace.npz"),
        "--advance-selected-env-horizons", "1",
    ]
    args = module.parse_args(base)
    assert args.race_both_envs is False
    assert args.advance_full_horizons == 0
    assert args.advance_selected_env_horizons == 1

    search_args = module.parse_args(
        base
        + [
            "--run-env-seed", "1012101",
            "--run-action-seed", "1012101",
        ]
    )
    assert module._seed_mode(search_args)["search"] is True
    with pytest.raises(SystemExit, match="mutually exclusive with --race-both-envs"):
        module.parse_args(base + ["--race-both-envs"])
    with pytest.raises(SystemExit, match="selected-env advancement requires"):
        module.parse_args(base + ["--action-source", "archived"])


def test_recorder_warmup_requires_exact_synchronized_healthy_horizon():
    module = _load("record_verified_policy_trace")
    module.STEPS = 4
    stats = {
        0: {
            "terminal_reason": "horizon",
            "scored": 101,
            "collected": 120,
            "cycles_completed": 2,
        },
        1: {
            "terminal_reason": "horizon",
            "scored": 191,
            "collected": 200,
            "cycles_completed": 4,
        },
    }
    summary = module._warmup_horizon_summary(
        np.asarray([True, True]), stats, transition=4
    )
    assert summary["normal_terminal_auto_reset"] is True
    assert summary["outcomes"]["1"]["scored"] == 191

    with pytest.raises(RuntimeError, match="not synchronized"):
        module._warmup_horizon_summary(
            np.asarray([True, False]), stats, transition=4
        )
    with pytest.raises(RuntimeError, match="expected 4"):
        module._warmup_horizon_summary(
            np.asarray([True, True]), stats, transition=3
        )
    unhealthy = {**stats, 1: {**stats[1], "terminal_reason": "unhealthy"}}
    with pytest.raises(RuntimeError, match="not a healthy horizon"):
        module._warmup_horizon_summary(
            np.asarray([True, True]), unhealthy, transition=4
        )


def test_recorder_selected_warmup_ignores_other_env_health():
    module = _load("record_verified_policy_trace")
    module.STEPS = 4
    stats = {
        0: {"terminal_reason": "unhealthy", "scored": 0},
        1: {
            "terminal_reason": "horizon",
            "scored": 202,
            "collected": 210,
            "cycles_completed": 5,
        },
    }
    summary = module._selected_env_warmup_summary(
        np.asarray([True, True]),
        stats,
        transition=4,
        selected_env_index=1,
    )
    assert summary["selected_terminal"]["scored"] == 202
    assert summary["other_env_done_on_selected_boundary"] is True
    assert summary["other_env_terminal_on_selected_boundary"]["terminal_reason"] == (
        "unhealthy"
    )

    summary = module._selected_env_warmup_summary(
        np.asarray([False, True]),
        {1: stats[1]},
        transition=4,
        selected_env_index=1,
    )
    assert summary["other_env_done_on_selected_boundary"] is False
    with pytest.raises(RuntimeError, match="expected 4"):
        module._selected_env_warmup_summary(
            np.asarray([False, True]),
            {1: stats[1]},
            transition=3,
            selected_env_index=1,
        )
    with pytest.raises(RuntimeError, match="not a healthy horizon"):
        module._selected_env_warmup_summary(
            np.asarray([False, True]),
            {1: {**stats[1], "terminal_reason": "unhealthy"}},
            transition=4,
            selected_env_index=1,
        )


def test_recorder_reports_reset_camera_transient_without_polling_contract():
    module = _load("record_verified_policy_trace")
    rgb = np.zeros((2, 3, 4, 5, 3), np.uint8)
    rgb[0] = np.arange(rgb[0].size, dtype=np.uint8).reshape(rgb[0].shape)
    assert module._black_policy_camera_envs({"rgb": rgb}) == [1]

    rgb[1] = np.arange(rgb[1].size, dtype=np.uint8).reshape(rgb[1].shape)
    assert module._black_policy_camera_envs({"rgb": rgb}) == []
    with pytest.raises(RuntimeError, match="batch shape changed"):
        module._black_policy_camera_envs({"rgb": np.zeros((2, 4, 5), np.uint8)})


def test_recorder_advanced_capture_retains_real_terminal_boundary():
    module = _load("record_verified_policy_trace")
    module.STEPS = 4
    arrays_by_env = {}
    for env_index, final_score in ((0, 202), (1, 211)):
        arrays_by_env[env_index] = {
            "score": np.asarray([0, 0, 40, 100, final_score], np.int32),
            "done": np.asarray([False, False, False, False, True], bool),
            "action": np.full((5, 2, 7), env_index, np.float32),
        }
    normalized, omitted = module._trim_advanced_capture(
        arrays_by_env, transitions=5
    )
    assert omitted == 1
    assert normalized[0]["score"].tolist() == [0, 40, 100, 202]
    assert normalized[1]["done"].tolist() == [False, False, False, True]
    assert normalized[1]["action"].shape == (4, 2, 7)

    exact_length = {
        env_index: {key: value[:4].copy() for key, value in arrays.items()}
        for env_index, arrays in arrays_by_env.items()
    }
    # Runtime allocations remain STEPS+1 even when the generation terminates
    # at exactly STEPS; the unused tail must not enter the published trace.
    exact_allocations = {}
    for env_index, arrays in exact_length.items():
        exact_allocations[env_index] = {
            key: np.concatenate((value, value[-1:]), axis=0)
            for key, value in arrays.items()
        }
        exact_allocations[env_index]["done"][:] = [False, False, False, True, True]
    exact_normalized, exact_omitted = module._trim_advanced_capture(
        exact_allocations, transitions=4
    )
    assert exact_omitted == 0
    assert exact_normalized[0]["score"].shape == (4,)

    with pytest.raises(RuntimeError, match="1600 or 1601"):
        module._trim_advanced_capture(arrays_by_env, transitions=3)
    bad = {
        0: {
            "score": np.asarray([0, 1, 40, 100, 202], np.int32),
            "done": np.asarray([False, False, False, False, True], bool),
        }
    }
    with pytest.raises(RuntimeError, match="start at score zero"):
        module._trim_advanced_capture(bad, transitions=5)
