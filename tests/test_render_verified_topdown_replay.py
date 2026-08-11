from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "tools" / "render_verified_topdown_replay.py"
    spec = importlib.util.spec_from_file_location("render_verified_topdown_replay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path: Path, *, steps: int, env_index: int) -> tuple[Path, str]:
    module = _module()
    arrays = {
        "action": np.zeros((steps, 7), np.float32),
        "proprio": np.zeros((steps, 30), np.float32),
        "privileged": np.zeros((steps, 26), np.float32),
        "reward": np.zeros(steps, np.float32),
        "done": np.zeros(steps, bool),
    }
    arrays["done"][-1] = True
    episode = {
        "checkpoint_sha256": "a" * 64,
        "env_seed": 123,
        "action_seed": 123,
        "env_index": env_index,
        "episode_steps": steps,
        "num_envs": 2,
        "mode": "full",
        "reset_mode": "full",
        "action_mode": "deterministic",
        "terminal_reason": "horizon",
        "episode_len_s": 160.0,
        "stage_d_contract": {
            **module.STRICT_CONTRACT,
            "policy_speed_scale": 1.0,
            "prefix_rescue_s": 35.0,
        },
    }
    metadata = {
        "schema": module.CAPTURE_SCHEMA,
        "length": steps,
        "episode": episode,
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    path = tmp_path / f"env{env_index}_{steps}.npz"
    np.savez_compressed(path, **arrays)
    return path, module.sha256_file(path)


def test_loads_target_and_1601_step_companion_with_strict_custody(tmp_path):
    module = _module()
    target_path, target_sha = _bundle(tmp_path, steps=1600, env_index=1)
    companion_path, companion_sha = _bundle(tmp_path, steps=1601, env_index=0)
    kwargs = {
        "expected_checkpoint_sha256": "a" * 64,
        "expected_env_seed": 123,
        "expected_action_seed": 123,
    }
    target = module.load_replay_bundle(
        target_path,
        expected_bundle_sha256=target_sha,
        expected_env_index=1,
        **kwargs,
    )
    companion = module.load_replay_bundle(
        companion_path,
        expected_bundle_sha256=companion_sha,
        expected_env_index=0,
        expected_steps=1601,
        **kwargs,
    )
    assert target.action.shape == (1600, 7)
    assert companion.action[:1600].shape == (1600, 7)


def test_replay_bundle_fails_closed_on_checkpoint_or_contract_drift(tmp_path):
    module = _module()
    path, digest = _bundle(tmp_path, steps=1600, env_index=1)
    common = {
        "path": path,
        "expected_bundle_sha256": digest,
        "expected_env_seed": 123,
        "expected_action_seed": 123,
        "expected_env_index": 1,
    }
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        module.load_replay_bundle(
            expected_checkpoint_sha256="b" * 64,
            **common,
        )


def test_stage_d_prefix_view_pins_historical_clock_and_hub_contract():
    module = _module()

    class FakeStageD:
        calls = []

        @classmethod
        def pin_prefix_view(cls, proprio, *, episode_len_s, legacy_dim):
            cls.calls.append((float(episode_len_s), int(legacy_dim)))
            view = np.array(proprio[:, :legacy_dim], dtype=np.float32, copy=True)
            view[:, 7] = np.minimum(view[:, 7] * episode_len_s / 90.0, 1.0)
            view[:, 12] = 1.0
            return view

    proprio = np.zeros((2, 30), np.float32)
    proprio[:, 7] = (45.0 / 160.0, 120.0 / 160.0)
    proprio[:, 12] = 0.0
    view = module.stage_d_prefix_view(
        FakeStageD,
        proprio,
        episode_len_s=160.0,
        legacy_dim=22,
    )

    assert FakeStageD.calls == [(160.0, 22)]
    assert view.shape == (2, 22)
    assert view[:, 7].tolist() == pytest.approx([0.5, 1.0])
    assert view[:, 12].tolist() == [1.0, 1.0]


def test_archived_action_batch_preserves_live_companion_target_order(tmp_path):
    module = _module()
    target_path, target_sha = _bundle(tmp_path, steps=1600, env_index=1)
    companion_path, companion_sha = _bundle(tmp_path, steps=1601, env_index=0)
    kwargs = {
        "expected_checkpoint_sha256": "a" * 64,
        "expected_env_seed": 123,
        "expected_action_seed": 123,
    }
    target = module.load_replay_bundle(
        target_path,
        expected_bundle_sha256=target_sha,
        expected_env_index=1,
        **kwargs,
    )
    companion = module.load_replay_bundle(
        companion_path,
        expected_bundle_sha256=companion_sha,
        expected_env_index=0,
        expected_steps=1601,
        **kwargs,
    )
    target.action[7] = np.arange(7, dtype=np.float32) + 10
    companion.action[7] = np.arange(7, dtype=np.float32) + 20

    actions = module.archived_action_batch(target, companion, 7)

    assert actions.shape == (2, 7)
    assert actions.dtype == np.float32
    assert actions.flags.c_contiguous
    assert actions[0].tolist() == companion.action[7].tolist()
    assert actions[1].tolist() == target.action[7].tolist()
    with pytest.raises(IndexError, match="out of range"):
        module.archived_action_batch(target, companion, 1600)


def test_gui_video_sink_preserves_hd_canvas_and_live_camera_panels(tmp_path):
    cv2 = pytest.importorskip("cv2")
    module = _module()
    height, width = module.CAMERA_SIZE[1], module.CAMERA_SIZE[0]
    x = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.repeat(x[None, :, None], height, axis=0)
    frame = np.repeat(frame, 3, axis=2)
    panels = np.zeros((3, 3, 90, 160), np.uint8)
    panels[0, 0] = 255
    panels[1, 1] = 255
    panels[2, 2] = 255
    output = tmp_path / "gui.avi"
    sink = module.VideoSink(lambda: frame, output)

    class Cv2Spy:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.text = []

        def putText(self, *args, **kwargs):
            self.text.append(str(args[1]))
            return self.wrapped.putText(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    cv2_spy = Cv2Spy(sink.cv2)
    sink.cv2 = cv2_spy
    sink.write(
        {
            "remaining_s": 160.0,
            "score": 0,
            "collected": 0,
            "magazine": 8,
            "phase": "AUTO",
            "hub": "ACTIVE",
            "cycles": 0,
        },
        panels,
    )
    sink.close()

    rendered = cv2.imread(str(output.with_suffix(".first-frame.png")))
    assert rendered.shape == (1080, 1920, 3)
    assert rendered[20, 20].mean() > 20
    assert rendered[950, 760, 2] > rendered[950, 760, 1]
    assert rendered[950, 1125, 1] > rendered[950, 1125, 2]
    assert rendered[950, 1490, 0] > rendered[950, 1490, 1]
    assert "FUEL scored   RED 0   BLUE 0" in cv2_spy.text
    assert "FUEL total 456  |  field 448 + robot 8" in cv2_spy.text
    assert not any("FRC" in value for value in cv2_spy.text)
    assert {
        "Viewport Intake",
        "Viewport Shooter",
        "Viewport Navigation",
    }.issubset(cv2_spy.text)


def test_saved_gui_pose_and_launcher_are_pinned_to_current_renderer():
    module = _module()
    root = Path(__file__).resolve().parents[1]
    renderer = root / "tools" / "render_verified_topdown_replay.py"
    camera_state = root / "runs" / "gui_camera_pose.json"
    launcher = (root / "tools" / "launch_score202_gui_video.sh").read_text(
        encoding="utf-8"
    )

    loaded, camera_sha256 = module.load_gui_camera_state(camera_state)
    renderer_sha256 = module.sha256_file(renderer)

    assert module.CAMERA_SIZE == (1920, 1080)
    assert module.STEPS == 1600
    assert module.FPS == 10.0
    assert loaded["fuel_template_count"] == 456
    assert renderer_sha256 == (
        "5598e3c0016807201e3ebb422491e1ca603d052e28eafef74f16e87fbdf270cf"
    )
    assert camera_sha256 == (
        "407477d3aa05016e26b960de0a1faa0807721fbf3eb0d1d5c06b51b7402501b0"
    )
    assert "gui_saved_pose_tools_20260811" in launcher
    assert f"check_sha {renderer_sha256} \"$renderer\"" in launcher
    assert f"check_sha {camera_sha256} \"$camera_state\"" in launcher
    assert '--camera-state-json "$camera_state"' in launcher


def test_approved_html_and_live_renderer_share_the_same_layout_geometry():
    root = Path(__file__).resolve().parents[1]
    html = (root / "oblique_456_gui_layout_preview.html").read_text(
        encoding="utf-8"
    )
    renderer = (root / "tools" / "render_verified_topdown_replay.py").read_text(
        encoding="utf-8"
    )

    assert "width: 1920px; height: 1080px" in html
    assert "CAMERA_SIZE = (1920, 1080)" in renderer
    assert "left: 28px; top: 72px; width: 500px; height: 620px" in html
    assert "x0, y0, sw, sh = 28, 72, 500, 620" in renderer
    for panel_class, x, title in (
        ("intake", 700, "Viewport Intake"),
        ("shooter", 1045, "Viewport Shooter"),
        ("navigation", 1390, "Viewport Navigation"),
    ):
        assert f".camera-window.{panel_class} {{ left: {x}px; }}" in html
        assert f'x={x}, title="{title}"' in renderer
    assert 'f"FUEL scored   RED 0   BLUE {data[\'score\']}"' in renderer


def test_gui_camera_state_loads_exact_saved_pose_and_sha(tmp_path):
    module = _module()
    payload = {
        "schema": module.GUI_CAMERA_STATE_SCHEMA,
        "camera_path": "/OmniverseKit_Persp",
        "eye_xyz": [6.0, -13.0, 10.0],
        "target_xyz": [-1.0, -4.5, -0.4],
        "up_xyz": [-0.42, 0.52, 0.74],
        "focal_length_mm": 15.8,
        "horizontal_aperture_mm": 20.955,
        "vertical_aperture_mm": 15.2908,
        "horizontal_aperture_offset_mm": 0.0,
        "vertical_aperture_offset_mm": 0.0,
        "exposure": 0.0,
        "clipping_range": [0.01, 10_000_000.0],
        "projection": "perspective",
        "viewport_resolution": [1600, 900],
        "fuel_template_count": 456,
    }
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, digest = module.load_gui_camera_state(path)

    assert loaded == payload
    assert digest == module.sha256_file(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fuel_template_count", 455, "456-FUEL"),
        ("projection", "orthographic", "perspective"),
        ("viewport_resolution", [1600, 1000], "aspect ratio"),
    ],
)
def test_gui_camera_state_fails_closed_on_contract_drift(
    tmp_path, field, value, message
):
    module = _module()
    payload = {
        "schema": module.GUI_CAMERA_STATE_SCHEMA,
        "eye_xyz": [6.0, -13.0, 10.0],
        "target_xyz": [-1.0, -4.5, -0.4],
        "up_xyz": [-0.42, 0.52, 0.74],
        "focal_length_mm": 15.8,
        "horizontal_aperture_mm": 20.955,
        "vertical_aperture_mm": 15.2908,
        "exposure": 0.0,
        "clipping_range": [0.01, 10_000_000.0],
        "projection": "perspective",
        "viewport_resolution": [1600, 900],
        "fuel_template_count": 456,
    }
    payload[field] = value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.load_gui_camera_state(path)
