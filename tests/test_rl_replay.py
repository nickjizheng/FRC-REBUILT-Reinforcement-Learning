"""RL-specific unit tests: replay trajectory separation, n-step math,
controller episode reset, and policy frame packing (no Isaac dependency)."""
from __future__ import annotations

import numpy as np
import pytest

from xrc_rebuilt.rl.replay import PerEnvReplay, ReplayRing


OBS_SHAPE = (2, 4, 4)


def _ring(n_step=3, gamma=0.5, capacity=64, seed=0) -> ReplayRing:
    return ReplayRing(
        capacity=capacity,
        obs_shape=OBS_SHAPE,
        proprio_dim=3,
        privileged_dim=2,
        action_dim=2,
        n_step=n_step,
        gamma=gamma,
        seed=seed,
    )


def _fill(ring: ReplayRing, rewards, dones, tag_start=0) -> None:
    for k, (reward, done) in enumerate(zip(rewards, dones)):
        obs = np.full(OBS_SHAPE, tag_start + k, np.uint8)
        ring.add(obs, np.zeros(3), np.zeros(2), np.zeros(2), reward, done)


def test_nstep_reward_accumulates_with_gamma():
    ring = _ring(n_step=3, gamma=0.5)
    _fill(ring, rewards=[1, 2, 4, 8, 16, 0, 0, 0], dones=[False] * 8)
    # force anchor index 0 through the sampler
    ring.rng = type("R", (), {"integers": staticmethod(lambda low, high, size: np.zeros(size, int))})()
    batch = ring.sample(1)
    # 1 + 0.5*2 + 0.25*4 = 3.0; bootstrap discount = 0.5^3
    assert batch.reward[0] == pytest.approx(3.0)
    assert batch.discount[0] == pytest.approx(0.125)
    # next_obs is the state 3 steps ahead (tag 3)
    assert int(batch.next_obs[0].flat[0]) == 3


def test_nstep_truncates_at_done_and_zeroes_bootstrap():
    ring = _ring(n_step=3, gamma=0.5)
    _fill(ring, rewards=[1, 2, 100, 100, 0, 0, 0, 0], dones=[False, True, False, False] + [False] * 4)
    ring.rng = type("R", (), {"integers": staticmethod(lambda low, high, size: np.zeros(size, int))})()
    batch = ring.sample(1)
    # chain stops after the done at k=1: 1 + 0.5*2, no third reward
    assert batch.reward[0] == pytest.approx(2.0)
    assert batch.discount[0] == pytest.approx(0.0)  # no bootstrap across done


def test_per_env_replay_never_mixes_env_streams():
    replay = PerEnvReplay(
        num_envs=2,
        capacity_per_env=64,
        seed=3,
        obs_shape=OBS_SHAPE,
        proprio_dim=3,
        privileged_dim=2,
        action_dim=2,
        n_step=3,
        gamma=1.0,
    )
    # env 0 stores obs tagged 0..19 with reward 0; env 1 tagged 100..119, reward 100
    for k in range(20):
        replay.add(0, np.full(OBS_SHAPE, k, np.uint8), np.zeros(3), np.zeros(2), np.zeros(2), 0.0, False)
        replay.add(1, np.full(OBS_SHAPE, 100 + k, np.uint8), np.zeros(3), np.zeros(2), np.zeros(2), 100.0, False)
    batch = replay.sample(64)
    for obs, next_obs, reward in zip(batch.obs, batch.next_obs, batch.reward):
        tag, next_tag = int(obs.flat[0]), int(next_obs.flat[0])
        same_stream = (tag < 100) == (next_tag < 100)
        assert same_stream, "n-step chain crossed env streams"
        # 3-step reward is 0 (env0) or 300 (env1); interleaving would give mixes
        assert reward in (0.0, 300.0)
        assert next_tag - tag == 3  # exactly n steps ahead within the stream


def test_controller_reset_match_state_clears_episode_state():
    from xrc_rebuilt.competition_robot import CompetitionRobotController

    controller = CompetitionRobotController(alliance_lock="blue")
    controller.magazine = [1, 2, 3]
    controller.balls_collected = 7
    controller.shots_fired = 5
    controller._driver_field_velocity = np.array([1.0, 2.0], np.float32)
    controller._driver_omega = 0.5
    controller._muzzle_watch = {4}
    controller.state_machine.press_hold()
    controller.reset_match_state()
    assert controller.magazine == []
    assert controller.balls_collected == 0
    assert controller.shots_fired == 0
    assert float(np.linalg.norm(controller._driver_field_velocity)) == 0.0
    assert controller._driver_omega == 0.0
    assert controller._muzzle_watch == set()
    assert not controller.state_machine.wants_fire


def test_to_policy_frames_packs_cameras_into_channels():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_drqv2",
        Path(__file__).resolve().parents[1] / "scripts" / "rl" / "train_drqv2.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rgb = np.zeros((2, 3, 360, 640, 3), np.uint8)
    rgb[1, 2, :, :, 1] = 77  # env1, cam2, green channel
    frames = module.to_policy_frames(rgb)
    assert frames.shape == (2, 9, 90, 160)
    assert frames.dtype == np.uint8
    assert frames[1, 2 * 3 + 1].max() == 77  # cam2's G channel landed at index 7
    assert frames[0].max() == 0
