"""Regression tests for the training-stability guards (audit hardening).

Pure-torch, CPU-only, no Isaac - runs in the standard `pytest` suite. Each test
pins one guard that a production run relied on and previously lacked.
"""
from __future__ import annotations

import types
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent


def _agent() -> DrQV2Agent:
    # small CPU config: decoupled from production frame size, fast to update
    return DrQV2Agent(
        DrQConfig(frame_channels=9, frame_h=48, frame_w=48, device="cpu")
    )


def _batch(agent: DrQV2Agent, b: int = 8, reward_scale: float = 1.0):
    cfg = agent.cfg
    rng = np.random.default_rng(0)
    shape = (b, cfg.frame_channels, cfg.frame_h, cfg.frame_w)
    return types.SimpleNamespace(
        obs=rng.integers(0, 255, shape, dtype=np.uint8),
        next_obs=rng.integers(0, 255, shape, dtype=np.uint8),
        proprio=rng.standard_normal((b, cfg.proprio_dim)).astype(np.float32),
        next_proprio=rng.standard_normal((b, cfg.proprio_dim)).astype(np.float32),
        privileged=rng.standard_normal((b, cfg.privileged_dim)).astype(np.float32),
        next_privileged=rng.standard_normal((b, cfg.privileged_dim)).astype(np.float32),
        action=rng.uniform(-1, 1, (b, cfg.action_dim)).astype(np.float32),
        reward=(rng.standard_normal(b) * reward_scale).astype(np.float32),
        discount=np.full(b, 0.997, np.float32),
    )


def test_act_sanitizes_nonfinite_proprio():
    """A NaN in the proprio input must not escape as a NaN action into PhysX."""
    agent = _agent()
    frames = np.random.randint(0, 255, (2, 9, 48, 48), dtype=np.uint8)
    proprio = np.zeros((2, agent.cfg.proprio_dim), np.float32)
    proprio[0, 0] = np.nan
    out = agent.act(frames, proprio, explore=True)
    assert np.isfinite(out).all(), "act() emitted a non-finite action"
    assert (np.abs(out) <= 1.0 + 1e-6).all(), "act() action out of [-1, 1]"


def test_act_sanitizes_poisoned_network():
    """Even a fully NaN-poisoned actor must yield a finite, clipped action."""
    agent = _agent()
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.copy_(torch.full_like(p, float("nan")))
    frames = np.random.randint(0, 255, (3, 9, 48, 48), dtype=np.uint8)
    proprio = np.zeros((3, agent.cfg.proprio_dim), np.float32)
    out = agent.act(frames, proprio, explore=False)
    assert np.isfinite(out).all()
    assert (np.abs(out) <= 1.0 + 1e-6).all()


def test_grad_clip_guard_blocks_nonfinite_norm():
    """A non-finite grad norm must skip the optimizer step, not write NaN weights.

    Mocks clip_grad_norm_ to report inf (what happens when a gradient is inf):
    the pre-fix code would compute clip_coef = max/inf = 0, then inf*0 = NaN and
    step NaN into the weights. The guard must instead skip and keep weights finite.
    """
    agent = _agent()
    batch = _batch(agent)
    before = [p.detach().clone() for p in agent.critic.parameters()]
    with patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("inf"))):
        metrics = agent.update(batch)
    assert agent.weights_finite(), "non-finite grad norm poisoned the weights"
    assert metrics.get("skipped", 0.0) >= 1.0, "guard did not record a skip"
    # critic weights must be unchanged (step was skipped)
    for a, b in zip(before, agent.critic.parameters()):
        assert torch.equal(a, b), "critic stepped despite non-finite grad norm"


def test_update_keeps_weights_finite_under_reward_spikes():
    """Many updates with large-but-finite reward spikes must not diverge to NaN."""
    agent = _agent()
    for _ in range(30):
        agent.update(_batch(agent, reward_scale=50.0))
    assert agent.weights_finite()


def test_explore_restart_rewarms_schedule():
    """Setting explore_offset = train_steps must return stddev to stddev_start."""
    agent = _agent()
    agent.train_steps = 500_000  # far past stddev_steps -> pinned at floor
    assert agent.stddev() == pytest.approx(agent.cfg.stddev_end)
    agent.explore_offset = agent.train_steps
    assert agent.stddev() == pytest.approx(agent.cfg.stddev_start)


def test_skipped_updates_do_not_advance_train_steps_on_bad_loss():
    """A non-finite critic loss must skip the whole update (no train_steps bump)."""
    agent = _agent()
    batch = _batch(agent)
    batch.reward = np.full_like(batch.reward, np.inf)  # -> target_q inf -> loss inf
    steps_before = agent.train_steps
    metrics = agent.update(batch)
    assert agent.weights_finite()
    assert agent.train_steps == steps_before
    assert metrics.get("skipped", 0.0) >= 1.0


def test_weights_finite_flags_poisoned_target_critic():
    """A NaN in the EMA target critic must fail the gate (it is saved/published)."""
    agent = _agent()
    assert agent.weights_finite()
    with torch.no_grad():
        next(agent.critic_target.parameters()).data.view(-1)[0] = float("nan")
    assert not agent.weights_finite(), "poisoned target critic passed the finite gate"


def test_weights_finite_flags_poisoned_optimizer_state():
    """A NaN in an Adam moment buffer must fail the gate (optimizer state is saved)."""
    agent = _agent()
    agent.update(_batch(agent))  # populate Adam state (exp_avg/exp_avg_sq)
    assert agent.weights_finite()
    state = next(iter(agent.critic_opt.state.values()))
    state["exp_avg"].fill_(float("inf"))
    assert not agent.weights_finite(), "poisoned Adam moment passed the finite gate"
