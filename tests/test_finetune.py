"""CPU tests for the reward-first fine-tune core.

Proves the two safety-critical properties: (1) the critic-only phase freezes actor AND
encoder so the deterministic policy is bit-identical (drive-L2 ≡ 0) while the critic
re-fits; (2) phase 2 moves the encoder+actor and the champion BC anchor pulls the actor
toward champion actions. Plus the beta schedule and NaN safety. No Isaac.
Run: C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_finetune.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent          # noqa: E402
from frc_rebuilt.rl.prefix_takeover import finetune_beta         # noqa: E402


def _agent(seed=0):
    torch.manual_seed(seed)
    return DrQV2Agent(DrQConfig(device="cpu"))


def _batch(cfg, B=4, seed=0, bad=False):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(B).astype(np.float32)
    if bad:
        r[0] = np.nan
    return SimpleNamespace(
        obs=rng.integers(0, 256, (B, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32),
        next_obs=rng.integers(0, 256, (B, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32),
        proprio=rng.standard_normal((B, cfg.proprio_dim)).astype(np.float32),
        next_proprio=rng.standard_normal((B, cfg.proprio_dim)).astype(np.float32),
        privileged=rng.standard_normal((B, cfg.privileged_dim)).astype(np.float32),
        next_privileged=rng.standard_normal((B, cfg.privileged_dim)).astype(np.float32),
        action=np.clip(rng.standard_normal((B, cfg.action_dim)), -1, 1).astype(np.float32),
        reward=r, discount=np.full(B, 0.99, np.float32),
    )


def _anchor(cfg, B=6, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (B, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32),
            rng.standard_normal((B, cfg.proprio_dim)).astype(np.float32),
            np.clip(rng.standard_normal((B, cfg.action_dim)), -1, 1).astype(np.float32))


def _snap(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.allclose(b, a) for b, a in zip(before, module.parameters()))


def test_beta_schedule():
    # 0 during critic-only, 0.3 at unlock, linear to 0 at 23k, 0 after
    assert finetune_beta(0, 3000) == 0.0
    assert finetune_beta(2999, 3000) == 0.0
    assert abs(finetune_beta(3000, 3000) - 0.3) < 1e-9
    assert abs(finetune_beta(13000, 3000) - 0.15) < 1e-6      # halfway of 3k..23k
    assert finetune_beta(23000, 3000) == 0.0
    assert finetune_beta(30000, 3000) == 0.0


def test_critic_only_freezes_policy_bit_identical():
    ag = _agent(); cfg = ag.cfg
    a = _anchor(cfg)
    enc0, act0, crit0 = _snap(ag.encoder), _snap(ag.actor), _snap(ag.critic)
    # a fixed probe: the deterministic action must be UNCHANGED after critic-only updates
    fr = np.random.default_rng(9).integers(0, 256, (3, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32)
    pr = np.random.default_rng(9).standard_normal((3, cfg.proprio_dim)).astype(np.float32)
    act_before = ag.act(fr, pr, explore=False)
    for k in range(4):
        ag.update_finetune(_batch(cfg, seed=k), *a, beta=0.0, critic_only=True)
    assert not _changed(enc0, ag.encoder), "critic-only moved the ENCODER (policy would drift)"
    assert not _changed(act0, ag.actor), "critic-only moved the actor"
    assert _changed(crit0, ag.critic), "critic-only did not re-fit the critic"
    act_after = ag.act(fr, pr, explore=False)
    # drive-L2 p50 ≡ 0 (the design note phase-1 identity assertion)
    assert float(np.abs(act_after - act_before).max()) == 0.0


def test_phase2_moves_encoder_and_actor():
    ag = _agent(seed=2); cfg = ag.cfg
    a = _anchor(cfg)
    enc0, act0 = _snap(ag.encoder), _snap(ag.actor)
    for k in range(3):
        ag.update_finetune(_batch(cfg, seed=k), *a, beta=0.3, critic_only=False)
    assert _changed(enc0, ag.encoder), "phase 2 left the encoder frozen (should be trainable)"
    assert _changed(act0, ag.actor), "phase 2 left the actor frozen"


def test_anchor_pulls_actor_toward_champion_actions():
    ag = _agent(seed=5); cfg = ag.cfg
    a_obs, a_pro, a_act = _anchor(cfg, B=8)
    first = ag.update_finetune(_batch(cfg, seed=0), a_obs, a_pro, a_act, beta=5.0, critic_only=False)
    for k in range(60):
        last = ag.update_finetune(_batch(cfg, seed=k), a_obs, a_pro, a_act, beta=5.0, critic_only=False)
    assert last["bc_anchor"] < 0.6 * first["bc_anchor"], \
        f"anchor did not pull actor: {first['bc_anchor']:.4f} -> {last['bc_anchor']:.4f}"


def test_nan_batch_skipped_and_weights_finite():
    ag = _agent(seed=3); cfg = ag.cfg
    a = _anchor(cfg)
    before = ag.skipped_updates
    out = ag.update_finetune(_batch(cfg, seed=0, bad=True), *a, beta=0.3, critic_only=False)
    assert np.isnan(out["critic_loss"]) and out["skipped"] == before + 1
    assert ag.weights_finite()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
