"""Pure-CPU regression tests for DrQV2Agent.update_suffix (prefix-takeover E core).

No Isaac, no GPU: builds a tiny agent and synthetic batches to prove the three
spec-mandated invariants of the suffix update plus its NaN safety.
Run: C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_update_suffix.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import frc_rebuilt.rl.drqv2 as drqv2_module  # noqa: E402
from frc_rebuilt.rl.drqv2 import (  # noqa: E402
    DrQConfig,
    DrQV2Agent,
    _stagec_v2_executed_action_policy_torch,
)


def _agent(seed=0):
    torch.manual_seed(seed)
    return DrQV2Agent(DrQConfig(device="cpu"))


def _batch(cfg, B=4, seed=0, bad=False):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 256, (B, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32)
    nxt = rng.integers(0, 256, (B, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32)
    reward = rng.standard_normal(B).astype(np.float32)
    if bad:
        reward[0] = np.nan
    return SimpleNamespace(
        obs=obs, next_obs=nxt,
        proprio=rng.standard_normal((B, cfg.proprio_dim)).astype(np.float32),
        next_proprio=rng.standard_normal((B, cfg.proprio_dim)).astype(np.float32),
        privileged=rng.standard_normal((B, cfg.privileged_dim)).astype(np.float32),
        next_privileged=rng.standard_normal((B, cfg.privileged_dim)).astype(np.float32),
        action=np.clip(rng.standard_normal((B, cfg.action_dim)), -1, 1).astype(np.float32),
        reward=reward,
        discount=np.full(B, 0.99, np.float32),
    )


def _anchor(cfg, B=4, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (B, cfg.frame_channels, cfg.frame_h, cfg.frame_w), np.uint8).astype(np.float32),
            rng.standard_normal((B, cfg.proprio_dim)).astype(np.float32),
            np.clip(rng.standard_normal((B, cfg.action_dim)), -1, 1).astype(np.float32))


def _enc_snapshot(agent):
    return [p.detach().clone() for p in agent.encoder.parameters()]


def _module_snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def test_frozen_encoder_never_moves():
    ag = _agent(); cfg = ag.cfg
    a_obs, a_pro, a_act = _anchor(cfg)
    before = _enc_snapshot(ag)
    for k in range(3):
        ag.update_suffix(_batch(cfg, seed=k), a_obs, a_pro, a_act, alpha=1.0, freeze_encoder=True)
    after = _enc_snapshot(ag)
    assert all(torch.equal(b, a) for b, a in zip(before, after)), "frozen encoder moved"


def test_unfrozen_encoder_does_move():
    ag = _agent(); cfg = ag.cfg
    a_obs, a_pro, a_act = _anchor(cfg)
    before = _enc_snapshot(ag)
    for k in range(3):
        ag.update_suffix(_batch(cfg, seed=k), a_obs, a_pro, a_act, alpha=1.0, freeze_encoder=False)
    after = _enc_snapshot(ag)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), "unfrozen encoder did not move"


def test_td3bc_lambda_scales_with_alpha():
    # identical agents + identical batch: the critic step is alpha-independent, so
    # lambda_q = alpha / mean(|Q|) must scale linearly with alpha.
    ag1 = _agent(seed=3)
    ag2 = _agent(seed=3)
    ag2.encoder.load_state_dict(ag1.encoder.state_dict())
    ag2.actor.load_state_dict(ag1.actor.state_dict())
    ag2.critic.load_state_dict(ag1.critic.state_dict())
    ag2.critic_target.load_state_dict(ag1.critic_target.state_dict())
    cfg = ag1.cfg
    a = _anchor(cfg)
    b = _batch(cfg, seed=7)
    # reset the global RNG identically before each call so the augmentation / target
    # noise (and hence the critic step and q_pi) are bit-identical; only alpha differs.
    torch.manual_seed(999); r1 = ag1.update_suffix(copy.deepcopy(b), *a, alpha=1.0, freeze_encoder=True)
    torch.manual_seed(999); r2 = ag2.update_suffix(copy.deepcopy(b), *a, alpha=2.5, freeze_encoder=True)
    assert r1["lambda_q"] > 0 and np.isfinite(r1["lambda_q"])
    ratio = r2["lambda_q"] / r1["lambda_q"]
    assert abs(ratio - 2.5) < 1e-3, f"lambda_q ratio {ratio} != 2.5"


def test_actor_q_center_default_is_bit_identical_to_explicit_zero():
    default = _agent(seed=81)
    explicit = _agent(seed=81)
    cfg = default.cfg
    batch = _batch(cfg, seed=82)
    anchor = _anchor(cfg, seed=83)

    torch.manual_seed(8484)
    default_metrics = default.update_suffix(
        copy.deepcopy(batch), *anchor, alpha=1.0
    )
    torch.manual_seed(8484)
    explicit_metrics = explicit.update_suffix(
        copy.deepcopy(batch),
        *anchor,
        alpha=1.0,
        actor_q_center_fraction=0.0,
    )

    assert default_metrics == explicit_metrics
    for default_module, explicit_module in (
        (default.encoder, explicit.encoder),
        (default.actor, explicit.actor),
        (default.critic, explicit.critic),
        (default.critic_target, explicit.critic_target),
    ):
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                default_module.parameters(), explicit_module.parameters()
            )
        )


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan"), float("inf")])
def test_actor_q_center_fraction_rejects_invalid_values(fraction):
    agent = _agent(seed=85)
    with pytest.raises(ValueError, match="actor_q_center_fraction"):
        agent.update_suffix(
            _batch(agent.cfg, seed=86),
            *_anchor(agent.cfg, seed=87),
            alpha=1.0,
            actor_q_center_fraction=fraction,
        )


def test_actor_q_center_fraction_blends_noisy_and_center_q():
    noisy = _agent(seed=88)
    center = _agent(seed=88)
    blended = _agent(seed=88)
    cfg = noisy.cfg
    batch = _batch(cfg, seed=89)
    anchor = _anchor(cfg, seed=90)

    torch.manual_seed(9191)
    noisy_metrics = noisy.update_suffix(
        copy.deepcopy(batch), *anchor, alpha=1.0, actor_q_center_fraction=0.0
    )
    torch.manual_seed(9191)
    center_metrics = center.update_suffix(
        copy.deepcopy(batch), *anchor, alpha=1.0, actor_q_center_fraction=1.0
    )
    torch.manual_seed(9191)
    blended_metrics = blended.update_suffix(
        copy.deepcopy(batch), *anchor, alpha=1.0, actor_q_center_fraction=0.5
    )

    assert noisy_metrics["q_pi"] == pytest.approx(noisy_metrics["q_pi_noisy"])
    assert center_metrics["q_pi"] == pytest.approx(center_metrics["q_pi_center"])
    assert blended_metrics["q_pi_noisy"] == pytest.approx(
        noisy_metrics["q_pi_noisy"]
    )
    assert blended_metrics["q_pi_center"] == pytest.approx(
        center_metrics["q_pi_center"]
    )
    assert blended_metrics["q_pi"] == pytest.approx(
        0.5
        * (
            blended_metrics["q_pi_noisy"]
            + blended_metrics["q_pi_center"]
        )
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(noisy.actor.parameters(), center.actor.parameters())
    )


def test_bc_anchor_pulls_actor_toward_champion_actions():
    # anchor-dominant (tiny alpha): pi(anchor_obs) should converge toward the
    # champion anchor actions -> the BC MSE drops.
    ag = _agent(seed=5); cfg = ag.cfg
    a_obs, a_pro, a_act = _anchor(cfg, B=6)
    first = ag.update_suffix(_batch(cfg, seed=0), a_obs, a_pro, a_act, alpha=1e-3, freeze_encoder=True)
    for k in range(60):
        last = ag.update_suffix(_batch(cfg, seed=k), a_obs, a_pro, a_act, alpha=1e-3, freeze_encoder=True)
    assert last["bc_anchor"] < 0.6 * first["bc_anchor"], \
        f"anchor did not pull actor: {first['bc_anchor']:.4f} -> {last['bc_anchor']:.4f}"


def test_nan_batch_is_skipped_and_weights_stay_finite():
    ag = _agent(seed=2); cfg = ag.cfg
    a_obs, a_pro, a_act = _anchor(cfg)
    skipped_before = ag.skipped_updates
    out = ag.update_suffix(_batch(cfg, seed=0, bad=True), a_obs, a_pro, a_act, alpha=1.0)
    assert np.isnan(out["critic_loss"]) and out["skipped"] == skipped_before + 1
    assert ag.weights_finite(), "a NaN suffix batch poisoned the weights"


def test_actor_mask_keeps_first_rows_out_of_suffix_critic():
    # Changing excluded FIRST rows cannot change a suffix-only critic step.
    left = _agent(seed=11)
    right = _agent(seed=11)
    cfg = left.cfg
    batch = _batch(cfg, B=4, seed=12)
    changed = copy.deepcopy(batch)
    changed.reward[[0, 2]] += 10_000.0
    anchor = _anchor(cfg, seed=13)
    mask = np.array([0, 1, 0, 1])

    torch.manual_seed(1414)
    out_left = left.update_suffix(
        copy.deepcopy(batch),
        *anchor,
        alpha=1.0,
        actor_mask=mask,
        critic_mask=mask,
    )
    torch.manual_seed(1414)
    out_right = right.update_suffix(
        changed,
        *anchor,
        alpha=1.0,
        actor_mask=mask,
        critic_mask=mask,
    )

    assert out_left["actor_rows"] == out_right["actor_rows"] == 2
    for left_param, right_param in zip(left.critic.parameters(), right.critic.parameters()):
        assert torch.equal(left_param, right_param), "excluded FIRST row changed suffix critic"


def test_stagec_torch_action_policy_matches_execution_contract_and_gradients():
    proprio = torch.zeros(5, 30)
    proprio[torch.arange(5), 22 + torch.arange(5)] = 1.0
    proposed = torch.tensor([[0.1, 0.2, 0.3, 0.4, -0.4, 0.6, 0.8]] * 5, requires_grad=True)

    executed = _stagec_v2_executed_action_policy_torch(proposed, proprio)

    torch.testing.assert_close(executed[:, :3], proposed[:, :3])
    torch.testing.assert_close(executed[1:, 6], torch.full((4,), -1.0))
    torch.testing.assert_close(executed[1:, 4], torch.ones(4))
    torch.testing.assert_close(executed[1:, 3], torch.tensor([-1.0, 1.0, -1.0, -1.0]))
    torch.testing.assert_close(executed[1:4, 5], torch.full((3,), -1.0))
    assert executed[4, 5] == proposed[4, 5]

    executed.sum().backward()
    # Fixed post-FIRST heads receive no actor gradient; SCORE shoot stays trainable.
    assert torch.equal(proposed.grad[1:, 4], torch.zeros(4))
    assert torch.equal(proposed.grad[1:, 6], torch.zeros(4))
    assert torch.equal(proposed.grad[1:4, 5], torch.zeros(3))
    assert proposed.grad[4, 5] == 1.0


def test_stagec_torch_return_intake_is_opt_in_and_matches_numpy_contract():
    proprio = torch.zeros(5, 30)
    proprio[torch.arange(5), 22 + torch.arange(5)] = 1.0
    proposed = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, -0.4, 0.6, 0.8]] * 5,
        requires_grad=True,
    )

    legacy = _stagec_v2_executed_action_policy_torch(proposed, proprio)
    enabled = _stagec_v2_executed_action_policy_torch(
        proposed,
        proprio,
        intake_during_return=True,
    )

    expected = legacy.detach().clone()
    expected[3, 3] = 1.0
    torch.testing.assert_close(enabled, expected)
    assert legacy[3, 3] == -1.0
    assert enabled[3, 3] == 1.0

    enabled.sum().backward()
    assert proposed.grad[3, 3] == 0.0


def test_update_suffix_threads_return_intake_through_every_action_query(monkeypatch):
    agent = DrQV2Agent(DrQConfig(device="cpu", proprio_dim=30))
    cfg = agent.cfg
    batch = _batch(cfg, B=4, seed=51)
    elite = _batch(cfg, B=3, seed=52)
    anchor = _anchor(cfg, B=4, seed=53)
    for proprio in (
        batch.proprio,
        batch.next_proprio,
        elite.proprio,
        elite.next_proprio,
    ):
        proprio[:, 22:27] = 0.0
        proprio[:, 25] = 1.0  # RETURN

    seen: list[tuple[bool, bool]] = []
    original = drqv2_module._stagec_v2_executed_action_policy_torch

    def recording_policy(
        actions,
        proprio,
        *,
        intake_during_return=False,
        stage_d_ferry=False,
    ):
        seen.append((bool(intake_during_return), bool(stage_d_ferry)))
        return original(
            actions,
            proprio,
            intake_during_return=intake_during_return,
            stage_d_ferry=stage_d_ferry,
        )

    monkeypatch.setattr(
        drqv2_module,
        "_stagec_v2_executed_action_policy_torch",
        recording_policy,
    )
    agent.update_suffix(
        batch,
        *anchor,
        alpha=1.0,
        elite_behavior_batch=elite,
        elite_behavior_weight=1.0,
        intake_during_return=True,
        stage_d_ferry=True,
    )

    # Bellman target, actor-Q sample, and elite-BC proposal all use the same
    # V10 legal-action manifold. The default objective remains exactly three
    # calls; the opt-in blend adds only the deterministic-center actor-Q query.
    assert seen == [(True, True)] * 3

    seen.clear()
    center_agent = DrQV2Agent(DrQConfig(device="cpu", proprio_dim=30))
    center_agent.update_suffix(
        copy.deepcopy(batch),
        *anchor,
        alpha=1.0,
        elite_behavior_batch=copy.deepcopy(elite),
        elite_behavior_weight=1.0,
        intake_during_return=True,
        stage_d_ferry=True,
        actor_q_center_fraction=0.5,
    )
    assert seen == [(True, True)] * 4


def test_elite_behavior_anchor_is_reported_and_changes_actor_step():
    left = DrQV2Agent(DrQConfig(device="cpu", proprio_dim=30))
    right = DrQV2Agent(DrQConfig(device="cpu", proprio_dim=30))
    right.encoder.load_state_dict(left.encoder.state_dict())
    right.actor.load_state_dict(left.actor.state_dict())
    right.critic.load_state_dict(left.critic.state_dict())
    right.critic_target.load_state_dict(left.critic_target.state_dict())
    cfg = left.cfg
    batch = _batch(cfg, B=4, seed=61)
    elite = _batch(cfg, B=3, seed=62)
    anchor = _anchor(cfg, B=4, seed=63)
    for proprio in (batch.proprio, batch.next_proprio, elite.proprio, elite.next_proprio):
        proprio[:, 22:27] = 0.0
        proprio[:, 26] = 1.0  # SCORE

    torch.manual_seed(6464)
    plain = left.update_suffix(copy.deepcopy(batch), *anchor, alpha=1.0)
    torch.manual_seed(6464)
    anchored = right.update_suffix(
        copy.deepcopy(batch),
        *anchor,
        alpha=1.0,
        elite_behavior_batch=elite,
        elite_behavior_weight=1.0,
    )

    assert plain["elite_behavior_rows"] == 0
    assert anchored["elite_behavior_rows"] == 3
    assert anchored["elite_behavior_bc"] > 0.0
    assert any(
        not torch.equal(a, b)
        for a, b in zip(left.actor.parameters(), right.actor.parameters())
    )


def test_anchor_weight_scales_the_suffix_champion_anchor():
    left = _agent(seed=71)
    right = _agent(seed=71)
    cfg = left.cfg
    batch = _batch(cfg, B=4, seed=72)
    anchor = _anchor(cfg, B=4, seed=73)

    torch.manual_seed(7474)
    unanchored = left.update_suffix(
        copy.deepcopy(batch), *anchor, alpha=1.0, anchor_weight=0.0
    )
    torch.manual_seed(7474)
    anchored = right.update_suffix(
        copy.deepcopy(batch), *anchor, alpha=1.0, anchor_weight=0.3
    )

    assert unanchored["anchor_weight"] == 0.0
    assert anchored["anchor_weight"] == 0.3
    assert any(
        not torch.equal(a, b)
        for a, b in zip(left.actor.parameters(), right.actor.parameters())
    )
    with pytest.raises(ValueError, match="anchor_weight"):
        right.update_suffix(
            copy.deepcopy(batch), *anchor, alpha=1.0, anchor_weight=-0.1
        )


def test_empty_actor_mask_is_intentional_critic_only_step():
    ag = _agent(seed=21)
    cfg = ag.cfg
    actor_before = _module_snapshot(ag.actor)
    encoder_before = _enc_snapshot(ag)
    critic_before = _module_snapshot(ag.critic)
    skipped_before = ag.skipped_updates
    steps_before = ag.train_steps

    out = ag.update_suffix(
        _batch(cfg, B=4, seed=22),
        *_anchor(cfg, seed=23),
        alpha=1.0,
        actor_mask=np.zeros(4, dtype=bool),
    )

    assert out["actor_rows"] == 0
    assert "actor_loss" not in out
    assert ag.skipped_updates == skipped_before
    assert ag.train_steps == steps_before + 1
    assert all(torch.equal(b, a) for b, a in zip(actor_before, ag.actor.parameters()))
    assert all(torch.equal(b, a) for b, a in zip(encoder_before, ag.encoder.parameters()))
    assert any(not torch.equal(b, a) for b, a in zip(critic_before, ag.critic.parameters()))


def test_actor_update_false_fits_only_critic():
    ag = _agent(seed=24)
    cfg = ag.cfg
    actor_before = _module_snapshot(ag.actor)
    critic_before = _module_snapshot(ag.critic)

    out = ag.update_suffix(
        _batch(cfg, B=4, seed=25),
        *_anchor(cfg, seed=26),
        alpha=1.0,
        actor_update=False,
    )

    assert out["actor_rows"] == 0
    assert out["critic_rows"] == 4
    assert "actor_loss" not in out
    assert all(torch.equal(b, a) for b, a in zip(actor_before, ag.actor.parameters()))
    assert any(not torch.equal(b, a) for b, a in zip(critic_before, ag.critic.parameters()))


def test_explicit_empty_critic_mask_is_safe_noop():
    ag = _agent(seed=27)
    cfg = ag.cfg
    actor_before = _module_snapshot(ag.actor)
    critic_before = _module_snapshot(ag.critic)
    steps_before = ag.train_steps

    out = ag.update_suffix(
        _batch(cfg, B=4, seed=28),
        *_anchor(cfg, seed=29),
        alpha=1.0,
        critic_mask=np.zeros(4, dtype=bool),
    )

    assert out["critic_rows"] == 0
    assert out["no_critic_rows"] == 1
    assert ag.train_steps == steps_before
    assert all(torch.equal(b, a) for b, a in zip(actor_before, ag.actor.parameters()))
    assert all(torch.equal(b, a) for b, a in zip(critic_before, ag.critic.parameters()))


def test_reset_optimizers_preserves_networks_and_discards_adam_momentum():
    ag = _agent(seed=30)
    cfg = ag.cfg
    ag.update_suffix(
        _batch(cfg, B=4, seed=31),
        *_anchor(cfg, seed=32),
        alpha=1.0,
    )
    modules_before = {
        "encoder": _module_snapshot(ag.encoder),
        "actor": _module_snapshot(ag.actor),
        "critic": _module_snapshot(ag.critic),
    }
    assert ag.actor_opt.state and ag.critic_opt.state

    ag.reset_optimizers(3e-7)

    assert not ag.encoder_opt.state
    assert not ag.actor_opt.state
    assert not ag.critic_opt.state
    assert ag.cfg.lr == pytest.approx(3e-7)
    for optimizer in (ag.encoder_opt, ag.actor_opt, ag.critic_opt):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-7)
    for name, module in (
        ("encoder", ag.encoder),
        ("actor", ag.actor),
        ("critic", ag.critic),
    ):
        assert all(
            torch.equal(before, after)
            for before, after in zip(modules_before[name], module.parameters())
        )


def test_v22_critic_warmup_and_suffix_keep_encoder_bit_identical():
    ag = _agent(seed=31)
    cfg = ag.cfg
    anchor = _anchor(cfg, seed=32)
    encoder_before = _enc_snapshot(ag)

    for seed in range(2):
        ag.update_finetune(
            _batch(cfg, seed=seed), *anchor, beta=0.3, critic_only=True
        )
    for seed in range(2, 4):
        ag.update_suffix(
            _batch(cfg, seed=seed),
            *anchor,
            alpha=1.0,
            freeze_encoder=True,
            actor_mask=np.array([False, True, True, False]),
        )

    assert all(
        torch.equal(before, after)
        for before, after in zip(encoder_before, ag.encoder.parameters())
    ), "v2.2 changed the champion encoder in one of its training phases"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
