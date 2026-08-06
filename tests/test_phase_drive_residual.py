"""CPU-only invariants for the exact-V4 phase-drive residual core."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from frc_rebuilt.rl.phase_drive_residual import (
    FrozenBasePhaseDrivePolicy,
    PhaseDriveResidual,
    RESIDUAL_ACTION_POLICY,
    RESIDUAL_CAP,
    RESIDUAL_CHECKPOINT_KEY,
    RESIDUAL_METADATA_KEY,
    apply_phase_drive_residual,
    build_phase_drive_residual_checkpoint,
    load_optional_phase_drive_residual,
)
from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
from frc_rebuilt.rl.policy_v2 import (
    ACTION_POLICY,
    FIELD_STRATEGY,
    LEGACY_PROPRIO_DIM,
    PolicyPhase,
    RETURN_SKILL_PRELOAD,
    SCHEMA_VERSION,
    V2_PROPRIO_DIM,
    validate_composite_metadata,
)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 5)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(frames))


class _TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(5 + V2_PROPRIO_DIM, 8),
            nn.LayerNorm(8),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(8, 11),
            nn.ReLU(inplace=False),
            nn.Linear(11, 7),
        )

    def forward(
        self, features: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.trunk(torch.cat((features, proprio), dim=-1))
        return torch.tanh(self.policy(hidden))


def _proprio(*phases: PolicyPhase) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260720)
    values = torch.randn(
        len(phases), V2_PROPRIO_DIM, generator=generator
    )
    values[:, LEGACY_PROPRIO_DIM : LEGACY_PROPRIO_DIM + 5] = 0.0
    for row, phase in enumerate(phases):
        values[row, LEGACY_PROPRIO_DIM + int(phase)] = 1.0
    return values


def _modules(seed: int = 0) -> tuple[_TinyEncoder, _TinyActor]:
    torch.manual_seed(seed)
    return _TinyEncoder(), _TinyActor()


def _base_payload(encoder: nn.Module, actor: nn.Module) -> dict:
    return {
        "encoder": copy.deepcopy(encoder.state_dict()),
        "actor": copy.deepcopy(actor.state_dict()),
        "critic": {},
        "critic_target": {},
        "train_steps": 985_000,
        "stagec_v2": {
            "schema_version": SCHEMA_VERSION,
            "proprio_dim": V2_PROPRIO_DIM,
            "legacy_proprio_dim": LEGACY_PROPRIO_DIM,
            "prefix_sha256": "b" * 64,
            "action_policy": ACTION_POLICY,
            "field_strategy": FIELD_STRATEGY,
            "return_skill_preload": RETURN_SKILL_PRELOAD,
            "reward_revision": "outer_rail_v4_ramp_out",
        },
    }


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
    }


def test_zero_init_is_bit_exact_and_freezes_base_without_changing_state():
    encoder, actor = _modules(seed=1)
    encoder_before = _snapshot(encoder)
    actor_before = _snapshot(actor)
    residual = PhaseDriveResidual(trunk_dim=8)
    policy = FrozenBasePhaseDrivePolicy(encoder, actor, residual)
    frames = torch.randn(5, 4)
    proprio = _proprio(*PolicyPhase)

    with torch.no_grad():
        base = actor(encoder(frames), proprio)
    composed = policy.mean_actions(frames, proprio)

    assert torch.equal(composed, base)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in actor.parameters())
    assert all(parameter.requires_grad for parameter in residual.parameters())
    for before, after in (
        (encoder_before, encoder.state_dict()),
        (actor_before, actor.state_dict()),
    ):
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key])


def test_real_drq_actor_uses_512_trunk_without_changing_actor_layout():
    torch.manual_seed(11)
    agent = DrQV2Agent(
        DrQConfig(
            device="cpu",
            proprio_dim=V2_PROPRIO_DIM,
            frame_channels=3,
            frame_h=48,
            frame_w=48,
        )
    )
    actor_keys = tuple(agent.actor.state_dict())
    trunk_dim = int(agent.actor.trunk[0].out_features)
    assert trunk_dim == 512
    policy = FrozenBasePhaseDrivePolicy(
        agent.encoder,
        agent.actor,
        PhaseDriveResidual(trunk_dim=trunk_dim),
    )
    frames = torch.randint(0, 256, (5, 3, 48, 48), dtype=torch.uint8)
    proprio = _proprio(*PolicyPhase)
    with torch.no_grad():
        exact_base = agent.actor(agent.encoder(frames), proprio)

    assert torch.equal(policy.mean_actions(frames, proprio), exact_base)
    assert tuple(agent.actor.state_dict()) == actor_keys


def test_heads_are_phase_specific_drive_only_and_hard_capped():
    encoder, actor = _modules(seed=2)
    residual = PhaseDriveResidual(trunk_dim=8)
    with torch.no_grad():
        residual.heads["leave"][-1].bias.copy_(
            torch.tensor([20.0, -20.0, 0.4])
        )
        residual.heads["collect"][-1].bias.copy_(
            torch.tensor([-20.0, 20.0, -0.4])
        )
        residual.heads["return"][-1].bias.copy_(
            torch.tensor([0.8, 0.2, -20.0])
        )
    policy = FrozenBasePhaseDrivePolicy(encoder, actor, residual)
    frames = torch.randn(5, 4)
    proprio = _proprio(*PolicyPhase)
    with torch.no_grad():
        features = encoder(frames)
        base = actor(features, proprio)
        hidden = actor.trunk(torch.cat((features, proprio), dim=-1))
        delta = residual(hidden, proprio)
    composed = policy.mean_actions(frames, proprio)

    assert torch.equal(delta[PolicyPhase.FIRST], torch.zeros(3))
    assert torch.equal(delta[PolicyPhase.SCORE], torch.zeros(3))
    assert not torch.equal(delta[PolicyPhase.LEAVE], delta[PolicyPhase.COLLECT])
    assert not torch.equal(delta[PolicyPhase.COLLECT], delta[PolicyPhase.RETURN])
    assert float(delta.abs().max()) <= RESIDUAL_CAP + 1e-7
    assert torch.equal(composed[PolicyPhase.FIRST], base[PolicyPhase.FIRST])
    assert torch.equal(composed[PolicyPhase.SCORE], base[PolicyPhase.SCORE])
    assert torch.equal(composed[:, 3:], base[:, 3:])
    drive_change = (
        composed[1:4, :3] - base[1:4, :3]
    ).abs()
    assert float(drive_change.max()) <= RESIDUAL_CAP + 1e-7


def test_gradient_reaches_only_residual_output_layers():
    encoder, actor = _modules(seed=3)
    residual = PhaseDriveResidual(trunk_dim=8)
    policy = FrozenBasePhaseDrivePolicy(encoder, actor, residual)
    frames = torch.randn(3, 4)
    proprio = _proprio(
        PolicyPhase.LEAVE, PolicyPhase.COLLECT, PolicyPhase.RETURN
    )

    policy.mean_actions(frames, proprio).sum().backward()

    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in actor.parameters())
    for head in residual.heads.values():
        assert head[-1].weight.grad is not None
        assert bool(torch.isfinite(head[-1].weight.grad).all())
        assert float(head[-1].weight.grad.abs().sum()) > 0.0


def test_action_semantics_reject_protected_or_over_cap_residuals():
    actions = torch.zeros(2, 7)
    proprio = _proprio(PolicyPhase.FIRST, PolicyPhase.LEAVE)
    malicious = torch.zeros(2, 3)
    malicious[0, 0] = 0.01
    with pytest.raises(ValueError, match="FIRST or SCORE"):
        apply_phase_drive_residual(actions, malicious, proprio)

    too_large = torch.zeros(2, 3)
    too_large[1, 0] = RESIDUAL_CAP + 0.001
    with pytest.raises(ValueError, match="hard cap"):
        apply_phase_drive_residual(actions, too_large, proprio)

    malformed = proprio.clone()
    malformed[1, LEGACY_PROPRIO_DIM + int(PolicyPhase.COLLECT)] = 1.0
    with pytest.raises(ValueError, match="strict one-hot"):
        apply_phase_drive_residual(actions, torch.zeros(2, 3), malformed)


def test_checkpoint_roundtrip_is_fail_closed_and_preserves_base_tensors():
    encoder, actor = _modules(seed=4)
    base = _base_payload(encoder, actor)
    residual = PhaseDriveResidual(trunk_dim=8)
    with torch.no_grad():
        residual.heads["return"][-1].bias.fill_(0.7)
    candidate = build_phase_drive_residual_checkpoint(
        base,
        residual,
        base_checkpoint_sha256="a" * 64,
        updates=17,
        optimizer_state={"state": {}, "param_groups": []},
    )

    assert RESIDUAL_CHECKPOINT_KEY not in base
    assert RESIDUAL_METADATA_KEY not in base["stagec_v2"]
    assert base["stagec_v2"]["action_policy"] == ACTION_POLICY
    assert candidate["stagec_v2"]["action_policy"] == RESIDUAL_ACTION_POLICY
    for name in ("encoder", "actor"):
        for key, value in base[name].items():
            assert torch.equal(candidate[name][key], value)
    # Legacy playback calls this validator and must reject, not silently run V4.
    with pytest.raises(ValueError, match="action_policy"):
        validate_composite_metadata(candidate["stagec_v2"], "b" * 64)

    assert load_optional_phase_drive_residual(base) is None
    loaded = load_optional_phase_drive_residual(
        candidate,
        expected_base_sha256="a" * 64,
        expected_trunk_dim=8,
    )
    assert loaded is not None
    assert loaded.updates == 17
    assert loaded.optimizer_state == {"state": {}, "param_groups": []}
    for key, value in residual.state_dict().items():
        assert torch.equal(loaded.module.state_dict()[key], value)

    broken = copy.deepcopy(candidate)
    broken.pop(RESIDUAL_CHECKPOINT_KEY)
    with pytest.raises(ValueError, match="partial"):
        load_optional_phase_drive_residual(broken)
    with pytest.raises(ValueError, match="parent SHA-256 mismatch"):
        load_optional_phase_drive_residual(
            candidate, expected_base_sha256="c" * 64
        )


def test_loaded_zero_init_checkpoint_remains_bit_exact():
    encoder, actor = _modules(seed=5)
    base = _base_payload(encoder, actor)
    candidate = build_phase_drive_residual_checkpoint(
        base,
        PhaseDriveResidual(trunk_dim=8),
        base_checkpoint_sha256="d" * 64,
    )
    loaded = load_optional_phase_drive_residual(
        candidate, expected_base_sha256="d" * 64
    )
    assert loaded is not None
    policy = FrozenBasePhaseDrivePolicy(encoder, actor, loaded.module)
    frames = torch.randn(5, 4)
    proprio = _proprio(*PolicyPhase)
    with torch.no_grad():
        exact_base = actor(encoder(frames), proprio)

    assert torch.equal(policy.mean_actions(frames, proprio), exact_base)
