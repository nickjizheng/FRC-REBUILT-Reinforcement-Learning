"""Focused tests for the legacy -> phase-augmented checkpoint migration."""
from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from frc_rebuilt.rl.checkpoint_v2 import load_legacy_checkpoint_into_v2
from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent


LEGACY_PROPRIO = 22
PRIVILEGED = 26
PHASE_FEATURES = 8


def _agent(proprio_dim: int) -> DrQV2Agent:
    return DrQV2Agent(
        DrQConfig(
            proprio_dim=proprio_dim,
            privileged_dim=PRIVILEGED,
            frame_channels=9,
            frame_h=48,
            frame_w=48,
            device="cpu",
        )
    )


def _stamp_modules(agent: DrQV2Agent) -> None:
    """Give every source tensor an unmistakable deterministic value."""
    with torch.no_grad():
        for module_index, module in enumerate(
            (agent.encoder, agent.actor, agent.critic, agent.critic_target), start=1
        ):
            for tensor_index, tensor in enumerate(module.state_dict().values(), start=1):
                tensor.fill_(module_index * 100 + tensor_index)

        # Column-dependent values make all three critic regions distinguishable.
        actor_weight = agent.actor.trunk[0].weight
        actor_weight.copy_(
            torch.arange(actor_weight.numel(), dtype=actor_weight.dtype).reshape_as(
                actor_weight
            )
        )
        for critic in (agent.critic, agent.critic_target):
            weight = critic.trunk[0].weight
            weight.copy_(
                torch.arange(weight.numel(), dtype=weight.dtype).reshape_as(weight)
            )


def _payload(agent: DrQV2Agent) -> dict:
    return {
        "encoder": deepcopy(agent.encoder.state_dict()),
        "actor": deepcopy(agent.actor.state_dict()),
        "critic": deepcopy(agent.critic.state_dict()),
        "critic_target": deepcopy(agent.critic_target.state_dict()),
        # These are intentionally invalid optimizer payloads.  The migration
        # must never inspect or restore them.
        "encoder_opt": {"must_not": "load"},
        "actor_opt": {"must_not": "load"},
        "critic_opt": {"must_not": "load"},
        "train_steps": 1_163_753,
        "explore_offset": 936_253,
        "skipped_updates": 7,
    }


def _assert_other_tensors_equal(source: dict, loaded: dict, expanded_key: str) -> None:
    assert source.keys() == loaded.keys()
    for key in source:
        if key != expanded_key:
            torch.testing.assert_close(loaded[key], source[key], rtol=0, atol=0)


def test_expands_inputs_shifts_privileged_and_starts_fresh_optimizers(tmp_path):
    legacy = _agent(LEGACY_PROPRIO)
    _stamp_modules(legacy)
    payload = _payload(legacy)
    checkpoint = tmp_path / "champion.pt"
    torch.save(payload, checkpoint)

    v2 = _agent(LEGACY_PROPRIO + PHASE_FEATURES)
    # Prove reused destination optimizer state is cleared as well as proving
    # the invalid optimizer dictionaries above are not loaded.
    for optimizer, parameter in (
        (v2.encoder_opt, next(v2.encoder.parameters())),
        (v2.actor_opt, next(v2.actor.parameters())),
        (v2.critic_opt, next(v2.critic.parameters())),
    ):
        optimizer.state[parameter]["preexisting"] = torch.tensor(1.0)

    report = load_legacy_checkpoint_into_v2(v2, checkpoint)

    assert report.feature_dim == legacy.feat_dim == v2.feat_dim
    assert report.appended_proprio_dim == PHASE_FEATURES
    assert report.train_steps == v2.train_steps == 1_163_753
    assert report.explore_offset == v2.explore_offset == 936_253
    assert report.skipped_updates == v2.skipped_updates == 7
    assert report.optimizer_state_restored is False
    assert all(not optimizer.state for optimizer in (v2.encoder_opt, v2.actor_opt, v2.critic_opt))

    for key, value in payload["encoder"].items():
        torch.testing.assert_close(v2.encoder.state_dict()[key], value, rtol=0, atol=0)

    loaded_actor = v2.actor.state_dict()
    _assert_other_tensors_equal(payload["actor"], loaded_actor, "trunk.0.weight")
    old_actor_weight = payload["actor"]["trunk.0.weight"]
    new_actor_weight = loaded_actor["trunk.0.weight"]
    old_actor_end = legacy.feat_dim + LEGACY_PROPRIO
    torch.testing.assert_close(
        new_actor_weight[:, :old_actor_end], old_actor_weight, rtol=0, atol=0
    )
    assert torch.count_nonzero(new_actor_weight[:, old_actor_end:]) == 0

    old_prefix_end = legacy.feat_dim + LEGACY_PROPRIO
    new_privileged_start = legacy.feat_dim + LEGACY_PROPRIO + PHASE_FEATURES
    for name, module in (("critic", v2.critic), ("critic_target", v2.critic_target)):
        loaded = module.state_dict()
        source = payload[name]
        _assert_other_tensors_equal(source, loaded, "trunk.0.weight")
        old_weight = source["trunk.0.weight"]
        new_weight = loaded["trunk.0.weight"]
        torch.testing.assert_close(
            new_weight[:, :old_prefix_end],
            old_weight[:, :old_prefix_end],
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(
            new_weight[:, old_prefix_end:new_privileged_start]
        ) == 0
        torch.testing.assert_close(
            new_weight[:, new_privileged_start:],
            old_weight[:, old_prefix_end:],
            rtol=0,
            atol=0,
        )

    # With zero-valued appended features, warm-start behavior is bit-for-bit
    # continuous for both the actor and asymmetric critics.
    generator = torch.Generator().manual_seed(20260714)
    feature = torch.randn(4, legacy.feat_dim, generator=generator)
    old_proprio = torch.randn(4, LEGACY_PROPRIO, generator=generator)
    # New inputs are zero-weighted by migration, so even the real non-zero
    # one-hot/load/time features leave the champion action bit-identical.
    appended = torch.randn(4, PHASE_FEATURES, generator=generator)
    v2_proprio = torch.cat([old_proprio, appended], dim=-1)
    privileged = torch.randn(4, PRIVILEGED, generator=generator)
    action = torch.randn(4, legacy.cfg.action_dim, generator=generator)
    torch.testing.assert_close(
        v2.actor(feature, v2_proprio),
        legacy.actor(feature, old_proprio),
        rtol=0,
        atol=0,
    )
    for old_critic, new_critic in (
        (legacy.critic, v2.critic),
        (legacy.critic_target, v2.critic_target),
    ):
        old_q = old_critic(feature, old_proprio, privileged, action)
        new_q = new_critic(feature, v2_proprio, privileged, action)
        for old_value, new_value in zip(old_q, new_q):
            torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)


def test_dimension_error_is_detected_before_any_module_is_changed():
    legacy = _agent(LEGACY_PROPRIO)
    _stamp_modules(legacy)
    payload = _payload(legacy)
    # Corrupt the critic's inferred legacy layout while keeping the actor valid.
    payload["critic"]["trunk.0.weight"] = payload["critic"][
        "trunk.0.weight"
    ][:, :-1]

    v2 = _agent(LEGACY_PROPRIO + PHASE_FEATURES)
    encoder_before = deepcopy(v2.encoder.state_dict())
    actor_before = deepcopy(v2.actor.state_dict())

    with pytest.raises(ValueError, match="legacy input has"):
        load_legacy_checkpoint_into_v2(v2, payload)

    for before, after in (
        (encoder_before, v2.encoder.state_dict()),
        (actor_before, v2.actor.state_dict()),
    ):
        for key in before:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)


def test_missing_schedule_metadata_uses_legacy_safe_defaults():
    legacy = _agent(LEGACY_PROPRIO)
    payload = _payload(legacy)
    payload.pop("train_steps")
    payload.pop("explore_offset")
    payload.pop("skipped_updates")

    v2 = _agent(LEGACY_PROPRIO + PHASE_FEATURES)
    v2.train_steps = 99
    v2.explore_offset = 88
    v2.skipped_updates = 77
    report = load_legacy_checkpoint_into_v2(v2, payload)

    assert (v2.train_steps, v2.explore_offset, v2.skipped_updates) == (0, 0, 0)
    assert (report.train_steps, report.explore_offset, report.skipped_updates) == (0, 0, 0)
