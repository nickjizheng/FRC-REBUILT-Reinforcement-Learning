import numpy as np
import pytest

from xrc_rebuilt.rl import CompetitionRLSpec, decode_policy_actions


def test_rl_timing_contract_matches_full_match():
    spec = CompetitionRLSpec()
    spec.validate()
    assert spec.physics_steps_per_action == 6
    assert spec.episode_policy_steps == 1600
    assert spec.alliance == "blue"


def test_rl_action_decode_clips_and_never_exposes_red_target():
    decoded = decode_policy_actions(
        np.array([[2.0, -2.0, 0.5, 0.7, -0.7, 0.8, 0.9]], dtype=np.float32)
    )
    np.testing.assert_allclose(decoded.driver, [[1.0, -1.0, 0.5]])
    assert decoded.intake_on.tolist() == [True]
    assert decoded.storage_extended.tolist() == [False]
    assert decoded.shoot_blue.tolist() == [True]
    assert decoded.ferry.tolist() == [False]
    assert not hasattr(decoded, "alliance")
    assert not hasattr(decoded, "shoot_red")


def test_rl_action_decode_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decode_policy_actions(np.zeros((4, 6), dtype=np.float32))


def test_rl_spec_rejects_non_blue_alliance():
    with pytest.raises(ValueError):
        CompetitionRLSpec(alliance="red").validate()
