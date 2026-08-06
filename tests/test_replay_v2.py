from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from frc_rebuilt.rl.replay_v2 import (
    CapturedEpisode,
    GroupedPerEnvReplay,
    UniformChunkIngestor,
)


RING_KWARGS = dict(
    obs_shape=(1, 2, 2),
    proprio_dim=2,
    privileged_dim=1,
    action_dim=1,
    n_step=2,
    gamma=0.5,
)


def _add(replay, stream: int, reward: float, count: int, done_at_end: bool = False):
    for step in range(count):
        replay.add(
            stream,
            np.full((1, 2, 2), stream, np.uint8),
            np.asarray([stream, step], np.float32),
            np.asarray([stream], np.float32),
            np.asarray([stream], np.float32),
            reward,
            done_at_end and step == count - 1,
        )


def _chunk(
    cid: int,
    seq: int,
    rewards: np.ndarray,
    *,
    bad_at=None,
    dones: np.ndarray | None = None,
    episodes: list[dict] | None = None,
):
    rewards = np.asarray(rewards, np.float32)
    envs, steps = rewards.shape
    arrays = {
        "obs": np.zeros((envs, steps, 1, 2, 2), np.uint8),
        "proprio": np.zeros((envs, steps, 2), np.float32),
        "privileged": np.zeros((envs, steps, 1), np.float32),
        "action": np.zeros((envs, steps, 1), np.float32),
        "reward": rewards.copy(),
        "done": (
            np.zeros((envs, steps), bool)
            if dones is None
            else np.asarray(dones, bool).copy()
        ),
    }
    if bad_at is not None:
        arrays["action"][bad_at] = np.nan
    return SimpleNamespace(
        collector_id=cid, seq=seq, arrays=arrays, episodes=list(episodes or [])
    )


def test_group_quotas_and_unavailable_group_redistribution():
    replay = GroupedPerEnvReplay(
        stream_groups=("full", "full", "collect", "return"),
        capacity_per_env=64,
        seed=7,
        **RING_KWARGS,
    )
    # Both full rings are live with very different fill; collect is live; return
    # is unavailable.  Equal available-group weights must yield an exact 6/6.
    _add(replay, 0, 10.0, 7)   # 4 valid anchors
    _add(replay, 1, 10.0, 27)  # 24 valid anchors
    _add(replay, 2, 20.0, 12)

    batch = replay.sample_grouped(12)

    assert replay.last_group_counts == {"full": 6, "collect": 6, "return": 0}
    assert sum(replay.last_ring_counts) == 12
    assert replay.last_ring_counts[1] > replay.last_ring_counts[0]
    assert np.count_nonzero(batch.reward == 15.0) == 6  # 10 + .5 * 10
    assert np.count_nonzero(batch.reward == 30.0) == 6  # 20 + .5 * 20


def test_configured_group_weights_are_exact_and_n_step_never_crosses_rings():
    replay = GroupedPerEnvReplay(
        stream_groups=("full", "collect"),
        capacity_per_env=64,
        group_weights={"full": 1.0, "collect": 3.0},
        seed=3,
        **RING_KWARGS,
    )
    _add(replay, 0, 1.0, 20)
    _add(replay, 1, 100.0, 20)

    batch = replay.sample(40)

    assert replay.last_group_counts == {"full": 10, "collect": 30}
    # A shared/interleaved ring would produce mixed values such as 51 or 100.5.
    assert set(np.unique(batch.reward)) == {1.5, 150.0}


def test_uniform_ingestor_closes_all_collector_streams_on_gap_and_reset():
    replay = GroupedPerEnvReplay(
        stream_groups=("full", "collect"),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(replay, collector_envs=2)

    assert not ingestor.ingest(_chunk(0, 5, [[1, 1], [2, 2]]))["restart_boundary"]
    gap = ingestor.ingest(_chunk(0, 7, [[3], [4]]))
    assert gap["restart_boundary"]
    assert replay.rings[0].done[1]
    assert replay.rings[1].done[1]

    reset = ingestor.ingest(_chunk(0, 0, [[5], [6]]))
    assert reset["restart_boundary"]
    assert replay.rings[0].done[2]
    assert replay.rings[1].done[2]


def test_uniform_ingestor_rejects_nonfinite_row_and_malformed_shape_with_boundaries():
    replay = GroupedPerEnvReplay(
        stream_groups=("full", "collect"),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(replay, collector_envs=2)

    result = ingestor.ingest(_chunk(0, 0, [[1, 2, 3], [4, 5, 6]], bad_at=(0, 1)))
    assert result == {
        "added": 5,
        "rejected": 1,
        "invalid_chunk": False,
        "restart_boundary": False,
    }
    # The last valid transition before the NaN was terminal-boundaried.
    assert replay.rings[0].done[0]

    malformed = _chunk(0, 1, [[7], [8]])
    malformed.arrays["proprio"] = np.zeros((2, 1, 99), np.float32)
    result = ingestor.ingest(malformed)
    assert result["invalid_chunk"]
    assert result["added"] == 0
    assert any("proprio" in error for error in result["schema_errors"])
    # Rejecting a whole chunk also closes each collector-owned open tail.
    assert replay.rings[0].done[(replay.rings[0].cursor - 1) % replay.rings[0].capacity]
    assert replay.rings[1].done[(replay.rings[1].cursor - 1) % replay.rings[1].capacity]


def test_opt_in_episode_capture_spans_chunks_and_uses_env_terminal_metadata():
    replay = GroupedPerEnvReplay(
        stream_groups=("full", "collect"),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(
        replay, collector_envs=2, capture_groups=("full",)
    )

    first = ingestor.ingest(_chunk(0, 0, [[1, 2], [8, 9]]))
    assert first["completed_episodes"] == []
    second = ingestor.ingest(
        _chunk(
            0,
            1,
            [[3, 4], [10, 11]],
            dones=[[False, True], [False, True]],
            episodes=[
                {"env_index": 1, "cycles_completed": 99},
                {"env_index": 0, "cycles_completed": 1, "episode_seq": 7},
            ],
        )
    )

    assert len(second["completed_episodes"]) == 1
    captured = second["completed_episodes"][0]
    assert isinstance(captured, CapturedEpisode)
    assert captured.stream_index == 0
    assert captured.group == "full"
    assert captured.arrays["reward"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert captured.arrays["done"].tolist() == [False, False, False, True]
    assert captured.stats["episode_seq"] == 7


def test_episode_capture_discards_entire_tainted_episode_across_forward_gap():
    replay = GroupedPerEnvReplay(
        stream_groups=("full",),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(
        replay, collector_envs=1, capture_groups=("full",)
    )
    ingestor.ingest(_chunk(0, 4, [[1, 2]]))

    result = ingestor.ingest(
        _chunk(
            0,
            6,
            [[9]],
            dones=[[True]],
            episodes=[{"env_index": 0, "cycles_completed": 1}],
        )
    )

    assert result["restart_boundary"]
    assert result["completed_episodes"] == []


def test_episode_capture_starts_fresh_after_collector_sequence_rewind():
    replay = GroupedPerEnvReplay(
        stream_groups=("full",),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(
        replay, collector_envs=1, capture_groups=("full",)
    )
    ingestor.ingest(_chunk(0, 4, [[1, 2]]))

    result = ingestor.ingest(
        _chunk(
            0,
            0,
            [[9]],
            dones=[[True]],
            episodes=[{"env_index": 0, "cycles_completed": 1}],
        )
    )

    assert result["restart_boundary"]
    assert len(result["completed_episodes"]) == 1
    assert result["completed_episodes"][0].arrays["reward"].tolist() == [9.0]


def test_capture_maps_two_same_group_envs_by_env_index_not_episode_list_order():
    replay = GroupedPerEnvReplay(
        stream_groups=("full", "full"),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(
        replay, collector_envs=2, capture_groups=("full",)
    )
    result = ingestor.ingest(
        _chunk(
            0,
            0,
            [[1], [2]],
            dones=[[True], [True]],
            episodes=[
                {"env_index": 1, "episode_seq": 21, "cycles_completed": 1},
                {"env_index": 0, "episode_seq": 10, "milestones": {"returned_home": 1}},
            ],
        )
    )

    by_stream = {ep.stream_index: ep for ep in result["completed_episodes"]}
    assert by_stream[0].stats["episode_seq"] == 10
    assert by_stream[1].stats["episode_seq"] == 21


def test_nonfinite_row_taints_capture_until_episode_boundary():
    replay = GroupedPerEnvReplay(
        stream_groups=("full",),
        capacity_per_env=32,
        seed=1,
        **RING_KWARGS,
    )
    ingestor = UniformChunkIngestor(
        replay, collector_envs=1, capture_groups=("full",)
    )
    result = ingestor.ingest(
        _chunk(
            0,
            0,
            [[1, 2, 3]],
            bad_at=(0, 1),
            dones=[[False, False, True]],
            episodes=[{"env_index": 0, "cycles_completed": 1}],
        )
    )
    assert result["completed_episodes"] == []

    clean = ingestor.ingest(
        _chunk(
            0,
            1,
            [[4]],
            dones=[[True]],
            episodes=[{"env_index": 0, "cycles_completed": 1}],
        )
    )
    assert clean["completed_episodes"][0].arrays["reward"].tolist() == [4.0]
