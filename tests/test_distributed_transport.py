"""Validate the distributed transport (chunk roundtrip, ordering, weight sync)
without Isaac or a GPU — this is the riskiest logic that must be correct before
the server ever runs it."""
from __future__ import annotations

import numpy as np
import pytest

from xrc_rebuilt.rl import distributed as D


def _arrays(num_envs, steps, tag):
    # obs values encode (env, step, tag) so we can prove ordering downstream
    obs = np.zeros((num_envs, steps, 9, 4, 4), np.uint8)
    for e in range(num_envs):
        for t in range(steps):
            obs[e, t] = (tag * 100 + e * 10 + t) % 256
    return {
        "obs": obs,
        "proprio": np.full((num_envs, steps, 3), tag, np.float32),
        "privileged": np.full((num_envs, steps, 2), tag, np.float32),
        "action": np.zeros((num_envs, steps, 7), np.float32),
        "reward": np.full((num_envs, steps), float(tag), np.float32),
        "done": np.zeros((num_envs, steps), bool),
    }


def test_chunk_roundtrip_and_delete(tmp_path):
    cdir = D.collector_dir(tmp_path, 0)
    D.write_chunk(cdir, 1, _arrays(4, 8, tag=7), episodes=[{"return": 1.0, "scored": 2}])
    consumed: set[str] = set()
    chunks = D.drain_chunks(tmp_path, num_collectors=1, consumed=consumed)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.collector_id == 0 and c.seq == 1
    assert c.arrays["obs"].shape == (4, 8, 9, 4, 4)
    assert c.episodes == [{"return": 1.0, "scored": 2}]
    # file is consumed (deleted) — a second drain sees nothing
    assert D.drain_chunks(tmp_path, 1, consumed) == []
    assert not list(cdir.glob("chunk_*.npz"))


def test_per_collector_sequence_order_preserved(tmp_path):
    # write chunks out of order on disk; drain must return them in seq order
    cdir = D.collector_dir(tmp_path, 0)
    for seq in (3, 1, 2):
        D.write_chunk(cdir, seq, _arrays(2, 4, tag=seq), episodes=[])
    chunks = D.drain_chunks(tmp_path, 1, set())
    assert [c.seq for c in chunks] == [1, 2, 3]


def test_multi_collector_streams_are_separable(tmp_path):
    for cid in (0, 1, 2):
        D.write_chunk(D.collector_dir(tmp_path, cid), 1, _arrays(2, 3, tag=cid), episodes=[])
    chunks = D.drain_chunks(tmp_path, num_collectors=3, consumed=set())
    seen = {c.collector_id for c in chunks}
    assert seen == {0, 1, 2}
    # reconstruct global stream id = cid*num_envs + env, verify env-contiguity
    num_envs = 2
    for c in chunks:
        for e in range(num_envs):
            stream = c.collector_id * num_envs + e
            # env e's obs are all tagged with this collector — no cross-mixing
            assert np.all(c.arrays["obs"][e] // 100 == c.collector_id)
            assert stream == c.collector_id * num_envs + e


def test_partial_chunk_not_read(tmp_path):
    # a lone .tmp (mid-write) must be invisible to drain
    cdir = D.collector_dir(tmp_path, 0)
    cdir.mkdir(parents=True)
    (cdir / ".chunk_000000001.tmp").write_bytes(b"partial")
    assert D.drain_chunks(tmp_path, 1, set()) == []


def test_weight_publish_and_latest(tmp_path):
    torch = pytest.importorskip("torch")
    wdir = D.weights_dir(tmp_path)
    for step in (10, 20, 30):
        D.publish_weights(
            wdir,
            {"encoder": {"w": torch.zeros(2)}, "actor": {"w": torch.ones(3)}, "train_steps": step},
            step,
            keep=2,
        )
    got = D.latest_weights(wdir)
    assert got is not None
    path, step = got
    assert step == 30
    # only the newest 2 are retained
    assert len(list(wdir.glob("weights_*.pt"))) == 2
    blob = torch.load(path, map_location="cpu")
    assert blob["train_steps"] == 30 and blob["actor"]["w"].tolist() == [1, 1, 1]
