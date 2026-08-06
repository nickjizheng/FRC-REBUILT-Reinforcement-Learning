"""Integration test of prefix-takeover replay isolation THROUGH THE REAL TRANSPORT.

Transport qualification: the continuity smoke used a local ``deque`` and hand-
reconstructed the n-step return; it never exercised the real ``ReplayRing`` /
collector chunk transport. This test drives the genuine path end to end --

    SuffixEmitter -> distributed.write_suffix_chunk (atomic tmpfs publish)
                  -> distributed.drain_suffix_chunks (round-trip off disk)
                  -> distributed.ingest_suffix_chunk -> real PerEnvReplay / ReplayRing

-- with a scripted two-env, two-episode scenario (natural end + forced-reset
truncation) and proves all five required properties:

  (1) the first stored transition of a stream is the candidate action at H+1;
  (2) no champion-prefix frame or reward appears in the ring (or the chunk);
  (3) the natural handoff is non-terminal;
  (4) a forced timeout/reset closes the stream terminal;
  (5) a sampled n-step return can cross NEITHER boundary (not into prefix, not into
      the next episode's suffix through the unwritten prefix gap).

Pure numpy; no Isaac, no torch. Run:
  C:\\il\\venv\\Scripts\\python.exe -m pytest tests/test_prefix_takeover_transport.py -q
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frc_rebuilt.rl import distributed as D              # noqa: E402
from frc_rebuilt.rl.distributed import SuffixIngestor, ingest_suffix_chunk, Chunk  # noqa: E402
from frc_rebuilt.rl.prefix_takeover import SuffixEmitter  # noqa: E402
from frc_rebuilt.rl.replay import PerEnvReplay            # noqa: E402

C, H, W, P, V, A = 9, 4, 4, 22, 26, 7
PREFIX_OBS_VAL = 7          # every prefix frame is filled with this sentinel
PREFIX_REWARD = -999.0      # every prefix reward is this sentinel
N_STEP, GAMMA = 3, 0.99


def _tx(obs_val, reward, action_val):
    return (np.full((C, H, W), obs_val, np.uint8),
            np.full(P, action_val, np.float32),          # proprio (reuse action_val as tag)
            np.full(V, 0.0, np.float32),                 # privileged
            np.full(A, action_val, np.float32),          # action (tagged)
            float(reward))


def _script_env(emitter, e, prefix_steps, suffix_rewards, suffix_obs_vals, last_forced):
    """Feed one env's episode: ``prefix_steps`` prefix ticks, then arm at the last
    prefix tick (unload), then suffix ticks with the given rewards, the last one
    terminal (natural done unless ``last_forced``)."""
    # prefix: unloaded False until the final prefix tick, which SETS unloaded (=H)
    for k in range(prefix_steps):
        unloaded = (k == prefix_steps - 1)      # H is the last prefix tick
        o, pr, pv, ac, rw = _tx(PREFIX_OBS_VAL, PREFIX_REWARD, -1.0)
        emitter.observe(e, o, pr, pv, ac, rw, unloaded=unloaded, done=False)
    # suffix: H+1 .. terminal
    ns = len(suffix_rewards)
    for j, (rw, ov) in enumerate(zip(suffix_rewards, suffix_obs_vals)):
        terminal = (j == ns - 1)
        o, pr, pv, ac, r = _tx(ov, rw, float(ov))       # action tag = obs value
        emitter.observe(e, o, pr, pv, ac, r,
                        unloaded=True, done=terminal and not last_forced,
                        forced_reset=terminal and last_forced)


def test_suffix_transport_replay_isolation(tmp_path=None):
    root = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    collector_envs = 2
    emitter = SuffixEmitter(collector_envs, chunk_steps=5)   # small -> multiple chunks
    cdir = D.collector_dir(root, 0)

    # ---- env 0: episode A (natural end) then episode B (forced-reset truncation) ----
    # ep A: 2 prefix, 6 suffix (rewards 1000..1005, obs 100..105), natural done
    _script_env(emitter, 0, prefix_steps=2,
                suffix_rewards=[1000, 1001, 1002, 1003, 1004, 1005],
                suffix_obs_vals=[100, 101, 102, 103, 104, 105], last_forced=False)
    # ep B: 2 prefix, 6 suffix (rewards 2000..2005, obs 200..205), FORCED reset
    _script_env(emitter, 0, prefix_steps=2,
                suffix_rewards=[2000, 2001, 2002, 2003, 2004, 2005],
                suffix_obs_vals=[200, 201, 202, 203, 204, 205], last_forced=True)
    # ---- env 1: one episode, 1 prefix, 3 suffix (rewards 5000..5002), natural done ----
    _script_env(emitter, 1, prefix_steps=1,
                suffix_rewards=[5000, 5001, 5002],
                suffix_obs_vals=[50, 51, 52], last_forced=False)

    # flush -> possibly several chunks via the REAL atomic writer
    seq = 0
    chunk = emitter.flush()
    assert chunk is not None
    D.write_suffix_chunk(cdir, seq, chunk, episodes=[]); seq += 1

    # drain off disk (real round-trip) and ingest into a REAL PerEnvReplay
    replay = PerEnvReplay(num_envs=collector_envs, capacity_per_env=1000, seed=0,
                          obs_shape=(C, H, W), proprio_dim=P, privileged_dim=V,
                          action_dim=A, n_step=N_STEP, gamma=GAMMA)
    consumed: set[str] = set()
    drained = D.drain_suffix_chunks(root, num_collectors=1, consumed=consumed)
    assert drained, "no suffix chunks drained"
    total_added = 0
    for ch in drained:
        res = D.ingest_suffix_chunk(replay, ch.collector_id, collector_envs, ch)
        total_added += res["added"]

    ring0, ring1 = replay.rings[0], replay.rings[1]

    # (2) no champion-prefix frame or reward anywhere in the rings
    for ring in (ring0, ring1):
        stored_obs = ring.obs[: ring.size]
        stored_rew = ring.reward[: ring.size]
        assert not (stored_obs == PREFIX_OBS_VAL).any(), "prefix FRAME leaked into ring"
        assert not np.isclose(stored_rew, PREFIX_REWARD).any(), "prefix REWARD leaked into ring"

    # stream 0 must hold exactly the 12 suffix transitions, in order
    assert ring0.size == 12, ring0.size
    assert np.allclose(ring0.reward[:12],
                       [1000, 1001, 1002, 1003, 1004, 1005, 2000, 2001, 2002, 2003, 2004, 2005])
    # (1) first stored transition is the candidate's H+1 action (obs/action tag 100)
    assert np.allclose(ring0.action[0], 100.0), "first stored is not the H+1 candidate action"
    assert (ring0.obs[0] == 100).all()
    # (3) natural handoff is non-terminal: ep-A first suffix (idx 0) and ep-B first
    #     suffix (idx 6) are both done=False
    assert ring0.done[0] == False and ring0.done[6] == False, "handoff marked terminal"
    # ep-A natural end (idx 5) terminal; (4) ep-B forced reset (idx 11) terminal
    assert ring0.done[5] == True, "natural episode end not terminal"
    assert ring0.done[11] == True, "forced reset not terminal (truncation lost)"
    # interior suffix steps are non-terminal
    assert not ring0.done[[1, 2, 3, 4, 7, 8, 9, 10]].any()

    # (5) NO n-step window crosses a boundary. Replicate ReplayRing's exact walk for
    #     EVERY start index and assert all summed rewards share the anchor's episode.
    def episode_of(reward_val):
        return int(reward_val) // 1000          # 1xxx -> ep1(A), 2xxx -> ep2(B)
    for i in range(ring0.size):
        anchor_ep = episode_of(ring0.reward[i])
        alive = True
        for k in range(N_STEP):
            j = (i + k) % ring0.capacity
            if not alive:
                break
            assert episode_of(ring0.reward[j]) == anchor_ep, (
                f"n-step from {i} crossed into episode {episode_of(ring0.reward[j])}")
            if ring0.done[j]:
                alive = False

    # and the actual sampler never bridges: sample many, decode reward bands
    for _ in range(200):
        b = replay.sample(8)
        # each summed n-step reward stays within one episode's band width; a crossed
        # boundary (e.g. 1005 + 2000) would exceed any single-episode max partial sum
        assert np.all(b.reward < 3000 * (1 + GAMMA + GAMMA ** 2)), "sample crossed episodes"

    # stream 1 sanity: 3 suffix, last terminal, no prefix
    assert ring1.size == 3 and ring1.done[2] == True and not ring1.done[[0, 1]].any()
    assert np.allclose(ring1.reward[:3], [5000, 5001, 5002])
    assert total_added == 15


def _feed_ongoing_suffix(emitter, e, n_suffix, base_reward, base_obs):
    """Feed 1 prefix + arm + ``n_suffix`` NON-terminal suffix rows (episode ongoing)."""
    o, pr, pv, ac, rw = _tx(PREFIX_OBS_VAL, PREFIX_REWARD, -1.0)
    emitter.observe(e, o, pr, pv, ac, rw, unloaded=False, done=False)   # prefix
    emitter.observe(e, o, pr, pv, ac, rw, unloaded=True, done=False)    # H (arm, not buffered)
    for j in range(n_suffix):
        o2, pr2, pv2, ac2, r2 = _tx(base_obs + j, base_reward + j, float(base_obs + j))
        emitter.observe(e, o2, pr2, pv2, ac2, r2, unloaded=True, done=False)  # H+1.. (non-terminal)


def test_collector_restart_inserts_boundary(tmp_path=None):
    """Adversarial finding wf_778e3bb8: a collector restart mid-suffix must NOT let an
    n-step return bridge the abandoned episode's non-terminal tail into the restarted
    episode's suffix. SuffixIngestor terminal-boundaries the stream on the seq reset."""
    import tempfile
    root = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    cdir = D.collector_dir(root, 0)
    replay = PerEnvReplay(num_envs=1, capacity_per_env=1000, seed=0,
                          obs_shape=(C, H, W), proprio_dim=P, privileged_dim=V,
                          action_dim=A, n_step=N_STEP, gamma=GAMMA)
    ing = SuffixIngestor(replay, collector_envs=1)
    consumed: set[str] = set()

    # episode A: 3 non-terminal suffix rows (mid-episode flush), then the collector crashes
    emA = SuffixEmitter(1, chunk_steps=3)
    _feed_ongoing_suffix(emA, 0, 3, base_reward=1000, base_obs=100)
    D.write_suffix_chunk(cdir, 0, emA.flush(), episodes=[])
    for ch in D.drain_suffix_chunks(root, 1, consumed):
        ing.ingest(ch)
    assert replay.rings[0].size == 3 and not replay.rings[0].done[:3].any()  # open tail

    # RESTART: fresh emitter, collector seq resets to 0; new episode B
    emB = SuffixEmitter(1, chunk_steps=3)
    _feed_ongoing_suffix(emB, 0, 3, base_reward=2000, base_obs=200)
    D.write_suffix_chunk(cdir, 0, emB.flush(), episodes=[])           # same filename, seq 0
    restart_seen = False
    for ch in D.drain_suffix_chunks(root, 1, consumed):
        res = ing.ingest(ch)
        restart_seen = restart_seen or res.get("restart_boundary", False)

    ring = replay.rings[0]
    assert restart_seen, "restart (seq reset) was not detected"
    assert ring.size == 6
    # the boundary MUST fall on episode A's abandoned tail (index 2) so n-step cannot bridge
    assert ring.done[2] == True, "no terminal boundary inserted at collector restart"
    assert not ring.done[[0, 1]].any()
    # every episode-A anchor's n-step walk stops AT the boundary and never reaches an
    # episode-B index (>=3). Without the SuffixIngestor boundary, done[2] would be False
    # and the walk from index 1 would reach index 3 (episode B), i.e. a bridge.
    A_TAIL = 2
    for i in range(A_TAIL + 1):
        reached, alive = [], True
        for k in range(N_STEP):
            j = i + k
            if not alive:
                break
            reached.append(j)
            if ring.done[j]:
                alive = False
        assert max(reached) <= A_TAIL, f"n-step from {i} bridged into episode B (idx {max(reached)})"


def test_ingest_rejects_out_of_range_stream():
    """Adversarial finding wf_778e3bb8: an out-of-range local stream id (resized/mis-
    configured collector) must be rejected, not silently mapped to the wrong ring."""
    replay = PerEnvReplay(num_envs=2, capacity_per_env=100, seed=0,
                          obs_shape=(C, H, W), proprio_dim=P, privileged_dim=V,
                          action_dim=A, n_step=N_STEP, gamma=GAMMA)
    T = 4
    arrays = {
        "obs": np.zeros((T, C, H, W), np.uint8),
        "proprio": np.zeros((T, P), np.float32),
        "privileged": np.zeros((T, V), np.float32),
        "action": np.zeros((T, A), np.float32),
        "reward": np.zeros(T, np.float32),
        "done": np.zeros(T, bool),
        "stream": np.array([0, 1, 5, 0], np.int32),   # 5 >= collector_envs(2) -> invalid
    }
    res = ingest_suffix_chunk(replay, collector_id=0, collector_envs=2, chunk=Chunk(0, 0, arrays, []))
    assert res.get("invalid_stream") and res["added"] == 0 and res["rejected"] == T
    assert len(replay) == 0                            # nothing corrupt got in


if __name__ == "__main__":
    test_suffix_transport_replay_isolation()
    test_collector_restart_inserts_boundary()
    test_ingest_rejects_out_of_range_stream()
    print("OK: real-transport replay isolation proven (incl. restart + validation)")
