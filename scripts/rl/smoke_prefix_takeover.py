"""GPU composition smoke for the prefix-takeover pilot. No learning.

Composes the REAL Isaac collector with the production suffix transport and asserts the
handoff/replay contract end to end, with a frozen champion + an IDENTITY candidate:

  * one Isaac env-set, two envs, champion plays the prefix (deterministic), candidate
    takes over at first unload (score>0 ∧ magazine empty);
  * reach >= 2 handoffs (retry-cap 6 episodes);
  * emit REAL suffix chunks through SuffixEmitter -> distributed.write_suffix_chunk,
    including at least one mid-suffix flush and at least one forced-reset terminal;
  * drain through distributed.SuffixIngestor into the real PerEnvReplay the learner uses;
  * assert: the first stored row of each suffix stream is the candidate's H+1 obs/action,
    no prefix frame appears in the ring, a terminal boundary exists, 3-step samples are
    valid, and the immutable champion SHA-256 is unchanged;
  * run the real AnchorSampler (holdout-excluding) and a no-update DriftGate publication
    check in the SAME composed process; assert PBRS γ == learner γ (0.999) and evaluate
    the leave potential Φ at the real handoff positions.

Writes runs/phase/smoke_prefix_takeover.json and prints SMOKE_PREFIX_TAKEOVER PASS/FAIL.
Needs Isaac + a GPU. Run: C:\\il\\venv\\Scripts\\python.exe scripts/rl/smoke_prefix_takeover.py
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np

LEARNER_GAMMA = 0.999            # scripts/rl/run_distributed.sh
CHUNK_STEPS = 8                  # small -> guarantees a mid-suffix flush (and several chunks)
SUFFIX_BEFORE_RESET = 40         # emit this many suffix steps, then force a terminal
                                 # (>=64 total across 2 envs so a real 3-step batch can sample)
ANCHOR_DIR = PROJECT_ROOT / "runs/phase/stageC_phase_timing_anchor_dev32_anchors"


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.replay import PerEnvReplay
        from frc_rebuilt.rl import distributed as D
        from frc_rebuilt.rl import prefix_takeover as pt

        champ_path = pt.CHAMPION_PATH
        sha0 = pt.immutable_champion_ok(champ_path)               # SHA guard at start
        pt.assert_pbrs_gamma(pt.STAGE_C_GAMMA, LEARNER_GAMMA)     # gamma footgun guard
        assert pt.STAGE_C_GAMMA == LEARNER_GAMMA

        n = 2
        env = VecCompetitionEnv(VecEnvCfg(
            num_envs=n, template_usd=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"),
            cameras=True, episode_len_s=90.0, preload_prob=0.0, seed=90001,
            spawn_under_trench=True, lock_storage_extended=False, collect_reward_weight=0.3,
        ))
        champion = DrQV2Agent(DrQConfig()); champion.load(champ_path)   # frozen prefix
        candidate = DrQV2Agent(DrQConfig()); candidate.load(champ_path) # identity suffix
        print(f"SMOKE_LOADED champ_steps={champion.train_steps} sha={sha0[:12]} "
              f"gamma={LEARNER_GAMMA}", flush=True)

        emitter = pt.SuffixEmitter(collector_envs=n, chunk_steps=CHUNK_STEPS)
        replay = PerEnvReplay(num_envs=n, capacity_per_env=4000, seed=5,
                              obs_shape=(9, 90, 160), proprio_dim=22, privileged_dim=26,
                              action_dim=7, n_step=3, gamma=LEARNER_GAMMA)
        ingestor = D.SuffixIngestor(replay, collector_envs=n)
        cdir = D.collector_dir(Path(tempfile.mkdtemp()), 0)
        leave = pt.LeavePotential("blue")

        env.reset_all()
        obs, *_ = env.step(np.zeros((n, 7), np.float32))
        seq = 0
        consumed: set = set()
        h1 = {e: None for e in range(n)}          # captured (obs_hash, action) at first H+1
        prefix_obs_hash = {e: set() for e in range(n)}
        suffix_count = {e: 0 for e in range(n)}
        handoff_pos = {e: None for e in range(n)}
        leave_phi = {e: None for e in range(n)}
        done_env = {e: False for e in range(n)}   # this env reached its forced terminal
        mid_flush_seen = False
        forced_terminal_seen = False
        reset_pending: list[int] = []
        ticks = 0
        MAX_TICKS = 2500

        def drain_ingest():
            nonlocal seq
            for ch in D.drain_suffix_chunks(cdir.parent, 1, consumed):
                ingestor.ingest(ch)

        while not all(done_env.values()) and ticks < MAX_TICKS:
            frames = D.to_policy_frames(obs["rgb"])
            champ_act = champion.act(frames, obs["proprio"], explore=False).astype(np.float32)
            actions = champ_act.copy()
            for e in range(n):
                if emitter.in_suffix(e):          # candidate drives once armed (from H+1)
                    actions[e] = candidate.act(frames[e:e+1], obs["proprio"][e:e+1],
                                               explore=False).astype(np.float32)[0]

            # per-env handoff signal (score>0 ∧ magazine empty) BEFORE stepping
            unloaded = np.zeros(n, bool)
            for e in range(n):
                c = env.slots[e].controller
                unloaded[e] = (int(env.slots[e].router.scored["blue"]) > 0 and len(c.magazine) == 0)

            nxt, rewards, dones, info = env.step(actions)

            for e in range(n):
                if done_env[e]:
                    continue
                was_suffix = emitter.in_suffix(e)
                force = was_suffix and suffix_count[e] + 1 >= SUFFIX_BEFORE_RESET
                emitter.observe(e, frames[e], obs["proprio"][e], obs["privileged"][e],
                                actions[e], float(rewards[e]),
                                unloaded=bool(unloaded[e]), done=bool(dones[e]), forced_reset=force)
                if was_suffix:
                    suffix_count[e] += 1
                    if h1[e] is None:             # first H+1 for this env -> capture ground truth
                        h1[e] = (_h(frames[e]), actions[e].copy())
                        pos, _ = env.slots[e].controller.chassis_pose()
                        handoff_pos[e] = (float(pos[0]), float(pos[1]))
                        leave.reset_segment()
                        if leave.select_gateway(handoff_pos[e]) is not None:
                            leave_phi[e] = leave.potential(handoff_pos[e])
                    if force:
                        forced_terminal_seen = True
                        reset_pending.append(e)
                        done_env[e] = True
                else:
                    prefix_obs_hash[e].add(_h(frames[e]))

            if emitter.ready():
                D.write_suffix_chunk(cdir, seq, emitter.flush(), episodes=[]); seq += 1
                mid_flush_seen = True
                drain_ingest()

            obs = nxt
            if reset_pending:
                obs = env.reset_slots(sorted(set(reset_pending)))   # forced terminal -> fresh obs
                reset_pending = []
            ticks += 1

        # flush any tail + final drain
        tail = emitter.flush()
        if tail is not None:
            D.write_suffix_chunk(cdir, seq, tail, episodes=[]); seq += 1
        drain_ingest()

        # -------------------- assertions --------------------
        checks = []
        def chk(name, cond, detail=""):
            checks.append({"name": name, "pass": bool(cond), "detail": str(detail)})
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + str(detail)) if detail else ''}", flush=True)

        handoffs = sum(1 for e in range(n) if h1[e] is not None)
        chk("reached >= 2 handoffs", handoffs >= 2, f"handoffs={handoffs}")
        chk("mid-suffix flush occurred", mid_flush_seen)
        chk("forced-reset terminal occurred", forced_terminal_seen)
        for e in range(n):
            ring = replay.rings[e]
            if h1[e] is None:
                chk(f"env{e} handoff", False, "no handoff"); continue
            # (1) first stored row = candidate H+1 obs + action
            first_ok = ring.size > 0 and _h(ring.obs[0]) == h1[e][0] and np.allclose(ring.action[0], h1[e][1])
            chk(f"env{e} first stored row is candidate H+1 obs+action", first_ok)
            # size equals the suffix steps that env emitted (no prefix rows)
            chk(f"env{e} ring size == suffix steps emitted", ring.size == suffix_count[e],
                f"ring={ring.size} suffix={suffix_count[e]}")
            # (2) NO prefix frame appears in the ring
            ring_hashes = {_h(ring.obs[i]) for i in range(ring.size)}
            chk(f"env{e} no prefix frame in ring", ring_hashes.isdisjoint(prefix_obs_hash[e]))
            # (4) a terminal boundary exists (the forced reset)
            chk(f"env{e} terminal boundary present", bool(ring.done[:ring.size].any()))
            # H+1 frame is fresh (non-black)
            chk(f"env{e} H+1 frame fresh (non-black)", float(np.asarray(ring.obs[0]).std()) > 1.0)

        # (5) valid 3-step samples from the composed ring
        if replay.ready(64, min_live_fraction=0.5):
            b = replay.sample(64)
            chk("valid 3-step samples (finite obs/reward/discount)",
                np.isfinite(b.reward).all() and np.isfinite(b.proprio).all()
                and np.isfinite(b.discount).all() and (b.discount >= 0).all())
        else:
            chk("replay warm enough to sample", False, f"len={len(replay)}")

        # champion SHA unchanged after the whole composed run
        chk("immutable champion SHA unchanged", pt.immutable_champion_ok(champ_path, sha0) == sha0)

        # real anchor sampler (holdout-excluding) in the same process
        if ANCHOR_DIR.exists():
            samp = pt.AnchorSampler(ANCHOR_DIR, pt.FROZEN_HOLDOUT_EPISODES, seed=0)
            fr, pr, ac = samp.sample(64)
            chk("anchor sampler excludes holdout + shapes ok",
                samp.excludes_holdout() and fr.shape[0] == 64 and ac.shape[1] == 7)
            # no-update DriftGate publication check: identity candidate must PASS the gate
            gate = pt.DriftGate(champion, ANCHOR_DIR, pt.FROZEN_HOLDOUT_EPISODES)
            gres = gate.check(candidate)
            chk("no-update DriftGate publication check: identity candidate passes",
                not gres["hard_stop"], f"driveL2p50={gres['drive_l2_p50']:.3f}")
        else:
            chk("anchor dir present", False, str(ANCHOR_DIR))

        # leave Φ produced valid values at the real handoff positions
        phis = [leave_phi[e] for e in range(n) if leave_phi[e] is not None]
        chk("leave Φ ∈ [-1,0] at real handoff positions", len(phis) >= 1 and all(-1.0 <= p <= 0.0 for p in phis),
            f"phis={[round(p,3) for p in phis]}")

        ok = all(c["pass"] for c in checks)
        out = {
            "pass": ok, "handoffs": handoffs, "seq_chunks": seq, "ticks": ticks,
            "champion_sha256": sha0, "gamma": {"learner": LEARNER_GAMMA, "pbrs": pt.STAGE_C_GAMMA},
            "leave_phi_config": {
                "formula": "Phi=-clip(d_graph(s,g*)/D,0,1); g* nearest legal gateway fixed at handoff; Phi=0 at terminal",
                "D_norm_m": pt.LeaveGeom().norm_distance_m, "gateway_y": -(pt.LeaveGeom().board_y - pt.LeaveGeom().gateway_margin_m),
                "gstar_per_env": {e: leave.gstar if leave_phi[e] is not None else None for e in range(n)},
            },
            "handoff_pos": handoff_pos, "leave_phi": leave_phi,
            "ring_sizes": {e: int(replay.rings[e].size) for e in range(n)},
            "suffix_counts": suffix_count, "checks": checks,
        }
        (PROJECT_ROOT / "runs/phase/smoke_prefix_takeover.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSMOKE_PREFIX_TAKEOVER {'PASS' if ok else 'FAIL'}  handoffs={handoffs} "
              f"chunks={seq} ticks={ticks}", flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
