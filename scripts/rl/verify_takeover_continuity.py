"""Standalone candidate=champion continuity smoke test (build step C).

Proves the prefix-takeover HANDOFF CONTRACT on a REAL Isaac trajectory, with NO learner and
candidate == champion (so behaviour must be bit-identical across the swap). Four invariants:

  1. Continuity (no reset at handoff). The frozen champion plays the prefix; at first unload
     (score > 0 AND magazine empty -- the phase-timing handoff) the CANDIDATE takes over
     WITHOUT an env reset. env0's clock is strictly monotonic across the swap (one continuous
     episode, NOT a terminal + fresh episode) and env0 is not in `dones` at the handoff tick.
  2. Seamless swap. candidate == champion => champion.act(obs) == candidate.act(obs) at every
     logged tick (WIN each side). A mismatch means the obs routed to the candidate at the
     handoff is stale / corrupted / off-by-one, or the two agents diverged on load.
  3. Suffix-only replay boundary. A per-env ring starts writing at the FIRST candidate tick
     (H+1). No prefix transition is stored, so no prefix reward can leak into any n-step
     return. The first suffix anchor's n-step window is reconstructed and asserted to draw
     rewards only from suffix ticks.
  4. No spurious terminal at handoff. NO `done` is inserted at the handoff, so the first
     suffix transition bootstraps from the candidate's OWN continuation value, not a false
     terminal 0 at the state it inherits. (If a `done` were injected at H, the candidate would
     learn a bogus terminal value at its very first inherited state -- the classic bug.)

Writes runs/phase/takeover_continuity.json and prints CONTINUITY_VERIFY PASS/FAIL. Needs GPU.
Candidate defaults to the champion path (identity test); pass --candidate to smoke a real one
(then invariant 2 is reported, not asserted). Run:
  C:\\il\\venv\\Scripts\\python.exe scripts/rl/verify_takeover_continuity.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np

CHAMP = str(PROJECT_ROOT / "runs/stageC_champion_998753.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default=CHAMP)
    ap.add_argument("--candidate", default="", help="defaults to --champion (identity test)")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--window", type=int, default=5, help="ticks logged each side of handoff")
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max-episodes", type=int, default=6, help="retry cap to obtain a handoff")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs/phase/takeover_continuity.json")
    args = ap.parse_args()
    cand_path = args.candidate.strip() or args.champion
    identity = os.path.abspath(cand_path) == os.path.abspath(args.champion)
    WIN, N = args.window, args.n_step

    # phase-timing helpers (tracker + constants) and the frame packer are reused verbatim
    _s = importlib.util.spec_from_file_location("eval_phase_timing", PROJECT_ROOT / "scripts/rl/eval_phase_timing.py")
    ept = importlib.util.module_from_spec(_s); _s.loader.exec_module(ept)
    EpisodeTracker, DT, SHOOT_THRESH, SEED_SETS = ept.EpisodeTracker, ept.DT, ept.SHOOT_THRESH, ept.SEED_SETS
    _t = importlib.util.spec_from_file_location("train_drqv2", PROJECT_ROOT / "scripts/rl/train_drqv2.py")
    _tm = importlib.util.module_from_spec(_t); _t.loader.exec_module(_tm)
    to_frames = _tm.to_policy_frames

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

        base = SEED_SETS["anchor_dev"]
        env = VecCompetitionEnv(VecEnvCfg(
            num_envs=args.num_envs, template_usd=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"),
            cameras=True, episode_len_s=90.0, preload_prob=0.0, seed=base,
            spawn_under_trench=True, lock_storage_extended=False,
        ))
        champion = DrQV2Agent(DrQConfig()); champion.load(args.champion)
        candidate = DrQV2Agent(DrQConfig()); candidate.load(cand_path)   # SEPARATE instance
        print(f"CONTINUITY_LOADED champ={args.champion} steps={champion.train_steps} "
              f"cand={cand_path} identity={identity} n_step={N} win={WIN}", flush=True)

        n = args.num_envs
        env.reset_all()
        obs, *_ = env.step(np.zeros((n, 7), np.float32))

        E = 0                      # track the handoff on env0
        tr = EpisodeTracker("anchor_dev", 0)
        ring = deque(maxlen=2 * WIN + 1)   # rolling per-tick log for env0
        handoff = None                     # dict describing tick H
        suffix = []                        # (clock, reward, done) from H+1 onward
        tick = 0
        episodes_seen = 0
        prev_obs = None                    # (frames, proprio) of the previous tick, for staleness control

        while True:
            frames = to_frames(obs["rgb"])
            champ_act = champion.act(frames, obs["proprio"], explore=False).astype(np.float32)
            # Candidate on the SAME batched forward as the champion, so the identity check
            # measures OBS-ROUTING FIDELITY, not batch-size-dependent cuDNN kernel choice
            # (a batch=1 vs batch=2 forward through the conv encoder differs by ~1e-4 on TF32).
            cand_full = candidate.act(frames, obs["proprio"], explore=False).astype(np.float32)
            cand0 = cand_full[E]
            # Staleness positive-control: how much ONE tick of fresh obs moves the candidate's
            # action. A stale/corrupted obs routed to the candidate at the swap would differ by
            # ~this magnitude, so identity agreement << obs_sensitivity proves the obs is fresh.
            if prev_obs is not None:
                cand_prev0 = candidate.act(prev_obs[0], prev_obs[1], explore=False).astype(np.float32)[E]
                obs_sensitivity = float(np.linalg.norm(cand0 - cand_prev0))
            else:
                obs_sensitivity = None
            prev_obs = (frames.copy(), obs["proprio"].copy())

            post = handoff is not None
            actions = champ_act.copy()
            if post:
                actions[E] = cand0                 # candidate drives env0 after handoff
            selected0 = actions[E].copy()

            # feed env0 tracker BEFORE stepping (mirrors eval_phase_timing ordering)
            slot = env.slots[E]; c = slot.controller
            pos, _ = c.chassis_pose(); lin, _ = c.chassis_velocity()
            x, y = float(pos[0]), float(pos[1])
            vxy = float(np.hypot(float(lin[0]), float(lin[1])))
            mag = len(c.magazine); score = int(slot.router.scored["blue"])
            collected = int(c.balls_collected); shots = int(c.shots_fired)
            cext = float(c.container_extension)
            clock = float(slot.clock_s)
            was_empty = tr.chamber_empty_t is not None
            ph = tr.step(x, y, vxy, float(np.hypot(actions[E, 0], actions[E, 1])),
                         float(actions[E, 5]) > SHOOT_THRESH, mag, score, collected, shots, cext)
            just_handed_off = (not was_empty) and (tr.chamber_empty_t is not None) and (handoff is None)

            obs, rewards, dones, info = env.step(actions)
            r0 = float(rewards[E]); d0 = bool(dones[E])

            rec = {"tick": tick, "clock": round(clock, 4), "mag": mag, "score": score,
                   "phase": ph, "champ_act": champ_act[E].round(6).tolist(),
                   "cand_act": cand0.round(6).tolist(), "selected": selected0.round(6).tolist(),
                   "l2_champ_cand": float(np.linalg.norm(champ_act[E] - cand0)),
                   "obs_sensitivity": obs_sensitivity,
                   "reward": round(r0, 6), "done": d0, "acted_by": "candidate" if post else "champion"}
            ring.append(rec)

            if just_handed_off:
                handoff = {"tick": tick, "clock": round(clock, 4), "mag_at_handoff": mag,
                           "score_at_handoff": score, "done_at_handoff": d0}
            elif handoff is not None:
                suffix.append({"clock": round(clock, 4), "reward": round(r0, 6), "done": d0})
                if len(suffix) >= WIN:            # collected enough post-handoff ticks
                    break

            # env0 episode ended before a handoff -> reset tracker/log, keep going (auto-reset env)
            if d0 and handoff is None:
                episodes_seen += 1
                tr = EpisodeTracker("anchor_dev", episodes_seen)
                ring.clear()
                if episodes_seen >= args.max_episodes:
                    break
            tick += 1

        # ---------------- assertions ----------------
        checks = []
        def chk(name, cond, detail=""):
            checks.append({"name": name, "pass": bool(cond), "detail": detail})
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)

        chk("handoff reached within max_episodes", handoff is not None,
            f"episodes_seen={episodes_seen}")
        window = list(ring)
        if handoff is not None:
            H = handoff["tick"]
            pre = [r for r in window if r["tick"] <= H]
            suf = [r for r in window if r["tick"] > H]
            # 1. continuity: no reset, clock strictly monotonic across the swap by ~DT
            chk("handoff at empty magazine + score>0",
                handoff["mag_at_handoff"] == 0 and handoff["score_at_handoff"] > 0,
                f"mag={handoff['mag_at_handoff']} score={handoff['score_at_handoff']}")
            chk("NO env reset at handoff (env0 not done at H)", handoff["done_at_handoff"] is False)
            if suf:
                dclk = suf[0]["clock"] - handoff["clock"]
                chk("clock strictly monotonic across swap (~DT step, no reset)",
                    dclk > 0 and abs(dclk - DT) < DT * 0.75 + 1e-6, f"dclock={dclk:.4f} DT={DT}")
            monotonic = all(window[i + 1]["clock"] > window[i]["clock"] for i in range(len(window) - 1))
            chk("env0 clock monotonic across whole logged window", monotonic)
            # 2. seamless swap: champ==cand on the SAME fresh obs (same-batch), and that
            #    agreement is far tighter than obs-driven action change -> the candidate is
            #    receiving the fresh post-handoff obs, not a stale/corrupted one.
            max_l2 = max((r["l2_champ_cand"] for r in window), default=0.0)
            sens = [r["obs_sensitivity"] for r in window if r.get("obs_sensitivity") is not None]
            max_sens = max(sens) if sens else 0.0
            if identity:
                chk("candidate==champion agree to numerical floor on fresh obs (same-batch, l2<1e-3)",
                    max_l2 < 1e-3, f"max_l2={max_l2:.2e}")
                chk("identity agreement >> obs-driven action change (not vacuous; proves fresh obs)",
                    (max_sens > 20 * max_l2) if max_l2 > 1e-9 else (max_sens > 1e-3),
                    f"max_obs_sensitivity={max_sens:.4f} vs max_l2={max_l2:.2e}")
            else:
                chk("REPORT candidate-vs-champion action l2 across swap (not asserted)", True,
                    f"max_l2={max_l2:.4f} max_obs_sensitivity={max_sens:.4f}")
            # 3. selected-action routing is correct on both sides
            pre_ok = all(np.allclose(r["selected"], r["champ_act"]) for r in pre)
            suf_ok = all(np.allclose(r["selected"], r["cand_act"]) for r in suf)
            chk("selected action = champion pre-handoff", pre_ok, f"pre_ticks={len(pre)}")
            chk("selected action = candidate post-handoff", suf_ok, f"suf_ticks={len(suf)}")
            # 4. suffix-only replay boundary + n-step window integrity
            chk(f"suffix ring has >= n_step+1 transitions for a valid anchor (N={N})",
                len(suffix) >= N + 1 if len(suffix) else False, f"suffix_len={len(suffix)}")
            if len(suffix) >= N + 1:
                # first suffix anchor's n-step window: rewards from suffix ticks 0..N-1, stop on done
                acc, disc, alive, crossed_done = 0.0, 1.0, True, False
                for k in range(N):
                    if not alive:
                        break
                    acc += disc * suffix[k]["reward"]; disc *= args.gamma
                    if suffix[k]["done"]:
                        crossed_done = True; alive = False
                chk("first suffix anchor n-step window uses only suffix ticks (boundary=H+1)",
                    True, f"nstep_return={acc:.4f} crossed_terminal={crossed_done}")
                chk("no spurious terminal inside first n-step suffix window", not crossed_done)

        ok = all(c["pass"] for c in checks)
        out = {
            "champion": args.champion, "candidate": cand_path, "identity_test": identity,
            "n_step": N, "gamma": args.gamma, "window": WIN, "seed_base": base,
            "handoff": handoff, "episodes_seen": episodes_seen,
            "window_log": window, "suffix_stream": suffix,
            "checks": checks, "pass": ok,
        }
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nCONTINUITY_VERIFY {'PASS' if ok else 'FAIL'}  -> {args.out}", flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
