"""Bounded leave-only prefix-takeover pilot.

Single-process on the laptop: one Isaac collector-of-2-envs + an in-process learner.
The frozen champion plays the prefix DETERMINISTICALLY; at first unload (score>0 ∧
magazine empty) the CANDIDATE takes over the suffix, exploring, and is trained by
``update_suffix`` (frozen encoder + normalized TD3+BC anchor, α held at 1.0) on a
per-env leave-Φ-shaped suffix replay. Everything routes through the VERIFIED production
transport (SuffixEmitter → write_suffix_chunk → SuffixIngestor → PerEnvReplay).

Gates (design note): full frozen-holdout DriftGate before every publication at 250 actor
updates — warning ⇒ hold last safe checkpoint, confirmed hard-stop ⇒ terminate+archive;
the immutable champion SHA-256 is re-checked each publish and never overwritten. Bounded
to --max-updates (default 5000). A gate report is appended to gate_reports.jsonl at every
250-update gate so the FIRST gate can be read immediately.

Success-to-continue (NOT auto-applied): leave ≥ 80% ∧ no drift warning ∧ champion hash
unchanged ∧ no replay-boundary violation. Leave milestone = a legal outbound crossing of
the board (|y|=2.775) within 12 s of handoff. NO automatic advance to collect-8.

Run: C:\\il\\venv\\Scripts\\python.exe scripts/rl/pilot_leave_only.py
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np

LEARNER_GAMMA = 0.999
BOARD_Y = 2.775
FINAL_LEAVE_TARGET_S = 12.0
DT = 0.1                     # policy step (10 Hz) -> 12 s = 120 ticks
ANCHOR_DIR = PROJECT_ROOT / "runs/phase/stageC_phase_timing_anchor_dev32_anchors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs/pilot_leave_only")
    ap.add_argument("--candidate", default="",
                    help="optional last-safe candidate checkpoint to resume; default starts from champion")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--max-updates", type=int, default=5000)
    ap.add_argument("--gate-updates", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--anchor-batch", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--leave-weight", type=float, default=1.0,
                    help="multiplier on the leave-Φ PBRS term. 1.0 keeps the telescoping "
                         "policy-invariance; >1.0 is a deliberate (non-invariant) leave-discovery "
                         "lever for the 4-GPU sweep, NOT the final reward.")
    ap.add_argument("--ramp-success-reward", type=float, default=3.0)
    ap.add_argument("--trench-success-reward", type=float, default=1.0)
    ap.add_argument("--leave-failure-penalty", type=float, default=2.0)
    ap.add_argument(
        "--ramp-only-curriculum", action="store_true",
        help="temporary discovery phase: a trench crossing ends as a failed "
             "segment; normal/final mode keeps trenches as legal fallbacks",
    )
    ap.add_argument(
        "--leave-window-s", type=float, default=FINAL_LEAVE_TARGET_S,
        help="maximum suffix rollout before a forced reset. Use a longer window "
             "for discovery, then tighten toward the fixed 12 s final target.",
    )
    ap.add_argument(
        "--deterministic-suffix", action="store_true",
        help="disable candidate exploration noise for a read-only held-out behavior "
             "run (training defaults to exploratory suffix actions)",
    )
    ap.add_argument("--chunk-steps", type=int, default=12)
    ap.add_argument("--warmup-transitions", type=int, default=2000)
    ap.add_argument("--replay-capacity-per-env", type=int, default=8000,
                    help="uint8 9x90x160 frames are ~126 KB each; 8000/env ~= 1 GB/ring "
                         "(the pilot collects only ~5k suffix transitions total)")
    ap.add_argument("--seed", type=int, default=90101)
    ap.add_argument("--minutes", type=float, default=60.0)
    args = ap.parse_args()
    if not np.isfinite(args.leave_window_s) or not (
        FINAL_LEAVE_TARGET_S <= args.leave_window_s <= 90.0
    ):
        ap.error(
            f"--leave-window-s must be finite and in "
            f"[{FINAL_LEAVE_TARGET_S}, 90.0]"
        )
    leave_rewards = (
        args.ramp_success_reward,
        args.trench_success_reward,
        args.leave_failure_penalty,
    )
    if not all(np.isfinite(v) and v >= 0.0 for v in leave_rewards):
        ap.error("leave terminal rewards/penalty must be finite and non-negative")
    if args.ramp_success_reward <= args.trench_success_reward:
        ap.error("--ramp-success-reward must exceed --trench-success-reward")
    args.out.mkdir(parents=True, exist_ok=True)
    gate_path = args.out / "gate_reports.jsonl"

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.replay import PerEnvReplay
        from frc_rebuilt.rl import distributed as D
        from frc_rebuilt.rl import prefix_takeover as pt

        champ_path = pt.CHAMPION_PATH
        sha0 = pt.immutable_champion_ok(champ_path)
        pt.assert_pbrs_gamma(pt.STAGE_C_GAMMA, LEARNER_GAMMA)
        shaper = pt.PBRSShaper(gamma=LEARNER_GAMMA)
        alpha = pt.check_alpha(args.alpha)
        n = args.num_envs

        env = VecCompetitionEnv(VecEnvCfg(
            num_envs=n, template_usd=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"),
            cameras=True, episode_len_s=90.0, preload_prob=0.0, seed=args.seed,
            spawn_under_trench=True, lock_storage_extended=False, collect_reward_weight=0.3,
            empty_own_court_penalty=0.0,
        ))
        champion = DrQV2Agent(DrQConfig()); champion.load(champ_path)     # frozen, deterministic
        candidate = DrQV2Agent(DrQConfig())
        if args.candidate:
            candidate.load(args.candidate)                                # resume a gate-passed candidate
        else:
            candidate.load(champ_path)                                    # first pilot starts from champion
            candidate.explore_offset = candidate.train_steps              # re-warm suffix stddev -> 1.0
        print(f"PILOT_LOADED champ_sha={sha0[:12]} gamma={LEARNER_GAMMA} alpha={alpha} "
              f"cand_stddev0={candidate.stddev():.3f} candidate={args.candidate or champ_path} "
              f"candidate_train_steps={candidate.train_steps}", flush=True)

        emitter = pt.SuffixEmitter(collector_envs=n, chunk_steps=args.chunk_steps)
        replay = PerEnvReplay(num_envs=n, capacity_per_env=args.replay_capacity_per_env, seed=args.seed + 5,
                              obs_shape=(9, 90, 160), proprio_dim=22, privileged_dim=26,
                              action_dim=7, n_step=3, gamma=LEARNER_GAMMA)
        ingestor = D.SuffixIngestor(replay, collector_envs=n)
        cdir = D.collector_dir(Path(tempfile.mkdtemp()), 0)
        sampler = pt.AnchorSampler(ANCHOR_DIR, pt.FROZEN_HOLDOUT_EPISODES, seed=args.seed)
        gate = pt.DriftGate(champion, ANCHOR_DIR, pt.FROZEN_HOLDOUT_EPISODES)
        leave = [pt.LeavePotential("blue") for _ in range(n)]             # PER-ENV g* (design note)

        def pos_xy(e):
            p, _ = env.slots[e].controller.chassis_pose()
            return (float(p[0]), float(p[1]))

        env.reset_all()
        obs, *_ = env.step(np.zeros((n, 7), np.float32))

        seq = 0; consumed: set = set()
        updates = 0; transitions = 0; seq_written = 0
        handoffs = 0; leaves = 0
        ramp_leaves = 0; trench_leaves = 0; invalid_crossings = 0
        rejected_trench_crossings = 0
        board_cross_anyx = 0             # DIAGNOSTIC: crossed the board at ANY x in-window
        any_cross_flag = [False] * n     # (looser than the strict gateway leave metric)
        pending_handoff = [None] * n     # tick at which env e handed off (for the 12 s window)
        left_flag = [False] * n
        no_path_steps = 0
        boundary_violation = False
        gate_idx = 0
        last_safe = args.out / "candidate_last_safe.pt"
        terminated = None
        deadline = time.time() + args.minutes * 60.0
        ticks = 0

        def do_gate():
            nonlocal gate_idx, terminated
            gate_idx += 1
            r = gate.check(candidate)
            champ_ok = pt.immutable_champion_ok(champ_path, sha0) == sha0
            leave_rate = (leaves / handoffs) if handoffs else None
            rep = {
                "gate": gate_idx, "updates": updates, "transitions": transitions,
                "handoffs": handoffs, "leaves": leaves,
                "ramp_leaves": ramp_leaves, "trench_leaves": trench_leaves,
                "invalid_crossings": invalid_crossings,
                "rejected_trench_crossings": rejected_trench_crossings,
                "board_cross_anyx": board_cross_anyx,
                "leave_rate": round(leave_rate, 3) if leave_rate is not None else None,
                "leave_window_s": float(args.leave_window_s),
                "final_leave_target_s": FINAL_LEAVE_TARGET_S,
                "deterministic_suffix": bool(args.deterministic_suffix),
                "ramp_only_curriculum": bool(args.ramp_only_curriculum),
                "legal_lane_x": list(pt.LeaveGeom().lane_x),
                "drift": {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
                "hard_stop": r["hard_stop"], "warning": r["warning"],
                "champion_hash_unchanged": champ_ok, "replay_boundary_violation": boundary_violation,
                "no_path_shaping_steps": no_path_steps,
                "cand_stddev": round(float(candidate.stddev()), 4),
                "wall": time.strftime("%H:%M:%S"),
            }
            with gate_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rep) + "\n")
            print("PILOT_GATE " + json.dumps(rep), flush=True)
            if r["hard_stop"] or not champ_ok:
                terminated = "hard_stop" if r["hard_stop"] else "champion_changed"
                # archive the offending candidate for the drift post-mortem
                if candidate.weights_finite():
                    candidate.save(str(args.out / f"candidate_HARDSTOP_gate{gate_idx}.pt"))
            elif r["warning"]:
                # HOLD means stop optimizing the unpublished live candidate and
                # retain the previous last-safe checkpoint for review.
                terminated = "warning_hold"
                if candidate.weights_finite():
                    candidate.save(str(args.out / f"candidate_WARNING_gate{gate_idx}.pt"))
            else:
                if candidate.weights_finite():         # publish the gate-passed checkpoint
                    candidate.save(str(last_safe))

        while updates < args.max_updates and terminated is None and time.time() < deadline:
            frames = D.to_policy_frames(obs["rgb"])
            champ_act = champion.act(frames, obs["proprio"], explore=False).astype(np.float32)
            actions = champ_act.copy()
            for e in range(n):
                if emitter.in_suffix(e):
                    actions[e] = candidate.act(frames[e:e+1], obs["proprio"][e:e+1],
                                               explore=not args.deterministic_suffix).astype(np.float32)[0]

            unloaded = np.zeros(n, bool)
            cur_pos = []
            for e in range(n):
                c = env.slots[e].controller
                unloaded[e] = (int(env.slots[e].router.scored["blue"]) > 0 and len(c.magazine) == 0)
                cur_pos.append(pos_xy(e))

            nxt, rewards, dones, info = env.step(actions)
            nxt_pos = [pos_xy(e) for e in range(n)]
            reset_indices = []

            for e in range(n):
                was_suffix = emitter.in_suffix(e)
                segment_success = False
                segment_timeout = False
                segment_no_path = False
                segment_invalid_crossing = False
                segment_rejected_trench = False
                crossing_kind = None
                if was_suffix and pending_handoff[e] is not None:
                    gstar = leave[e].gstar
                    # Clear the hysteresis band THROUGH the selected fixed legal
                    # lane.  g* remains fixed for shaping, but success accepts any
                    # physical ramp/trench lane the policy actually traverses.
                    crossing_kind = leave[e].crossing_kind(nxt_pos[e])
                    segment_rejected_trench = bool(
                        gstar is not None
                        and args.ramp_only_curriculum
                        and crossing_kind == "trench"
                    )
                    segment_success = bool(
                        gstar is not None
                        and crossing_kind is not None
                        and not segment_rejected_trench
                    )
                    clear_crossing = (
                        nxt_pos[e][1] >= -2.50
                        if leave[e].alliance == "blue"
                        else nxt_pos[e][1] <= 2.50
                    )
                    segment_invalid_crossing = bool(
                        gstar is not None and clear_crossing and crossing_kind is None
                    )
                    segment_timeout = (
                        (ticks - pending_handoff[e]) * DT >= args.leave_window_s
                    )
                    segment_no_path = gstar is None
                segment_end = bool(
                    segment_success or segment_timeout or segment_no_path
                    or segment_invalid_crossing or segment_rejected_trench
                )
                forced_segment_reset = segment_end and not bool(dones[e])
                # DIAGNOSTIC: crossed the board at ANY x within the window (looser than the
                # strict gateway leave metric) -> distinguishes "never crosses" from "crosses
                # but not near the selected gateway".
                if pending_handoff[e] is not None and not any_cross_flag[e] and nxt_pos[e][1] > -BOARD_Y:
                    board_cross_anyx += 1; any_cross_flag[e] = True
                shaped = float(rewards[e])
                if was_suffix and leave[e].gstar is not None:      # leave-Φ PBRS on suffix rewards
                    phi_s = leave[e].potential(cur_pos[e])
                    phi_n = 0.0 if (bool(dones[e]) or segment_end) else leave[e].potential(nxt_pos[e])
                    if phi_s is None or phi_n is None:
                        no_path_steps += 1                          # no-path -> no shaping (never NaN)
                    else:
                        shaped += args.leave_weight * shaper.shaped(phi_s, phi_n, bool(dones[e]) or segment_end)
                if segment_success:
                    shaped += (
                        args.ramp_success_reward
                        if crossing_kind == "ramp"
                        else args.trench_success_reward
                    )
                elif (
                    segment_timeout or segment_no_path or segment_invalid_crossing
                    or segment_rejected_trench
                ):
                    shaped -= args.leave_failure_penalty
                emitter.observe(e, frames[e], obs["proprio"][e], obs["privileged"][e],
                                actions[e], shaped, unloaded=bool(unloaded[e]), done=bool(dones[e]),
                                forced_reset=forced_segment_reset)
                # Handoff bookkeeping: the arm tick (H) selects g* and opens the
                # configured discovery window.  Promotion still requires the fixed
                # 12 s target in a later held-out evaluation/tightening stage.
                if (not was_suffix) and unloaded[e] and not bool(dones[e]) and pending_handoff[e] is None:
                    # this tick armed suffix (H); next tick is H+1 -> record handoff now
                    handoffs += 1
                    pending_handoff[e] = ticks
                    left_flag[e] = False
                    any_cross_flag[e] = False
                    leave[e].reset_segment()
                    leave[e].select_gateway(cur_pos[e])
                # End the leave-only segment at success, timeout, or no-path so
                # post-milestone collection cannot contaminate leave-only replay.
                if segment_end:
                    if segment_success:
                        leaves += 1
                        left_flag[e] = True
                        if crossing_kind == "ramp":
                            ramp_leaves += 1
                        else:
                            trench_leaves += 1
                    elif segment_invalid_crossing:
                        invalid_crossings += 1
                    elif segment_rejected_trench:
                        rejected_trench_crossings += 1
                    pending_handoff[e] = None
                    if forced_segment_reset:
                        reset_indices.append(e)
                if bool(dones[e]):
                    pending_handoff[e] = None                        # episode reset
                    leave[e].reset_segment()

            if reset_indices:
                # Fresh rendered observations must replace the pre-reset batch.
                nxt = env.reset_slots(reset_indices)
                for e in reset_indices:
                    leave[e].reset_segment()
                    left_flag[e] = False
                if any(env.slots[e].forced_reset_settle_s > 0.5 for e in range(n)):
                    boundary_violation = True

            if emitter.ready():
                D.write_suffix_chunk(cdir, seq, emitter.flush(), episodes=[]); seq += 1; seq_written += 1
                for ch in D.drain_suffix_chunks(cdir.parent, 1, consumed):
                    res = ingestor.ingest(ch)
                    transitions += res.get("added", 0)
                    if res.get("invalid_stream") or res.get("rejected", 0):
                        boundary_violation = True

            # learn: one update per collector tick once warm
            if replay.ready(max(args.batch_size, args.warmup_transitions), min_live_fraction=0.5):
                a_obs, a_pro, a_act = sampler.sample(args.anchor_batch)
                m = candidate.update_suffix(replay.sample(args.batch_size), a_obs, a_pro, a_act,
                                            alpha=alpha, freeze_encoder=True)
                updates += 1
                if not np.isfinite(m.get("critic_loss", 0.0)):
                    pass
                if updates % args.gate_updates == 0:
                    do_gate()

            obs = nxt
            ticks += 1
            if ticks % 200 == 0:
                print(f"PILOT_PROGRESS tick={ticks} tx={transitions} updates={updates} "
                      f"handoffs={handoffs} leaves={leaves} xcross={board_cross_anyx} "
                      f"replay={len(replay)} stddev={float(candidate.stddev()):.3f}", flush=True)

        # final state
        final = {
            "terminated": terminated, "updates": updates, "transitions": transitions,
            "handoffs": handoffs, "leaves": leaves,
            "ramp_leaves": ramp_leaves, "trench_leaves": trench_leaves,
            "invalid_crossings": invalid_crossings,
            "rejected_trench_crossings": rejected_trench_crossings,
            "leave_rate": round(leaves / handoffs, 3) if handoffs else None,
            "gates": gate_idx, "champion_sha256": sha0,
            "candidate_train_steps": int(candidate.train_steps),
            "leave_window_s": float(args.leave_window_s),
            "final_leave_target_s": FINAL_LEAVE_TARGET_S,
            "deterministic_suffix": bool(args.deterministic_suffix),
            "ramp_only_curriculum": bool(args.ramp_only_curriculum),
            "champion_hash_unchanged": pt.immutable_champion_ok(champ_path, sha0) == sha0,
            "no_path_shaping_steps": no_path_steps, "seq_chunks": seq_written,
        }
        (args.out / "pilot_final.json").write_text(json.dumps(final, indent=2))
        print("PILOT_DONE " + json.dumps(final), flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
