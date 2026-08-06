"""Distributed DrQ-v2 collector: render one env-set on one GPU, push transitions.

Runs inference only (no learning): loads the newest actor+encoder weights the
learner publishes, steps the full-physics vision env, and drops transition
chunks onto tmpfs for the learner to drain. Launch one per GPU with
CUDA_VISIBLE_DEVICES set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collector-id", type=int, required=True)
    ap.add_argument("--root", default="/dev/shm/frc_dist")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--stage", choices=("A", "B", "C"), default="C")
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_96.usd"))
    ap.add_argument("--episode-len-s", type=float, default=90.0)
    ap.add_argument("--preload-prob", type=float, default=0.4)
    ap.add_argument("--spawn-under-trench", action="store_true",
                    help="stage C: spawn compact under the trench (implies storage unlocked)")
    ap.add_argument("--mask-illegal-fire", action="store_true",
                    help="illegal fire/ferry is a no-op (matches the trainer's Stage-C env)")
    ap.add_argument("--chunk-steps", type=int, default=12)
    ap.add_argument("--weight-reload-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stddev-end", type=float, default=0.2,
                    help="exploration-noise floor (Stage C contract 0.2)")
    ap.add_argument("--collect-weight", type=float, default=0.3,
                    help="Stage-C collect reward weight; MUST match the resumed champion's "
                         "training weight (0.3, constant). The vec_env default 1.5 is the "
                         "fresh-agent teaching value and poisons a resumed replay.")
    ap.add_argument("--rho-score", type=float, default=1.0,
                    help="custody discount for RE-scoring the same ball (reward-first "
                         "fine-tune, design note). 1.0 = original raw reward; 0.2 = anti-recycle.")
    ap.add_argument("--rho-collect", type=float, default=1.0,
                    help="custody discount for RE-collecting a previously-scored ball.")
    ap.add_argument("--phase-pbrs", action="store_true",
                    help="second-cycle phase-PBRS shaping (design note): reward progress through "
                         "the leave->collect->return->unload cycle after the first unload.")
    ap.add_argument("--pbrs-weight", type=float, default=1.0,
                    help="multiplier on the phase-PBRS term (1.0 = policy-invariant).")
    ap.add_argument("--pbrs-gamma", type=float, default=0.999,
                    help="gamma for F=g*Phi'-Phi; MUST equal the learner/replay gamma.")
    # aggressive multi-cycle reward (user directive 2026-07-13; alternative to --phase-pbrs)
    ap.add_argument("--aggressive-cycle", action="store_true",
                    help="DIRECT aggressive multi-cycle reward: full-clear bonus + gated "
                         "abandon penalty + cross-over linger + escalating ordered-cycle bonus. "
                         "Non-invariant (biases the policy). Use INSTEAD of --phase-pbrs.")
    ap.add_argument("--score-floor", type=float, default=25.0)
    ap.add_argument("--atonce-weight", type=float, default=1.0)
    ap.add_argument("--atonce-cap", type=float, default=50.0)
    ap.add_argument("--atonce-min-load", type=int, default=4)
    ap.add_argument("--abandon-load", type=int, default=8)
    ap.add_argument("--abandon-weight", type=float, default=0.3)
    ap.add_argument("--linger-penalty", type=float, default=0.35)
    ap.add_argument("--linger-grace", type=int, default=5)
    ap.add_argument("--leave-bonus", type=float, default=15.0)
    ap.add_argument("--collect-bonus", type=float, default=15.0)
    ap.add_argument("--return-bonus", type=float, default=15.0)
    ap.add_argument("--mc-per-ball", type=float, default=2.0)
    ap.add_argument("--mc-cap", type=float, default=40.0)
    ap.add_argument("--mc-escalation", type=float, default=1.2)
    ap.add_argument("--mc-episode-cap", type=float, default=150.0)
    ap.add_argument("--mc-min-cycle-score", type=int, default=2)
    ap.add_argument("--neutral-deep-y", type=float, default=-1.5)
    # Term D — bounded time-decay ramps (2026-07-14). Defaults are no-op (slopes/pref 0,
    # deadline off); pass live values to enable the 2nd-cycle time pressure + ramp pref.
    ap.add_argument("--core-slope", type=float, default=0.0)
    ap.add_argument("--core-step-cap", type=float, default=0.50)
    ap.add_argument("--core-grace-steps", type=int, default=8)
    ap.add_argument("--core-freeze-confirm", type=int, default=15)
    ap.add_argument("--core-freeze-cap", type=int, default=30)
    ap.add_argument("--arm-deadline-steps", type=int, default=10**9)
    ap.add_argument("--rampB-slope", type=float, default=0.0)
    ap.add_argument("--rampB-step-cap", type=float, default=0.25)
    ap.add_argument("--rampB-grace", type=int, default=30)
    ap.add_argument("--rampB-budget", type=float, default=10.0)
    ap.add_argument("--rampC-slope", type=float, default=0.0)
    ap.add_argument("--rampC-step-cap", type=float, default=0.20)
    ap.add_argument("--rampC-grace", type=int, default=40)
    ap.add_argument("--rampC-budget", type=float, default=8.0)
    ap.add_argument("--ramp-episode-cap", type=float, default=60.0)
    ap.add_argument("--ramp-pref", type=float, default=0.0)
    ap.add_argument("--ramp-center", type=float, default=1.55)
    ap.add_argument("--ramp-tol", type=float, default=0.9)
    # 2nd-cycle curriculum: relocate N balls to a shallow-neutral cluster on P of episodes
    ap.add_argument("--neutral-refill-count", type=int, default=0)
    ap.add_argument("--neutral-refill-prob", type=float, default=0.0)
    ap.add_argument("--neutral-loaded-prob", type=float, default=0.0,
                    help="fraction of episodes that start in neutral, LOADED (return->shoot curriculum)")
    ap.add_argument("--empty-own-court-penalty", type=float, default=None,
                    help="override the env empty-own-court penalty (default 0.02); set 0.0 "
                         "with --aggressive-cycle since Term B supersedes it (no double count).")
    ap.add_argument("--minutes", type=float, default=600.0)
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import torch  # noqa: F401
        from frc_rebuilt.rl import distributed as D
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        cdir = D.collector_dir(args.root, args.collector_id)
        wdir = D.weights_dir(args.root)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                preload_prob=args.preload_prob if args.stage in ("B", "C") else 0.0,
                spawn_under_trench=bool(args.spawn_under_trench),
                lock_storage_extended=not args.spawn_under_trench,
                mask_illegal_fire=bool(args.mask_illegal_fire),
                collect_reward_weight=float(args.collect_weight),
                rho_score=float(args.rho_score),
                rho_collect=float(args.rho_collect),
                empty_own_court_penalty=(0.02 if args.empty_own_court_penalty is None
                                         else float(args.empty_own_court_penalty)),
                neutral_refill_count=int(args.neutral_refill_count),
                neutral_refill_prob=float(args.neutral_refill_prob),
                neutral_loaded_prob=float(args.neutral_loaded_prob),
                seed=args.seed,
            )
        )
        if not bool(getattr(env, "_camera_ready", False)):
            raise RuntimeError(
                "collector camera initialization failed; refusing to publish black frames"
            )
        n = args.num_envs
        obs, _, _, _ = env.step(np.zeros((n, 7), np.float32))
        initial_camera_std = obs["rgb"].std(axis=(2, 3, 4))
        if bool((initial_camera_std <= 1.0).any()):
            bad = np.argwhere(initial_camera_std <= 1.0).tolist()
            raise RuntimeError(f"collector camera black frames at startup: {bad}")
        frames = D.to_policy_frames(obs["rgb"])
        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=frames.shape[1],
                frame_h=frames.shape[2],
                frame_w=frames.shape[3],
                proprio_dim=obs["proprio"].shape[1],
                privileged_dim=obs["privileged"].shape[1],
                stddev_end=args.stddev_end,
            )
        )

        # wait for the learner's first weights before acting
        for _ in range(600):
            got = D.latest_weights(wdir)
            if got:
                break
            time.sleep(1.0)
        loaded_step = -1

        def maybe_reload():
            nonlocal loaded_step
            got = D.latest_weights(wdir)
            if not got:
                return
            path, step = got
            if step == loaded_step:
                return
            try:
                blob = torch.load(path, map_location=agent.device)
                agent.encoder.load_state_dict(blob["encoder"])
                agent.actor.load_state_dict(blob["actor"])
                agent.train_steps = int(blob.get("train_steps", agent.train_steps))
                # restore the exploration anchor so stddev() matches the learner's
                # re-warmed schedule; older blobs without it fall back (audit).
                agent.explore_offset = int(blob.get("explore_offset", agent.explore_offset))
                loaded_step = step
            except Exception as exc:
                print(f"COLLECTOR{args.collector_id} weight reload skipped: {exc}", flush=True)

        maybe_reload()
        print(f"COLLECTOR{args.collector_id} READY on {agent.device}, "
              f"frames={list(frames.shape[1:])} collect_weight={args.collect_weight}", flush=True)

        deadline = time.time() + args.minutes * 60.0
        seq = 0
        step = 0
        buf: dict[str, list] = {k: [] for k in D.FIELD_KEYS}
        ep_return = np.zeros(n, np.float32)
        ep_score = np.zeros(n, np.float32)
        ep_collect = np.zeros(n, np.float32)
        cyc_completed_ep = np.zeros(n, int)   # aggressive-cycle: completed ordered cycles / ep
        route_ep: list[list] = [[] for _ in range(n)]   # per-slot downsampled path for the trace gallery
        pending_eps: list[dict] = []
        black_frame_streak = 0

        # Second-cycle phase-PBRS rewards progress through the
        # leave->collect->return->unload cycle AFTER the first unload. F = pbrs_gamma*Phi(s')
        # - Phi(s) is added to the RAW score reward in the training buffer only; the
        # reported ep_score/ep_collect components stay raw. One shared occupancy grid.
        cyc = None
        no_path_pbrs = 0
        if args.phase_pbrs:
            from frc_rebuilt.field_map import OccupancyGrid
            from frc_rebuilt.rl.prefix_takeover import CyclePotential, assert_pbrs_gamma
            assert_pbrs_gamma(args.pbrs_gamma, args.pbrs_gamma)  # finiteness/consistency guard
            _grid = OccupancyGrid()
            cyc = [CyclePotential("blue", grid=_grid) for _ in range(n)]
            print(f"COLLECTOR{args.collector_id} phase-PBRS ON weight={args.pbrs_weight} "
                  f"gamma={args.pbrs_gamma}", flush=True)

        # aggressive multi-cycle reward (user directive): direct, non-invariant terms added
        # to the RAW reward in the training buffer only (ep_score/ep_collect stay raw).
        shp = None
        if args.aggressive_cycle:
            from frc_rebuilt.rl.prefix_takeover import AggressiveCycleCfg, AggressiveCycleShaper
            _acfg = AggressiveCycleCfg(
                score_floor=args.score_floor, atonce_weight=args.atonce_weight,
                atonce_cap=args.atonce_cap, atonce_min_load=args.atonce_min_load,
                abandon_load=args.abandon_load, abandon_weight=args.abandon_weight,
                linger_penalty=args.linger_penalty, linger_grace_steps=args.linger_grace,
                leave_bonus=args.leave_bonus, collect_bonus=args.collect_bonus,
                return_bonus=args.return_bonus,
                mc_per_ball=args.mc_per_ball, mc_cap=args.mc_cap,
                mc_escalation=args.mc_escalation, mc_episode_cap=args.mc_episode_cap,
                mc_min_cycle_score=args.mc_min_cycle_score, neutral_deep_y=args.neutral_deep_y,
                core_slope=args.core_slope, core_step_cap=args.core_step_cap,
                core_grace_steps=args.core_grace_steps, core_freeze_confirm=args.core_freeze_confirm,
                core_freeze_cap=args.core_freeze_cap, arm_deadline_steps=args.arm_deadline_steps,
                rampB_slope=args.rampB_slope, rampB_step_cap=args.rampB_step_cap,
                rampB_grace=args.rampB_grace, rampB_budget=args.rampB_budget,
                rampC_slope=args.rampC_slope, rampC_step_cap=args.rampC_step_cap,
                rampC_grace=args.rampC_grace, rampC_budget=args.rampC_budget,
                ramp_episode_cap=args.ramp_episode_cap, ramp_pref=args.ramp_pref,
                ramp_center=args.ramp_center, ramp_tol=args.ramp_tol)
            shp = [AggressiveCycleShaper(_acfg) for _ in range(n)]
            cyc_tel_path = PROJECT_ROOT / "runs" / "cycle_telemetry.jsonl"   # live cycle monitor feed
            cyc_tel_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"COLLECTOR{args.collector_id} AGGRESSIVE-CYCLE ON floor={args.score_floor} "
                  f"atonce_w={args.atonce_weight}/cap{args.atonce_cap} abandon={args.abandon_weight}"
                  f"@{args.abandon_load} linger={args.linger_penalty} mc={args.mc_per_ball}/ball"
                  f"x{args.mc_escalation} empty_court_pen={args.empty_own_court_penalty}", flush=True)

        def _pos(i):
            p, _ = env.slots[i].controller.chassis_pose()
            return (float(p[0]), float(p[1]))

        while time.time() < deadline:
            actions = agent.act(frames, obs["proprio"], explore=True).astype(np.float32)
            # Phi(s) BEFORE stepping (env.slots reflect the current state)
            phi_cur = None
            if cyc is not None:
                phi_cur = []
                for i in range(n):
                    pc = _pos(i)
                    cyc[i].update_leg(pc, len(env.slots[i].controller.magazine))
                    phi_cur.append(cyc[i].potential(pc))
            next_obs, rewards, dones, info = env.step(actions)
            camera_std = next_obs["rgb"].std(axis=(2, 3, 4))
            black_frame_streak = (
                black_frame_streak + 1
                if bool((camera_std <= 1.0).any())
                else 0
            )
            if black_frame_streak >= 10:
                bad = np.argwhere(camera_std <= 1.0).tolist()
                raise RuntimeError(
                    f"collector cameras stayed black for {black_frame_streak} steps: {bad}"
                )
            next_frames = D.to_policy_frames(next_obs["rgb"])
            # add the phase-PBRS term to the (raw) reward before buffering for training
            if cyc is not None:
                for i in range(n):
                    slot = env.slots[i]
                    mag = len(slot.controller.magazine)
                    if int(slot.router.scored["blue"]) > 0 and mag == 0:
                        cyc[i].note_unload()            # first unload complete -> shaping starts
                    pn = _pos(i)
                    cyc[i].update_leg(pn, mag)
                    phi_n = 0.0 if bool(dones[i]) else cyc[i].potential(pn)
                    if phi_cur[i] is None or phi_n is None:
                        no_path_pbrs += 1               # no legal path -> no shaping (never NaN)
                    else:
                        rewards[i] += args.pbrs_weight * (args.pbrs_gamma * phi_n - phi_cur[i])
                    if bool(dones[i]):
                        cyc[i].reset()                  # new episode: clear cycle state
            # aggressive multi-cycle shaping (direct terms; added to the raw reward, buffer
            # only). On done the env has already auto-reset the slot, so capture the finished
            # episode's completed-cycle count BEFORE resetting the shaper and skip the update
            # (its state now reflects the fresh episode, not the terminal step).
            if shp is not None:
                for i in range(n):
                    if bool(dones[i]):
                        rep = shp[i].phase_report()          # capture BEFORE reset
                        cyc_completed_ep[i] = shp[i].cycles_completed
                        _st = info.get("episode_stats", {}).get(int(i), {})
                        _sc = int(_st.get("scored", 0))
                        rep["scored"] = _sc
                        rep["collected"] = int(_st.get("collected", 0))
                        rep["train_steps"] = int(agent.train_steps)
                        rep["collector"] = int(args.collector_id)
                        rep["neutral_loaded"] = bool(getattr(env.slots[i], "episode_neutral_loaded", False))
                        # attach the downsampled robot path for trace-worthy episodes so the live
                        # top-8 trace gallery can render it Isaac-free; others stay small.
                        if (_sc >= 40 or int(rep.get("max_stage", 0)) >= 1) and route_ep[i]:
                            rep["route"] = route_ep[i]
                        route_ep[i] = []
                        try:                                  # atomic small-line append (18 writers)
                            with open(cyc_tel_path, "a", encoding="utf-8") as _fh:
                                _fh.write(json.dumps(rep) + "\n")
                        except Exception:
                            pass
                        shp[i].reset()
                        continue
                    slot = env.slots[i]
                    _px, _py = _pos(i)
                    rewards[i] += shp[i].update(
                        len(slot.controller.magazine),
                        int(slot.router.scored["blue"]),
                        int(slot.controller.balls_collected),
                        int(slot.custody.fresh_score),
                        _py, done=False, x=_px)
                    if step % 3 == 0:                          # downsampled path for the trace gallery
                        route_ep[i].append([round(_px, 2), round(_py, 2),
                                            int(slot.router.scored["blue"])])
            # store the transition (current frame -> action -> reward/done)
            buf["obs"].append(frames)
            buf["proprio"].append(obs["proprio"].copy())
            buf["privileged"].append(obs["privileged"].copy())
            buf["action"].append(actions)
            buf["reward"].append(rewards.astype(np.float32))
            buf["done"].append(dones.copy())
            for i in range(n):
                parts = info["reward_components"][i]
                ep_return[i] += rewards[i]
                ep_score[i] += parts["score"]
                ep_collect[i] += parts["collect"]
            for i in np.flatnonzero(dones):
                st = info.get("episode_stats", {}).get(int(i), {})
                pending_eps.append(
                    {
                        "return": round(float(ep_return[i]), 3),
                        "score_reward": round(float(ep_score[i]), 3),
                        "collect_reward": round(float(ep_collect[i]), 3),
                        "scored": int(st.get("scored", 0)),
                        "collected": int(st.get("collected", 0)),
                        # custody-bite telemetry (design note): forward the fresh/recycled
                        # score credits so the learner can report recycled_share.
                        "fresh_score": int(st.get("fresh_score", 0)),
                        "recycled_score": int(st.get("recycled_score", 0)),
                        "fresh_collect": int(st.get("fresh_collect", 0)),
                        "recycled_collect": int(st.get("recycled_collect", 0)),
                        # aggressive-cycle: completed ordered 2nd/3rd... cycles this episode
                        # (the live success signal; 0 unless --aggressive-cycle is on)
                        "cycles_completed": int(cyc_completed_ep[i]),
                    }
                )
                ep_return[i] = ep_score[i] = ep_collect[i] = 0.0
                cyc_completed_ep[i] = 0
            obs, frames = next_obs, next_frames
            step += 1

            if step % args.weight_reload_steps == 0:
                maybe_reload()

            if len(buf["reward"]) >= args.chunk_steps:
                # stack to (num_envs, steps, ...) so each env stream is contiguous
                arrays = {
                    k: np.stack(buf[k], axis=1) for k in D.FIELD_KEYS
                }
                D.write_chunk(cdir, seq, arrays, pending_eps)
                seq += 1
                for k in buf:
                    buf[k].clear()
                pending_eps = []

        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
