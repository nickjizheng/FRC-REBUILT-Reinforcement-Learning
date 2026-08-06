"""Stage-A DrQ-v2 training on the vectorized full-physics competition env.

The converged first baseline uses off-policy DrQ-v2 with pixel and
proprio actor, asymmetric privileged critic, n-step returns, curriculum
stage A (short acquisition episodes, 32-FUEL template).  Prints one JSON
metrics line per interval and checkpoints the agent + a rolling metrics log
under runs/drqv2_stageA/.

Policy view: the three 640x360 frames are 4x-downsampled to 160x90 and
channel-stacked -> (9, 90, 160) uint8.  The full-resolution frames remain the
sensor contract; downsampling is part of the policy, not privileged access.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


# One named, versioned contract per curriculum stage. Stage C is reproducible
# from `--stage C` alone -> run_config.json records "C", not "B + hand flags".
STAGE_C_CONFIG = {
    "version": "C.v1-2026-07-12",
    "episode_len_s": 90.0,        # 90 s continuous HUB (sandbox scoring kept)
    "sandbox_scoring": True,      # KEPT for Stage C (both hubs always eligible)
    "spawn_under_trench": True,   # source-faithful compact-under-trench start
    "unlock_storage": True,       # storage active: policy drives compact/extend
    "preload_prob": 0.0,          # no shooting starts: full escape->collect->score
    "spawn_ramp_prob": 0.0,       # ramp-teaching is a Stage-B device; off in C
    "collect_weight_start": 0.3,  # constant 0.3 -- matches the resumed champion's
    "collect_weight_end": 0.3,    # weight (the 1.5 anneal poisons a resumed replay)
    "action_penalty": 0.001,      # drive energy penalty (<< collect reward)
    "stddev_end": 0.2,            # trench escape is exploration-hard
}


def to_policy_frames(rgb: np.ndarray) -> np.ndarray:
    """(N, C_cam, 360, 640, 3) uint8 -> (N, 9, 90, 160) uint8 (4x downsample)."""
    small = rgb[:, :, ::4, ::4, :]                       # (N, cams, 90, 160, 3)
    n, cams, h, w, c = small.shape
    return (
        small.transpose(0, 1, 4, 2, 3).reshape(n, cams * c, h, w).copy()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_32.usd"))
    ap.add_argument(
        "--replay-capacity",
        type=int,
        default=60_000,
        help="total transitions in RAM (~130 KB each; 60k = ~7.8 GB on the "
        "32 GB machine - the D:-NVMe chunk store is the planned larger tier)",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.997,
        help="converged plan: 0.997 for short curriculum stages, annealed "
        "toward 0.999 for full matches",
    )
    ap.add_argument("--seed-transitions", type=int, default=1_000)
    ap.add_argument("--updates-per-tx", type=float, default=1.0)
    ap.add_argument("--episode-len-s", type=float, default=20.0)
    ap.add_argument(
        "--stage",
        choices=("A", "B", "C"),
        default="A",
        help="B = acquire-and-score (36 s, preloaded shooting starts, collect "
        "annealed 1.5->0.3). C = trench start: compact-under-trench, 90 s "
        "continuous HUB (sandbox scoring kept), storage active, constant 0.3 "
        "collect weight -- the reproducible Stage C contract (see STAGE_C_CONFIG).",
    )
    ap.add_argument("--preload-prob", type=float, default=None)
    ap.add_argument(
        "--spawn-ramp-prob",
        type=float,
        default=None,
        help="stage B ramp-teaching: fraction of COLD (non-preloaded) episodes that "
        "start in the neutral zone aligned to a hub ramp lane, facing the hub, "
        "preloaded, so the shortest path to a legal shot runs over the drive-over "
        "ramp instead of the compact-only trench lanes (0.0 = off)",
    )
    ap.add_argument(
        "--mask-illegal-fire",
        action="store_true",
        help="stage B: gate firing on a legal HUB shot THIS tick (and mask ferry "
        "off) so an illegal fire is a no-op instead of auto-stopping the chassis - "
        "prevents the always-fire freeze/collapse",
    )
    ap.add_argument(
        "--action-penalty",
        type=float,
        default=None,
        help="drive action-penalty weight (stage B default 0.0 so it does not tax "
        "the navigation we teach; otherwise 0.001)",
    )
    ap.add_argument(
        "--stddev-end",
        type=float,
        default=None,
        help="exploration stddev floor (stage B default 0.2; otherwise 0.1)",
    )
    ap.add_argument(
        "--spawn-under-trench",
        action="store_true",
        help="stage C: begin every episode compact, fully beneath the blue "
        "trench (the source-faithful match start); implies --unlock-storage",
    )
    ap.add_argument(
        "--unlock-storage",
        action="store_true",
        help="let the policy drive the compact/extend (dim-4 storage) action",
    )
    ap.add_argument("--collect-weight-start", type=float, default=1.5)
    ap.add_argument("--collect-weight-end", type=float, default=None)
    ap.add_argument("--collect-anneal-tx", type=int, default=30_000)
    ap.add_argument(
        "--checkpoint-every-tx",
        type=int,
        default=10_000,
        help="numbered finite-only checkpoints (ckpt_<transitions>.pt)",
    )
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="master seed (env resets, replay sampling, torch, exploration) - "
        "vary per parallel run so seeds are real replications",
    )
    ap.add_argument(
        "--watchdog-stall-s",
        type=float,
        default=180.0,
        help="exit(3) if no finite transition is accepted for this many seconds "
        "(supervisor then restarts from latest.pt with a fresh sim)",
    )
    ap.add_argument(
        "--explore-restart",
        action="store_true",
        help="re-warm the exploration-noise schedule for this stage on --resume "
        "(a resumed agent past stddev_steps is otherwise pinned at the floor)",
    )
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "drqv2_stageA")
    args = ap.parse_args()
    if args.stage == "B":
        if args.episode_len_s == 20.0:
            args.episode_len_s = 36.0
        if args.preload_prob is None:
            args.preload_prob = 0.3  # rebalanced from 0.5: less shoot-heavy start mix
        if args.spawn_ramp_prob is None:
            args.spawn_ramp_prob = 0.4  # rebalanced from 0.6 (still teaches the ramp)
        if args.collect_weight_end is None:
            args.collect_weight_end = 0.3
        if args.action_penalty is None:
            args.action_penalty = 0.0  # do not tax the driving we are teaching
        if args.stddev_end is None:
            args.stddev_end = 0.2  # keep exploring so it can discover the crossing
    elif args.stage == "C":
        c = STAGE_C_CONFIG
        if args.episode_len_s == 20.0:       # only if left at the parser default
            args.episode_len_s = c["episode_len_s"]
        args.spawn_under_trench = True        # force the contract regardless of flag
        args.unlock_storage = True
        if args.preload_prob is None:
            args.preload_prob = c["preload_prob"]
        if args.spawn_ramp_prob is None:
            args.spawn_ramp_prob = c["spawn_ramp_prob"]
        # Stage C resumes a champion -> constant 0.3 collect weight (the 1.5 anneal
        # is a fresh-agent teaching device that poisons a resumed replay; audit).
        if args.collect_weight_start == 1.5:  # only if left at the parser default
            args.collect_weight_start = c["collect_weight_start"]
        if args.collect_weight_end is None:
            args.collect_weight_end = c["collect_weight_end"]
        if args.action_penalty is None:
            args.action_penalty = c["action_penalty"]
        if args.stddev_end is None:
            args.stddev_end = c["stddev_end"]
    args.preload_prob = args.preload_prob or 0.0
    args.spawn_ramp_prob = args.spawn_ramp_prob or 0.0
    if args.action_penalty is None:
        args.action_penalty = 0.001
    if args.stddev_end is None:
        args.stddev_end = 0.1
    args.collect_weight_end = (
        args.collect_weight_start
        if args.collect_weight_end is None
        else args.collect_weight_end
    )

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from frc_rebuilt.rl.replay import PerEnvReplay
        from frc_rebuilt.rl.vec_env import (
            SimulationUnstable,
            VecCompetitionEnv,
            VecEnvCfg,
        )

        args.out.mkdir(parents=True, exist_ok=True)
        run_started_at = time.time()
        run_config = {
            **vars(args),
            "out": str(args.out.resolve()),
            "template": str(Path(args.template).resolve()),
            "started_at_unix": run_started_at,
            "started_at": datetime.fromtimestamp(run_started_at).astimezone().isoformat(),
            "stage_config_version": STAGE_C_CONFIG["version"] if args.stage == "C" else None,
            "pid": os.getpid(),
        }
        (args.out / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                preload_prob=float(args.preload_prob),
                spawn_ramp_prob=float(args.spawn_ramp_prob),
                mask_illegal_fire=bool(args.mask_illegal_fire),
                action_penalty=float(args.action_penalty),
                collect_reward_weight=float(args.collect_weight_start),
                seed=args.seed,
                spawn_under_trench=bool(args.spawn_under_trench),
                lock_storage_extended=not (
                    args.unlock_storage or args.spawn_under_trench
                ),
            )
        )
        n = args.num_envs
        zero_actions = np.zeros((n, 7), np.float32)
        obs, _, _, _ = env.step(zero_actions)
        frames = to_policy_frames(obs["rgb"])
        cams = frames.shape[1]
        print(
            f"TRAIN_ENV_READY envs={n} frame_shape={list(frames.shape[1:])} "
            f"frame_std={[round(float(frames[i].std()), 1) for i in range(n)]}",
            flush=True,
        )

        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=cams,
                frame_h=frames.shape[2],
                frame_w=frames.shape[3],
                proprio_dim=obs["proprio"].shape[1],
                privileged_dim=obs["privileged"].shape[1],
                stddev_end=float(args.stddev_end),
            )
        )
        if args.resume:
            agent.load(args.resume)
            print(f"TRAIN_RESUMED {args.resume} steps={agent.train_steps}", flush=True)
            if args.explore_restart:
                # One-shot re-warm: skip when auto-restarting from our OWN latest.pt
                # (the supervisor passes --explore-restart on every relaunch, but the
                # anchor is already saved in latest.pt, so re-firing would reset
                # exploration to the start on every crash-recovery; audit).
                resuming_own_latest = (
                    Path(args.resume).resolve() == (args.out / "latest.pt").resolve()
                )
                if resuming_own_latest:
                    print(
                        "TRAIN_EXPLORE_RESTART skipped (auto-restart from own "
                        f"latest.pt; schedule anchor offset={agent.explore_offset} "
                        "preserved)",
                        flush=True,
                    )
                else:
                    agent.explore_offset = agent.train_steps
                    print(
                        f"TRAIN_EXPLORE_RESTART offset={agent.explore_offset} "
                        "(exploration noise re-warmed for this stage)",
                        flush=True,
                    )
        replay = PerEnvReplay(
            num_envs=n,
            capacity_per_env=max(1000, args.replay_capacity // n),
            seed=args.seed + 11,
            obs_shape=tuple(frames.shape[1:]),
            proprio_dim=obs["proprio"].shape[1],
            privileged_dim=obs["privileged"].shape[1],
            action_dim=7,
            n_step=3,
            gamma=args.gamma,
        )

        deadline = time.time() + args.minutes * 60.0
        transitions = 0
        updates = 0
        episode_return = np.zeros(n, np.float32)
        episode_score = np.zeros(n, np.float32)
        episode_collect = np.zeros(n, np.float32)
        finished_returns: list[float] = []
        finished_scores: list[float] = []
        finished_collects: list[float] = []
        metrics_path = args.out / "metrics.jsonl"
        last_report = time.time()
        report_every_s = 60.0
        update_debt = 0.0
        train_metrics: dict[str, float] = {}
        best_metric = float("-inf")
        rejected_transitions = 0
        last_accept_time = time.time()
        next_numbered_ckpt = args.checkpoint_every_tx

        current = {
            "frames": frames,
            "proprio": obs["proprio"].copy(),
            "privileged": obs["privileged"].copy(),
        }
        while time.time() < deadline:
            if transitions < args.seed_transitions and not args.resume:
                # random seeding is only for FRESH training; a resumed (already
                # trained) agent should act from step 0 so a supervisor restart
                # doesn't dump ~1000 random steps into replay and stall recovery.
                actions = np.random.uniform(-1, 1, (n, 7)).astype(np.float32)
                actions[:, 3] = 1.0  # keep intake on while seeding
            else:
                actions = agent.act(
                    current["frames"], current["proprio"], explore=True
                ).astype(np.float32)
            try:
                obs, rewards, dones, info = env.step(actions)
            except SimulationUnstable as exc:
                print(
                    f"TRAIN_SIM_UNSTABLE {exc}; saving latest.pt + exit(4) for "
                    "supervisor restart",
                    flush=True,
                )
                if agent.weights_finite():
                    agent.save(str(args.out / "latest.pt"))
                sys.exit(4)
            next_frames = to_policy_frames(obs["rgb"])
            # reject non-finite inputs at the boundary: a corrupted sim step
            # must never enter the replay buffer or the running statistics
            finite_rewards = np.isfinite(rewards)
            finite_obs = np.array(
                [
                    np.isfinite(obs["proprio"][i]).all()
                    and np.isfinite(obs["privileged"][i]).all()
                    for i in range(n)
                ]
            )
            corrupt = ~(finite_rewards & finite_obs)
            accepted = int((~corrupt).sum())
            if corrupt.any():
                rejected_transitions += int(corrupt.sum())
                rewards = np.where(finite_rewards, rewards, 0.0).astype(np.float32)
            for i in range(n):
                if corrupt[i]:
                    continue
                replay.add(
                    i,
                    current["frames"][i],
                    current["proprio"][i],
                    current["privileged"][i],
                    actions[i],
                    rewards[i],
                    dones[i],
                )
            episode_return += rewards
            for i in range(n):
                parts = info["reward_components"][i]
                episode_score[i] += parts["score"]
                episode_collect[i] += parts["collect"]
            for i in np.flatnonzero(dones):
                finished_returns.append(float(episode_return[i]))
                finished_scores.append(float(episode_score[i]))
                finished_collects.append(float(episode_collect[i]))
                episode_return[i] = 0.0
                episode_score[i] = 0.0
                episode_collect[i] = 0.0
            # defensive: never carry a non-finite obs into the next act() call
            # (the env force-resets poisoned slots, so this is a backstop only).
            current = {
                "frames": next_frames,
                "proprio": np.nan_to_num(obs["proprio"], copy=True).astype(np.float32),
                "privileged": np.nan_to_num(obs["privileged"], copy=True).astype(np.float32),
            }
            transitions += accepted
            # watchdog: if the sim is stuck emitting only rejected transitions,
            # exit(3) so the supervisor restarts from latest.pt - a globally
            # poisoned PhysX scene never recovers in-process.
            if accepted > 0:
                last_accept_time = time.time()
            elif time.time() - last_accept_time > args.watchdog_stall_s:
                print(
                    "TRAIN_WATCHDOG_STALL no accepted transitions for "
                    f"{args.watchdog_stall_s:.0f}s; saving latest.pt + exit(3) "
                    "for supervisor restart",
                    flush=True,
                )
                if agent.weights_finite():
                    agent.save(str(args.out / "latest.pt"))
                sys.exit(3)
            # collection-weight anneal (stage B): 1.5 -> 0.3 over the window
            if args.collect_weight_end != args.collect_weight_start:
                mix = min(1.0, transitions / max(1, args.collect_anneal_tx))
                env.collect_weight = float(
                    args.collect_weight_start
                    + (args.collect_weight_end - args.collect_weight_start) * mix
                )
            if transitions >= next_numbered_ckpt:
                next_numbered_ckpt += args.checkpoint_every_tx
                if agent.weights_finite():
                    # key on the monotonic gradient-step counter (survives restarts),
                    # not the per-process transition counter (resets to 0 each life ->
                    # numbered checkpoints would collide/overwrite across crash-restarts)
                    agent.save(str(args.out / f"ckpt_{agent.train_steps:09d}.pt"))

            if replay.ready(max(args.batch_size, args.seed_transitions)) and accepted > 0:
                update_debt += args.updates_per_tx * accepted
                # anti-runaway: never let debt balloon (e.g. after a stall) into a
                # burst of updates on a stale buffer
                update_debt = min(update_debt, args.updates_per_tx * n * 4.0)
                while update_debt >= 1.0:
                    train_metrics = agent.update(replay.sample(args.batch_size))
                    updates += 1
                    update_debt -= 1.0

            if time.time() - last_report >= report_every_s:
                last_report = time.time()
                recent = finished_returns[-20:]
                elapsed_s = time.time() - run_started_at
                line = {
                    "wall_time": datetime.now().astimezone().isoformat(),
                    "elapsed_s": round(elapsed_s, 1),
                    "transitions": transitions,
                    "transitions_per_s": round(transitions / max(elapsed_s, 1e-6), 3),
                    "updates": updates,
                    "replay": len(replay),
                    "recent_return_mean": round(float(np.mean(recent)), 2) if recent else None,
                    "recent_return_max": round(float(np.max(recent)), 2) if recent else None,
                    "recent_score_reward": round(float(np.mean(finished_scores[-20:])), 2)
                    if finished_scores
                    else None,
                    "recent_collect_reward": round(float(np.mean(finished_collects[-20:])), 2)
                    if finished_collects
                    else None,
                    "episodes": len(finished_returns),
                    "collect_weight": round(float(env.collect_weight), 3),
                    "rejected_transitions": rejected_transitions,
                    **{k: round(v, 4) for k, v in train_metrics.items()},
                }
                print("TRAIN " + json.dumps(line), flush=True)
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line) + "\n")
                # checkpoint hygiene: never persist non-finite weights (a 4 h
                # run once diverged in its final minute and poisoned final.pt),
                # and keep the best-return checkpoint separately.
                if agent.weights_finite():
                    agent.save(str(args.out / "latest.pt"))
                    # rank by recent SCORE reward in stage B (total return is
                    # confounded by the collect-weight anneal that inflates early
                    # episodes); stage A has no scoring, so fall back to return.
                    recent_scores = finished_scores[-20:]
                    if args.stage in ("B", "C") and recent_scores:
                        metric = float(np.mean(recent_scores))
                    elif recent:
                        metric = float(np.mean(recent))
                    else:
                        metric = None
                    if metric is not None and metric > best_metric:
                        best_metric = metric
                        agent.save(str(args.out / "best.pt"))
                else:
                    print("TRAIN_WEIGHTS_NONFINITE latest.pt NOT overwritten", flush=True)

        if agent.weights_finite():
            agent.save(str(args.out / "final.pt"))
        else:
            print(
                "TRAIN_FINAL_NONFINITE final.pt skipped; use latest.pt/best.pt",
                flush=True,
            )
        elapsed_s = time.time() - run_started_at
        summary = {
            "started_at": run_config["started_at"],
            "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_s": round(elapsed_s, 1),
            "transitions": transitions,
            "transitions_per_s": round(transitions / max(elapsed_s, 1e-6), 3),
            "updates": updates,
            "episodes": len(finished_returns),
            "mean_return_last20": round(float(np.mean(finished_returns[-20:])), 2)
            if finished_returns
            else None,
            "mean_score_reward_last20": round(float(np.mean(finished_scores[-20:])), 2)
            if finished_scores
            else None,
            "mean_collect_reward_last20": round(float(np.mean(finished_collects[-20:])), 2)
            if finished_collects
            else None,
            "first5_returns": [round(r, 2) for r in finished_returns[:5]],
            "last5_returns": [round(r, 2) for r in finished_returns[-5:]],
        }
        print("TRAIN_DONE " + json.dumps(summary), flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
