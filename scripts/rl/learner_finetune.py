"""Reward-first warm-start fine-tune learner.

Same distributed transport/replay/publish as `learner.py`, but:
  * REQUIRES a warm-start champion (`--resume`); clears all three Adam moment states at
    launch (stale w.r.t. the custody-weighted objective) and applies LR 5e-5;
  * re-warms exploration to a chosen mid-point via `--explore-warm-steps` (NOT the
    all-or-nothing `--explore-restart`, which set stddev 1.0 and collapsed a champion);
  * runs `agent.update_finetune`: a critic-only + encoder-frozen re-fit for the first
    `--critic-only-updates`, then encoder+actor unlocked with an end-to-end champion BC
    anchor `beta*MSE(pi(enc(f_a)),a_champ)` annealed 0.3 -> 0 over
    `[--critic-only-updates, --anchor-beta-end-updates]`.
The custody reward itself lives in the ENV (collectors set `--rho-score/--rho-collect`);
the learner records them for reproducibility. DriftGate is diagnostic here (drift away
from the champion is the goal); promotion is decided by external deterministic phase evals.
Pure torch, no Isaac.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/dev/shm/frc_dist")
    ap.add_argument("--num-collectors", type=int, required=True)
    ap.add_argument("--collector-envs", type=int, default=4)
    ap.add_argument("--resume", required=True, help="immutable champion to warm-start from")
    ap.add_argument("--minutes", type=float, default=600.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--updates-per-tx", type=float, default=1.0)
    ap.add_argument("--replay-capacity", type=int, default=400_000)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--seed-transitions", type=int, default=2_000)
    ap.add_argument("--weight-publish-updates", type=int, default=400)
    ap.add_argument("--eval-snapshot-updates", type=int, default=5_000)
    ap.add_argument("--max-updates-per-tick", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--stddev-end", type=float, default=0.2)
    ap.add_argument("--min-live-fraction", type=float, default=0.5)
    ap.add_argument("--stream-stall-s", type=float, default=180.0)
    # --- reward-first fine-tune knobs (design note) ---
    ap.add_argument("--critic-only-updates", type=int, default=3000,
                    help="phase 1: critic head + EMA on FROZEN champion features (actor+encoder frozen)")
    ap.add_argument("--explore-warm-steps", type=int, default=62500,
                    help="explore_offset = train_steps - this; 62500 -> stddev 0.50 at launch")
    ap.add_argument("--anchor-beta-start", type=float, default=0.3)
    ap.add_argument("--anchor-beta-end-updates", type=int, default=23000,
                    help="beta anneals from anchor-beta-start (at critic-only-updates) to 0 here")
    ap.add_argument("--anchor-dir", required=True, help="champion anchor npz dump (holdout auto-excluded)")
    ap.add_argument("--anchor-batch", type=int, default=128)
    ap.add_argument("--rho-score", type=float, default=0.2, help="record-only; collectors set the env reward")
    ap.add_argument("--rho-collect", type=float, default=0.2, help="record-only; collectors set the env reward")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "drqv2_rewardfirst")
    args = ap.parse_args()

    import torch

    from frc_rebuilt.rl import distributed as D
    from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
    from frc_rebuilt.rl.prefix_takeover import AnchorSampler, FROZEN_HOLDOUT_EPISODES, finetune_beta
    from frc_rebuilt.rl.replay import PerEnvReplay

    args.out.mkdir(parents=True, exist_ok=True)
    eval_queue = args.out / "eval_queue"; eval_queue.mkdir(parents=True, exist_ok=True)
    wdir = D.weights_dir(args.root); wdir.mkdir(parents=True, exist_ok=True)
    streams = args.num_collectors * args.collector_envs
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = DrQConfig(stddev_end=args.stddev_end, lr=args.learning_rate)
    agent = DrQV2Agent(cfg)
    agent.load(args.resume)
    # clear stale Adam moments (calibrated to the OLD reward scale) + apply the fine-tune LR
    for optimizer in (agent.encoder_opt, agent.actor_opt, agent.critic_opt):
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    # re-warm exploration to a mid-point (design note): explore_offset = train_steps - warm_steps
    agent.explore_offset = agent.train_steps - args.explore_warm_steps
    finetune_start_steps = agent.train_steps
    print(f"LEARNER_FT resumed {args.resume} steps={agent.train_steps} "
          f"stddev0={agent.stddev():.3f} gamma={args.gamma} lr={args.learning_rate} "
          f"critic_only={args.critic_only_updates} beta={args.anchor_beta_start}->0@{args.anchor_beta_end_updates}",
          flush=True)

    sampler = AnchorSampler(args.anchor_dir, FROZEN_HOLDOUT_EPISODES, seed=args.seed + 7)
    print(f"LEARNER_FT anchors: {len(sampler.train_episodes)} train eps, "
          f"{sampler.frames.shape[0]} states, phases={sampler.phases}", flush=True)

    def publish(step: int) -> None:
        if agent.weights_finite():
            D.publish_weights(wdir, {
                "encoder": agent.encoder.state_dict(), "actor": agent.actor.state_dict(),
                "train_steps": agent.train_steps, "explore_offset": agent.explore_offset,
            }, step)

    publish(agent.train_steps)
    print(f"LEARNER_FT published initial weights; streams={streams}", flush=True)

    replay = PerEnvReplay(
        num_envs=streams, capacity_per_env=max(1000, args.replay_capacity // streams),
        seed=args.seed + 5, obs_shape=(cfg.frame_channels, cfg.frame_h, cfg.frame_w),
        proprio_dim=cfg.proprio_dim, privileged_dim=cfg.privileged_dim,
        action_dim=cfg.action_dim, n_step=args.n_step, gamma=args.gamma)

    run_started = time.time()
    (args.out / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "streams": streams,
         "mode": "rewardfirst_finetune", "started_at_unix": run_started,
         "champion": str(args.resume)}, indent=2))

    def save_atomic(path: Path) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            agent.save(str(tmp)); os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    consumed: set = set()
    transitions = updates = 0
    update_debt = 0.0
    finished_scores: list[float] = []
    finished_scored: list[int] = []
    finished_returns: list[float] = []
    finished_collects: list[float] = []
    fresh_score_ep: list[int] = []
    recycled_score_ep: list[int] = []
    finished_cycles: list[int] = []      # aggressive-cycle: completed ordered cycles / ep
    metrics_path = args.out / "metrics.jsonl"
    last_report = time.time()
    train_metrics: dict = {}
    deadline = time.time() + args.minutes * 60.0

    while time.time() < deadline:
        chunks = D.drain_chunks(args.root, args.num_collectors, consumed)
        new_tx = 0
        for chunk in chunks:
            a = chunk.arrays
            steps = a["reward"].shape[1]; chunk_envs = int(a["reward"].shape[0])
            if chunk_envs > args.collector_envs:
                for e in range(args.collector_envs):
                    replay.mark_boundary(chunk.collector_id * args.collector_envs + e)
                continue
            for e in range(chunk_envs):
                stream = chunk.collector_id * args.collector_envs + e
                for t in range(steps):
                    if not (np.isfinite(a["proprio"][e, t]).all() and np.isfinite(a["privileged"][e, t]).all()
                            and np.isfinite(a["action"][e, t]).all() and np.isfinite(a["reward"][e, t])):
                        replay.mark_boundary(stream); continue
                    replay.add(stream, a["obs"][e, t], a["proprio"][e, t], a["privileged"][e, t],
                               a["action"][e, t], float(a["reward"][e, t]), bool(a["done"][e, t]))
                    new_tx += 1
            for ep in chunk.episodes:
                finished_scores.append(ep.get("score_reward", 0.0))
                finished_scored.append(ep.get("scored", 0))
                finished_returns.append(ep.get("return", 0.0))
                finished_collects.append(ep.get("collect_reward", 0.0))
                finished_cycles.append(ep.get("cycles_completed", 0))
                if "fresh_score" in ep:
                    fresh_score_ep.append(ep["fresh_score"]); recycled_score_ep.append(ep.get("recycled_score", 0))
        transitions += new_tx

        if replay.ready(max(args.batch_size, args.seed_transitions), min_live_fraction=args.min_live_fraction):
            update_debt += args.updates_per_tx * new_tx
            done_this_tick = 0
            while update_debt >= 1.0 and done_this_tick < args.max_updates_per_tick:
                critic_only = updates < args.critic_only_updates
                beta = finetune_beta(updates, args.critic_only_updates,
                                     args.anchor_beta_start, args.anchor_beta_end_updates)
                a_obs, a_pro, a_act = sampler.sample(args.anchor_batch)
                train_metrics = agent.update_finetune(replay.sample(args.batch_size),
                                                       a_obs, a_pro, a_act, beta=beta, critic_only=critic_only)
                updates += 1; done_this_tick += 1; update_debt -= 1.0
                if updates % args.weight_publish_updates == 0:
                    publish(agent.train_steps)
                if (args.eval_snapshot_updates > 0
                        and (agent.train_steps - finetune_start_steps) % args.eval_snapshot_updates == 0
                        and agent.weights_finite()):
                    save_atomic(eval_queue / f"ft_{updates:09d}.pt")

        if new_tx == 0:
            time.sleep(0.25)

        if time.time() - last_report >= 60.0:
            last_report = time.time()
            elapsed = time.time() - run_started
            n_fresh = float(np.mean(fresh_score_ep[-40:])) if fresh_score_ep else None
            n_recyc = float(np.mean(recycled_score_ep[-40:])) if recycled_score_ep else None
            recyc_share = (n_recyc / (n_fresh + n_recyc)) if (n_fresh is not None and (n_fresh + n_recyc) > 0) else None
            line = {
                "wall_time": datetime.now().astimezone().isoformat(), "elapsed_s": round(elapsed, 1),
                "transitions": transitions, "updates": updates, "replay": len(replay),
                "phase": "critic_only" if updates < args.critic_only_updates else "finetune",
                "beta": round(finetune_beta(updates, args.critic_only_updates,
                                            args.anchor_beta_start, args.anchor_beta_end_updates), 4),
                "recent_scored_balls": round(float(np.mean(finished_scored[-40:])), 2) if finished_scored else None,
                "recent_scored_max": int(np.max(finished_scored[-40:])) if finished_scored else None,
                # fields the training dashboard charts (score/return/collect panels)
                "recent_score_reward": round(float(np.mean(finished_scores[-40:])), 2) if finished_scores else None,
                "recent_return_mean": round(float(np.mean(finished_returns[-40:])), 2) if finished_returns else None,
                "recent_return_max": round(float(np.max(finished_returns[-40:])), 2) if finished_returns else None,
                "recent_collect_reward": round(float(np.mean(finished_collects[-40:])), 2) if finished_collects else None,
                # aggressive-cycle success signal: mean/max completed ordered cycles + the
                # fraction of recent episodes with >=1 real 2nd cycle (baseline 0)
                "recent_cycles_mean": round(float(np.mean(finished_cycles[-40:])), 3) if finished_cycles else None,
                "recent_cycles_max": int(np.max(finished_cycles[-40:])) if finished_cycles else None,
                "cycle2_rate": round(float(np.mean([c >= 1 for c in finished_cycles[-40:]])), 3) if finished_cycles else None,
                "mean_fresh_score": round(n_fresh, 2) if n_fresh is not None else None,
                "mean_recycled_score": round(n_recyc, 2) if n_recyc is not None else None,
                "recycled_share": round(recyc_share, 3) if recyc_share is not None else None,
                "episodes": len(finished_scored),
                **{k: round(v, 4) for k, v in train_metrics.items()},
            }
            print("TRAIN_FT " + json.dumps(line), flush=True)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
            if agent.weights_finite():
                save_atomic(args.out / "latest.pt")

    if agent.weights_finite():
        save_atomic(args.out / "final.pt")
    print("LEARNER_FT_DONE " + json.dumps({"transitions": transitions, "updates": updates}), flush=True)


if __name__ == "__main__":
    main()
