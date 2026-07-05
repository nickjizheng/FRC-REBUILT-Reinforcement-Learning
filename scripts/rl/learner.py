"""Distributed DrQ-v2 learner: drain collector chunks, train, publish weights.

Pure torch — NO Isaac/env. Owns the single policy + replay; consumes transitions
from every collector via tmpfs and republishes fresh actor/encoder weights.

Deadlock note: collectors block until the learner publishes its FIRST weights, so
the learner builds the agent from the fixed DrQConfig architecture and publishes
immediately (before draining any chunks) to unblock them.
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
    ap.add_argument("--root", default="/dev/shm/xrc_dist")
    ap.add_argument("--num-collectors", type=int, required=True)
    ap.add_argument("--collector-envs", type=int, default=4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--minutes", type=float, default=240.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--updates-per-tx", type=float, default=1.0)
    ap.add_argument("--replay-capacity", type=int, default=400_000)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--seed-transitions", type=int, default=2_000)
    ap.add_argument("--weight-publish-updates", type=int, default=400)
    ap.add_argument("--max-updates-per-tick", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "drqv2_stageC_dist")
    args = ap.parse_args()

    import torch

    from xrc_rebuilt.rl import distributed as D
    from xrc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
    from xrc_rebuilt.rl.replay import PerEnvReplay

    args.out.mkdir(parents=True, exist_ok=True)
    wdir = D.weights_dir(args.root)
    wdir.mkdir(parents=True, exist_ok=True)
    streams = args.num_collectors * args.collector_envs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = DrQConfig()
    agent = DrQV2Agent(cfg)
    if args.resume:
        agent.load(args.resume)
        print(f"LEARNER resumed {args.resume} steps={agent.train_steps}", flush=True)

    def publish(step: int) -> None:
        if agent.weights_finite():
            D.publish_weights(
                wdir,
                {
                    "encoder": agent.encoder.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "train_steps": agent.train_steps,
                },
                step,
            )

    publish(agent.train_steps)  # unblock collectors immediately
    print(f"LEARNER published initial weights; streams={streams}", flush=True)

    replay = PerEnvReplay(
        num_envs=streams,
        capacity_per_env=max(1000, args.replay_capacity // streams),
        seed=args.seed + 5,
        obs_shape=(cfg.frame_channels, cfg.frame_h, cfg.frame_w),
        proprio_dim=cfg.proprio_dim,
        privileged_dim=cfg.privileged_dim,
        action_dim=cfg.action_dim,
        n_step=args.n_step,
        gamma=args.gamma,
    )

    run_started = time.time()
    (args.out / "run_config.json").write_text(
        json.dumps(
            {**{k: str(v) for k, v in vars(args).items()}, "streams": streams, "mode": "distributed"},
            indent=2,
        )
    )

    consumed: set[str] = set()
    transitions = 0
    updates = 0
    update_debt = 0.0
    best_return = float("-inf")
    finished_returns: list[float] = []
    finished_scores: list[float] = []
    finished_collects: list[float] = []
    finished_scored: list[int] = []
    metrics_path = args.out / "metrics.jsonl"
    last_report = time.time()
    train_metrics: dict[str, float] = {}
    deadline = time.time() + args.minutes * 60.0

    while time.time() < deadline:
        chunks = D.drain_chunks(args.root, args.num_collectors, consumed)
        new_tx = 0
        for chunk in chunks:
            a = chunk.arrays
            steps = a["reward"].shape[1]
            for e in range(args.collector_envs):
                stream = chunk.collector_id * args.collector_envs + e
                for t in range(steps):
                    replay.add(
                        stream,
                        a["obs"][e, t],
                        a["proprio"][e, t],
                        a["privileged"][e, t],
                        a["action"][e, t],
                        float(a["reward"][e, t]),
                        bool(a["done"][e, t]),
                    )
                    new_tx += 1
            for ep in chunk.episodes:
                finished_returns.append(ep["return"])
                finished_scores.append(ep["score_reward"])
                finished_collects.append(ep["collect_reward"])
                finished_scored.append(ep["scored"])
        transitions += new_tx

        if replay.ready(max(args.batch_size, args.seed_transitions)):
            update_debt += args.updates_per_tx * new_tx
            done_this_tick = 0
            while update_debt >= 1.0 and done_this_tick < args.max_updates_per_tick:
                train_metrics = agent.update(replay.sample(args.batch_size))
                updates += 1
                done_this_tick += 1
                update_debt -= 1.0
                if updates % args.weight_publish_updates == 0:
                    publish(agent.train_steps)

        if new_tx == 0:
            time.sleep(0.25)  # idle: don't spin waiting on collectors

        if time.time() - last_report >= 60.0:
            last_report = time.time()
            elapsed = time.time() - run_started
            recent = finished_returns[-40:]
            line = {
                "wall_time": datetime.now().astimezone().isoformat(),
                "elapsed_s": round(elapsed, 1),
                "transitions": transitions,
                "transitions_per_s": round(transitions / max(elapsed, 1e-6), 3),
                "updates": updates,
                "replay": len(replay),
                "recent_return_mean": round(float(np.mean(recent)), 2) if recent else None,
                "recent_return_max": round(float(np.max(recent)), 2) if recent else None,
                "recent_score_reward": round(float(np.mean(finished_scores[-40:])), 2) if finished_scores else None,
                "recent_collect_reward": round(float(np.mean(finished_collects[-40:])), 2) if finished_collects else None,
                "recent_scored_balls": round(float(np.mean(finished_scored[-40:])), 2) if finished_scored else None,
                "episodes": len(finished_returns),
                **{k: round(v, 4) for k, v in train_metrics.items()},
            }
            print("TRAIN " + json.dumps(line), flush=True)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
            if agent.weights_finite():
                agent.save(str(args.out / "latest.pt"))
                if recent and float(np.mean(recent)) > best_return:
                    best_return = float(np.mean(recent))
                    agent.save(str(args.out / "best.pt"))

    if agent.weights_finite():
        agent.save(str(args.out / "final.pt"))
    print("LEARNER_DONE " + json.dumps({"transitions": transitions, "updates": updates, "episodes": len(finished_returns)}), flush=True)


if __name__ == "__main__":
    main()
