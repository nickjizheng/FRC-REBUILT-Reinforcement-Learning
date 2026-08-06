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
    ap.add_argument("--root", default="/dev/shm/frc_dist")
    ap.add_argument("--num-collectors", type=int, required=True)
    ap.add_argument("--collector-envs", type=int, default=4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--minutes", type=float, default=240.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--updates-per-tx", type=float, default=1.0)
    ap.add_argument("--replay-capacity", type=int, default=400_000)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--seed-transitions", type=int, default=2_000)
    ap.add_argument("--weight-publish-updates", type=int, default=400)
    ap.add_argument("--eval-snapshot-updates", type=int, default=5_000,
                    help="atomically queue a deterministic-eval candidate every N updates")
    ap.add_argument("--max-updates-per-tick", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--stddev-end", type=float, default=0.2,
                    help="target-smoothing / exploration floor (Stage C contract 0.2)")
    ap.add_argument("--explore-restart", action="store_true",
                    help="on resume from an EXTERNAL champion (not our own latest.pt), "
                         "re-warm exploration (explore_offset=train_steps) so Stage C "
                         "re-explores for trench escape; one-shot -- skipped on an "
                         "auto-restart from latest.pt so a crash loop can't reset it.")
    ap.add_argument("--collect-weight", type=float, default=0.3,
                    help="record-only (learner builds no env): the collect_reward_weight "
                         "the collectors MUST launch with; lands in run_config.json.")
    ap.add_argument("--stream-stall-s", type=float, default=180.0,
                    help="flag a collector-env stream stale after this long with no finite tx")
    ap.add_argument("--min-live-fraction", type=float, default=0.5,
                    help="learn once this fraction of streams is warm; a dead stream cannot block all learning")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "drqv2_stageC_dist")
    args = ap.parse_args()

    import torch

    from frc_rebuilt.rl import distributed as D
    from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
    from frc_rebuilt.rl.replay import PerEnvReplay

    args.out.mkdir(parents=True, exist_ok=True)
    eval_queue = args.out / "eval_queue"
    eval_queue.mkdir(parents=True, exist_ok=True)
    wdir = D.weights_dir(args.root)
    wdir.mkdir(parents=True, exist_ok=True)
    streams = args.num_collectors * args.collector_envs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = DrQConfig(stddev_end=args.stddev_end, lr=args.learning_rate)
    agent = DrQV2Agent(cfg)
    if args.resume:
        agent.load(args.resume)
        # Optimizer state carries the old param-group LR.  A fine-tuning run
        # must explicitly apply its requested LR after restoring moments.
        for optimizer in (agent.encoder_opt, agent.actor_opt, agent.critic_opt):
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        if args.explore_restart:
            # One-shot re-warm: only when resuming an EXTERNAL champion (the seed),
            # not our own latest.pt on a crash-restart (which would keep resetting
            # exploration to the start). The re-warmed offset is published to the
            # collectors by publish() below (audit).
            resuming_own_latest = (
                Path(args.resume).resolve() == (args.out / "latest.pt").resolve()
            )
            if resuming_own_latest:
                print(f"LEARNER explore-restart skipped (own latest.pt; "
                      f"offset={agent.explore_offset} preserved)", flush=True)
            else:
                agent.explore_offset = agent.train_steps
                print(f"LEARNER explore-restart offset={agent.explore_offset} "
                      "(exploration re-warmed for Stage C trench escape)", flush=True)
        print(f"LEARNER resumed {args.resume} steps={agent.train_steps}", flush=True)

    def save_atomic(path: Path) -> None:
        """Publish a complete checkpoint or leave the previous one untouched."""
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            agent.save(str(tmp))
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def publish(step: int) -> None:
        if agent.weights_finite():
            D.publish_weights(
                wdir,
                {
                    "encoder": agent.encoder.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "train_steps": agent.train_steps,
                    # carry the exploration-schedule anchor so collectors compute
                    # stddev(train_steps - explore_offset), not the floor (audit).
                    "explore_offset": agent.explore_offset,
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
            {
                **{k: str(v) for k, v in vars(args).items()},
                "streams": streams,
                "mode": "distributed",
                "started_at_unix": run_started,
            },
            indent=2,
        )
    )

    consumed: set[str] = set()
    transitions = 0
    updates = 0
    update_debt = 0.0
    best_metric_path = args.out / "best_training_score.json"
    try:
        best_training_score = float(json.loads(best_metric_path.read_text())["mean_scored"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        best_training_score = float("-inf")
    finished_returns: list[float] = []
    finished_scores: list[float] = []
    finished_collects: list[float] = []
    finished_scored: list[int] = []
    metrics_path = args.out / "metrics.jsonl"
    last_report = time.time()
    train_metrics: dict[str, float] = {}
    deadline = time.time() + args.minutes * 60.0
    rejected = 0
    stream_last_tx = np.full(streams, time.time(), dtype=np.float64)
    health_path = args.out / "stream_health.json"

    while time.time() < deadline:
        chunks = D.drain_chunks(args.root, args.num_collectors, consumed)
        new_tx = 0
        for chunk in chunks:
            a = chunk.arrays
            steps = a["reward"].shape[1]
            chunk_envs = int(a["reward"].shape[0])
            if chunk_envs > args.collector_envs:
                print(
                    f"LEARNER rejected collector={chunk.collector_id} seq={chunk.seq}: "
                    f"chunk_envs={chunk_envs} configured={args.collector_envs}",
                    flush=True,
                )
                rejected += int(a["reward"].size)
                for e in range(args.collector_envs):
                    replay.mark_boundary(chunk.collector_id * args.collector_envs + e)
                continue
            for e in range(chunk_envs):
                stream = chunk.collector_id * args.collector_envs + e
                stream_added = 0
                for t in range(steps):
                    # finite gate: a collector's PhysX can go non-finite; never
                    # let a poisoned transition into the shared replay (audit).
                    if not (
                        np.isfinite(a["proprio"][e, t]).all()
                        and np.isfinite(a["privileged"][e, t]).all()
                        and np.isfinite(a["action"][e, t]).all()
                        and np.isfinite(a["reward"][e, t])
                    ):
                        rejected += 1
                        # terminal boundary so an n-step return never bridges the
                        # gap left by the dropped step (audit).
                        replay.mark_boundary(stream)
                        continue
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
                    stream_added += 1
                # freshness stamp only on a finite add: a NaN-spewing collector is
                # still flagged stale even though its chunks keep arriving (audit).
                if stream_added:
                    stream_last_tx[stream] = time.time()
            for ep in chunk.episodes:
                finished_returns.append(ep["return"])
                finished_scores.append(ep["score_reward"])
                finished_collects.append(ep["collect_reward"])
                finished_scored.append(ep["scored"])
        transitions += new_tx

        if replay.ready(max(args.batch_size, args.seed_transitions), min_live_fraction=args.min_live_fraction):
            update_debt += args.updates_per_tx * new_tx
            done_this_tick = 0
            while update_debt >= 1.0 and done_this_tick < args.max_updates_per_tick:
                train_metrics = agent.update(replay.sample(args.batch_size))
                updates += 1
                done_this_tick += 1
                update_debt -= 1.0
                if updates % args.weight_publish_updates == 0:
                    publish(agent.train_steps)
                if (
                    args.eval_snapshot_updates > 0
                    and agent.train_steps % args.eval_snapshot_updates == 0
                    and agent.weights_finite()
                ):
                    save_atomic(eval_queue / f"step_{agent.train_steps:09d}.pt")

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
                "recent_scored_max": int(np.max(finished_scored[-40:])) if finished_scored else None,
                "episodes": len(finished_returns),
                **{k: round(v, 4) for k, v in train_metrics.items()},
            }
            print("TRAIN " + json.dumps(line), flush=True)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
            # per-stream freshness -> stream_health.json so the shell watchdog can
            # restart an alive-but-silent collector (ps-liveness alone misses it).
            now = time.time()
            stale_streams = [s for s in range(streams) if now - stream_last_tx[s] > args.stream_stall_s]
            stale_collectors = sorted({s // args.collector_envs for s in stale_streams})
            warm_rings = int(sum(len(r) > r.n_step + 1 for r in replay.rings))
            if stale_streams:
                print(f"STREAM_STALL collectors={stale_collectors} streams={stale_streams} "
                      f"warm_rings={warm_rings}/{streams} updates={updates}", flush=True)
            _hp_tmp = health_path.with_name(".stream_health.json.tmp")
            _hp_tmp.write_text(json.dumps({"wall_unix": now, "stale_collectors": stale_collectors,
                                           "stale_streams": stale_streams, "warm_rings": warm_rings,
                                           "streams": streams, "updates": updates}))
            os.replace(_hp_tmp, health_path)
            if agent.weights_finite():
                save_atomic(args.out / "latest.pt")
                recent_scored = finished_scored[-40:]
                score_metric = (
                    float(np.mean(recent_scored)) if recent_scored else float("-inf")
                )
                if score_metric > best_training_score:
                    best_training_score = score_metric
                    save_atomic(args.out / "best.pt")
                    # atomic write so a crash mid-write can't corrupt the best-score
                    # marker that gates best.pt overwrites across restarts (audit).
                    _bm_tmp = best_metric_path.with_name(".best_training_score.json.tmp")
                    _bm_tmp.write_text(
                        json.dumps(
                            {
                                "mean_scored": best_training_score,
                                "window": len(recent_scored),
                                "train_steps": agent.train_steps,
                                "wall_time": datetime.now().astimezone().isoformat(),
                            },
                            indent=2,
                        )
                    )
                    os.replace(_bm_tmp, best_metric_path)

    if agent.weights_finite():
        save_atomic(args.out / "final.pt")
    print("LEARNER_DONE " + json.dumps({"transitions": transitions, "updates": updates, "episodes": len(finished_returns)}), flush=True)


if __name__ == "__main__":
    main()
