"""Stage-A DrQ-v2 training on the vectorized full-physics competition env.

The converged first baseline (RL_BRAINSTORM.md): off-policy DrQ-v2, pixel+
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
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


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
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "drqv2_stageA")
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import torch  # noqa: F401  (fail fast if the RL stack is broken)

        from xrc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from xrc_rebuilt.rl.replay import PerEnvReplay
        from xrc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        args.out.mkdir(parents=True, exist_ok=True)
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
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
            )
        )
        replay = PerEnvReplay(
            num_envs=n,
            capacity_per_env=max(1000, args.replay_capacity // n),
            seed=11,
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

        current = {
            "frames": frames,
            "proprio": obs["proprio"].copy(),
            "privileged": obs["privileged"].copy(),
        }
        while time.time() < deadline:
            if transitions < args.seed_transitions:
                actions = np.random.uniform(-1, 1, (n, 7)).astype(np.float32)
                actions[:, 3] = 1.0  # keep intake on while seeding
            else:
                actions = agent.act(
                    current["frames"], current["proprio"], explore=True
                ).astype(np.float32)
            obs, rewards, dones, info = env.step(actions)
            next_frames = to_policy_frames(obs["rgb"])
            for i in range(n):
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
            current = {
                "frames": next_frames,
                "proprio": obs["proprio"].copy(),
                "privileged": obs["privileged"].copy(),
            }
            transitions += n

            if replay.ready(max(args.batch_size, args.seed_transitions)):
                update_debt += args.updates_per_tx * n
                while update_debt >= 1.0:
                    train_metrics = agent.update(replay.sample(args.batch_size))
                    updates += 1
                    update_debt -= 1.0

            if time.time() - last_report >= report_every_s:
                last_report = time.time()
                recent = finished_returns[-20:]
                line = {
                    "transitions": transitions,
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
                    **{k: round(v, 4) for k, v in train_metrics.items()},
                }
                print("TRAIN " + json.dumps(line), flush=True)
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line) + "\n")
                agent.save(str(args.out / "latest.pt"))

        agent.save(str(args.out / "final.pt"))
        summary = {
            "transitions": transitions,
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
