"""Deterministic checkpoint evaluation vs random/zero baselines.

Runs E fixed-seed Stage-A episodes per policy with exploration OFF and reports
raw FUEL collected + legally scored (not just shaped return), so "did it learn
to score or only to intake" is answered directly (reviewer requirement).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "runs/drqv2_stageA_v2/final.pt"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--episode-len-s", type=float, default=20.0)
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_32.usd"))
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--policies", default="checkpoint,random,zero")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "eval_stageA.json")
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from xrc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

        results: dict[str, dict] = {}
        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=args.num_envs,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                seed=args.seed,
            )
        )
        steps_per_episode = int(args.episode_len_s * 10)

        agent = None
        to_frames = None
        if "checkpoint" in args.policies:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "train_drqv2", PROJECT_ROOT / "scripts" / "rl" / "train_drqv2.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            to_frames = module.to_policy_frames
            from xrc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

            agent = DrQV2Agent(DrQConfig())
            agent.load(args.checkpoint)
            print(f"EVAL_LOADED {args.checkpoint} steps={agent.train_steps}", flush=True)

        for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
            rng = np.random.default_rng(args.seed)
            env.rng = np.random.default_rng(args.seed)  # identical resets per policy
            env.reset_all()
            obs, *_ = env.step(np.zeros((args.num_envs, 7), np.float32))
            episode_stats: list[dict] = []
            returns = np.zeros(args.num_envs, np.float32)
            needed = args.episodes
            while len(episode_stats) < needed:
                if policy == "checkpoint":
                    frames = to_frames(obs["rgb"])
                    actions = agent.act(frames, obs["proprio"], explore=False)
                elif policy == "random":
                    actions = rng.uniform(-1, 1, (args.num_envs, 7)).astype(np.float32)
                    actions[:, 3] = 1.0
                else:  # zero: sit still, intake off
                    actions = np.zeros((args.num_envs, 7), np.float32)
                obs, rewards, dones, info = env.step(actions.astype(np.float32))
                returns += rewards
                for i in np.flatnonzero(dones):
                    terminal = info["episode_stats"][int(i)]
                    episode_stats.append(
                        {
                            "return": round(float(returns[i]), 2),
                            "scored": terminal["scored"],
                            "collected": terminal["collected"],
                            "shots_fired": terminal["shots_fired"],
                        }
                    )
                    returns[i] = 0.0
            scored = [e["scored"] for e in episode_stats[:needed]]
            collected = [e["collected"] for e in episode_stats[:needed]]
            rets = [e["return"] for e in episode_stats[:needed]]
            results[policy] = {
                "episodes": needed,
                "mean_return": round(float(np.mean(rets)), 2),
                "mean_scored": round(float(np.mean(scored)), 2),
                "max_scored": int(np.max(scored)),
                "pct_episodes_scored": round(
                    100.0 * float(np.mean([s >= 1 for s in scored])), 1
                ),
                "mean_collected": round(float(np.mean(collected)), 2),
                "max_collected": int(np.max(collected)),
                "per_episode": episode_stats[:needed],
            }
            print(f"EVAL {policy} " + json.dumps(results[policy]), flush=True)

        args.out.write_text(json.dumps(results, indent=2))
        print("EVAL_DONE " + str(args.out), flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
