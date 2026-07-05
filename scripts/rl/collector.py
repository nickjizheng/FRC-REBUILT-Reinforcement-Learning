"""Distributed DrQ-v2 collector: render one env-set on one GPU, push transitions.

Runs inference only (no learning): loads the newest actor+encoder weights the
learner publishes, steps the full-physics vision env, and drops transition
chunks onto tmpfs for the learner to drain. Launch one per GPU with
CUDA_VISIBLE_DEVICES set.
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--root", default="/dev/shm/xrc_dist")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--stage", choices=("A", "B", "C"), default="C")
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_96.usd"))
    ap.add_argument("--episode-len-s", type=float, default=90.0)
    ap.add_argument("--preload-prob", type=float, default=0.4)
    ap.add_argument("--chunk-steps", type=int, default=12)
    ap.add_argument("--weight-reload-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=600.0)
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import torch  # noqa: F401
        from xrc_rebuilt.rl import distributed as D
        from xrc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent
        from xrc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg

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
                seed=args.seed,
            )
        )
        n = args.num_envs
        obs, _, _, _ = env.step(np.zeros((n, 7), np.float32))
        frames = D.to_policy_frames(obs["rgb"])
        agent = DrQV2Agent(
            DrQConfig(
                frame_channels=frames.shape[1],
                frame_h=frames.shape[2],
                frame_w=frames.shape[3],
                proprio_dim=obs["proprio"].shape[1],
                privileged_dim=obs["privileged"].shape[1],
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
                loaded_step = step
            except Exception as exc:
                print(f"COLLECTOR{args.collector_id} weight reload skipped: {exc}", flush=True)

        maybe_reload()
        print(f"COLLECTOR{args.collector_id} READY on {agent.device}, frames={list(frames.shape[1:])}", flush=True)

        deadline = time.time() + args.minutes * 60.0
        seq = 0
        step = 0
        buf: dict[str, list] = {k: [] for k in D.FIELD_KEYS}
        ep_return = np.zeros(n, np.float32)
        ep_score = np.zeros(n, np.float32)
        ep_collect = np.zeros(n, np.float32)
        pending_eps: list[dict] = []

        while time.time() < deadline:
            actions = agent.act(frames, obs["proprio"], explore=True).astype(np.float32)
            next_obs, rewards, dones, info = env.step(actions)
            next_frames = D.to_policy_frames(next_obs["rgb"])
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
                    }
                )
                ep_return[i] = ep_score[i] = ep_collect[i] = 0.0
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
