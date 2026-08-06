"""Record the robot's route (trajectory + storage/action/score events) over
deterministic eval episodes, so the ACTUAL behavior can be drawn on a top-down
field diagram instead of inferred from score numbers.

Logs env-0 each policy step: robot x/y/yaw, storage extension (0=compact,
1=extended), cumulative scored/collected, drive+shoot action. Also dumps the
static field layout (fuel home positions, hub, trench box).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--episode-len-s", type=float, default=90.0)
    ap.add_argument("--template", default=str(PROJECT_ROOT / "assets/rl/env_template_200.usd"))
    ap.add_argument("--preload-prob", type=float, default=0.0)
    ap.add_argument("--spawn-under-trench", action="store_true")
    ap.add_argument("--label", default="model")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mask-illegal-fire", action="store_true",
                    help="match Stage-B training: an illegal fire is a no-op (does not "
                    "auto-stop the chassis); required to faithfully eval a masked policy")
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from frc_rebuilt.rl.vec_env import VecCompetitionEnv, VecEnvCfg
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_drqv2", PROJECT_ROOT / "scripts" / "rl" / "train_drqv2.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        to_frames = module.to_policy_frames
        from frc_rebuilt.rl.drqv2 import DrQConfig, DrQV2Agent

        env = VecCompetitionEnv(
            VecEnvCfg(
                num_envs=2,
                template_usd=args.template,
                cameras=True,
                episode_len_s=args.episode_len_s,
                preload_prob=args.preload_prob,
                mask_illegal_fire=args.mask_illegal_fire,
                seed=args.seed,
                spawn_under_trench=args.spawn_under_trench,
                lock_storage_extended=not args.spawn_under_trench,
            )
        )
        agent = DrQV2Agent(DrQConfig())
        agent.load(args.checkpoint)
        print(f"ROUTE_LOADED {args.checkpoint} steps={agent.train_steps}", flush=True)

        fuel_home = np.asarray(env._fuel_home)[:, :2].round(3).tolist()

        def state(i: int):
            pos, quat = env.slots[i].articulation.get_world_pose()
            w, x, y, z = (float(v) for v in quat)
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            ext = float(getattr(env.slots[i].controller, "container_extension", 1.0))
            return float(pos[0]), float(pos[1]), yaw, ext

        episodes: list[dict] = []
        route: list[list] = []
        env.reset_all()
        obs, *_ = env.step(np.zeros((2, 7), np.float32))
        while len(episodes) < args.episodes:
            frames = to_frames(obs["rgb"])
            actions = agent.act(frames, obs["proprio"], explore=False)
            x, y, yaw, ext = state(0)
            s = env.slots[0]
            route.append(
                [round(x, 3), round(y, 3), round(yaw, 3), round(ext, 2),
                 int(s.score_seen), int(s.collected_seen),
                 round(float(actions[0, 0]), 2), round(float(actions[0, 1]), 2),
                 round(float(actions[0, 2]), 2), int(float(actions[0, 5]) > 0.25)]
            )
            obs, rewards, dones, info = env.step(actions.astype(np.float32))
            if bool(dones[0]):
                st = info["episode_stats"][0]
                episodes.append({
                    "scored": st["scored"], "collected": st["collected"],
                    "shots_fired": st.get("shots_fired", 0), "route": route,
                })
                route = []

        out = {
            "label": args.label,
            "checkpoint": args.checkpoint,
            "train_steps": agent.train_steps,
            "spawn_under_trench": bool(args.spawn_under_trench),
            "field": {
                "fuel_home": fuel_home,
                "hub": [-0.0199, -3.6874],
                "trench": {"x_min": 2.7189, "x_max": 3.9975,
                           "y_neutral": -3.05, "y_alliance": -4.24,
                           "start": [3.3582, -3.8850]},
            },
            "route_cols": ["x", "y", "yaw", "storage_ext", "score_seen",
                           "collected_seen", "drive_x", "drive_y", "turn", "shoot"],
            "episodes": episodes,
        }
        args.out.write_text(json.dumps(out))
        print(f"ROUTE_DONE {args.out} eps={len(episodes)} "
              f"scored={[e['scored'] for e in episodes]}", flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
